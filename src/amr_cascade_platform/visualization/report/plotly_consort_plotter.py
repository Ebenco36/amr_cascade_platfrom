"""CONSORT-style data flow diagram: result rows to episodes to pairs to validated patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from amr_cascade_platform.visualization.report.cohort_rules import CohortRules
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_BOX_FILL   = "#EBF3FB"
_BOX_LINE   = "#2C6FAC"
_EXCL_FILL  = "#FEF0E7"
_EXCL_LINE  = "#D35400"
_FINAL_FILL = "#EAF7EA"
_FINAL_LINE = "#27AE60"
_TEXT_DARK  = "#1a1a2e"
_ARROW      = "#555555"
_BG         = "white"  # static PNG/PDF exports composite transparency onto black in some viewers/renderers -- explicit white matches every other figure module in this codebase


def _fmt(n: int | None) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


class PlotlyConsortPlotter:
    """
    CONSORT-style cohort data flow diagram.

    Row-level counts come from ``flow_table`` (ManuscriptTableBuilder.build_data_quality_flow_table:
    site, stage, row_count). Pair-level counts come from the cascade outputs themselves:
    ``cotesting_pairs`` (pairs the co-testing screen removed), ``escalation_results`` (ordered
    pairs with both upstream branches, evaluated against the support thresholds), and
    ``edge_report`` (retained candidates and their validation labels). Each count is taken from
    the stage that produced it rather than inferred from a later one, so a pair removed by an
    earlier rule is never reported as passing it. Missing inputs render as "—".
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int, rules: CohortRules) -> None:
        self._exporter = exporter
        self._template = template
        self._width    = width
        self._height   = height
        self._rules    = rules

    # ── Public API ────────────────────────────────────────────────────────

    def export(
        self,
        flow_table: pd.DataFrame,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        escalation_results: pd.DataFrame | None = None,
        cotesting_pairs: pd.DataFrame | None = None,
    ) -> dict[str, Path]:
        stats = self.collect_stats(flow_table, edge_report, escalation_results, cotesting_pairs)
        text = self._texts(stats)
        fig = self._build(stats, text)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(stats, text, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Stats extraction ──────────────────────────────────────────────────

    def collect_stats(
        self,
        flow_table: pd.DataFrame,
        edge_report: pd.DataFrame,
        escalation_results: pd.DataFrame | None = None,
        cotesting_pairs: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        rules = self._rules

        def stage_total(stage: str) -> int | None:
            if flow_table is None or flow_table.empty or "stage" not in flow_table.columns:
                return None
            rows = flow_table.loc[flow_table["stage"].eq(stage), "row_count"]
            if rows.empty:
                return None
            return int(pd.to_numeric(rows, errors="coerce").sum())

        evaluated = below_support = no_event = None
        if escalation_results is not None and not escalation_results.empty:
            counts = escalation_results[["total_support_n", "positive_support_n", "negative_support_n"]].apply(pd.to_numeric, errors="coerce")
            passing = (
                counts["total_support_n"].ge(rules.min_total_support)
                & counts["positive_support_n"].ge(rules.min_result_support)
                & counts["negative_support_n"].ge(rules.min_result_support)
            )
            events = (
                pd.to_numeric(escalation_results["positive_tested_n"], errors="coerce").fillna(0)
                + pd.to_numeric(escalation_results["negative_tested_n"], errors="coerce").fillna(0)
            )
            evaluated = int(len(escalation_results))
            below_support = int((~passing).sum())
            no_event = int((passing & events.lt(rules.min_downstream_events)).sum())

        retained = robust = supported = not_validated = None
        if edge_report is not None and not edge_report.empty:
            report = edge_report
            if "retained_edge" in report.columns:
                report = report.loc[report["retained_edge"].fillna(False).astype(bool)]
            retained = int(len(report))
            if "validation_status" in report.columns:
                status = report["validation_status"].astype(str)
                robust = int(status.eq("robust").sum())
                supported = int(status.eq("supported").sum())
                not_validated = int(status.isin(["mixed", "insufficient"]).sum())

        sites = (
            sorted(flow_table["site"].dropna().astype(str).unique().tolist())
            if flow_table is not None and not flow_table.empty and "site" in flow_table.columns
            else []
        )
        return {
            "raw_ast": stage_total("raw_ast_rows"),
            "episodes": stage_total("culture_episodes"),
            "pair_rows": stage_total("binary_upstream_pair_rows"),
            "screened_out": int(len(cotesting_pairs)) if cotesting_pairs is not None else None,
            "evaluated": evaluated,
            "below_support": below_support,
            "no_event": no_event,
            "retained": retained,
            "robust": robust,
            "supported": supported,
            "validated": robust + supported if robust is not None and supported is not None else None,
            "not_validated": not_validated,
            "sites": sites,
        }

    # ── Shared wording ────────────────────────────────────────────────────

    def _texts(self, s: dict[str, Any]) -> dict[str, Any]:
        r = self._rules
        return {
            "boxes": [
                ("Raw AST result rows", f"N = {_fmt(s['raw_ast'])} rows", ", ".join(s["sites"]) if s["sites"] else "multi-site source archives"),
                ("Culture episodes", f"N = {_fmt(s['episodes'])} episodes", "One per culture order of the target organism"),
                ("Eligible episode-pair rows", f"N = {_fmt(s['pair_rows'])} rows", "Upstream resistant or susceptible; downstream eligible"),
                ("Ordered antibiotic pairs evaluated", f"N = {_fmt(s['evaluated'])} pairs", "Resistant and susceptible upstream branches both present"),
                ("Retained candidate pairs", f"N = {_fmt(s['retained'])} pairs", "Checked by permutation, bootstrap, and replication"),
            ],
            "exclusions": [
                ["Not in the observed layer: other organisms,", "results other than R/S/I, ESBL confirmation", "assays, exact duplicate rows"],
                ["Downstream antibiotic intrinsically resistant or", "unavailable at that site and era;", "intermediate upstream results"],
                [f"Removed by the co-testing screen: {_fmt(s['screened_out'])} pairs", f"(both co-observation probabilities ≥ {r.cotesting_threshold:.2f},", f"≥ {r.min_total_support} rows in each direction)"],
                [f"Below support thresholds: {_fmt(s['below_support'])} pairs", f"(n < {r.min_total_support}, n_R < {r.min_result_support}, or n_S < {r.min_result_support})", self._event_rule_line(s)],
                [f"Not validated (mixed or insufficient): {_fmt(s['not_validated'])}", f"(FDR-adjusted permutation q > {r.q_threshold:g}", f"or bootstrap sign stability < {r.stability_threshold:.2f})"],
            ],
            "robust": ("Robust", f"N = {_fmt(s['robust'])}", "All three checks passed"),
            "supported": ("Supported", f"N = {_fmt(s['supported'])}", "Permutation and bootstrap"),
            "total": f"Total validated: {_fmt(s['validated'])} pairs" if s["validated"] is not None else None,
        }

    def _event_rule_line(self, s: dict[str, Any]) -> str:
        minimum = self._rules.min_downstream_events
        rule = "No downstream-observed episode" if minimum <= 1 else f"Fewer than {minimum} downstream-observed episodes"
        return f"{rule}: {_fmt(s['no_event'])} pairs"

    # ── Interactive rendering ─────────────────────────────────────────────

    def _build(self, s: dict[str, Any], text: dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="Study Cohort Data Flow", x=0.5, font=dict(size=18, color=_TEXT_DARK)),
            xaxis=dict(range=[0, 10], visible=False),
            # Top-to-bottom flow: an explicit reversed range (Plotly ignores autorange="reversed"
            # when a range is also given).
            yaxis=dict(range=[10.4, 0], visible=False),
            margin=dict(t=60, b=30, l=20, r=20),
            showlegend=False,
        )
        cx, bw, bh = 3.4, 3.1, 0.52
        ys = [0.75, 2.3, 3.85, 5.4, 6.95, 9.05]

        def box(x: float, y: float, title: str, line1: str, line2: str, fill: str = _BOX_FILL, line_col: str = _BOX_LINE, half_width: float = bw) -> None:
            fig.add_shape(type="rect", x0=x - half_width, y0=y - bh, x1=x + half_width, y1=y + bh, xref="x", yref="y",
                          fillcolor=fill, line=dict(color=line_col, width=1.8), layer="below")
            fig.add_annotation(x=x, y=y, xref="x", yref="y", showarrow=False, align="center",
                               text=f"<b>{title}</b><br>{line1}<br><span style='font-size:11px;color:#555'>{line2}</span>",
                               font=dict(size=13, color=_TEXT_DARK), bgcolor="rgba(0,0,0,0)")

        def arrow(y_from: float, y_to: float, x: float = cx) -> None:
            fig.add_annotation(x=x, y=y_to - bh - 0.04, ax=x, ay=y_from + bh + 0.04, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8, arrowcolor=_ARROW, text="")

        exclusion_left, exclusion_right, exclusion_half_height = 6.75, 9.9, 0.42

        def exclusion(y: float, lines: list[str]) -> None:
            fig.add_shape(type="line", x0=cx, y0=y, x1=exclusion_left, y1=y, xref="x", yref="y",
                          line=dict(color=_ARROW, width=1.4, dash="dot"))
            fig.add_shape(type="rect", x0=exclusion_left, y0=y - exclusion_half_height, x1=exclusion_right, y1=y + exclusion_half_height,
                          xref="x", yref="y", fillcolor=_EXCL_FILL, line=dict(color=_EXCL_LINE, width=1.2), layer="below")
            fig.add_annotation(x=(exclusion_left + exclusion_right) / 2, y=y, xref="x", yref="y", showarrow=False, align="center",
                               text="<br>".join(line.replace("<", "&lt;") for line in lines),
                               font=dict(size=11, color="#6B2C00"), bgcolor="rgba(0,0,0,0)")

        for index, (title, line1, line2) in enumerate(text["boxes"]):
            box(cx, ys[index], title, line1, line2)
        for index in range(len(text["boxes"]) - 1):
            arrow(ys[index], ys[index + 1])
        for index, lines in enumerate(text["exclusions"]):
            exclusion((ys[index] + ys[index + 1]) / 2, lines)

        box(cx - 1.65, ys[5], *text["robust"], fill=_FINAL_FILL, line_col=_FINAL_LINE, half_width=1.45)
        box(cx + 1.65, ys[5], *text["supported"], fill=_FINAL_FILL, line_col=_FINAL_LINE, half_width=1.45)
        bracket_y = ys[5] - bh - 0.3
        fig.add_shape(type="line", x0=cx, y0=ys[4] + bh, x1=cx, y1=bracket_y, xref="x", yref="y", line=dict(color=_ARROW, width=1.8))
        for x in (cx - 1.65, cx + 1.65):
            fig.add_shape(type="line", x0=cx, y0=bracket_y, x1=x, y1=bracket_y, xref="x", yref="y", line=dict(color=_ARROW, width=1.8))
            arrow(bracket_y - bh - 0.02, ys[5], x=x)
        if text["total"]:
            fig.add_annotation(x=cx, y=ys[5] + bh + 0.35, xref="x", yref="y", showarrow=False,
                               text=f"<b>{text['total']}</b>", font=dict(size=14, color=_FINAL_LINE))
        return fig

    # ── Publication static rendering ─────────────────────────────────────

    def _write_static(self, s: dict[str, Any], text: dict[str, Any], fmt: str, path: Path) -> None:
        """Draw the static cohort-flow diagram for PNG/PDF/SVG export.

        The interactive HTML remains Plotly. Static outputs use Matplotlib
        because annotation-heavy Plotly diagrams can collapse text under
        Kaleido/generic fallback in constrained runtimes.
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.99, "Study Cohort Data Flow", ha="center", va="top", fontsize=18, fontweight="bold", color=_TEXT_DARK)

        pad = 0.012
        box_x, box_w, box_h = 0.04, 0.50, 0.095
        centre = box_x + box_w / 2
        ys = [0.895, 0.75, 0.605, 0.46, 0.315]

        def draw_box(x: float, y: float, w: float, title: str, line1: str, line2: str, fill: str, edge: str) -> None:
            ax.add_patch(FancyBboxPatch((x, y - box_h / 2), w, box_h, boxstyle=f"round,pad={pad},rounding_size=0.015",
                                        linewidth=1.8, edgecolor=edge, facecolor=fill))
            ax.text(x + 0.016, y + 0.024, title, ha="left", va="center", fontsize=12, fontweight="bold", color=_TEXT_DARK)
            ax.text(x + 0.016, y - 0.003, line1, ha="left", va="center", fontsize=11, color=_TEXT_DARK)
            ax.text(x + 0.016, y - 0.029, line2, ha="left", va="center", fontsize=9, color="#555555")

        def arrow(x: float, y_from: float, y_to: float) -> None:
            ax.annotate("", xy=(x, y_to), xytext=(x, y_from), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=_ARROW))

        top = lambda y: y + box_h / 2 + pad  # noqa: E731
        bottom = lambda y: y - box_h / 2 - pad  # noqa: E731

        for index, (title, line1, line2) in enumerate(text["boxes"]):
            draw_box(box_x, ys[index], box_w, title, line1, line2, _BOX_FILL, _BOX_LINE)
            if index < len(ys) - 1:
                arrow(centre, bottom(ys[index]) - 0.004, top(ys[index + 1]) + 0.004)

        final_y = 0.10
        final_w = 0.225
        final_x = (box_x, box_x + box_w - final_w)
        junction = top(final_y) + 0.04
        exclusion_ys = [(ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1)] + [(bottom(ys[-1]) + junction) / 2]
        exclusion_x, exclusion_w, exclusion_h = 0.585, 0.395, 0.078
        for y, lines in zip(exclusion_ys, text["exclusions"]):
            ax.plot([centre, exclusion_x - pad], [y, y], color=_ARROW, lw=1.2, linestyle=":")
            ax.add_patch(FancyBboxPatch((exclusion_x, y - exclusion_h / 2), exclusion_w, exclusion_h,
                                        boxstyle=f"round,pad=0.008,rounding_size=0.012", linewidth=1.3,
                                        edgecolor=_EXCL_LINE, facecolor=_EXCL_FILL))
            for offset, line in zip((0.021, 0.0, -0.021), lines):
                ax.text(exclusion_x + 0.012, y + offset, line, ha="left", va="center", fontsize=8.8, color="#6B2C00")

        ax.plot([centre, centre], [bottom(ys[-1]) - 0.004, junction], color=_ARROW, lw=1.8)
        centres = [x + final_w / 2 for x in final_x]
        ax.plot(centres, [junction, junction], color=_ARROW, lw=1.8)
        for x, key in zip(final_x, ("robust", "supported")):
            arrow(x + final_w / 2, junction, top(final_y) + 0.004)
            draw_box(x, final_y, final_w, *text[key], _FINAL_FILL, _FINAL_LINE)
        if text["total"]:
            ax.text(centre, bottom(final_y) - 0.012, text["total"], ha="center", va="top", fontsize=10.5, color=_FINAL_LINE, fontweight="bold")

        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
