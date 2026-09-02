"""CONSORT-style data flow diagram — patient/episode/pair cohort attrition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_BOX_FILL   = "#EBF3FB"
_BOX_LINE   = "#2C6FAC"
_EXCL_FILL  = "#FEF0E7"
_EXCL_LINE  = "#D35400"
_FINAL_FILL = "#EAF7EA"
_FINAL_LINE = "#27AE60"
_TEXT_DARK  = "#1a1a2e"
_ARROW      = "#555555"
_BG         = "rgba(0,0,0,0)"


def _fmt(n: int | None) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


class PlotlyConsortPlotter:
    """
    CONSORT-style cohort data flow diagram.

    Derives counts from:
      ``flow_table``    — output of ManuscriptTableBuilder.build_data_quality_flow_table()
                          Columns: site, stage, row_count
                          Stages: raw_ast_rows, culture_episodes, observed_episode_drug_rows,
                                  eligible_episode_drug_rows, eligible_directed_pair_rows
      ``edge_report``   — cascade analysis output; supplies candidate / validated counts.

    Missing counts render as "—" so the figure is always displayable.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width    = width
        self._height   = height

    # ── Public API ────────────────────────────────────────────────────────

    def export(
        self,
        flow_table:  pd.DataFrame,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        stats = self._collect_stats(flow_table, edge_report)
        fig   = self._build(stats)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(stats, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Stats extraction ──────────────────────────────────────────────────

    @staticmethod
    def _collect_stats(
        flow_table:  pd.DataFrame,
        edge_report: pd.DataFrame,
    ) -> dict[str, Any]:

        def _sum_stage(stage: str) -> int | None:
            if flow_table.empty or "stage" not in flow_table.columns:
                return None
            sub = flow_table[flow_table["stage"].eq(stage)]
            if sub.empty or "row_count" not in sub.columns:
                return None
            val = pd.to_numeric(sub["row_count"], errors="coerce").sum()
            return int(val) if pd.notna(val) else None

        raw_ast       = _sum_stage("raw_ast_rows")
        episodes      = _sum_stage("culture_episodes")
        eligible_rows = _sum_stage("eligible_directed_pair_rows")

        candidates = validated = robust = supported = excluded_pbi = None

        if not edge_report.empty:
            if "escalation_ratio" in edge_report.columns:
                mask = edge_report["escalation_ratio"].notna()
                if "passes_support_threshold" in edge_report.columns:
                    mask = mask & edge_report["passes_support_threshold"].eq(True)
                candidates = int(mask.sum())
            if "validation_status" in edge_report.columns:
                vs = edge_report["validation_status"]
                robust    = int(vs.eq("robust").sum())
                supported = int(vs.eq("supported").sum())
                validated = (robust or 0) + (supported or 0)
            if "panel_bundling_index" in edge_report.columns:
                import numpy as _np
                pbi = pd.to_numeric(edge_report["panel_bundling_index"], errors="coerce")
                excluded_pbi = int((pbi >= 0.95).sum())

        sites = (
            flow_table["site"].dropna().unique().tolist()
            if not flow_table.empty and "site" in flow_table.columns
            else []
        )

        return {
            "raw_ast":       raw_ast,
            "episodes":      episodes,
            "eligible_rows": eligible_rows,
            "candidates":    candidates,
            "validated":     validated,
            "robust":        robust,
            "supported":     supported,
            "excluded_pbi":  excluded_pbi,
            "n_sites":       len(sites),
            "sites":         sites,
        }

    # ── Figure construction ───────────────────────────────────────────────

    def _build(self, s: dict[str, Any]) -> go.Figure:
        fig = go.Figure()

        W, H = self._width, self._height
        # Work in pixel coordinates, normalised to 0-1 via paper refs
        # Use a virtual canvas 0-10 × 0-10 with axis range set accordingly.

        fig.update_layout(
            template=self._template,
            width=W,
            height=H,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="Study Cohort Data Flow",
                x=0.5,
                font=dict(size=18, color=_TEXT_DARK),
            ),
            xaxis=dict(range=[0, 10], visible=False),
            yaxis=dict(range=[0, 10], visible=False, autorange="reversed"),
            margin=dict(t=60, b=30, l=20, r=20),
            showlegend=False,
        )

        # ── Layout constants ─────────────────────────────────────────────
        CX   = 5.0   # centre x of main flow
        BW   = 4.0   # box half-width
        BH   = 0.55  # box half-height
        EX   = 8.6   # x-centre of exclusion side-boxes
        EBW  = 1.3
        GAP  = 1.55  # vertical spacing between boxes

        # Y positions (top-to-bottom)
        Y = [1.0, 2.55, 4.1, 5.65, 7.45]
        # Y[0] = raw records, [1] = episodes, [2] = eligible pairs,
        # [3] = candidates, [4] = validated (split into robust/supported)

        # ── Helper: add a box ────────────────────────────────────────────
        def box(cx, cy, text,
                fill=_BOX_FILL, line_col=_BOX_LINE,
                bw=BW, bh=BH, fontsize=14):
            fig.add_shape(
                type="rect",
                x0=cx - bw, y0=cy - bh, x1=cx + bw, y1=cy + bh,
                xref="x", yref="y",
                fillcolor=fill, line=dict(color=line_col, width=1.8),
                layer="below",
            )
            fig.add_annotation(
                x=cx, y=cy,
                xref="x", yref="y",
                text=text,
                showarrow=False,
                font=dict(size=fontsize, color=_TEXT_DARK),
                align="center",
                bgcolor="rgba(0,0,0,0)",
            )

        # ── Helper: vertical arrow ───────────────────────────────────────
        def arrow(y_from, y_to, cx=CX):
            fig.add_annotation(
                x=cx, y=y_to - BH - 0.04,
                ax=cx, ay=y_from + BH + 0.04,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
                arrowcolor=_ARROW,
                text="",
            )

        # ── Helper: side exclusion box with connecting line ──────────────
        def excl(from_y, at_y, text):
            # Horizontal connector: from main flow to exclusion box
            mid_y = (from_y + at_y) / 2
            fig.add_shape(
                type="line",
                x0=CX + BW, y0=mid_y,
                x1=EX - EBW - 0.05, y1=mid_y,
                xref="x", yref="y",
                line=dict(color=_ARROW, width=1.4, dash="dot"),
            )
            fig.add_annotation(
                x=EX, y=at_y,
                xref="x", yref="y",
                text=text,
                showarrow=False,
                font=dict(size=13, color="#6B2C00"),
                align="center",
                bgcolor=_EXCL_FILL,
                bordercolor=_EXCL_LINE,
                borderwidth=1.2,
                borderpad=5,
            )

        # ── Box 1: Raw AST Records ───────────────────────────────────────
        sites_str = (
            f"({', '.join(s['sites'][:3])})" if s["sites"] else "(multi-site)"
        )
        box(
            CX, Y[0],
            f"<b>Raw AST Culture Records</b><br>"
            f"N = {_fmt(s['raw_ast'])} rows<br>"
            f"<span style='font-size:12px;color:#555'>{sites_str}</span>",
            fontsize=14,
        )

        # ── Arrow 1→2 ────────────────────────────────────────────────────
        arrow(Y[0], Y[1])
        excl(Y[0], (Y[0] + Y[1]) / 2,
             "Excluded: repeated specimens,<br>non-target organisms")

        # ── Box 2: Culture Episodes ──────────────────────────────────────
        box(
            CX, Y[1],
            f"<b>Culture Episodes Constructed</b><br>"
            f"N = {_fmt(s['episodes'])} episodes<br>"
            f"<span style='font-size:12px;color:#555'>First-per-admission, organism-level dedup</span>",
            fontsize=14,
        )

        # ── Arrow 2→3 ────────────────────────────────────────────────────
        arrow(Y[1], Y[2])
        excl(Y[1], (Y[1] + Y[2]) / 2,
             "Excluded: intrinsic resistance pairs,<br>missing antibiotic data")

        # ── Box 3: Eligible Directed Pairs ───────────────────────────────
        box(
            CX, Y[2],
            f"<b>Eligible Directed Antibiotic Pair Rows</b><br>"
            f"N = {_fmt(s['eligible_rows'])} episode-pair rows<br>"
            f"<span style='font-size:12px;color:#555'>Upstream tested → downstream observable</span>",
            fontsize=14,
        )

        # ── Arrow 3→4 ────────────────────────────────────────────────────
        arrow(Y[2], Y[3])
        pbi_note = (
            f"Panel bundles excluded (PBI ≥ 0.95): {_fmt(s['excluded_pbi'])}<br>"
            if s["excluded_pbi"] is not None else ""
        )
        excl(Y[2], (Y[2] + Y[3]) / 2,
             f"Excluded: below support threshold<br>"
             f"(n&lt;25, n_R&lt;5, or n_S&lt;5)<br>"
             f"{pbi_note}")

        # ── Box 4: Candidate Pairs ───────────────────────────────────────
        box(
            CX, Y[3],
            f"<b>Candidate Pairs (Support Criteria Met)</b><br>"
            f"N = {_fmt(s['candidates'])} pairs<br>"
            f"<span style='font-size:12px;color:#555'>"
            f"Permutation → Bootstrap → Site/temporal replication</span>",
            fontsize=14,
        )

        # ── Arrow 4→5 ────────────────────────────────────────────────────
        arrow(Y[3], Y[4])
        excl(Y[3], (Y[3] + Y[4]) / 2,
             "Excluded: failed permutation (p ≥ 0.05),<br>"
             "bootstrap instability (S &lt; 0.80),<br>"
             "or insufficient site/temporal replication")

        # ── Box 5L: Robust ───────────────────────────────────────────────
        box(
            3.2, Y[4],
            f"<b>Robust</b><br>N = {_fmt(s['robust'])}<br>"
            f"<span style='font-size:12px;color:#1a6b2a'>All 3 tests pass</span>",
            fill=_FINAL_FILL, line_col=_FINAL_LINE,
            bw=1.8, fontsize=13,
        )

        # ── Box 5R: Supported ────────────────────────────────────────────
        box(
            6.8, Y[4],
            f"<b>Supported</b><br>N = {_fmt(s['supported'])}<br>"
            f"<span style='font-size:12px;color:#1a6b2a'>"
            f"Permutation + Bootstrap</span>",
            fill=_FINAL_FILL, line_col=_FINAL_LINE,
            bw=1.8, fontsize=13,
        )

        # ── Bracket connecting box 5L and 5R ─────────────────────────────
        bracket_y = Y[4] - BH - 0.25
        fig.add_shape(
            type="line",
            x0=CX, y0=Y[3] + BH + 0.55,
            x1=CX, y1=bracket_y,
            xref="x", yref="y",
            line=dict(color=_ARROW, width=1.8),
        )
        for tx in (3.2, 6.8):
            fig.add_shape(
                type="line",
                x0=CX, y0=bracket_y,
                x1=tx, y1=bracket_y,
                xref="x", yref="y",
                line=dict(color=_ARROW, width=1.8),
            )
            fig.add_annotation(
                x=tx, y=Y[4] - BH - 0.08,
                ax=tx, ay=bracket_y + 0.02,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3,
                arrowsize=1.1, arrowwidth=1.6,
                arrowcolor=_ARROW, text="",
            )

        # ── Total validated label ─────────────────────────────────────────
        if s["validated"] is not None:
            fig.add_annotation(
                x=CX, y=Y[4] + BH + 0.32,
                xref="x", yref="y",
                text=f"<b>Total validated: {_fmt(s['validated'])} pairs</b>",
                showarrow=False,
                font=dict(size=14, color=_FINAL_LINE),
                align="center",
            )

        return fig

    # ── Publication static rendering ─────────────────────────────────────

    def _write_static(self, s: dict[str, Any], fmt: str, path: Path) -> None:
        """Draw a clean static cohort-flow diagram for PNG/PDF/SVG export.

        The interactive HTML remains Plotly. Static outputs use Matplotlib
        because annotation-heavy Plotly diagrams can collapse text under
        Kaleido/generic fallback in constrained runtimes.
        """

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(
            0.5,
            0.965,
            "Study Cohort Data Flow",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=_TEXT_DARK,
        )

        main_boxes = [
            (
                0.84,
                "Raw AST culture records",
                f"N = {_fmt(s['raw_ast'])} rows",
                ", ".join(s["sites"][:3]) if s["sites"] else "multi-site source archives",
                _BOX_FILL,
                _BOX_LINE,
            ),
            (
                0.66,
                "Culture episodes constructed",
                f"N = {_fmt(s['episodes'])} episodes",
                "Configured organism-level episode key",
                _BOX_FILL,
                _BOX_LINE,
            ),
            (
                0.48,
                "Eligible directed pair rows",
                f"N = {_fmt(s['eligible_rows'])} episode-pair rows",
                "Upstream observed; downstream operationally eligible",
                _BOX_FILL,
                _BOX_LINE,
            ),
            (
                0.30,
                "Candidate pairs",
                f"N = {_fmt(s['candidates'])} support-passing pairs",
                "Support screen and panel-bundling filter passed",
                _BOX_FILL,
                _BOX_LINE,
            ),
        ]

        box_x = 0.18
        box_w = 0.46
        box_h = 0.105

        def draw_box(x: float, y: float, w: float, h: float, title: str, line1: str, line2: str, fill: str, edge: str) -> None:
            patch = FancyBboxPatch(
                (x, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.018",
                linewidth=1.8,
                edgecolor=edge,
                facecolor=fill,
            )
            ax.add_patch(patch)
            ax.text(x + 0.025, y + 0.026, title, ha="left", va="center", fontsize=12.5, fontweight="bold", color=_TEXT_DARK)
            ax.text(x + 0.025, y - 0.004, line1, ha="left", va="center", fontsize=11.5, color=_TEXT_DARK)
            ax.text(x + 0.025, y - 0.032, line2, ha="left", va="center", fontsize=9.5, color="#555555")

        for idx, (y, title, line1, line2, fill, edge) in enumerate(main_boxes):
            draw_box(box_x, y, box_w, box_h, title, line1, line2, fill, edge)
            if idx < len(main_boxes) - 1:
                y_next = main_boxes[idx + 1][0]
                ax.annotate(
                    "",
                    xy=(box_x + box_w / 2, y_next + box_h / 2 + 0.008),
                    xytext=(box_x + box_w / 2, y - box_h / 2 - 0.008),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=_ARROW),
                )

        exclusions = [
            (0.75, "Deduplication and target-cohort restrictions"),
            (0.57, "Intrinsic resistance and unavailable opportunities removed"),
            (
                0.39,
                "Sparse pairs removed; near-deterministic panel bundles excluded"
                + (f" (PBI >= 0.95: {_fmt(s['excluded_pbi'])})" if s["excluded_pbi"] is not None else ""),
            ),
        ]
        for y, label in exclusions:
            ax.plot([box_x + box_w, 0.72], [y, y], color=_ARROW, lw=1.2, linestyle=":")
            draw_box(0.73, y, 0.22, 0.082, "Excluded / filtered", label, "", _EXCL_FILL, _EXCL_LINE)

        final_y = 0.12
        ax.annotate(
            "",
            xy=(box_x + box_w / 2, final_y + 0.07),
            xytext=(box_x + box_w / 2, main_boxes[-1][0] - box_h / 2 - 0.01),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=_ARROW),
        )
        draw_box(
            0.14,
            final_y,
            0.25,
            0.095,
            "Robust edges",
            f"N = {_fmt(s['robust'])}",
            "Permutation + bootstrap + replication",
            _FINAL_FILL,
            _FINAL_LINE,
        )
        draw_box(
            0.43,
            final_y,
            0.25,
            0.095,
            "Supported edges",
            f"N = {_fmt(s['supported'])}",
            "Permutation + bootstrap",
            _FINAL_FILL,
            _FINAL_LINE,
        )
        if s["validated"] is not None:
            ax.text(
                0.41,
                0.025,
                f"Primary interpretation carries robust and supported edges only: {_fmt(s['validated'])} validated pairs",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=_FINAL_LINE,
                fontweight="bold",
            )

        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
