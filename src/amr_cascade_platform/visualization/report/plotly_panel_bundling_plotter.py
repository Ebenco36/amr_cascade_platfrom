"""Panel bundling index chart: which pairs the co-testing screen removed, and what was retained."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.cohort_rules import CohortRules
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_KEPT_COLOR = "rgba(31,119,180,0.80)"
_REMOVED_COLOR = "rgba(214,39,40,0.80)"
_GRID = "#E8EEF5"


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper", font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyPanelBundlingPlotter:
    """
    Two-panel co-testing screen diagnostic.

    Panel A: PBI distribution over every ordered pair the co-testing screen assessed (both
        directions observed), read from ``cotesting_probabilities``. Red: removed by the screen
        (both co-observation probabilities at or above the threshold, with the minimum number
        of rows in each direction). Blue: kept. A kept pair can lie above the threshold only
        when one direction has fewer rows than the minimum; the panel title counts these.
    Panel B: PBI against escalation ratio for the retained candidate pairs (``edge_report``).
        Removed pairs have no escalation ratio, because the screen runs before estimation.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int, rules: CohortRules) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._rules = rules

    def screened_pairs(self, cotesting_probabilities: pd.DataFrame) -> pd.DataFrame:
        """Assessed pairs with their PBI and whether the screen removed them."""
        columns = ["p_downstream_given_upstream", "p_upstream_given_downstream", "support_n", "reverse_support_n"]
        if cotesting_probabilities is None or cotesting_probabilities.empty or not set(columns) <= set(cotesting_probabilities.columns):
            return pd.DataFrame(columns=["upstream_antibiotic", "downstream_antibiotic", "panel_bundling_index", "removed", "exempt"])
        frame = cotesting_probabilities.copy()
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        frame["panel_bundling_index"] = values[columns[:2]].min(axis=1, skipna=False)
        threshold, minimum = self._rules.cotesting_threshold, self._rules.min_total_support
        frame["removed"] = (
            values[columns[0]].ge(threshold)
            & values[columns[1]].ge(threshold)
            & values["support_n"].ge(minimum)
            & values["reverse_support_n"].ge(minimum)
        )
        frame["exempt"] = frame["panel_bundling_index"].ge(threshold) & ~frame["removed"]
        return frame.loc[frame["panel_bundling_index"].notna()].reset_index(drop=True)

    def export(
        self,
        cotesting_probabilities: pd.DataFrame,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        screened = self.screened_pairs(cotesting_probabilities)
        retained = self._retained(edge_report)
        if screened.empty and retained.empty:
            fig = _empty(self._template, self._width, self._height,
                         "Panel bundling chart unavailable: no co-testing probabilities or retained pairs")
            return self._exporter.write(fig, output_stem, formats)

        titles = self._titles(screened)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>A</b>   {titles['a']}", f"<b>B</b>   {titles['b']}"), horizontal_spacing=0.12)
        # make_subplots's auto-generated subplot-title annotations carry no
        # explicit font size, so the print-export font scale (which only
        # multiplies sizes already present in the figure JSON) never touches
        # them -- give them one explicitly so they participate in it too.
        for annotation in fig.layout.annotations:
            annotation.font = {"size": 11}
        self._panel_a_histogram(fig, screened)
        self._panel_b_scatter(fig, retained)
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(text="Panel Bundling Index (PBI) and the Co-Testing Screen", x=0.5, font=dict(size=16)),
            barmode="stack",
            legend=dict(bordercolor="#cccccc", borderwidth=1, bgcolor="rgba(255,255,255,0.9)"),
            margin=dict(t=100, b=70, l=60, r=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(screened, retained, titles, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Shared pieces ─────────────────────────────────────────────────────

    @staticmethod
    def _retained(edge_report: pd.DataFrame) -> pd.DataFrame:
        if edge_report is None or edge_report.empty or "panel_bundling_index" not in edge_report.columns:
            return pd.DataFrame(columns=["upstream_antibiotic", "downstream_antibiotic", "panel_bundling_index", "escalation_ratio"])
        frame = edge_report.copy()
        frame["panel_bundling_index"] = pd.to_numeric(frame["panel_bundling_index"], errors="coerce")
        frame["escalation_ratio"] = pd.to_numeric(frame.get("escalation_ratio", pd.Series(dtype=float)), errors="coerce")
        return frame.loc[frame["panel_bundling_index"].notna()].reset_index(drop=True)

    def _titles(self, screened: pd.DataFrame) -> dict[str, str]:
        removed = int(screened["removed"].sum()) if not screened.empty else 0
        kept = int((~screened["removed"]).sum()) if not screened.empty else 0
        exempt = int(screened["exempt"].sum()) if not screened.empty else 0
        note = (
            f"{exempt:,} kept {'pair has' if exempt == 1 else 'pairs have'} PBI ≥ {self._rules.cotesting_threshold:.2f} "
            f"but fewer than {self._rules.min_total_support} rows in one direction"
            if exempt
            else ""
        )
        return {"a": f"Assessed pairs: {removed:,} removed, {kept:,} kept", "b": "Retained pairs: PBI vs escalation ratio", "note": note}

    def _labels(self) -> tuple[str, str]:
        r = self._rules
        return (
            "Kept by the screen",
            f"Removed (both ≥ {r.cotesting_threshold:.2f}, ≥ {r.min_total_support} rows each way)",
        )

    # ── Panel A ───────────────────────────────────────────────────────────

    def _panel_a_histogram(self, fig: go.Figure, screened: pd.DataFrame) -> None:
        threshold = self._rules.cotesting_threshold
        kept_label, removed_label = self._labels()
        for removed, color, name in [(False, _KEPT_COLOR, kept_label), (True, _REMOVED_COLOR, removed_label)]:
            sub = screened.loc[screened["removed"].eq(removed)] if not screened.empty else screened
            if sub.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub["panel_bundling_index"],
                    name=name,
                    marker_color=color,
                    marker_line=dict(color="white", width=0.5),
                    xbins=dict(start=0.0, end=1.0 + 1e-9, size=0.05),
                    hovertemplate="PBI %{x}: %{y} pairs<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_shape(type="line", x0=threshold, x1=threshold, y0=0, y1=1, xref="x", yref="y domain",
                      line=dict(color="#d62728", dash="dash", width=2.2))
        fig.add_annotation(x=threshold, y=1.0, xref="x", yref="y domain", text=f"  PBI = {threshold:.2f}",
                           showarrow=False, font=dict(size=11, color="#d62728"), xanchor="left", yanchor="top")
        fig.update_xaxes(title_text="Panel Bundling Index (PBI)", range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(title_text="Number of ordered pairs", row=1, col=1)
        note = self._titles(screened)["note"]
        if note:
            fig.add_annotation(x=0.5, y=0.78, xref="x domain", yref="y domain", text=note, showarrow=False,
                               font=dict(size=10, color="#555555"), xanchor="center")

    # ── Panel B ───────────────────────────────────────────────────────────

    def _panel_b_scatter(self, fig: go.Figure, retained: pd.DataFrame) -> None:
        threshold = self._rules.cotesting_threshold
        if not retained.empty:
            er = retained["escalation_ratio"].replace([math.inf, -math.inf], np.nan)
            labels = retained["upstream_antibiotic"].astype(str) + " → " + retained["downstream_antibiotic"].astype(str)
            fig.add_trace(
                go.Scatter(
                    x=retained["panel_bundling_index"], y=er, mode="markers", name="Retained", showlegend=False,
                    marker=dict(color=_KEPT_COLOR, size=8, line=dict(color="white", width=0.6)),
                    customdata=list(zip(labels, retained["panel_bundling_index"].round(3), retained["escalation_ratio"].round(3))),
                    hovertemplate="<b>%{customdata[0]}</b><br>PBI: %{customdata[1]}<br>ER: %{customdata[2]}<extra></extra>",
                ),
                row=1, col=2,
            )
        fig.add_shape(type="line", x0=threshold, x1=threshold, y0=0, y1=1, xref="x2", yref="y2 domain",
                      line=dict(color="#d62728", dash="dash", width=2.0))
        fig.add_shape(type="line", x0=0, x1=1, y0=1.0, y1=1.0, xref="x2 domain", yref="y2",
                      line=dict(color="#bbbbbb", dash="dot", width=1.5))
        fig.update_xaxes(title_text="Panel Bundling Index (PBI)", range=[0, 1.05], row=1, col=2)
        # Log scale: escalation ratios span orders of magnitude in both directions.
        fig.update_yaxes(title_text="Escalation ratio (ER, log scale)", type="log", row=1, col=2)

    # ── Static rendering ──────────────────────────────────────────────────

    def _write_static(self, screened: pd.DataFrame, retained: pd.DataFrame, titles: dict[str, str], fmt: str, path: Path) -> None:
        threshold = self._rules.cotesting_threshold
        kept_label, removed_label = self._labels()
        fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.0, 1.05]})

        if screened.empty:
            ax_hist.text(0.5, 0.5, "Co-testing probabilities unavailable.", ha="center", va="center", transform=ax_hist.transAxes)
        else:
            bins = np.linspace(0.0, 1.0, 21)
            kept = screened.loc[~screened["removed"], "panel_bundling_index"]
            removed = screened.loc[screened["removed"], "panel_bundling_index"]
            ax_hist.hist([kept, removed], bins=bins, stacked=True, color=["#1f77b4", "#d62728"], label=[kept_label, removed_label], alpha=0.82)
            ax_hist.legend(frameon=False, fontsize=8, loc="upper center")
            if titles["note"]:
                ax_hist.text(0.5, 0.84, titles["note"], transform=ax_hist.transAxes, ha="center", va="top", fontsize=8, color="#555555")
        ax_hist.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.6)
        ax_hist.text(threshold - 0.01, 0.97, f"PBI = {threshold:.2f}", color="#d62728", fontsize=8, va="top", ha="right", transform=ax_hist.get_xaxis_transform())
        ax_hist.set_xlim(0, 1.03)
        ax_hist.set_xlabel("Panel Bundling Index (PBI)")
        ax_hist.set_ylabel("Number of ordered pairs")
        ax_hist.set_title(f"A. {titles['a']}", loc="left", fontweight="bold", fontsize=10)
        ax_hist.grid(True, axis="y", alpha=0.18)

        if retained.empty:
            ax_scatter.text(0.5, 0.5, "No retained pairs.", ha="center", va="center", transform=ax_scatter.transAxes)
        else:
            ratios = retained["escalation_ratio"].replace([math.inf, -math.inf], np.nan)
            usable = retained.loc[ratios.gt(0)]
            ax_scatter.scatter(usable["panel_bundling_index"], ratios[ratios.gt(0)],
                               s=42, color="#1f77b4", alpha=0.82, edgecolor="white", linewidth=0.6, label="Retained")
            ax_scatter.set_yscale("log")
        ax_scatter.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.4)
        ax_scatter.axhline(1.0, color="#98A2B3", linestyle=":", linewidth=1.2)
        ax_scatter.set_xlim(0, 1.03)
        ax_scatter.set_xlabel("Panel Bundling Index (PBI)")
        ax_scatter.set_ylabel("Escalation ratio (ER, log scale)")
        ax_scatter.set_title(f"B. {titles['b']}", loc="left", fontweight="bold", fontsize=10)
        ax_scatter.grid(True, which="both", alpha=0.18)

        fig.text(0.06, 0.025, "Static export omits pair labels. Pair identities are available in the interactive HTML hover text.",
                 ha="left", va="bottom", fontsize=8, color="#344054")
        fig.suptitle("Panel Bundling Index and the Co-Testing Screen", fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
