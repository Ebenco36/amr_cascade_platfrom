"""Panel bundling index chart — shows which pairs were excluded as co-ordered panels."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_RETAINED_COLOR = "rgba(31,119,180,0.80)"
_EXCLUDED_COLOR = "rgba(214,39,40,0.80)"
_THRESHOLD = 0.95
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
    Two-panel panel bundling diagnostic.

    Panel A — PBI distribution histogram
        X = Panel Bundling Index (PBI), binned in 0.05 steps.
        Red bars (PBI ≥ 0.95) = excluded as panel bundles.
        Blue bars (PBI < 0.95) = retained for cascade analysis.
        Vertical dashed line at PBI = 0.95 threshold.

    Panel B — PBI scatter vs. Escalation Ratio
        X = PBI, Y = ER.
        Excluded pairs (red) cluster at high PBI and varied ER —
        they co-occur not because of cascade logic but because they are always
        ordered together.
        Retained pairs (blue) show the cascade signal the analysis is built on.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        if edge_report.empty or "panel_bundling_index" not in edge_report.columns:
            fig = _empty(self._template, self._width, self._height,
                         "Panel bundling chart unavailable — panel_bundling_index column missing")
            return self._exporter.write(fig, output_stem, formats)

        df = edge_report.copy()
        df["panel_bundling_index"] = pd.to_numeric(df["panel_bundling_index"], errors="coerce")
        df = df[df["panel_bundling_index"].notna()].copy()
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No pairs with a computed panel bundling index")
            return self._exporter.write(fig, output_stem, formats)

        df["excluded"] = df["panel_bundling_index"].ge(_THRESHOLD)
        df["escalation_ratio"] = pd.to_numeric(df.get("escalation_ratio", pd.Series(dtype=float)),
                                                errors="coerce")
        pair_labels = (
            df["upstream_antibiotic"].fillna("?")
            + " → "
            + df["downstream_antibiotic"].fillna("?")
        )

        n_excluded = int(df["excluded"].sum())
        n_retained = int((~df["excluded"]).sum())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"<b>A</b>   PBI Distribution  "
                f"(excluded: {n_excluded}, retained: {n_retained})",
                "<b>B</b>   PBI vs. Escalation Ratio",
            ),
            horizontal_spacing=0.12,
        )

        self._panel_a_histogram(fig, df)
        self._panel_b_scatter(fig, df, pair_labels)

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text=(
                    "Panel Bundling Index (PBI) — "
                    "Identifying Pairs Excluded as Co-Ordered Test Panels"
                ),
                x=0.5,
                font=dict(size=16),
            ),
            barmode="stack",
            legend=dict(
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(t=100, b=70, l=60, r=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(df, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Panel A ───────────────────────────────────────────────────────────

    def _panel_a_histogram(self, fig: go.Figure, df: pd.DataFrame) -> None:
        for excluded, color, name in [
            (False, _RETAINED_COLOR, f"Retained  (PBI < {_THRESHOLD})"),
            (True,  _EXCLUDED_COLOR, f"Excluded  (PBI ≥ {_THRESHOLD})"),
        ]:
            sub = df[df["excluded"].eq(excluded)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub["panel_bundling_index"],
                    name=name,
                    marker_color=color,
                    marker_line=dict(color="white", width=0.5),
                    xbins=dict(start=0.0, end=1.02, size=0.05),
                    hovertemplate="PBI ∈ %{x:.2f}: %{y} pairs<extra></extra>",
                ),
                row=1, col=1,
            )

        # Exclusion threshold line
        fig.add_shape(
            type="line",
            x0=_THRESHOLD, x1=_THRESHOLD, y0=0, y1=1,
            xref="x", yref="y domain",
            line=dict(color="#d62728", dash="dash", width=2.2),
        )
        fig.add_annotation(
            x=_THRESHOLD, y=1.0,
            xref="x", yref="y domain",
            text=f"  PBI = {_THRESHOLD} threshold",
            showarrow=False,
            font=dict(size=11, color="#d62728"),
            xanchor="left",
            yanchor="top",
        )
        fig.update_xaxes(
            title_text="Panel Bundling Index (PBI)",
            range=[0, 1.05],
            row=1, col=1,
        )
        fig.update_yaxes(title_text="Number of pairs", row=1, col=1)

    # ── Panel B ───────────────────────────────────────────────────────────

    def _panel_b_scatter(
        self, fig: go.Figure, df: pd.DataFrame, pair_labels: pd.Series
    ) -> None:
        er_finite = df["escalation_ratio"].replace([math.inf, -math.inf], pd.NA).dropna()
        er_max = er_finite.quantile(0.99) if not er_finite.empty else 10.0

        for excluded, color, name in [
            (False, _RETAINED_COLOR, "Retained"),
            (True,  _EXCLUDED_COLOR, "Excluded (panel bundle)"),
        ]:
            mask = df["excluded"].eq(excluded)
            sub  = df[mask]
            if sub.empty:
                continue
            er = sub["escalation_ratio"].clip(upper=er_max * 1.05).clip(lower=0.9)
            fig.add_trace(
                go.Scatter(
                    x=sub["panel_bundling_index"],
                    y=er,
                    mode="markers",
                    name=name,
                    showlegend=False,
                    marker=dict(
                        color=color,
                        size=8,
                        line=dict(color="white", width=0.6),
                    ),
                    customdata=list(zip(
                        pair_labels[mask].values,
                        sub["panel_bundling_index"].round(3),
                        er.round(3),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "PBI: %{customdata[1]}<br>"
                        "ER: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

        # Exclusion threshold vertical
        fig.add_shape(
            type="line",
            x0=_THRESHOLD, x1=_THRESHOLD, y0=0, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color="#d62728", dash="dash", width=2.0),
        )
        # ER = 1 horizontal
        fig.add_shape(
            type="line",
            x0=0, x1=1, y0=1.0, y1=1.0,
            xref="x2 domain", yref="y2",
            line=dict(color="#bbbbbb", dash="dot", width=1.5),
        )
        fig.update_xaxes(
            title_text="Panel Bundling Index (PBI)",
            range=[0, 1.05],
            row=1, col=2,
        )
        # Log scale: escalation ratios span ~1 to 1000s here, and a linear
        # axis crushes nearly every point into a sliver near zero.
        fig.update_yaxes(title_text="Escalation ratio (ER, log scale)", type="log", row=1, col=2)

    def _write_static(self, df: pd.DataFrame, fmt: str, path: Path) -> None:
        plot_data = df.copy()
        plot_data["pair_label"] = (
            plot_data["upstream_antibiotic"].fillna("?")
            + " to "
            + plot_data["downstream_antibiotic"].fillna("?")
        )
        plot_data["panel_bundling_index"] = pd.to_numeric(
            plot_data["panel_bundling_index"],
            errors="coerce",
        )
        plot_data["escalation_ratio"] = pd.to_numeric(
            plot_data.get("escalation_ratio", pd.Series(dtype=float)),
            errors="coerce",
        )
        plot_data = plot_data[plot_data["panel_bundling_index"].notna()].copy()

        fig, (ax_hist, ax_scatter) = plt.subplots(
            1,
            2,
            figsize=(16, 7),
            gridspec_kw={"width_ratios": [1.0, 1.05]},
        )
        if plot_data.empty:
            ax_hist.axis("off")
            ax_scatter.axis("off")
            ax_hist.text(0.5, 0.5, "Panel bundling diagnostics unavailable.", ha="center", va="center")
            fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return

        retained = plot_data[~plot_data["excluded"]]
        excluded = plot_data[plot_data["excluded"]]
        bins = np.linspace(0.0, 1.0, 21)
        ax_hist.hist(
            [retained["panel_bundling_index"], excluded["panel_bundling_index"]],
            bins=bins,
            stacked=True,
            color=["#1f77b4", "#d62728"],
            label=[f"Retained (PBI < {_THRESHOLD})", f"Excluded (PBI >= {_THRESHOLD})"],
            alpha=0.82,
        )
        ax_hist.axvline(_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.6)
        ax_hist.text(_THRESHOLD + 0.01, ax_hist.get_ylim()[1] * 0.95, "PBI = 0.95", color="#d62728", fontsize=8, va="top")
        ax_hist.set_xlabel("Panel Bundling Index (PBI)")
        ax_hist.set_ylabel("Number of antibiotic pairs")
        n_excluded, n_retained = len(excluded), len(retained)
        ax_hist.set_title(
            f"A. Co-testing distribution  (excluded: {n_excluded}, retained: {n_retained})",
            loc="left", fontweight="bold",
        )
        if n_excluded == 0:
            # A visibly empty "excluded" category reads as broken unless the
            # reader can tell it's a real finding: at this cohort/sample size
            # no pair was tested together deterministically enough to cross
            # the PBI >= 0.95 bundling threshold, not that the filter failed
            # to run. Centered in the plot's own empty interior (away from
            # both the bar at the left edge and the "PBI = 0.95" label
            # pinned to the threshold line) so the two annotations never
            # compete for the same corner.
            ax_hist.text(
                0.5, 0.55,
                "No pairs crossed the PBI ≥ 0.95\nbundling threshold in this cohort.",
                transform=ax_hist.transAxes, ha="center", va="center",
                fontsize=9, color="#667085", style="italic",
            )
        ax_hist.grid(True, axis="y", alpha=0.18)
        ax_hist.legend(frameon=False, fontsize=8)

        finite_er = plot_data["escalation_ratio"].replace([math.inf, -math.inf], pd.NA).dropna()
        er_max = float(finite_er.quantile(0.99)) * 1.10 if not finite_er.empty else 10.0
        for mask, color, label in [
            (~plot_data["excluded"], "#1f77b4", "Retained"),
            (plot_data["excluded"], "#d62728", "Excluded panel bundle"),
        ]:
            sub = plot_data[mask & plot_data["escalation_ratio"].notna()]
            if sub.empty:
                continue
            ax_scatter.scatter(
                sub["panel_bundling_index"],
                sub["escalation_ratio"].clip(lower=0.9, upper=er_max),
                s=42,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
                label=label,
            )

        ax_scatter.axvline(_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.4)
        ax_scatter.axhline(1.0, color="#98A2B3", linestyle=":", linewidth=1.2)
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlim(0, 1.03)
        ax_scatter.set_xlabel("Panel Bundling Index (PBI)")
        ax_scatter.set_ylabel("Escalation ratio (ER, log scale)")
        ax_scatter.set_title("B. PBI vs escalation ratio", loc="left", fontweight="bold")
        ax_scatter.grid(True, which="both", alpha=0.18)
        ax_scatter.legend(frameon=False, fontsize=8)

        fig.text(
            0.06,
            0.025,
            "Static export omits pair labels. Pair identities are available in the interactive HTML hover text.",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#344054",
        )
        fig.suptitle("Panel Bundling Index — Excluding Near-Deterministic Co-Testing", fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
