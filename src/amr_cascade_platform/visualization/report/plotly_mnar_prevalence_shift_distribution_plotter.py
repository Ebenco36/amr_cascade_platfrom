"""MNAR prevalence-shift distribution — diverging waterfall and scatter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from amr_cascade_platform.visualization.report.organism_labels import format_organism_label
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_OVER_COLOR  = "rgba(214,39,40,0.82)"   # positive shift — naive overstates resistance
_UNDER_COLOR = "rgba(31,119,180,0.82)"  # negative shift — naive understates resistance
_OVER_MPL = "#D62728"
_UNDER_MPL = "#1F77B4"
_GRID = "#E8EEF5"


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper", font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyMNARPrevalenceShiftDistributionPlotter:
    """
    Two-panel MNAR prevalence-shift figure.

    Panel A — Diverging waterfall (sorted by absolute shift)
        Each bar = one organism-antibiotic pair.
        Bar length = naive prevalence minus MNAR lambda-zero prevalence.
        Red = naive rate overstates resistance.
        Blue = naive rate understates resistance.

    Panel B — Naive prevalence vs. MNAR lambda-zero prevalence scatter
        Each point = one pair.
        Points above the diagonal: naive > shifted (overestimate).
        Points below the diagonal: naive < shifted (underestimate).
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        prevalence_shift: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        required = {"naive_prevalence_pct", "mnar_lambda0_prevalence_pct", "mnar_lambda0_shift_from_naive_pct"}
        if prevalence_shift.empty or not required.issubset(prevalence_shift.columns):
            fig = _empty(self._template, self._width, self._height,
                         "MNAR prevalence-shift distribution unavailable — required columns are missing")
            return self._exporter.write(fig, output_stem, formats)

        df = prevalence_shift.copy()
        df["naive_prevalence_pct"] = pd.to_numeric(df["naive_prevalence_pct"], errors="coerce")
        df["mnar_lambda0_prevalence_pct"] = pd.to_numeric(df["mnar_lambda0_prevalence_pct"], errors="coerce")
        df["shift_pct"] = pd.to_numeric(df["mnar_lambda0_shift_from_naive_pct"], errors="coerce")

        df = df[df["shift_pct"].notna() & df["naive_prevalence_pct"].notna() & df["mnar_lambda0_prevalence_pct"].notna()].copy()
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No prevalence shift results with valid MNAR shift values")
            return self._exporter.write(fig, output_stem, formats)

        # Pair label
        up_col   = "upstream_antibiotic"   if "upstream_antibiotic"   in df.columns else None
        down_col = "downstream_antibiotic" if "downstream_antibiotic" in df.columns else None
        # prevalence_shift.parquet (the input to this figure) names its drug
        # column "drug", not "antibiotic" -- the old antibiotic-only check
        # always missed it and fell through to using the raw row index as
        # the label (e.g. "KLEBSIELLA PNEUMONIAE / 0"), which conveys
        # nothing about which drug each bar is.
        drug_col = next((c for c in ("drug", "antibiotic") if c in df.columns), None)

        if up_col and down_col:
            df["pair_label"] = df[up_col].fillna("?") + " → " + df[down_col].fillna("?")
        elif drug_col:
            df["pair_label"] = df[drug_col].fillna("?")
        else:
            df["pair_label"] = df.index.astype(str)

        # Add organism prefix only when the figure spans multiple organisms --
        # repeating the same organism name on every one of ~30 bars (the common
        # single-organism report case) is pure redundant clutter.
        organism_title = ""
        if "organism" in df.columns:
            organisms = df["organism"].dropna().unique()
            if len(organisms) == 1:
                organism_title = format_organism_label(str(organisms[0]))
            else:
                df["pair_label"] = df["organism"].fillna("") + " / " + df["pair_label"]

        df = df.sort_values("shift_pct", ascending=True).reset_index(drop=True)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "<b>A</b>   Naive − MNAR Prevalence Shift per Drug",
                "<b>B</b>   Naive vs. MNAR Lambda-Zero Prevalence",
            ),
            column_widths=[0.55, 0.45],
            horizontal_spacing=0.12,
        )
        self._panel_a_waterfall(fig, df)
        self._panel_b_scatter(fig, df)

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=max(self._height, 80 + 22 * len(df)),
            title=dict(
                text=(
                    "MNAR Prevalence Shift Under Selective AST Observation"
                    + (f" ({organism_title})" if organism_title else "")
                ),
                x=0.5,
                font=dict(size=16),
            ),
            showlegend=True,
            legend=dict(
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(t=110, b=60, l=220, r=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(df, organism_title, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Panel A ───────────────────────────────────────────────────────────

    def _panel_a_waterfall(self, fig: go.Figure, df: pd.DataFrame) -> None:
        colors = [_OVER_COLOR if v > 0 else _UNDER_COLOR for v in df["shift_pct"]]

        fig.add_trace(
            go.Bar(
                x=df["shift_pct"],
                y=df["pair_label"],
                orientation="h",
                marker_color=colors,
                marker_line=dict(color="white", width=0.4),
                customdata=list(zip(
                    df["pair_label"],
                    df["shift_pct"].round(2),
                    df["naive_prevalence_pct"].round(1),
                    df["mnar_lambda0_prevalence_pct"].round(1),
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Shift: %{customdata[1]:+.2f} pp<br>"
                    "Naive prevalence: %{customdata[2]:.1f}%<br>"
                    "MNAR prevalence (lambda=0): %{customdata[3]:.1f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Zero reference
        fig.add_shape(
            type="line",
            x0=0, x1=0, y0=-0.5, y1=len(df) - 0.5,
            xref="x", yref="y",
            line=dict(color="#555555", width=1.5),
        )

        # Legend proxies
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h",
                             marker_color=_OVER_COLOR, name="Naive overstates resistance",
                             showlegend=True), row=1, col=1)
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h",
                             marker_color=_UNDER_COLOR, name="Naive understates resistance",
                             showlegend=True), row=1, col=1)

        fig.update_xaxes(title_text="Naive − MNAR prevalence (percentage points)", zeroline=False, row=1, col=1)
        fig.update_yaxes(title_text="", tickfont=dict(size=10), row=1, col=1)

    # ── Panel B ───────────────────────────────────────────────────────────

    def _panel_b_scatter(self, fig: go.Figure, df: pd.DataFrame) -> None:
        ax_max = max(
            df["naive_prevalence_pct"].max(),
            df["mnar_lambda0_prevalence_pct"].max(),
            1.0,
        ) * 1.10

        # Diagonal — naive = MNAR lambda-zero prevalence.
        fig.add_trace(
            go.Scatter(
                x=[0, ax_max], y=[0, ax_max],
                mode="lines",
                line=dict(color="#bbbbbb", dash="dot", width=1.5),
                name="No distortion",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

        over_mask  = df["shift_pct"] > 0
        under_mask = df["shift_pct"] <= 0

        for mask, color, label in [
            (over_mask,  _OVER_COLOR,  "Naive overstates"),
            (under_mask, _UNDER_COLOR, "Naive understates"),
        ]:
            sub = df[mask]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["naive_prevalence_pct"],
                    y=sub["mnar_lambda0_prevalence_pct"],
                    mode="markers",
                    name=label,
                    showlegend=False,
                    marker=dict(color=color, size=10, line=dict(color="white", width=0.8)),
                    customdata=list(zip(
                        sub["pair_label"],
                        sub["naive_prevalence_pct"].round(1),
                        sub["mnar_lambda0_prevalence_pct"].round(1),
                        sub["shift_pct"].round(2),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Naive: %{customdata[1]:.1f}%<br>"
                        "MNAR lambda=0: %{customdata[2]:.1f}%<br>"
                        "Shift: %{customdata[3]:+.2f} pp<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

        fig.update_xaxes(
            title_text="Naive resistance prevalence (%)",
            range=[0, ax_max],
            row=1, col=2,
        )
        fig.update_yaxes(
            title_text="MNAR prevalence at lambda=0 (%)",
            range=[0, ax_max],
            row=1, col=2,
        )

    # ── Publication static rendering ─────────────────────────────────────

    def _write_static(self, df: pd.DataFrame, organism_title: str, fmt: str, path: Path) -> None:
        """Render a static figure with controlled title spacing."""

        height = max(6.8, 2.0 + 0.32 * len(df))
        fig, (ax_bar, ax_scatter) = plt.subplots(
            1,
            2,
            figsize=(13.5, height),
            dpi=300,
            gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.34},
        )
        title = "MNAR Prevalence Shift Under Selective AST Observation"
        if organism_title:
            title += f" ({organism_title})"
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

        colors = [_OVER_MPL if value > 0 else _UNDER_MPL for value in df["shift_pct"]]
        y_pos = np.arange(len(df))
        ax_bar.barh(y_pos, df["shift_pct"], color=colors, alpha=0.85)
        ax_bar.axvline(0, color="#475467", linewidth=1.2)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(df["pair_label"], fontsize=8)
        ax_bar.set_xlabel("Naive minus MNAR prevalence (percentage points)")
        ax_bar.set_title("A. Shift per drug", loc="left", fontsize=11, fontweight="bold")
        ax_bar.grid(True, axis="x", color=_GRID, linewidth=0.7)

        ax_max = max(
            float(df["naive_prevalence_pct"].max()),
            float(df["mnar_lambda0_prevalence_pct"].max()),
            1.0,
        ) * 1.10
        ax_scatter.plot([0, ax_max], [0, ax_max], linestyle=":", color="#98A2B3", linewidth=1.3, label="No distortion")
        over_mask = df["shift_pct"] > 0
        under_mask = ~over_mask
        if over_mask.any():
            ax_scatter.scatter(
                df.loc[over_mask, "naive_prevalence_pct"],
                df.loc[over_mask, "mnar_lambda0_prevalence_pct"],
                color=_OVER_MPL,
                edgecolors="white",
                linewidths=0.6,
                s=38,
                label="Naive overstates",
                alpha=0.85,
            )
        if under_mask.any():
            ax_scatter.scatter(
                df.loc[under_mask, "naive_prevalence_pct"],
                df.loc[under_mask, "mnar_lambda0_prevalence_pct"],
                color=_UNDER_MPL,
                edgecolors="white",
                linewidths=0.6,
                s=38,
                label="Naive understates",
                alpha=0.85,
            )
        ax_scatter.set_xlim(0, ax_max)
        ax_scatter.set_ylim(0, ax_max)
        ax_scatter.set_xlabel("Naive prevalence (%)")
        ax_scatter.set_ylabel("MNAR prevalence at lambda=0 (%)")
        ax_scatter.set_title("B. Naive vs MNAR estimate", loc="left", fontsize=11, fontweight="bold")
        ax_scatter.grid(True, color=_GRID, linewidth=0.7)
        ax_scatter.legend(frameon=False, fontsize=8, loc="lower right")

        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
