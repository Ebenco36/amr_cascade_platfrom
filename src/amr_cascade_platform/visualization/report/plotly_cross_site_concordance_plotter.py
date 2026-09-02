"""Cross-site concordance scatter — site-level ER vs combined ER per pair."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_GRID = "#E8EEF5"
_NEUTRAL = "#aaaaaa"

_SITE_PALETTE = [
    "#1f77b4", "#2ca02c", "#d62728",
    "#ff7f0e", "#9467bd", "#8c564b",
]


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper", font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyCrossSiteConcordancePlotter:
    """
    Two-panel cross-site concordance figure.

    Panel A — Site ER vs Combined ER scatter.
        Each point = one (pair, site) combination.
        X = combined_escalation_ratio (pooled across sites).
        Y = site_escalation_ratio (this site only).
        Colour = site.
        Diagonal y = x = perfect concordance.
        Points clustered near the diagonal = sites agree.

    Panel B — ER delta distribution by site.
        X = escalation_ratio_delta  (site ER − combined ER).
        Colour = site.
        Dashed reference at Δ = 0.
        Distribution width = site-to-combined disagreement.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        comparison_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        needed = {"site_escalation_ratio", "combined_escalation_ratio"}
        if comparison_summary.empty or not needed.issubset(comparison_summary.columns):
            fig = _empty(
                self._template, self._width, self._height,
                "Cross-site concordance unavailable — "
                "site_escalation_ratio / combined_escalation_ratio columns missing",
            )
            return self._exporter.write(fig, output_stem, formats)

        df = comparison_summary.copy()
        for col in ("site_escalation_ratio", "combined_escalation_ratio",
                    "escalation_ratio_delta", "site_total_support_n"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["site_escalation_ratio"].notna() & df["combined_escalation_ratio"].notna()].copy()
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No pairs with site-level and combined ER data")
            return self._exporter.write(fig, output_stem, formats)

        df["pair_label"] = (
            df.get("upstream_antibiotic", pd.Series("?", index=df.index)).fillna("?")
            + " → "
            + df.get("downstream_antibiotic", pd.Series("?", index=df.index)).fillna("?")
        )

        sites = sorted(df["site"].dropna().unique().tolist()) if "site" in df.columns else ["combined"]
        site_colors = {s: _SITE_PALETTE[i % len(_SITE_PALETTE)] for i, s in enumerate(sites)}

        # Pearson r for subtitle
        try:
            from scipy.stats import pearsonr
            r, _ = pearsonr(df["combined_escalation_ratio"].replace([math.inf, -math.inf], pd.NA).dropna(),
                            df["site_escalation_ratio"].replace([math.inf, -math.inf], pd.NA).dropna())
            r_text = f"Pearson r = {r:.3f}"
        except Exception:
            r_text = ""

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"<b>A</b>   Site ER vs Combined ER  {('— ' + r_text) if r_text else ''}",
                "<b>B</b>   ER Delta by Site (Site ER − Combined ER)",
            ),
            horizontal_spacing=0.12,
        )

        self._panel_a(fig, df, sites, site_colors)
        self._panel_b(fig, df, sites, site_colors)

        n_pairs = df["pair_label"].nunique()
        n_rows  = len(df)

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text=(
                    f"Cross-Site Concordance — Site-Level vs Combined Escalation Ratio  "
                    f"({n_pairs} unique pairs, {n_rows} site observations)"
                ),
                x=0.5,
                font=dict(size=15),
            ),
            legend=dict(
                title="Site",
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(t=100, b=70, l=70, r=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID)
        return self._exporter.write(fig, output_stem, formats)

    # ── Panel A: site vs combined scatter ─────────────────────────────────

    def _panel_a(
        self, fig: go.Figure, df: pd.DataFrame,
        sites: list[str], site_colors: dict[str, str],
    ) -> None:
        combined_col = df["combined_escalation_ratio"].replace([math.inf, -math.inf], pd.NA)
        site_col     = df["site_escalation_ratio"].replace([math.inf, -math.inf], pd.NA)
        size_col     = df.get("site_total_support_n", pd.Series(50, index=df.index)).fillna(50)
        marker_sizes = np.sqrt(size_col.clip(lower=4)).clip(4, 20)

        site_groups = df.get("site", pd.Series("combined", index=df.index))
        for site in sites:
            mask = site_groups.eq(site) & combined_col.notna() & site_col.notna()
            if not mask.any():
                continue
            sub = df[mask]
            fig.add_trace(
                go.Scatter(
                    x=combined_col[mask].values,
                    y=site_col[mask].values,
                    mode="markers",
                    name=site,
                    legendgroup=site,
                    marker=dict(
                        color=site_colors[site],
                        size=marker_sizes[mask].values,
                        opacity=0.80,
                        line=dict(color="white", width=0.6),
                    ),
                    customdata=list(zip(
                        sub["pair_label"].values,
                        combined_col[mask].round(3).values,
                        site_col[mask].round(3).values,
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>  [" + site + "]<br>"
                        "Combined ER: %{customdata[1]:.3f}<br>"
                        "Site ER:     %{customdata[2]:.3f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Diagonal reference
        all_vals = pd.concat([combined_col, site_col]).dropna()
        if not all_vals.empty:
            vmin = max(all_vals.min() * 0.8, 1e-3)
            vmax = all_vals.max() * 1.2
            fig.add_shape(
                type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                xref="x", yref="y",
                line=dict(color=_NEUTRAL, dash="dash", width=1.5),
            )
            # ER = 1 references
            for ref_axis, is_x in [("x", True), ("y", False)]:
                fig.add_shape(
                    type="line",
                    x0=1 if is_x else vmin, y0=1 if not is_x else vmin,
                    x1=1 if is_x else vmax, y1=1 if not is_x else vmax,
                    xref="x", yref="y",
                    line=dict(color="#dddddd", dash="dot", width=1.0),
                )

        fig.update_xaxes(title_text="Combined ER (pooled)", row=1, col=1)
        fig.update_yaxes(title_text="Site ER (site-specific)", row=1, col=1)

    # ── Panel B: delta distribution ───────────────────────────────────────

    def _panel_b(
        self, fig: go.Figure, df: pd.DataFrame,
        sites: list[str], site_colors: dict[str, str],
    ) -> None:
        delta_col = "escalation_ratio_delta"
        if delta_col not in df.columns:
            df = df.copy()
            df[delta_col] = (
                df["site_escalation_ratio"] - df["combined_escalation_ratio"]
            )

        site_groups = df.get("site", pd.Series("combined", index=df.index))
        for site in sites:
            mask = site_groups.eq(site) & df[delta_col].notna()
            if not mask.any():
                continue
            fig.add_trace(
                go.Violin(
                    x=df[delta_col][mask].values,
                    name=site,
                    legendgroup=site,
                    showlegend=False,
                    line_color=site_colors[site],
                    fillcolor=site_colors[site],
                    opacity=0.55,
                    box_visible=True,
                    meanline_visible=True,
                    orientation="h",
                    hoverinfo="skip",
                ),
                row=1, col=2,
            )

        # Reference at Δ = 0
        fig.add_shape(
            type="line", x0=0, y0=0, x1=0, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color="#d62728", dash="dash", width=2),
        )
        fig.add_annotation(
            x=0, y=1.0, xref="x2", yref="y2 domain",
            text="  Δ = 0 (perfect concordance)", showarrow=False,
            font=dict(size=10, color="#d62728"), xanchor="left", yanchor="top",
        )
        fig.update_xaxes(title_text="ER delta (site − combined)", row=1, col=2)
        fig.update_yaxes(title_text="Site", row=1, col=2)
