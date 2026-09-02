"""Permutation null-distribution scatter and bootstrap sign-stability histogram."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_STATUS_COLOR: dict[str, str] = {
    "robust":       "#2ca02c",
    "supported":    "#1f77b4",
    "mixed":        "#ff7f0e",
    "insufficient": "#d62728",
    "unavailable":  "#aec7e8",
}
_STATUS_ORDER = ["robust", "supported", "mixed", "insufficient", "unavailable"]
_STATUS_LABEL: dict[str, str] = {
    "robust":       "Robust",
    "supported":    "Supported",
    "mixed":        "Mixed",
    "insufficient": "Insufficient",
    "unavailable":  "Unavailable",
}
_GRID = "#E8EEF5"


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
                       font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyValidationDiagnosticsPlotter:
    """
    Two-panel publication figure.

    Panel A — Permutation scatter
        X = direction-specific permutation null boundary
            (95th percentile for escalation edges; 5th percentile for suppression edges)
        Y = observed ER
        Escalation edges beyond the null fall above the diagonal; suppression edges beyond
        the null fall below the diagonal.
        Color = validation status.

    Panel B — Bootstrap sign-stability histogram
        Distribution of bootstrap_sign_stability (S) across all support-passing pairs.
        Vertical line at S = 0.80 threshold.
        Color = validation status.
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
        required = {
            "escalation_ratio", "permutation_null_q05_er", "permutation_null_q95_er",
            "bootstrap_sign_stability", "validation_status", "passes_support_threshold",
        }
        if edge_report.empty or not required.issubset(edge_report.columns):
            fig = _empty(self._template, self._width, self._height,
                         "Validation diagnostics unavailable — required columns missing from edge report")
            return self._exporter.write(fig, output_stem, formats)

        df = edge_report[edge_report["passes_support_threshold"].eq(True)].copy()
        df["validation_status"] = df["validation_status"].fillna("unavailable")
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No support-passing pairs — diagnostics cannot be drawn")
            return self._exporter.write(fig, output_stem, formats)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "<b>A</b>   Observed ER vs. Direction-Specific Permutation Null Boundary",
                "<b>B</b>   Bootstrap Sign-Stability Distribution",
            ),
            horizontal_spacing=0.14,
        )
        # Shared across both panels so a status already legended in panel A
        # isn't legended a second time by panel B.
        seen_statuses: set[str] = set()
        self._panel_a_permutation_scatter(fig, df, seen_statuses)
        self._panel_b_bootstrap_histogram(fig, df, seen_statuses)

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text=(
                    "Validation Diagnostics — Gates 1–2"
                    "<br><sup>Permutation null test and bootstrap sign stability; "
                    "site and temporal replication are reported separately</sup>"
                ),
                x=0.5,
                font=dict(size=17),
            ),
            legend=dict(
                title="Validation status",
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
            barmode="stack",
            margin=dict(t=110, b=70, l=70, r=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(df, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Panel A ──────────────────────────────────────────────────────────────

    def _panel_a_permutation_scatter(self, fig: go.Figure, df: pd.DataFrame, seen_statuses: set[str]) -> None:
        er = pd.to_numeric(df["escalation_ratio"], errors="coerce")
        null_q05 = pd.to_numeric(df["permutation_null_q05_er"], errors="coerce")
        null_q95 = pd.to_numeric(df["permutation_null_q95_er"], errors="coerce")
        null_boundary = null_q95.where(er.ge(1.0), null_q05)

        finite_er = er.replace([math.inf, -math.inf], pd.NA).dropna()
        finite_null = null_boundary.replace([math.inf, -math.inf], pd.NA).dropna()
        # Escalation ratios are heavily right-skewed (a handful of pairs can be
        # 100-1000x while most sit near 1) -- a linear axis crushes nearly every
        # point into a sliver near the origin. Log scale is required for the
        # bulk of pairs to be distinguishable at all, not just the outliers.
        ax_min = min(
            finite_er.quantile(0.01) if not finite_er.empty else 1.0,
            finite_null.quantile(0.01) if not finite_null.empty else 1.0,
            0.95,
        ) * 0.92
        ax_min = max(ax_min, 0.05)
        ax_max = max(
            finite_er.quantile(0.99) if not finite_er.empty else 2.0,
            finite_null.quantile(0.99) if not finite_null.empty else 2.0,
            2.0,
        ) * 1.12

        # Diagonal reference — escalation points beyond the null appear above it;
        # suppression points beyond the null appear below it.
        fig.add_trace(
            go.Scatter(
                x=[ax_min, ax_max], y=[ax_min, ax_max],
                mode="lines",
                line=dict(color="#bbbbbb", dash="dot", width=1.5),
                name="Null boundary = observed ER",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

        for status in _STATUS_ORDER:
            sub = df[df["validation_status"] == status].copy()
            if sub.empty:
                continue
            sub_er = pd.to_numeric(sub["escalation_ratio"], errors="coerce")
            sub_boundary = pd.to_numeric(sub["permutation_null_q95_er"], errors="coerce").where(
                sub_er.ge(1.0),
                pd.to_numeric(sub["permutation_null_q05_er"], errors="coerce"),
            )
            x_vals = sub_boundary.clip(lower=ax_min, upper=ax_max)
            y_vals = sub_er.clip(lower=ax_min, upper=ax_max)
            pair_labels = (
                sub["upstream_antibiotic"].fillna("?")
                + " → "
                + sub["downstream_antibiotic"].fillna("?")
            )
            pvals = pd.to_numeric(sub.get("permutation_p_value", pd.Series(dtype=float)), errors="coerce")
            show_legend = status not in seen_statuses
            seen_statuses.add(status)
            fig.add_trace(
                go.Scatter(
                    x=x_vals.clip(lower=ax_min),
                    y=y_vals.clip(lower=ax_min),
                    mode="markers",
                    name=_STATUS_LABEL[status],
                    legendgroup=status,
                    showlegend=show_legend,
                    marker=dict(
                        color=_STATUS_COLORS_RGBA(status, 0.82),
                        size=9,
                        line=dict(color="white", width=0.8),
                    ),
                    customdata=list(zip(
                        pair_labels,
                        y_vals.round(3),
                        x_vals.round(3),
                        np.where(sub_er.ge(1.0), "escalation", "suppression"),
                        pvals.round(4) if pvals.notna().any() else ["—"] * len(sub),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Observed ER: %{customdata[1]}<br>"
                        "Null boundary ER: %{customdata[2]}<br>"
                        "Direction: %{customdata[3]}<br>"
                        "Permutation p: %{customdata[4]}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        fig.update_xaxes(
            title_text="Direction-specific permutation null boundary ER (log scale)",
            type="log",
            range=[math.log10(ax_min), math.log10(ax_max)],
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Observed escalation ratio (ER, log scale)",
            type="log",
            range=[math.log10(ax_min), math.log10(ax_max)],
            row=1, col=1,
        )

    # ── Panel B ──────────────────────────────────────────────────────────────

    def _panel_b_bootstrap_histogram(self, fig: go.Figure, df: pd.DataFrame, seen_statuses: set[str]) -> None:
        for status in _STATUS_ORDER:
            sub = df[
                (df["validation_status"] == status)
                & df["bootstrap_sign_stability"].notna()
            ].copy()
            if sub.empty:
                continue
            show_legend = status not in seen_statuses
            seen_statuses.add(status)
            fig.add_trace(
                go.Histogram(
                    x=pd.to_numeric(sub["bootstrap_sign_stability"], errors="coerce"),
                    name=_STATUS_LABEL[status],
                    legendgroup=status,
                    showlegend=show_legend,
                    marker_color=_STATUS_COLORS_RGBA(status, 0.78),
                    marker_line=dict(color="white", width=0.6),
                    xbins=dict(start=0.0, end=1.02, size=0.04),
                    hovertemplate="S ∈ %{x:.2f}: %{y} pairs<extra></extra>",
                ),
                row=1, col=2,
            )

        # S = 0.80 threshold line
        fig.add_shape(
            type="line",
            x0=0.80, x1=0.80, y0=0, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color="#d62728", dash="dash", width=2.2),
        )
        fig.add_annotation(
            x=0.80, y=1.0,
            xref="x2", yref="y2 domain",
            text="  S = 0.80 threshold",
            showarrow=False,
            font=dict(size=11, color="#d62728"),
            xanchor="left",
            yanchor="top",
        )
        fig.update_xaxes(
            title_text="Bootstrap sign-stability score (S)",
            range=[0.0, 1.05],
            row=1, col=2,
        )
        fig.update_yaxes(title_text="Number of candidate pairs", row=1, col=2)

    def _write_static(self, df: pd.DataFrame, fmt: str, path: Path) -> None:
        plot_data = df.copy()
        plot_data["pair_label"] = (
            plot_data["upstream_antibiotic"].fillna("?")
            + " to "
            + plot_data["downstream_antibiotic"].fillna("?")
        )
        plot_data["escalation_ratio"] = pd.to_numeric(plot_data["escalation_ratio"], errors="coerce")
        plot_data["permutation_null_q05_er"] = pd.to_numeric(plot_data["permutation_null_q05_er"], errors="coerce")
        plot_data["permutation_null_q95_er"] = pd.to_numeric(plot_data["permutation_null_q95_er"], errors="coerce")
        plot_data["bootstrap_sign_stability"] = pd.to_numeric(plot_data["bootstrap_sign_stability"], errors="coerce")
        plot_data["permutation_null_boundary_er"] = plot_data["permutation_null_q95_er"].where(
            plot_data["escalation_ratio"].ge(1.0),
            plot_data["permutation_null_q05_er"],
        )
        plot_data = plot_data[
            plot_data["escalation_ratio"].notna()
            & plot_data["permutation_null_boundary_er"].notna()
        ].copy()

        fig, (ax_perm, ax_boot) = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.8),
            gridspec_kw={"width_ratios": [1.08, 1.0]},
        )
        if plot_data.empty:
            ax_perm.axis("off")
            ax_boot.axis("off")
            ax_perm.text(0.5, 0.5, "Validation diagnostics unavailable.", ha="center", va="center")
            fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return

        # Keep extreme values visible without letting a few outliers determine
        # all axis scaling in the static export.
        ax_min = min(
            float(plot_data["escalation_ratio"].quantile(0.01)),
            float(plot_data["permutation_null_boundary_er"].quantile(0.01)),
            0.95,
        ) * 0.92
        ax_min = max(ax_min, 0.05)
        ax_max = max(
            float(plot_data["escalation_ratio"].quantile(0.99)),
            float(plot_data["permutation_null_boundary_er"].quantile(0.99)),
            2.0,
        ) * 1.15
        ax_perm.plot([ax_min, ax_max], [ax_min, ax_max], linestyle=":", color="#98A2B3", linewidth=1.4)
        perm_supported = (
            (
                plot_data["escalation_ratio"].ge(1.0)
                & plot_data["escalation_ratio"].gt(plot_data["permutation_null_boundary_er"])
            )
            | (
                plot_data["escalation_ratio"].lt(1.0)
                & plot_data["escalation_ratio"].lt(plot_data["permutation_null_boundary_er"])
            )
        )
        for supported, color, alpha in ((False, "#98A2B3", 0.62), (True, "#1f77b4", 0.86)):
            sub = plot_data[perm_supported.eq(supported)]
            if sub.empty:
                continue
            ax_perm.scatter(
                sub["permutation_null_boundary_er"].clip(lower=ax_min, upper=ax_max),
                sub["escalation_ratio"].clip(lower=ax_min, upper=ax_max),
                s=42,
                color=color,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.6,
            )
        ax_perm.set_xscale("log")
        ax_perm.set_yscale("log")
        ax_perm.set_xlim(ax_min, ax_max)
        ax_perm.set_ylim(ax_min, ax_max)
        ax_perm.set_xlabel("Direction-specific permutation null boundary ER", fontsize=9)
        ax_perm.set_ylabel("Observed escalation ratio", fontsize=9)
        ax_perm.set_title("A. Observed ER vs direction-specific permutation null", loc="left", fontweight="bold", fontsize=10)
        ax_perm.tick_params(axis="both", labelsize=8)
        ax_perm.grid(True, which="both", alpha=0.18)
        ax_perm.text(
            0.02,
            0.98,
            "Blue points: observed ER is beyond the relevant one-sided permutation boundary",
            transform=ax_perm.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#475467",
        )

        stability_values = plot_data["bootstrap_sign_stability"].dropna()
        if not stability_values.empty:
            ax_boot.hist(stability_values.to_numpy(), bins=np.linspace(0, 1.0, 21), color="#667085", alpha=0.82)
        ax_boot.axvline(0.80, color="#d62728", linestyle="--", linewidth=1.6)
        ax_boot.text(0.805, ax_boot.get_ylim()[1] * 0.95, "S = 0.80", color="#d62728", fontsize=8, va="top")
        ax_boot.set_xlabel("Bootstrap sign-stability score", fontsize=9)
        ax_boot.set_ylabel("Number of candidate pairs", fontsize=9)
        ax_boot.set_title("B. Bootstrap sign-stability distribution", loc="left", fontweight="bold", fontsize=10)
        ax_boot.tick_params(axis="both", labelsize=8)
        ax_boot.grid(True, axis="y", alpha=0.18)
        fig.suptitle("Validation Diagnostics — Gates 1–2: Permutation and Bootstrap", fontsize=12, fontweight="bold", y=0.98)
        fig.text(
            0.06,
            0.025,
            "Static export intentionally omits pair labels and status legend. Pair identities and validation status are available in the interactive HTML.",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#344054",
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _STATUS_COLORS_RGBA(status: str, alpha: float) -> str:
    """Convert hex colour to rgba string."""
    hex_c = _STATUS_COLOR.get(status, "#aaaaaa").lstrip("#")
    r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
