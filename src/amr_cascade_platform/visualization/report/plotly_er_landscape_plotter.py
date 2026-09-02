"""ER landscape — full distribution of escalation ratios across all candidate pairs."""

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


class PlotlyERLandscapePlotter:
    """
    Two-panel ER landscape figure.

    Panel A — Strip chart
        Every support-passing pair as a dot, Y = ER, sorted by ER descending.
        Color = validation status.  Error whiskers = bootstrap 95 % CI.
        ER = 1 reference line.

    Panel B — Histogram
        Distribution of ERs binned by validation status (stacked).
        Shows the full landscape including non-validated candidates.
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
        required = {"escalation_ratio", "validation_status", "passes_support_threshold"}
        if edge_report.empty or not required.issubset(edge_report.columns):
            fig = _empty(self._template, self._width, self._height,
                         "ER landscape unavailable — required columns missing from edge report")
            return self._exporter.write(fig, output_stem, formats)

        df = edge_report[edge_report["passes_support_threshold"].eq(True)].copy()
        df["validation_status"] = df["validation_status"].fillna("unavailable")
        df["escalation_ratio"] = pd.to_numeric(df["escalation_ratio"], errors="coerce")
        df = df[
            df["escalation_ratio"].notna()
            & ~df["escalation_ratio"].isin([math.inf, -math.inf])
            & df["escalation_ratio"].gt(0)
        ]
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No support-passing pairs with finite ER — landscape cannot be drawn")
            return self._exporter.write(fig, output_stem, formats)

        df = df.sort_values("escalation_ratio", ascending=False).reset_index(drop=True)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "<b>A</b>   Escalation Ratio per Candidate Pair",
                "<b>B</b>   ER Distribution by Validation Status",
            ),
            column_widths=[0.62, 0.38],
            horizontal_spacing=0.10,
        )
        # Shared across both panels so a status already legended in panel A
        # (e.g. "Robust") isn't legended a second time by panel B.
        seen: set[str] = set()
        self._panel_a_strip(fig, df, seen)
        self._panel_b_histogram(fig, df, seen)

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text="Escalation Ratio Landscape — All Support-Passing Candidate Pairs",
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
            margin=dict(t=110, b=135, l=70, r=50),
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

    # ── Panel A — strip chart ─────────────────────────────────────────────

    def _panel_a_strip(self, fig: go.Figure, df: pd.DataFrame, seen: set[str]) -> None:
        has_ci = {"bootstrap_er_ci_lower", "bootstrap_er_ci_upper"}.issubset(df.columns)
        pair_labels = (
            df["upstream_antibiotic"].fillna("?")
            + " → "
            + df["downstream_antibiotic"].fillna("?")
        )
        x_index = np.arange(len(df))

        # ER = 1 reference
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(df) - 0.5,
            y0=1.0, y1=1.0,
            xref="x", yref="y",
            line=dict(color="#bbbbbb", dash="dot", width=1.5),
        )

        for status in _STATUS_ORDER:
            mask = df["validation_status"] == status
            sub = df[mask]
            if sub.empty:
                continue
            idx = x_index[mask.values]
            er = sub["escalation_ratio"].values
            error_y: dict | None = None
            if has_ci:
                lo = pd.to_numeric(sub["bootstrap_er_ci_lower"], errors="coerce").values
                hi = pd.to_numeric(sub["bootstrap_er_ci_upper"], errors="coerce").values
                error_y = dict(
                    type="data",
                    symmetric=False,
                    array=(hi - er).clip(min=0),
                    arrayminus=(er - lo).clip(min=0),
                    color=_hex_rgba(_STATUS_COLOR[status], 0.35),
                    thickness=1.2,
                    width=3,
                )
            support = pd.to_numeric(sub.get("total_support_n", pd.Series([10] * len(sub))), errors="coerce").fillna(10)
            size = (np.sqrt(support).clip(3, 30)).astype(float)

            show_leg = status not in seen
            seen.add(status)
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=er,
                    mode="markers",
                    name=_STATUS_LABEL[status],
                    legendgroup=status,
                    showlegend=show_leg,
                    marker=dict(
                        color=_hex_rgba(_STATUS_COLOR[status], 0.85),
                        size=size,
                        line=dict(color="white", width=0.5),
                    ),
                    error_y=error_y,
                    customdata=list(zip(
                        pair_labels[mask].values,
                        er.round(3),
                        support.astype(int).values,
                        [status] * len(sub),
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "ER: %{customdata[1]}<br>"
                        "Support n: %{customdata[2]}<br>"
                        "Status: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        fig.update_xaxes(
            title_text="Candidate pairs ranked by ER (descending); pair names are available in HTML hover text",
            showticklabels=False,
            row=1, col=1,
        )
        # Escalation ratios are heavily right-skewed (a few pairs reach
        # 1000x+ while most sit near 1) -- a linear axis crushes nearly
        # every point into a sliver near zero. Log scale is required for
        # the bulk of pairs to be distinguishable, not just the outliers.
        fig.update_yaxes(title_text="Escalation ratio (ER, log scale)", type="log", row=1, col=1)

    # ── Panel B — histogram ────────────────────────────────────────────────

    def _panel_b_histogram(self, fig: go.Figure, df: pd.DataFrame, seen: set[str]) -> None:
        # Bin in log10(ER) space, not raw ER. Do not clip values below 1:
        # suppression-direction edges (ER < 1) are valid when suppression
        # validation is enabled and should remain visible left of the ER=1 line.
        log_er_all = np.log10(df["escalation_ratio"])
        log_er_all = log_er_all.replace([math.inf, -math.inf], pd.NA).dropna()
        log_min = min(float(log_er_all.quantile(0.01)) if not log_er_all.empty else 0.0, -0.3)
        log_max = max(float(log_er_all.quantile(0.99)) if not log_er_all.empty else 0.0, 0.3)
        span = max(log_max - log_min, 0.6)
        bin_size = max(0.05, round(span / 24, 3))

        for status in _STATUS_ORDER:
            sub = df[df["validation_status"] == status]
            if sub.empty:
                continue
            show_leg = status not in seen
            seen.add(status)
            log_er = np.log10(sub["escalation_ratio"]).clip(lower=log_min, upper=log_max)
            fig.add_trace(
                go.Histogram(
                    x=log_er,
                    name=_STATUS_LABEL[status],
                    legendgroup=status,
                    showlegend=show_leg,
                    marker_color=_hex_rgba(_STATUS_COLOR[status], 0.78),
                    marker_line=dict(color="white", width=0.5),
                    xbins=dict(start=log_min, end=log_max, size=bin_size),
                    customdata=(10**log_er).round(2),
                    hovertemplate="ER ≈ %{customdata}: %{y} pairs<extra></extra>",
                ),
                row=1, col=2,
            )

        # ER = 1 reference line — add as shape anchored to col 2 axes
        fig.add_shape(
            type="line",
            x0=0.0, x1=0.0, y0=0, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color="#bbbbbb", dash="dot", width=1.5),
        )
        tick_vals = list(np.linspace(log_min, log_max, 6))
        fig.update_xaxes(
            title_text="Escalation ratio (ER, log scale)",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[_format_er_tick(10**v) for v in tick_vals],
            row=1, col=2,
        )
        fig.update_yaxes(title_text="Number of pairs", row=1, col=2)

    def _write_static(self, df: pd.DataFrame, fmt: str, path: Path) -> None:
        """Matplotlib static renderer for the distribution-level ER landscape."""

        plot_data = df.copy()
        if plot_data.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.axis("off")
            ax.text(0.5, 0.5, "No support-passing pairs with finite ER.", ha="center", va="center")
            fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return

        x_index = np.arange(len(plot_data))
        fig, (ax_strip, ax_hist) = plt.subplots(
            1,
            2,
            figsize=(17, 8),
            gridspec_kw={"width_ratios": [1.55, 1.0]},
        )

        for status in _STATUS_ORDER:
            sub = plot_data[plot_data["validation_status"].eq(status)]
            if sub.empty:
                continue
            idx = sub.index.to_numpy()
            er = pd.to_numeric(sub["escalation_ratio"], errors="coerce")
            support = pd.to_numeric(sub.get("total_support_n", pd.Series(10, index=sub.index)), errors="coerce").fillna(10)
            sizes = np.sqrt(support).clip(4, 22) * 8
            color = _STATUS_COLOR[status]

            if {"bootstrap_er_ci_lower", "bootstrap_er_ci_upper"}.issubset(sub.columns):
                lo = pd.to_numeric(sub["bootstrap_er_ci_lower"], errors="coerce")
                hi = pd.to_numeric(sub["bootstrap_er_ci_upper"], errors="coerce")
                yerr = np.vstack([(er - lo).clip(lower=0).fillna(0), (hi - er).clip(lower=0).fillna(0)])
                ax_strip.errorbar(idx, er, yerr=yerr, fmt="none", ecolor=color, alpha=0.25, linewidth=1)

            ax_strip.scatter(
                idx,
                er,
                s=sizes,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.7,
                label=_STATUS_LABEL[status],
                zorder=3,
            )

        ax_strip.set_xticks([])
        ax_strip.axhline(1.0, color="#98A2B3", linestyle=":", linewidth=1.3)
        ax_strip.set_yscale("log")
        ax_strip.set_ylabel("Escalation ratio (ER, log scale)")
        ax_strip.set_xlabel("Candidate pairs ranked by ER")
        ax_strip.set_title("A. Escalation ratio per candidate pair", loc="left", fontweight="bold")
        ax_strip.grid(True, which="both", axis="y", alpha=0.18)

        handles, labels = ax_strip.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=False))
            ax_strip.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, loc="upper right")

        log_values_by_status: list[np.ndarray] = []
        hist_labels: list[str] = []
        hist_colors: list[str] = []
        for status in _STATUS_ORDER:
            sub = plot_data[plot_data["validation_status"].eq(status)]
            if sub.empty:
                continue
            er_values = pd.to_numeric(sub["escalation_ratio"], errors="coerce")
            log_values_by_status.append(np.log10(er_values[er_values.gt(0)]).dropna().to_numpy())
            hist_labels.append(_STATUS_LABEL[status])
            hist_colors.append(_STATUS_COLOR[status])
        if log_values_by_status:
            all_log = np.concatenate([vals for vals in log_values_by_status if len(vals)])
            min_log = min(float(np.nanquantile(all_log, 0.01)), -0.3)
            max_log = max(float(np.nanquantile(all_log, 0.99)), 0.3)
            bins = np.linspace(min_log, max_log, 18)
            ax_hist.hist(log_values_by_status, bins=bins, stacked=True, label=hist_labels, color=hist_colors, alpha=0.82)
            tick_vals = np.linspace(min_log, max_log, 6)
            ax_hist.set_xticks(tick_vals)
            ax_hist.set_xticklabels([_format_er_tick(10 ** val) for val in tick_vals])
        ax_hist.axvline(0.0, color="#98A2B3", linestyle=":", linewidth=1.3)
        ax_hist.set_xlabel("Escalation ratio (ER, log scale)")
        ax_hist.set_ylabel("Number of pairs")
        ax_hist.set_title("B. ER distribution by validation status", loc="left", fontweight="bold")
        ax_hist.grid(True, axis="y", alpha=0.18)
        ax_hist.legend(frameon=False, fontsize=9)

        fig.suptitle(
            "Escalation Ratio Landscape — All Support-Passing Candidate Pairs",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.06,
            0.025,
            "Static export omits pair labels. Pair identities are available in the interactive HTML hover text.",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#344054",
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _format_er_tick(value: float) -> str:
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value < 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"
