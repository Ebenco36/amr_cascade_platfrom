"""Temporal stability scatter — early-period vs late-period escalation ratio."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_GRID = "#E8EEF5"
_NEUTRAL = "#aaaaaa"

_STATUS_ORDER = ["robust", "supported", "mixed", "insufficient", "unavailable"]
_STATUS_COLORS = {
    "robust":       "#2ca02c",
    "supported":    "#1f77b4",
    "mixed":        "#ff7f0e",
    "insufficient": "#9467bd",
    "unavailable":  "#cccccc",
}
_STATUS_LABELS = {
    "robust":       "Robust (all 3 tests)",
    "supported":    "Supported (permutation + bootstrap)",
    "mixed":        "Mixed (≥1 test)",
    "insufficient": "Insufficient support",
    "unavailable":  "Unavailable",
}


def _safe_log(s: pd.Series) -> pd.Series:
    return np.log10(s.clip(lower=1e-6))


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper", font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyTemporalStabilityPlotter:
    """
    Two-panel temporal stability figure.

    Panel A — Early-period vs late-period ER scatter (log-log).
        Each dot = one candidate pair.
        Colour = validation_status; size = sqrt(total_support_n).
        Diagonal y = x is perfect temporal stability.
        Points above diagonal: signal strengthened over time (late ER > early ER).
        Points below diagonal: signal weakened.

    Panel B — Temporal direction-agreement strip chart.
        X = temporal_direction_agreement (fraction of site-halves where ER > 1).
        Y = pair label, sorted ascending.
        Colour = validation_status.
        Vertical dashed reference at 0.50 (random-chance agreement).
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
        needed = {"temporal_early_er", "temporal_late_er"}
        if edge_report.empty or not needed.issubset(edge_report.columns):
            fig = _empty(
                self._template, self._width, self._height,
                "Temporal stability chart unavailable — "
                "temporal_early_er / temporal_late_er columns missing",
            )
            return self._exporter.write(fig, output_stem, formats)

        df = edge_report.copy()
        for col in ("temporal_early_er", "temporal_late_er", "total_support_n",
                    "temporal_direction_agreement"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["temporal_early_er"].notna() & df["temporal_late_er"].notna()].copy()
        if df.empty:
            fig = _empty(self._template, self._width, self._height,
                         "No pairs with temporal early/late ER data")
            return self._exporter.write(fig, output_stem, formats)

        df["pair_label"] = (
            df["upstream_antibiotic"].fillna("?") + " → " + df["downstream_antibiotic"].fillna("?")
        )
        df["validation_status"] = df.get("validation_status", pd.Series("unavailable", index=df.index)).fillna("unavailable")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "<b>A</b>   Early-Period vs Late-Period Escalation Ratio",
                "<b>B</b>   Direction-Concordant Fraction by Validation Status"
                "<br><sup>Robust and Mixed only — Supported/Insufficient are 0% by definition, see note below</sup>",
            ),
            horizontal_spacing=0.12,
        )

        # Shared across both panels so a status already legended in panel A
        # isn't legended a second time by panel B.
        shown: set[str] = set()
        self._panel_a(fig, df, shown)
        self._panel_b(fig, df, shown)

        n_concordant = int((df.get("temporal_direction_agreement", pd.Series(dtype=float)) >= 0.5).sum()) if "temporal_direction_agreement" in df.columns else "—"

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text=(
                    f"Temporal Stability — Cascade Signal Consistency Across Time Periods  "
                    f"(n = {len(df)} pairs, direction-concordant = {n_concordant})"
                ),
                x=0.5,
                font=dict(size=15),
            ),
            legend=dict(
                title="Validation status",
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
            margin=dict(t=100, b=150, l=70, r=50),
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

    # ── Panel A: early vs late scatter ────────────────────────────────────

    def _panel_a(self, fig: go.Figure, df: pd.DataFrame, shown: set[str]) -> None:
        # Upper clip guards against near-zero-denominator blowups in the early/late
        # split (finite but astronomically large, e.g. 1e100+) that would otherwise
        # stretch the shared log axis until every real pair collapses to one pixel.
        # 1e3 is far above any physically plausible escalation ratio, so genuine
        # values are never touched -- this only pins runaway artifacts to a
        # readable ceiling, mirroring the existing near-zero floor below.
        early = df["temporal_early_er"].replace([math.inf, -math.inf], pd.NA).clip(lower=1e-3, upper=1e3)
        late  = df["temporal_late_er"].replace([math.inf, -math.inf], pd.NA).clip(lower=1e-3, upper=1e3)
        size_col = df.get("total_support_n", pd.Series(100, index=df.index)).fillna(100)
        marker_sizes = np.sqrt(size_col.clip(lower=4)).clip(4, 20)

        for status in _STATUS_ORDER:
            mask = df["validation_status"].eq(status) & early.notna() & late.notna()
            if not mask.any():
                continue
            label = _STATUS_LABELS.get(status, status)
            show_legend = label not in shown
            shown.add(label)
            sub = df[mask]
            fig.add_trace(
                go.Scatter(
                    x=early[mask].values,
                    y=late[mask].values,
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    marker=dict(
                        color=_STATUS_COLORS.get(status, _NEUTRAL),
                        size=marker_sizes[mask].values,
                        opacity=0.80,
                        line=dict(color="white", width=0.6),
                    ),
                    customdata=list(zip(
                        sub["pair_label"].values,
                        early[mask].round(3).values,
                        late[mask].round(3).values,
                        sub.get("total_support_n", pd.Series(dtype=float)).reindex(sub.index).fillna(0).astype(int).values,
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Early ER: %{customdata[1]:.3f}<br>"
                        "Late ER:  %{customdata[2]:.3f}<br>"
                        "N: %{customdata[3]:,}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Diagonal reference y = x
        all_vals = pd.concat([early, late]).dropna()
        vmin, vmax = 1e-3, 1.0
        if not all_vals.empty:
            vmin = max(all_vals.min() * 0.8, 1e-3)
            vmax = all_vals.max() * 1.2
            fig.add_shape(
                type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                xref="x", yref="y",
                line=dict(color=_NEUTRAL, dash="dash", width=1.5),
            )
            # Domain-relative (not data-coordinate) placement: a data-coordinate
            # annotation on a log axis inside a multi-subplot figure makes
            # kaleido's autorange computation blow up by dozens of orders of
            # magnitude (observed: axis stretching to 1e161) even though every
            # trace/shape value stays within [1e-3, 1e3]. Anchoring to the
            # panel's own top-right corner sidesteps that autorange path
            # entirely and is also the more correct intent -- this label marks
            # a fixed reference corner, not the literal (vmax, vmax) point.
            fig.add_annotation(
                x=0.98, y=0.98, xref="x domain", yref="y domain",
                text="y = x (stable)", showarrow=False,
                font=dict(size=13, color=_NEUTRAL), xanchor="right",
            )
            # ER = 1 reference lines
            fig.add_shape(type="line", x0=1, y0=vmin, x1=1, y1=vmax,
                          xref="x", yref="y", line=dict(color="#dddddd", dash="dot", width=1.0))
            fig.add_shape(type="line", x0=vmin, y0=1, x1=vmax, y1=1,
                          xref="x", yref="y", line=dict(color="#dddddd", dash="dot", width=1.0))

        # Explicit range (log10 units) rather than relying on autorange -- belt
        # and suspenders alongside the domain-relative annotation above, and
        # keeps the plotted extent tied to real data instead of whatever
        # kaleido's own padding heuristic would otherwise pick.
        log_range = [math.log10(vmin), math.log10(vmax)]
        fig.update_xaxes(
            title_text="Early-period ER",
            type="log",
            range=log_range,
            showgrid=True, gridcolor=_GRID,
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Late-period ER",
            type="log",
            range=log_range,
            showgrid=True, gridcolor=_GRID,
            row=1, col=1,
        )

    # ── Panel B: direction-concordance rate by validation status ───────────
    # A per-pair scatter with one y-tick label per pair is unreadable once
    # there are more than a few dozen pairs (this dataset routinely has
    # 300+) -- every label collides into an unreadable block. Aggregating to
    # a concordance rate per validation_status answers the actual question
    # (does temporal stability track validation robustness?) directly,
    # rather than showing per-pair detail no reader can parse visually.
    #
    # "supported" and "insufficient" are excluded here, not merely filtered
    # for tidiness: _classify_validation() (cascade_validation_analyzer.py)
    # is an if/elif chain where "robust" requires replication_supported
    # (site-agreement OR temporal_direction_agreement) but "supported" is
    # only reached when that same check fails -- which forces temporal
    # agreement to False for every "supported" row. "insufficient" fails
    # replication_supported too, for the same reason. Their concordance
    # rate is therefore 0% by construction of the classification, not an
    # empirical result, so plotting it alongside "robust"/"mixed" (which are
    # not determined this way) would misrepresent a tautology as a finding.
    _INFORMATIVE_STATUSES = ("robust", "mixed")

    def _panel_b(self, fig: go.Figure, df: pd.DataFrame, shown: set[str]) -> None:
        if "temporal_direction_agreement" not in df.columns:
            fig.add_annotation(
                text="temporal_direction_agreement not available",
                x=0.75, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color="#888"),
            )
            return

        sub = df[df["temporal_direction_agreement"].notna()].copy()
        if sub.empty:
            return

        rows = []
        for status in self._INFORMATIVE_STATUSES:
            chunk = sub[sub["validation_status"].eq(status)]
            if chunk.empty:
                continue
            concordant = int((chunk["temporal_direction_agreement"] >= 0.5).sum())
            rows.append(
                {
                    "status": status,
                    "label": _STATUS_LABELS.get(status, status),
                    "n": len(chunk),
                    "concordant": concordant,
                    "rate": concordant / len(chunk),
                }
            )
        if not rows:
            return

        for row in rows:
            label = row["label"]
            show_legend = label not in shown
            shown.add(label)
            fig.add_trace(
                go.Bar(
                    x=[row["rate"]],
                    y=[label],
                    orientation="h",
                    name=label,
                    legendgroup=row["status"],
                    showlegend=show_legend,
                    marker_color=_STATUS_COLORS.get(row["status"], _NEUTRAL),
                    text=[f"{row['concordant']}/{row['n']}"],
                    textposition="outside",
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Direction-concordant: %{x:.0%}<br>"
                        f"{row['concordant']} of {row['n']} pairs<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

        # Reference at 0.50
        fig.add_shape(
            type="line", x0=0.5, y0=0, x1=0.5, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color="#d62728", dash="dash", width=2),
        )
        fig.add_annotation(
            x=0.5, y=1.0, xref="x2", yref="y2 domain",
            text="  0.5 (chance)", showarrow=False,
            font=dict(size=13, color="#d62728"), xanchor="left", yanchor="top",
        )
        fig.add_annotation(
            x=0.0, y=-0.13, xref="paper", yref="paper",
            text=(
                "‘Supported’/‘insufficient’ omitted from panel B: their temporal concordance is 0% by "
                "definition of the validation classification, not an empirical result."
            ),
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=13, color="#888888"),
        )
        # "outside" bar-end text needs real room past the bar, not just enough
        # for the axis line: a bar sitting near 100% (as "Robust" typically
        # does here) left the "N/M" label with only ~5% of axis width to
        # render in and got clipped by the figure edge. 1.05 -> 1.30 gives it
        # the same margin regardless of how close to 100% any given bar is.
        fig.update_xaxes(
            title_text="Direction-concordant fraction",
            range=[0, 1.30],
            tickformat=".0%",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            row=1, col=2,
        )
        fig.update_yaxes(
            title_text="",
            tickfont=dict(size=13),
            row=1, col=2,
        )

    # ── Publication static rendering ─────────────────────────────────────

    def _write_static(self, df: pd.DataFrame, fmt: str, path: Path) -> None:
        """Draw a compact static version without Plotly subplot-title overlap."""

        fig, (ax_scatter, ax_bar) = plt.subplots(
            1,
            2,
            figsize=(13, 6.8),
            dpi=300,
            gridspec_kw={"width_ratios": [1.05, 0.95], "wspace": 0.32},
        )

        fig.suptitle(
            f"Temporal Stability of Candidate Cascade Edges (n = {len(df)} pairs)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        early = pd.to_numeric(df["temporal_early_er"], errors="coerce").replace([math.inf, -math.inf], np.nan).clip(lower=1e-3, upper=1e3)
        late = pd.to_numeric(df["temporal_late_er"], errors="coerce").replace([math.inf, -math.inf], np.nan).clip(lower=1e-3, upper=1e3)
        support = pd.to_numeric(df.get("total_support_n", pd.Series(100, index=df.index)), errors="coerce").fillna(100)
        sizes = np.sqrt(support.clip(lower=4)).clip(14, 70)

        for status in _STATUS_ORDER:
            mask = df["validation_status"].eq(status) & early.notna() & late.notna()
            if not mask.any():
                continue
            ax_scatter.scatter(
                early[mask],
                late[mask],
                s=sizes[mask],
                c=_STATUS_COLORS.get(status, _NEUTRAL),
                alpha=0.82,
                edgecolors="white",
                linewidths=0.6,
                label=_STATUS_LABELS.get(status, status),
            )

        all_vals = pd.concat([early, late]).dropna()
        if all_vals.empty:
            vmin, vmax = 1e-3, 1
        else:
            vmin = max(float(all_vals.min()) * 0.8, 1e-3)
            vmax = max(float(all_vals.max()) * 1.2, 1.2)
        ax_scatter.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="#9AA4B2", linewidth=1.4)
        ax_scatter.axvline(1, linestyle=":", color="#D0D5DD", linewidth=1.0)
        ax_scatter.axhline(1, linestyle=":", color="#D0D5DD", linewidth=1.0)
        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlim(vmin, vmax)
        ax_scatter.set_ylim(vmin, vmax)
        ax_scatter.set_title("A. Early vs late escalation ratio", loc="left", fontsize=11, fontweight="bold")
        ax_scatter.set_xlabel("Early-period ER")
        ax_scatter.set_ylabel("Late-period ER")
        ax_scatter.grid(True, which="both", color=_GRID, linewidth=0.7)
        ax_scatter.legend(frameon=False, fontsize=7.5, loc="lower right")

        rows = []
        if "temporal_direction_agreement" in df.columns:
            sub = df[pd.to_numeric(df["temporal_direction_agreement"], errors="coerce").notna()].copy()
            sub["temporal_direction_agreement"] = pd.to_numeric(sub["temporal_direction_agreement"], errors="coerce")
            for status in self._INFORMATIVE_STATUSES:
                chunk = sub[sub["validation_status"].eq(status)]
                if chunk.empty:
                    continue
                concordant = int((chunk["temporal_direction_agreement"] >= 0.5).sum())
                rows.append((_STATUS_LABELS.get(status, status), concordant, len(chunk), concordant / len(chunk), _STATUS_COLORS.get(status, _NEUTRAL)))

        ax_bar.set_title("B. Direction-concordant fraction", loc="left", fontsize=11, fontweight="bold")
        if rows:
            labels = [r[0] for r in rows]
            rates = [r[3] for r in rows]
            colors = [r[4] for r in rows]
            y = np.arange(len(rows))
            ax_bar.barh(y, rates, color=colors, alpha=0.85)
            for yy, (_, concordant, total, rate, _) in zip(y, rows, strict=False):
                ax_bar.text(min(rate + 0.025, 1.16), yy, f"{concordant}/{total}", va="center", fontsize=10)
            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels(labels, fontsize=9)
            ax_bar.set_xlim(0, 1.2)
            ax_bar.axvline(0.5, linestyle="--", color="#D62728", linewidth=1.4)
            ax_bar.text(0.505, len(rows) - 0.52, "0.5", color="#D62728", fontsize=9, va="top")
            ax_bar.set_xlabel("Fraction direction-concordant")
            ax_bar.grid(True, axis="x", color=_GRID, linewidth=0.7)
        else:
            ax_bar.axis("off")
            ax_bar.text(0.5, 0.5, "Temporal direction-agreement not evaluable", ha="center", va="center", fontsize=11, color="#667085")

        fig.text(
            0.02,
            0.02,
            "Panel B omits supported/insufficient labels because their temporal concordance is fixed by the validation rule, not an empirical comparison.",
            fontsize=8.5,
            color="#667085",
        )
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
