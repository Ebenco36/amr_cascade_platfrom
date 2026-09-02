"""Plotly figures for validated AWaRe transition summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyAwareTransitionPlotter:
    """Export publication-facing AWaRe transition summary figures."""

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        classification_resolver: AntibioticClassificationResolver,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._classification = classification_resolver

    def export_transition_heatmap(
        self,
        transition_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        if transition_summary.empty:
            fig = self._empty_figure("No validated AWaRe transition data available")
            return self._exporter.write(
                fig,
                output_stem,
                formats,
                static_fallback=lambda fmt, path: self._write_static(transition_summary, fmt, path),
            )

        ordered_categories = [
            category
            for category in AntibioticClassificationResolver.CATEGORY_ORDER
            if category in set(transition_summary["upstream_aware_category"]).union(
                set(transition_summary["downstream_aware_category"])
            )
        ]
        if not ordered_categories:
            ordered_categories = list(AntibioticClassificationResolver.CATEGORY_ORDER)

        heatmap = (
            transition_summary.pivot_table(
                index="upstream_aware_category",
                columns="downstream_aware_category",
                values="support_weighted_mean_escalation_ratio",
                aggfunc="mean",
            )
            .reindex(index=ordered_categories, columns=ordered_categories)
        )
        counts = (
            transition_summary.pivot_table(
                index="upstream_aware_category",
                columns="downstream_aware_category",
                values="validated_edge_n",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=ordered_categories, columns=ordered_categories, fill_value=0)
        )
        support = (
            transition_summary.pivot_table(
                index="upstream_aware_category",
                columns="downstream_aware_category",
                values="total_support_sum",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=ordered_categories, columns=ordered_categories, fill_value=0)
        )
        hover = []
        text = []
        for upstream in ordered_categories:
            hover_row: list[str] = []
            text_row: list[str] = []
            for downstream in ordered_categories:
                subset = transition_summary.loc[
                    (transition_summary["upstream_aware_category"] == upstream)
                    & (transition_summary["downstream_aware_category"] == downstream)
                ]
                edge_n = int(counts.loc[upstream, downstream]) if downstream in counts.columns else 0
                support_sum = int(support.loc[upstream, downstream]) if downstream in support.columns else 0
                weighted_er = heatmap.loc[upstream, downstream]
                robust_n = int(subset["robust_edge_n"].sum()) if not subset.empty else 0
                supported_n = int(subset["supported_edge_n"].sum()) if not subset.empty else 0
                hover_row.append(
                    "<b>"
                    + self._classification.transition_label(upstream, downstream)
                    + "</b><br>"
                    + f"Validated edges={edge_n}<br>"
                    + f"Robust={robust_n} | Supported={supported_n}<br>"
                    + f"Total support={support_sum}<br>"
                    + (
                        "Support-weighted mean ER=NA"
                        if pd.isna(weighted_er)
                        else f"Support-weighted mean ER={weighted_er:.3f}"
                    )
                )
                text_row.append("" if edge_n == 0 else f"{edge_n}")
            hover.append(hover_row)
            text.append(text_row)

        # Support-weighted mean ER is a ratio -- it cannot be negative, and every
        # edge here already passed the ER>=1.0 retention gate, so real values
        # cluster at the low end of a heavily right-skewed range (a few Watch->
        # Reserve cells reach the hundreds). zmid=1.0 previously forced Plotly to
        # build a *symmetric* color range around 1.0, which silently allocates
        # roughly half the colorscale to negative values that can never occur --
        # exactly the wasted-dynamic-range problem the escalation-matrix heatmap
        # fix addressed earlier. Color on log10 instead, with colorbar ticks
        # remapped back to raw ER so it still reads as "escalation ratio."
        z_raw = heatmap.values.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_color = np.where(z_raw > 0, np.log10(np.clip(z_raw, 1e-9, None)), np.nan)
        finite_log = z_color[~np.isnan(z_color)]
        colorbar_ticks: dict[str, list] = {}
        if finite_log.size:
            lo, hi = float(finite_log.min()), float(finite_log.max())
            tick_positions = np.linspace(lo, hi, num=min(6, max(2, int(round(hi - lo)) + 1)))
            colorbar_ticks = {
                "tickvals": tick_positions.tolist(),
                "ticktext": [f"{10 ** value:,.1f}" for value in tick_positions],
            }
        fig = go.Figure(
            data=go.Heatmap(
                z=z_color,
                x=ordered_categories,
                y=ordered_categories,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "#0F172A"},
                colorscale="YlOrRd",
                colorbar={"title": "Support-weighted<br>mean ER", **colorbar_ticks},
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title={
                "text": "Validated AWaRe Transition Heatmap",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            xaxis_title="Downstream AWaRe category",
            yaxis_title="Upstream AWaRe category",
            margin={"l": 110, "r": 70, "t": 110, "b": 120},
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"size": 16, "family": "Arial"},
        )
        fig.add_annotation(
            text=(
                "Cells show the number of validated edges; color encodes the support-weighted mean escalation ratio for "
                "robust and supported edges only."
            ),
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            xanchor="left",
            showarrow=False,
            font={"size": 13, "color": "#475467"},
        )
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(transition_summary, fmt, path),
        )

    def _empty_figure(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    @staticmethod
    def _write_static(transition_summary: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9.5, 7.2))
        if transition_summary.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No validated AWaRe transition data available", ha="center", va="center")
        else:
            ordered_categories = [
                category
                for category in AntibioticClassificationResolver.CATEGORY_ORDER
                if category in set(transition_summary["upstream_aware_category"]).union(
                    set(transition_summary["downstream_aware_category"])
                )
            ]
            if not ordered_categories:
                ordered_categories = list(AntibioticClassificationResolver.CATEGORY_ORDER)

            heatmap = (
                transition_summary.pivot_table(
                    index="upstream_aware_category",
                    columns="downstream_aware_category",
                    values="support_weighted_mean_escalation_ratio",
                    aggfunc="mean",
                )
                .reindex(index=ordered_categories, columns=ordered_categories)
            )
            counts = (
                transition_summary.pivot_table(
                    index="upstream_aware_category",
                    columns="downstream_aware_category",
                    values="validated_edge_n",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(index=ordered_categories, columns=ordered_categories, fill_value=0)
            )
            masked = np.ma.masked_invalid(heatmap.to_numpy(dtype=float))
            cmap = plt.get_cmap("YlOrRd").copy()
            cmap.set_bad(color="#F3F4F6")
            image = ax.imshow(masked, cmap=cmap, aspect="auto")
            colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
            colorbar.set_label("Support-weighted mean ER", fontsize=13)

            ax.set_xticks(np.arange(len(ordered_categories)))
            ax.set_yticks(np.arange(len(ordered_categories)))
            ax.set_xticklabels(ordered_categories, rotation=0, fontsize=13)
            ax.set_yticklabels(ordered_categories, fontsize=13)
            ax.set_xlabel("Downstream AWaRe category", fontsize=13)
            ax.set_ylabel("Upstream AWaRe category", fontsize=13)
            ax.set_title("Validated AWaRe Transition Heatmap", fontsize=18, fontweight="bold", pad=16)
            ax.tick_params(axis="both", labelsize=13)

            for i, upstream in enumerate(ordered_categories):
                for j, downstream in enumerate(ordered_categories):
                    count = int(counts.loc[upstream, downstream])
                    if count == 0:
                        continue
                    value = heatmap.loc[upstream, downstream]
                    label = f"n={count}"
                    if pd.notna(value):
                        label += f"\nER {value:.2f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=13, color="#111827")

            ax.set_xticks(np.arange(-0.5, len(ordered_categories), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(ordered_categories), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.text(
                0.02,
                0.01,
                "Cell labels show validated edge count and support-weighted mean ER for robust and supported edges only.",
                fontsize=12,
                color="#475467",
            )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
