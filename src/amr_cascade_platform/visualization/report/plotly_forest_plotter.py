"""Plotly forest plots for cascade reporting."""

from __future__ import annotations

import gc
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from amr_cascade_platform.core.utils.text import safe_feature_name
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class EdgeReportNormalizer:
    """Normalize retained-edge and edge-report tables into one plotting schema."""

    @staticmethod
    def normalize(edge_report: pd.DataFrame) -> pd.DataFrame:
        if edge_report.empty:
            return edge_report
        normalized = edge_report.rename(
            columns={
                "positive_support_n": "resistant_support_n",
                "positive_tested_n": "resistant_tested_n",
                "negative_support_n": "susceptible_support_n",
                "negative_tested_n": "susceptible_tested_n",
                "positive_probability": "resistant_downstream_test_probability",
                "negative_probability": "susceptible_downstream_test_probability",
                "is_retained_edge": "retained_edge",
            }
        ).copy()
        defaults = {
            "adjusted_odds_ratio": 1.0,
            "total_support_n": 0,
            "resistant_support_n": 0,
            "resistant_tested_n": 0,
            "susceptible_support_n": 0,
            "susceptible_tested_n": 0,
            "resistant_downstream_test_probability": 0.0,
            "susceptible_downstream_test_probability": 0.0,
        }
        for column, default in defaults.items():
            if column not in normalized.columns:
                normalized[column] = default
        return normalized


class PlotlyForestPlotter:
    """Create Plotly forest plots for cascade edge summaries."""

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        top_n: int,
        classification_resolver: AntibioticClassificationResolver,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._top_n = top_n
        self._classification = classification_resolver

    @staticmethod
    def _marker_sizes(values: pd.Series) -> pd.Series:
        support = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0, upper=500)
        return (8 + np.sqrt(support)).clip(lower=8, upper=18)

    def export_primary_forest(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        plot_data = self._canonicalize_pair_table(EdgeReportNormalizer.normalize(edge_report)).sort_values(
            by="escalation_ratio", ascending=False
        ).head(self._top_n)
        plot_data = self._attach_rr_confidence_intervals(plot_data)
        if plot_data.empty:
            fig = self._empty_figure("No retained edges available")
        else:
            ordered = self._classification.add_metadata(plot_data.iloc[::-1].copy(), "downstream_antibiotic", "downstream")
            ordered["edge_label"] = ordered["upstream_antibiotic"] + " -> " + ordered["downstream_antibiotic"]
            ordered["marker_symbol"] = "square"
            fig = self._build_forest_figure(
                ordered,
                x_column="escalation_ratio",
                y_column="edge_label",
                title="Primary Cascade Forest: Top Retained Directed Pairs",
                xaxis_title="Escalation Ratio",
                include_adjusted_or=True,
                metadata_prefix="downstream",
                subtitle="Showing the strongest retained directed pairs after support, panel-bundling, and raw-ER screening.",
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#c0392b")
            fig.update_layout(
                xaxis_type="log",
                height=max(720, 36 * len(ordered) + 220),
                margin={"l": 260, "r": 80, "t": 110, "b": 140},
            )
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_primary_static(plot_data, fmt, path),
        )

    def export_validated_primary_forest(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> dict[str, Path]:
        plot_data = self._canonicalize_pair_table(EdgeReportNormalizer.normalize(edge_report))
        if "validation_status" in plot_data.columns:
            plot_data = plot_data.loc[plot_data["validation_status"].isin(allowed_statuses)].copy()
        else:
            plot_data = plot_data.iloc[0:0].copy()
        plot_data = plot_data.sort_values(
            by=["validation_status", "escalation_ratio", "adjusted_odds_ratio", "total_support_n"],
            ascending=[True, False, False, False],
        ).head(self._top_n)
        plot_data = self._attach_rr_confidence_intervals(plot_data)
        if plot_data.empty:
            fig = self._empty_figure("No robust or supported retained edges available")
        else:
            ordered = self._classification.add_metadata(plot_data.iloc[::-1].copy(), "downstream_antibiotic", "downstream")
            ordered["edge_label"] = ordered["upstream_antibiotic"] + " -> " + ordered["downstream_antibiotic"]
            ordered["marker_symbol"] = "square"
            ordered["marker_fill_color"] = ordered["downstream_aware_color"]
            ordered["marker_line_color"] = ordered["validation_status"].map(
                {
                    "robust": "#111827",
                    "supported": "#C2410C",
                }
            ).fillna("#64748B")
            ordered["marker_opacity"] = ordered["validation_status"].map(
                {
                    "robust": 0.95,
                    "supported": 0.85,
                }
            ).fillna(0.75)
            fig = self._build_forest_figure(
                ordered,
                x_column="escalation_ratio",
                y_column="edge_label",
                title="Validated Cascade Forest: Robust and Supported Directed Pairs",
                xaxis_title="Escalation Ratio",
                include_adjusted_or=True,
                metadata_prefix="downstream",
                subtitle="Displaying only retained edges that survive the validation layer; outline color encodes validation status.",
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#c0392b")
            fig.update_layout(
                xaxis_type="log",
                height=max(720, 36 * len(ordered) + 220),
                margin={"l": 260, "r": 80, "t": 110, "b": 140},
            )
            self._add_validation_legend(fig)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_primary_static(plot_data, fmt, path),
        )

    def export_downstream_trigger_suite(
        self,
        escalation_results: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        plot_data = self._canonicalize_pair_table(EdgeReportNormalizer.normalize(escalation_results))
        if plot_data.empty:
            return {}

        self._remove_stale_suite_exports(output_dir)
        outputs: dict[str, Path] = {}
        for downstream, subset in plot_data.groupby("downstream_antibiotic", dropna=False, observed=True):
            clean_downstream = str(downstream) if pd.notna(downstream) else "unknown"
            downstream_key = self._classification.normalize_label(clean_downstream)
            upstream_universe = sorted(subset["upstream_antibiotic"].dropna().unique().tolist())
            filtered_universe = [
                antibiotic
                for antibiotic in upstream_universe
                if self._classification.normalize_label(antibiotic) != downstream_key
            ]
            stem = output_dir / f"figure_downstream_trigger_forest__{safe_feature_name(clean_downstream)}"
            figure_data = self._prepare_downstream_plot_data(
                clean_downstream,
                subset.copy(),
                filtered_universe,
            )
            figure = self._build_downstream_figure(clean_downstream, figure_data, len(filtered_universe))
            exported = self._exporter.write(
                figure,
                stem,
                formats,
                static_fallback=(
                    lambda fmt, path, d=clean_downstream, s=figure_data.copy(), n=len(filtered_universe): (
                        self._write_downstream_static(d, s, n, fmt, path)
                    )
                ),
                prefer_static_fallback=True,
            )
            outputs.update(exported)
            del figure, figure_data, exported
            gc.collect()
        return outputs

    def export_upstream_suite(
        self,
        escalation_results: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Backward-compatible alias for the downstream-trigger suite."""
        return self.export_downstream_trigger_suite(escalation_results, output_dir, formats)

    def export_forest_summary(self, edge_report: pd.DataFrame, output_path: Path) -> Path:
        fmt = output_path.suffix.lstrip(".") or "png"
        outputs = self.export_primary_forest(edge_report, output_path.with_suffix(""), (fmt,))
        return outputs[output_path.name]

    def _build_downstream_figure(self, downstream: str, subset: pd.DataFrame, universe_size: int) -> go.Figure:
        ordered = subset.copy()
        ordered["upstream_antibiotic_display"] = ordered["upstream_antibiotic"].map(
            lambda value: self._wrap_label(value, width=28, separator="<br>")
        )
        available_n = int(len(ordered))
        fig = self._build_forest_figure(
            ordered,
            x_column="plot_position_value",
            y_column="upstream_antibiotic_display",
            title=f"Validated Downstream Testing Response Forest:<br>{self._wrap_label(downstream, width=46, separator='<br>')}",
            xaxis_title="Raw Escalation Ratio",
            include_adjusted_or=False,
            metadata_prefix="upstream",
            subtitle=(
                f"Top {available_n} robust/supported upstream antibiotics from {universe_size} validated upstreams, ranked by raw downstream testing contrast."
            ),
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#c0392b")
        fig.add_annotation(
            text=(
                "Square = ER > 1; diamond = ER < 1; filled circle = ER = 1; open circle = non-estimable or no downstream testing.<br>"
                "Marker size reflects total support."
            ),
            x=0.0,
            y=-0.25,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            font={"size": 14, "color": "#475467"},
        )
        fig.update_layout(
            yaxis_title="Upstream trigger antibiotic",
            width=self._width,
            height=max(980, 54 * len(ordered) + 300),
            margin={"l": 370, "r": 95, "t": 135, "b": 230},
        )
        fig.update_yaxes(tickfont={"size": 13})
        fig.update_xaxes(tickfont={"size": 13}, title_font={"size": 15})
        max_x = max(1.25, float(ordered["plot_position_value"].max()) * 1.10 if not ordered.empty else 1.25)
        fig.update_xaxes(
            range=[0, max_x],
            showgrid=True,
            gridcolor="#E8EEF5",
            zeroline=False,
        )
        return fig

    def _prepare_downstream_plot_data(
        self,
        downstream: str,
        subset: pd.DataFrame,
        upstream_universe: list[str],
    ) -> pd.DataFrame:
        full = pd.DataFrame({"upstream_antibiotic": upstream_universe})
        merged = full.merge(subset, on="upstream_antibiotic", how="left")
        merged["downstream_antibiotic"] = downstream
        merged["total_support_n"] = merged["total_support_n"].fillna(0).astype(int)
        merged["resistant_support_n"] = merged["resistant_support_n"].fillna(0).astype(int)
        merged["susceptible_support_n"] = merged["susceptible_support_n"].fillna(0).astype(int)
        merged["resistant_tested_n"] = merged["resistant_tested_n"].fillna(0).astype(int)
        merged["susceptible_tested_n"] = merged["susceptible_tested_n"].fillna(0).astype(int)
        merged["resistant_downstream_test_probability"] = merged["resistant_downstream_test_probability"].fillna(0.0)
        merged["susceptible_downstream_test_probability"] = merged["susceptible_downstream_test_probability"].fillna(0.0)
        merged["escalation_ratio"] = pd.to_numeric(merged["escalation_ratio"], errors="coerce")
        merged = self._classification.add_metadata(merged, "upstream_antibiotic", "upstream")
        merged = self._attach_display_effects(merged)
        merged["ranking_bucket"] = merged["effect_status"].map(
            {
                "trigger": 0,
                "deescalation": 1,
                "deescalation_raw_zero": 1,
                "neutral": 2,
                "non_estimable": 3,
                "no_downstream_testing": 4,
                "not_observed_pair": 5,
            }
        ).fillna(5)
        merged["signal_strength"] = merged["plot_signal_strength"]
        ranked = merged.sort_values(
            by=["ranking_bucket", "signal_strength", "total_support_n", "upstream_antibiotic"],
            ascending=[True, False, False, True],
        ).head(self._top_n)
        return ranked.iloc[::-1].reset_index(drop=True)

    def _build_forest_figure(
        self,
        plot_data: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str,
        xaxis_title: str,
        include_adjusted_or: bool,
        metadata_prefix: str,
        subtitle: str | None = None,
    ) -> go.Figure:
        fig = go.Figure()
        ci_lower_column = "er_ci_lower" if "er_ci_lower" in plot_data.columns else None
        ci_upper_column = "er_ci_upper" if "er_ci_upper" in plot_data.columns else None
        for row in plot_data.itertuples(index=False):
            x_value = getattr(row, x_column)
            if pd.isna(x_value):
                continue
            ci_lower = getattr(row, ci_lower_column) if ci_lower_column else np.nan
            ci_upper = getattr(row, ci_upper_column) if ci_upper_column else np.nan
            if pd.notna(ci_lower) and pd.notna(ci_upper) and ci_lower > 0 and ci_upper > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[ci_lower, ci_upper],
                        y=[getattr(row, y_column), getattr(row, y_column)],
                        mode="lines",
                        line={"color": "#98A2B3", "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        category_order = list(AntibioticClassificationResolver.CATEGORY_ORDER)
        category_column = f"{metadata_prefix}_aware_category"
        color_column = f"{metadata_prefix}_aware_color"
        for category in category_order:
            subset = plot_data.loc[plot_data[category_column] == category].copy()
            if subset.empty:
                continue
            hover_lines = [
                self._build_hover_line(row, y_column, include_adjusted_or, metadata_prefix)
                for _, row in subset.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=subset[x_column],
                    y=subset[y_column],
                    mode="markers",
                    name=f"AWaRe: {category}",
                    legendgroup="aware",
                    marker={
                        "size": self._marker_sizes(subset["total_support_n"]),
                        "color": subset.get("marker_fill_color", subset[color_column]),
                        "symbol": subset["marker_symbol"] if "marker_symbol" in subset.columns else "circle",
                        "line": {
                            "width": 1.0,
                            "color": subset.get("marker_line_color", subset[color_column]),
                        },
                        "opacity": subset.get("marker_opacity", pd.Series([0.90] * len(subset))),
                    },
                    text=hover_lines,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )
        self._add_status_legend(fig)
        fig.update_layout(
            template=self._template,
            xaxis_title=xaxis_title,
            yaxis_title="Directed antibiotic pair" if y_column == "edge_label" else "Antibiotic",
            width=self._width,
            legend_title="Figure encoding",
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.11,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 13},
                "itemsizing": "constant",
            },
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"size": 14, "family": "Arial"},
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E8EEF5", zeroline=False)
        fig.update_yaxes(showgrid=False)
        if subtitle:
            fig.add_annotation(
                text=subtitle,
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="left",
                showarrow=False,
                font={"size": 13, "color": "#475467"},
            )
        return fig

    @staticmethod
    def _add_status_legend(fig: go.Figure) -> None:
        status_legend_items = (
            {
                "name": "Status: Relative escalation",
                "symbol": "square",
                "fill": "#111827",
                "line": "#111827",
            },
            {
                "name": "Status: Relative de-escalation",
                "symbol": "diamond",
                "fill": "#111827",
                "line": "#111827",
            },
            {
                "name": "Status: Neutral (raw ER = 1)",
                "symbol": "circle",
                "fill": "#111827",
                "line": "#111827",
            },
            {
                "name": "Status: Non-estimable / anchored at 0",
                "symbol": "circle-open",
                "fill": "#FFFFFF",
                "line": "#111827",
            },
        )
        for item in status_legend_items:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=item["name"],
                    legendgroup="status",
                    marker={
                        "size": 11,
                        "symbol": item["symbol"],
                        "color": item["fill"],
                        "line": {"width": 1.0, "color": item["line"]},
                    },
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    @staticmethod
    def _add_validation_legend(fig: go.Figure) -> None:
        for item in (
            {
                "name": "Validation: robust",
                "line": "#111827",
            },
            {
                "name": "Validation: supported",
                "line": "#C2410C",
            },
        ):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=item["name"],
                    legendgroup="validation",
                    marker={
                        "size": 11,
                        "symbol": "square",
                        "color": "#FFFFFF",
                        "line": {"width": 2.0, "color": item["line"]},
                    },
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    @staticmethod
    def _build_hover_line(row: pd.Series, label_column: str, include_adjusted_or: bool, metadata_prefix: str) -> str:
        er_value = "undefined" if pd.isna(row["escalation_ratio"]) else f"{row['escalation_ratio']:.4f}"
        ci_value = ""
        if "er_ci_lower" in row.index and "er_ci_upper" in row.index and pd.notna(row["er_ci_lower"]) and pd.notna(row["er_ci_upper"]):
            ci_value = f"<br>95% CI={row['er_ci_lower']:.4f} to {row['er_ci_upper']:.4f}"
        aware_column = f"{metadata_prefix}_aware_category"
        status_label = row.get("effect_status_label", "screened")
        message = (
            f"<b>{row[label_column]}</b><br>"
            f"AWaRe={row[aware_column]}<br>"
            f"Status={status_label}<br>"
            f"Support={int(row['total_support_n'])}<br>"
            f"tested_R={int(row.get('resistant_tested_n', 0))} | tested_S={int(row.get('susceptible_tested_n', 0))}<br>"
            f"n_R={int(row['resistant_support_n'])} | n_S={int(row['susceptible_support_n'])}<br>"
            f"p_R={row['resistant_downstream_test_probability']:.4f} | p_S={row['susceptible_downstream_test_probability']:.4f}<br>"
            f"Raw ER={er_value}<br>"
            f"Plotted position={row.get('plot_position_label', 'n/a')}"
        )
        message += ci_value
        if include_adjusted_or and "adjusted_odds_ratio" in row.index and pd.notna(row["adjusted_odds_ratio"]):
            message += f"<br>Adjusted OR={row['adjusted_odds_ratio']:.4f}"
        return message

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

    def _write_primary_static(self, plot_data: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * max(len(plot_data), 1))))
        if plot_data.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No retained edges available", ha="center", va="center")
        else:
            ordered = self._classification.add_metadata(plot_data.iloc[::-1].copy(), "downstream_antibiotic", "downstream")
            labels = ordered["upstream_antibiotic"] + " -> " + ordered["downstream_antibiotic"]
            positions = np.arange(len(ordered))
            marker_sizes = self._marker_sizes(ordered["total_support_n"]) * 7
            ax.hlines(positions, 1.0, ordered["escalation_ratio"], color="#CBD5E1", linewidth=2.0, zorder=1)
            ax.scatter(
                ordered["escalation_ratio"],
                positions,
                s=marker_sizes,
                color=ordered["downstream_aware_color"],
                edgecolors="#1F2937",
                linewidths=0.9,
                zorder=2,
                marker="s",
            )
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.set_xscale("log")
            ax.grid(axis="x", color="#E8EEF5", linewidth=0.8)
            ax.axvline(1.0, linestyle="--", color="#c0392b", linewidth=1.2)
            ax.set_xlabel("Escalation Ratio (log scale)")
            ax.set_title("Primary Cascade Results: Top Retained Directed Pairs")
            ax.tick_params(axis="both", labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            fig.legend(
                handles=self._build_static_legend_handles(),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=4,
                frameon=False,
                fontsize=8,
            )
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_downstream_static(
        self,
        downstream: str,
        subset: pd.DataFrame,
        universe_size: int,
        fmt: str,
        path: Path,
    ) -> None:
        ordered = subset.copy()
        fig, ax = plt.subplots(figsize=(16, max(8.5, 0.64 * max(len(subset), 1) + 2.4)))
        positions = np.arange(len(ordered))
        marker_sizes = self._marker_sizes(ordered["total_support_n"]) * 7
        if "er_ci_lower" in ordered.columns and "er_ci_upper" in ordered.columns:
            ci_mask = ordered["er_ci_lower"].notna() & ordered["er_ci_upper"].notna()
            ax.hlines(
                positions[ci_mask.to_numpy()],
                ordered.loc[ci_mask, "er_ci_lower"],
                ordered.loc[ci_mask, "er_ci_upper"],
                color="#98A2B3",
                linewidth=2.0,
                zorder=1,
            )
        marker_map = {
            "trigger": "s",
            "deescalation": "D",
            "deescalation_raw_zero": "D",
            "neutral": "o",
            "non_estimable": "o",
            "no_downstream_testing": "o",
            "not_observed_pair": "o",
        }
        for idx, row in ordered.iterrows():
            ax.scatter(
                row["plot_position_value"],
                positions[idx],
                color=row.get("marker_fill_color", row["upstream_aware_color"]),
                s=marker_sizes.iloc[idx],
                edgecolors=row.get("marker_line_color", row["upstream_aware_color"]),
                linewidths=1.1,
                marker=marker_map.get(row.get("effect_status", "neutral"), "o"),
                zorder=2,
            )
        ax.set_yticks(positions)
        wrapped_labels = [
            self._wrap_label(label, width=30, separator="\n")
            for label in ordered["upstream_antibiotic"]
        ]
        ax.set_yticklabels(wrapped_labels, fontsize=11)
        ax.grid(axis="x", color="#E8EEF5", linewidth=0.8)
        max_x = max(1.25, float(ordered["plot_position_value"].max()) * 1.10 if not ordered.empty else 1.25)
        ax.set_xlim(0, max_x)
        ax.set_xlabel("Raw Escalation Ratio", fontsize=12, labelpad=8)
        ax.set_ylabel("Upstream trigger antibiotic", fontsize=12, labelpad=10)
        ax.axvline(1.0, linestyle="--", color="#c0392b", linewidth=1.2)
        ax.set_title(
            f"Validated Downstream Testing Response Forest: {self._wrap_label(downstream, width=52, separator=chr(10))}\n"
            f"Top {len(ordered)} robust/supported upstream antibiotics from {universe_size} validated upstreams",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        ax.tick_params(axis="x", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(
            handles=self._build_static_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
            fontsize=10.5,
        )
        fig.subplots_adjust(left=0.32, right=0.97, top=0.89, bottom=0.17)
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _wrap_label(value: object, *, width: int, separator: str) -> str:
        """Wrap antibiotic labels at readable boundaries for static and HTML plots."""

        label = str(value)
        label = label.replace("/", "/ ").replace("-", "- ")
        lines = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
        return separator.join(line.replace("/ ", "/").replace("- ", "-").strip() for line in lines) or label

    def _write_placeholder_trigger(self, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No downstream trigger data available", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def summarize_downstream_trigger_availability(self, escalation_results: pd.DataFrame) -> pd.DataFrame:
        plot_data = self._canonicalize_pair_table(EdgeReportNormalizer.normalize(escalation_results))
        if plot_data.empty:
            return pd.DataFrame(
                columns=[
                    "downstream_antibiotic",
                    "screened_upstream_n",
                    "estimable_upstream_n",
                    "max_escalation_ratio",
                    "max_total_support_n",
                    "has_estimable_trigger_forest",
                ]
            )
        records: list[dict[str, object]] = []
        for downstream, subset in plot_data.groupby("downstream_antibiotic", dropna=False, observed=True):
            valid = subset.loc[subset["escalation_ratio"].notna() & subset["escalation_ratio"].gt(1.0)].copy()
            records.append(
                {
                    "downstream_antibiotic": downstream,
                    "screened_upstream_n": int(subset["upstream_antibiotic"].nunique()),
                    "estimable_upstream_n": int(valid["upstream_antibiotic"].nunique()),
                    "max_escalation_ratio": float(valid["escalation_ratio"].max()) if not valid.empty else np.nan,
                    "max_total_support_n": int(valid["total_support_n"].max()) if not valid.empty else 0,
                    "has_estimable_trigger_forest": bool(not valid.empty),
                }
            )
        summary = pd.DataFrame.from_records(records)
        return summary.sort_values(
            by=["has_estimable_trigger_forest", "estimable_upstream_n", "max_escalation_ratio", "downstream_antibiotic"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def _build_static_legend_handles() -> list[Line2D]:
        aware_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=8,
                markerfacecolor=AntibioticClassificationResolver.CATEGORY_COLORS[category],
                markeredgecolor=AntibioticClassificationResolver.CATEGORY_COLORS[category],
                label=f"AWaRe: {category}",
            )
            for category in AntibioticClassificationResolver.CATEGORY_ORDER
            if category in AntibioticClassificationResolver.CATEGORY_COLORS
        ]
        status_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=8,
                markerfacecolor="#111827",
                markeredgecolor="#111827",
                label="Status: Relative escalation",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markersize=8,
                markerfacecolor="#111827",
                markeredgecolor="#111827",
                label="Status: Relative de-escalation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=8,
                markerfacecolor="#111827",
                markeredgecolor="#111827",
                label="Status: Neutral (raw ER = 1)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=8,
                markerfacecolor="#FFFFFF",
                markeredgecolor="#111827",
                label="Status: Non-estimable / anchored at 0",
            ),
        ]
        return aware_handles + status_handles

    @staticmethod
    def _attach_rr_confidence_intervals(dataframe: pd.DataFrame) -> pd.DataFrame:
        enriched = dataframe.copy()
        required = {
            "resistant_tested_n",
            "resistant_support_n",
            "susceptible_tested_n",
            "susceptible_support_n",
            "escalation_ratio",
        }
        if enriched.empty or not required.issubset(enriched.columns):
            return enriched
        for column in required:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
        valid = (
            enriched["escalation_ratio"].notna()
            & enriched["escalation_ratio"].gt(1.0)
            & enriched["resistant_tested_n"].gt(0)
            & enriched["susceptible_tested_n"].gt(0)
            & enriched["resistant_support_n"].gt(0)
            & enriched["susceptible_support_n"].gt(0)
        )
        se = np.sqrt(
            (1 / enriched.loc[valid, "resistant_tested_n"])
            - (1 / enriched.loc[valid, "resistant_support_n"])
            + (1 / enriched.loc[valid, "susceptible_tested_n"])
            - (1 / enriched.loc[valid, "susceptible_support_n"])
        )
        log_rr = np.log(enriched.loc[valid, "escalation_ratio"])
        enriched["er_ci_lower"] = np.nan
        enriched["er_ci_upper"] = np.nan
        enriched.loc[valid, "er_ci_lower"] = np.exp(log_rr - 1.96 * se)
        enriched.loc[valid, "er_ci_upper"] = np.exp(log_rr + 1.96 * se)
        return enriched

    def _attach_display_effects(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        enriched = dataframe.copy()
        a = enriched["resistant_tested_n"].astype(float)
        n1 = enriched["resistant_support_n"].astype(float)
        c = enriched["susceptible_tested_n"].astype(float)
        n0 = enriched["susceptible_support_n"].astype(float)
        valid_support = (n1 > 0) & (n0 > 0)
        raw_ratio = pd.to_numeric(enriched["escalation_ratio"], errors="coerce")
        enriched["plot_position_value"] = 0.0
        enriched["plot_position_label"] = "0 (non-estimable anchor)"
        enriched["plot_signal_strength"] = 0.0
        enriched["er_ci_lower"] = np.nan
        enriched["er_ci_upper"] = np.nan
        estimable = raw_ratio.notna()
        enriched.loc[estimable, "plot_position_value"] = raw_ratio.loc[estimable]
        enriched.loc[estimable, "plot_position_label"] = raw_ratio.loc[estimable].map(lambda value: f"{value:.4f}")
        finite_positive = estimable & raw_ratio.gt(0)
        enriched.loc[finite_positive, "plot_signal_strength"] = np.where(
            raw_ratio.loc[finite_positive] >= 1.0,
            raw_ratio.loc[finite_positive],
            1.0 / raw_ratio.loc[finite_positive].clip(lower=1e-12),
        )
        zero_raw = estimable & raw_ratio.eq(0)
        enriched.loc[zero_raw, "plot_signal_strength"] = 1_000.0

        # Approximate CI only for finite positive raw ratios.
        ci_valid = (
            raw_ratio.notna()
            & raw_ratio.gt(0)
            & a.gt(0)
            & c.gt(0)
            & n1.gt(0)
            & n0.gt(0)
        )
        se = np.sqrt((1 / a.loc[ci_valid]) - (1 / n1.loc[ci_valid]) + (1 / c.loc[ci_valid]) - (1 / n0.loc[ci_valid]))
        log_rr = np.log(raw_ratio.loc[ci_valid])
        enriched.loc[ci_valid, "er_ci_lower"] = np.exp(log_rr - 1.96 * se)
        enriched.loc[ci_valid, "er_ci_upper"] = np.exp(log_rr + 1.96 * se)

        enriched["effect_status"] = "neutral"
        enriched.loc[(a == 0) & (c == 0) & valid_support, "effect_status"] = "no_downstream_testing"
        enriched.loc[raw_ratio.notna() & raw_ratio.gt(1.0), "effect_status"] = "trigger"
        enriched.loc[raw_ratio.notna() & raw_ratio.ge(0) & raw_ratio.lt(1.0), "effect_status"] = "deescalation"
        enriched.loc[raw_ratio.notna() & raw_ratio.eq(1.0), "effect_status"] = "neutral"
        enriched.loc[raw_ratio.isna() & valid_support, "effect_status"] = "non_estimable"
        enriched.loc[~valid_support, "effect_status"] = "not_observed_pair"

        enriched["effect_status_label"] = enriched["effect_status"].map(
            {
                "trigger": "raw ER > 1 (escalation)",
                "deescalation": "0 <= raw ER < 1 (de-escalation)",
                "deescalation_raw_zero": "raw ER = 0",
                "neutral": "raw ER = 1 (neutral)",
                "non_estimable": "raw ER not estimable",
                "no_downstream_testing": "no downstream tests in either branch; anchored at 0",
                "not_observed_pair": "pair not observed with both contrast branches",
            }
        ).fillna("screened")
        enriched["marker_symbol"] = enriched["effect_status"].map(
            {
                "trigger": "square",
                "deescalation": "diamond",
                "deescalation_raw_zero": "diamond",
                "neutral": "circle",
                "non_estimable": "circle-open",
                "no_downstream_testing": "circle-open",
                "not_observed_pair": "circle-open",
            }
        ).fillna("circle")
        enriched["marker_fill_color"] = enriched["upstream_aware_color"]
        open_mask = enriched["effect_status"].isin(["non_estimable", "no_downstream_testing", "not_observed_pair"])
        enriched.loc[open_mask, "marker_fill_color"] = "#FFFFFF"
        enriched["marker_line_color"] = enriched["upstream_aware_color"]
        enriched["marker_opacity"] = 0.9
        enriched.loc[open_mask, "marker_opacity"] = 0.75
        return enriched

    @staticmethod
    def _remove_stale_suite_exports(output_dir: Path) -> None:
        for pattern in (
            "figure_upstream_contribution_forest__*",
            "figure_downstream_trigger_forest__*",
        ):
            for path in output_dir.glob(pattern):
                if path.is_file():
                    path.unlink(missing_ok=True)

    def _canonicalize_pair_table(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        canonical = dataframe.copy()
        if canonical.empty:
            return canonical
        for column in ("upstream_antibiotic", "downstream_antibiotic"):
            if column in canonical.columns:
                key_column = f"__{column}_key"
                raw_column = f"__{column}_raw"
                canonical[raw_column] = canonical[column].astype("string").fillna("Unknown").str.upper()
                canonical[key_column] = canonical[column].map(self._classification.normalize_label)
                display_map = (
                    canonical.groupby(key_column, observed=True)[raw_column]
                    .apply(lambda values: self._choose_preferred_display_label(values, values.name))
                    .to_dict()
                )
                canonical[column] = canonical[key_column].map(display_map)
        sort_columns = []
        ascending = []
        if "escalation_ratio" in canonical.columns:
            canonical["has_estimable_ratio"] = canonical["escalation_ratio"].notna() & canonical["escalation_ratio"].gt(1.0)
            sort_columns.append("has_estimable_ratio")
            ascending.append(False)
        if "total_support_n" in canonical.columns:
            sort_columns.append("total_support_n")
            ascending.append(False)
        sort_columns.extend(["__upstream_antibiotic_key", "__downstream_antibiotic_key"])
        ascending.extend([True, True])
        canonical = canonical.sort_values(sort_columns, ascending=ascending)
        canonical = canonical.drop_duplicates(["__upstream_antibiotic_key", "__downstream_antibiotic_key"], keep="first")
        return canonical.drop(
            columns=[
                "has_estimable_ratio",
                "__upstream_antibiotic_key",
                "__downstream_antibiotic_key",
                "__upstream_antibiotic_raw",
                "__downstream_antibiotic_raw",
            ],
            errors="ignore",
        ).reset_index(drop=True)

    @staticmethod
    def _choose_preferred_display_label(values: pd.Series, canonical_key: str | None = None) -> str:
        candidates = sorted({str(value).strip().upper() for value in values if str(value).strip()})
        if not candidates:
            return "UNKNOWN"
        if canonical_key:
            for candidate in candidates:
                if candidate == canonical_key:
                    return candidate

        def score(label: str) -> tuple[int, int, int, int]:
            alpha_count = sum(character.isalpha() for character in label)
            separator_bonus = label.count("/") + label.count(" ")
            rm_penalty = 0 if " RM" not in label else -1
            abbreviation_penalty = 0 if len(label) > 4 else -1
            return (abbreviation_penalty, rm_penalty, alpha_count + separator_bonus, len(label))

        return max(candidates, key=score)
