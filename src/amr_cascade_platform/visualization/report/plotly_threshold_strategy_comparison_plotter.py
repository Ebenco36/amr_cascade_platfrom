"""Threshold-selection strategy comparison figure."""

from __future__ import annotations

from amr_cascade_platform.visualization.report.figure_presentation import save_matplotlib

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyThresholdStrategyComparisonPlotter:
    """Compare threshold development and sealed-test operating characteristics."""

    COLORS = {
        "validation_holdout": "#2563EB",
        "grouped_oof": "#D97706",
    }
    LABELS = {
        "validation_holdout": "Validation-site holdout",
        "grouped_oof": "Patient-grouped OOF",
    }
    METRIC_LABELS = {"balanced_accuracy": "Balanced accuracy", "mcc": "MCC", "f1": "F1"}
    TITLE = "Threshold-selection strategy comparison"
    SUBTITLE = (
        "Prespecified primary estimator with leakage-safe clinical features; "
        "only the threshold-development data change"
    )
    # The no-skill level of each selection metric: a coin-flip classifier's balanced
    # accuracy and MCC; for F1 the score of predicting every row positive, 2p/(1+p).
    TEST_METRICS = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("mcc", "MCC"),
    )

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = max(width, 1800)
        self._height = max(height, 950)

    def export(
        self,
        curves: pd.DataFrame,
        selected: pd.DataFrame,
        test_metrics: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        self._validate(curves, selected, test_metrics)
        figure = self._build_figure(curves, selected, test_metrics)
        return self._exporter.write(
            figure,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(
                curves, selected, test_metrics, fmt, path
            ),
            prefer_static_fallback=True,
        )

    @classmethod
    def _selection_metric(cls, selected: pd.DataFrame) -> tuple[str, str]:
        metric = "f1"
        if "selection_metric" in selected.columns and selected["selection_metric"].notna().any():
            metric = str(selected["selection_metric"].dropna().iloc[0])
        return metric, cls.METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())

    @staticmethod
    def _reference_level(metric: str, subset: pd.DataFrame) -> float | None:
        if metric == "balanced_accuracy":
            return 0.5
        if metric == "mcc":
            return 0.0
        if metric == "f1" and "prevalence" in subset.columns and subset["prevalence"].notna().any():
            prevalence = float(subset["prevalence"].dropna().iloc[0])
            return 2 * prevalence / (1 + prevalence)
        return None

    @staticmethod
    def _boundary(selected_threshold: float, subset: pd.DataFrame) -> bool:
        return bool(len(subset)) and (
            np.isclose(selected_threshold, subset["threshold"].min()) or np.isclose(selected_threshold, subset["threshold"].max())
        )

    def _build_figure(
        self,
        curves: pd.DataFrame,
        selected: pd.DataFrame,
        test_metrics: pd.DataFrame,
    ) -> go.Figure:
        metric, metric_label = self._selection_metric(selected)
        figure = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.56, 0.44],
            horizontal_spacing=0.13,
            subplot_titles=(
                f"A. {metric_label} across candidate thresholds (selection metric)",
                "B. Performance on the sealed test split",
            ),
        )

        strategy_order = [strategy for strategy in self.LABELS if strategy in set(curves["comparison_strategy"])]
        for strategy in strategy_order:
            label = self.LABELS[strategy]
            color = self.COLORS[strategy]
            subset = curves.loc[curves["comparison_strategy"].eq(strategy)].sort_values("threshold")
            selected_threshold = float(
                selected.loc[selected["comparison_strategy"].eq(strategy), "selected_threshold"].iloc[0]
            )
            selected_point = subset.loc[np.isclose(subset["threshold"], selected_threshold)]
            edge = " (grid edge)" if self._boundary(selected_threshold, subset) else ""
            reference = self._reference_level(metric, subset)
            if reference is not None:
                figure.add_trace(
                    go.Scatter(
                        x=[subset["threshold"].min(), subset["threshold"].max()],
                        y=[reference, reference],
                        mode="lines",
                        line={"color": color, "width": 1.5, "dash": "dot"},
                        name=f"{label}: no-skill level",
                        legendgroup=strategy,
                        showlegend=False,
                        hovertemplate=f"No-skill {metric_label}=%{{y:.3f}}<extra>{label}</extra>",
                    ),
                    row=1,
                    col=1,
                )
            figure.add_trace(
                go.Scatter(
                    x=subset["threshold"],
                    y=subset[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=strategy,
                    line={"color": color, "width": 3},
                    marker={"color": color, "size": 9},
                    customdata=np.column_stack(
                        [subset["precision"], subset["recall"], subset["mcc"]]
                    ),
                    hovertemplate=(
                        "Threshold=%{x:.2f}<br>" + metric_label + "=%{y:.3f}<br>"
                        "Precision=%{customdata[0]:.3f}<br>Recall=%{customdata[1]:.3f}<br>"
                        "MCC=%{customdata[2]:.3f}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=selected_point["threshold"],
                    y=selected_point[metric],
                    mode="markers+text",
                    text=[f"selected {selected_threshold:.2f}{edge}"] * len(selected_point),
                    textposition=(
                        "top center" if strategy == "validation_holdout" else "bottom center"
                    ),
                    textfont={"size": 12, "color": color},
                    marker={
                        "symbol": "diamond",
                        "size": 15,
                        "color": color,
                        "line": {"color": "white", "width": 2},
                    },
                    name=f"{label} selected",
                    legendgroup=strategy,
                    showlegend=False,
                    hovertemplate="Selected threshold=%{x:.2f}<br>Selection " + metric_label + "=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            test_row = test_metrics.loc[test_metrics["comparison_strategy"].eq(strategy)].iloc[0]
            figure.add_trace(
                go.Bar(
                    x=[label for _, label in self.TEST_METRICS],
                    y=[float(test_row[column]) for column, _ in self.TEST_METRICS],
                    name=label,
                    legendgroup=strategy,
                    # The line trace already names this strategy in the legend.
                    showlegend=False,
                    marker_color=color,
                    text=[f"{float(test_row[column]):.3f}" for column, _ in self.TEST_METRICS],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}: %{y:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=2,
            )

        comparison_note = self._selection_note(selected)
        figure.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title={
                "text": f"{self.TITLE}<br><sup>{self.SUBTITLE}</sup>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 22},
            },
            barmode="group",
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.18,
                "xanchor": "center",
                "x": 0.5,
                "title": {"text": "Threshold-selection strategy"},
            },
            margin={"l": 95, "r": 55, "t": 170, "b": 190},
            font={"family": "Arial", "size": 15, "color": "#243B5A"},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        figure.update_xaxes(
            title_text="Probability threshold",
            range=[0.02, 0.98],
            tickformat=".2f",
            showgrid=True,
            gridcolor="#E8EEF5",
            row=1,
            col=1,
        )
        figure.update_yaxes(
            title_text=f"{metric_label} on strategy-specific development predictions",
            range=[0, 1.03],
            tickformat=".2f",
            showgrid=True,
            gridcolor="#E8EEF5",
            row=1,
            col=1,
        )
        figure.update_xaxes(tickangle=-20, row=1, col=2)
        figure.update_yaxes(
            title_text="Metric value",
            range=[0, 1.08],
            tickformat=".2f",
            showgrid=True,
            gridcolor="#E8EEF5",
            row=1,
            col=2,
        )
        figure.add_annotation(
            x=0.5,
            y=1.075,
            xref="paper",
            yref="paper",
            yanchor="bottom",
            text=comparison_note,
            showarrow=False,
            font={"size": 13, "color": "#475467"},
        )
        figure.add_annotation(
            x=0.99,
            y=0.02,
            xref="x domain",
            yref="y domain",
            xanchor="right",
            yanchor="bottom",
            text="dotted: no-skill level",
            showarrow=False,
            font={"size": 12, "color": "#475467"},
        )
        figure.add_annotation(
            x=0.5,
            y=-0.30,
            xref="paper",
            yref="paper",
            text=self._caption(metric_label),
            showarrow=False,
            font={"size": 12, "color": "#475467"},
        )
        return figure

    @staticmethod
    def _caption(metric_label: str) -> str:
        return (
            f"Panel A: {metric_label} on each strategy's own development predictions, which come from different "
            "populations, so the curves are not a head-to-head comparison. Panel B is descriptive only and does "
            "not choose the strategy."
        )

    def _write_static(
        self,
        curves: pd.DataFrame,
        selected: pd.DataFrame,
        test_metrics: pd.DataFrame,
        fmt: str,
        path: Path,
    ) -> None:
        metric, metric_label = self._selection_metric(selected)
        fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.4), gridspec_kw={"width_ratios": [1.2, 1]})
        strategy_order = [strategy for strategy in self.LABELS if strategy in set(curves["comparison_strategy"])]
        for strategy_index, strategy in enumerate(strategy_order):
            subset = curves.loc[curves["comparison_strategy"].eq(strategy)].sort_values("threshold")
            color = self.COLORS[strategy]
            label = self.LABELS[strategy]
            reference = self._reference_level(metric, subset)
            if reference is not None:
                axes[0].axhline(reference, color=color, linestyle=":", linewidth=1.3, alpha=0.8)
            axes[0].plot(
                subset["threshold"], subset[metric], color=color, marker="o", linewidth=2.4, label=label
            )
            chosen = float(
                selected.loc[selected["comparison_strategy"].eq(strategy), "selected_threshold"].iloc[0]
            )
            edge = " (grid edge)" if self._boundary(chosen, subset) else ""
            point = subset.loc[np.isclose(subset["threshold"], chosen)]
            axes[0].scatter(point["threshold"], point[metric], color=color, marker="D", s=90, zorder=4)
            if not point.empty:
                axes[0].annotate(
                    f"selected {chosen:.2f}{edge}",
                    (float(point["threshold"].iloc[0]), float(point[metric].iloc[0])),
                    xytext=(0, 14 if strategy_index == 0 else -18),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=10,
                )
        axes[0].set_title(f"A. {metric_label} across candidate thresholds (selection metric)", fontweight="bold")
        axes[0].set_xlabel("Probability threshold")
        axes[0].set_ylabel(f"{metric_label} on strategy-specific development predictions")
        axes[0].set_xlim(0.02, 0.98)
        axes[0].text(0.99, 0.02, "dotted: no-skill level", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9, color="#475467")
        axes[0].set_ylim(0, 1.03)
        axes[0].grid(color="#E8EEF5", linewidth=0.8)

        metric_labels = [label for _, label in self.TEST_METRICS]
        positions = np.arange(len(metric_labels), dtype=float)
        width = 0.36
        for index, strategy in enumerate(strategy_order):
            row = test_metrics.loc[test_metrics["comparison_strategy"].eq(strategy)].iloc[0]
            values = [float(row[column]) for column, _ in self.TEST_METRICS]
            offset = (index - (len(strategy_order) - 1) / 2) * width
            bars = axes[1].bar(
                positions + offset,
                values,
                width=width,
                color=self.COLORS[strategy],
                label=self.LABELS[strategy],
            )
            axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
        axes[1].set_title("B. Performance on the sealed test split", fontweight="bold")
        axes[1].set_ylabel("Metric value")
        axes[1].set_xticks(positions, metric_labels, rotation=20, ha="right")
        axes[1].set_ylim(0, 1.08)
        axes[1].grid(axis="y", color="#E8EEF5", linewidth=0.8)

        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.055))
        # The header is three stacked figure-text lines above the panels' rect. A
        # suptitle would make tight_layout reserve room below it on top of the rect,
        # leaving a wide gap, and a fixed-position note collided with its second line.
        fig.text(0.5, 0.985, self.TITLE, ha="center", va="top", fontsize=17, fontweight="bold", gid="figure_title")
        fig.text(0.5, 0.940, self.SUBTITLE, ha="center", va="top", fontsize=12, color="#243B5A", gid="figure_subtitle")
        fig.text(0.5, 0.905, self._selection_note(selected), ha="center", va="top", fontsize=10.5, color="#475467", gid="figure_subtitle")
        fig.text(0.5, 0.015, self._caption(metric_label), ha="center", fontsize=9, color="#475467")
        fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.865))
        save_matplotlib(fig, path, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    @classmethod
    def _selection_note(cls, selected: pd.DataFrame) -> str:
        notes: list[str] = []
        for strategy in cls.LABELS:
            rows = selected.loc[selected["comparison_strategy"].eq(strategy)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            detail = f"{cls.LABELS[strategy]}: threshold {float(row['selected_threshold']):.2f}, n={int(row['selection_rows']):,}"
            boundary = row.get("at_grid_boundary")
            if pd.notna(boundary) and bool(boundary):
                detail += " (grid edge)"
            if pd.notna(row.get("selection_groups")):
                detail += f", {int(row['selection_groups']):,} patients"
            if pd.notna(row.get("cv_folds")):
                detail += f", {int(row['cv_folds'])} folds"
            notes.append(detail)
        return " | ".join(notes)

    @classmethod
    def _validate(
        cls,
        curves: pd.DataFrame,
        selected: pd.DataFrame,
        test_metrics: pd.DataFrame,
    ) -> None:
        curve_required = {
            "comparison_strategy",
            "threshold",
            "f1",
            "precision",
            "recall",
            "mcc",
        }
        selected_required = {
            "comparison_strategy",
            "selected_threshold",
            "selection_rows",
        }
        test_required = {"comparison_strategy", *(column for column, _ in cls.TEST_METRICS)}
        curve_required.add(cls._selection_metric(selected)[0])
        missing = {
            "curves": sorted(curve_required.difference(curves.columns)),
            "selected": sorted(selected_required.difference(selected.columns)),
            "test_metrics": sorted(test_required.difference(test_metrics.columns)),
        }
        failures = {name: columns for name, columns in missing.items() if columns}
        if failures:
            raise ValidationError(f"Threshold-strategy comparison figure is missing columns: {failures}.")
        expected = set(cls.LABELS)
        for name, frame in (("curves", curves), ("selected", selected), ("test_metrics", test_metrics)):
            present = set(frame["comparison_strategy"].dropna().astype(str))
            if present != expected:
                raise ValidationError(
                    f"Threshold-strategy comparison {name} must contain exactly {sorted(expected)}; "
                    f"found {sorted(present)}."
                )
