"""Standard Plotly model-comparison figures for downstream-testing evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyModelEvaluationPlotter:
    """Export standard model-comparison figures from real evaluation artifacts."""

    MODEL_COLORS = {
        "logistic_regression": "#1F77B4",
        "random_forest": "#2E8B57",
        "xgboost": "#E69F00",
    }
    MODEL_LABELS = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export_metrics_comparison(
        self,
        metrics: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        split: str = "test",
    ) -> dict[str, Path]:
        plot_data = metrics.loc[metrics["split"] == split].copy()
        if plot_data.empty:
            fig = self._empty_figure(f"No model metrics available for split='{split}'")
        else:
            metric_specs = [
                ("roc_auc", "ROC-AUC"),
                ("pr_auc", "PR-AUC"),
                ("brier_score", "Brier"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("mcc", "MCC"),
            ]
            cols = 3
            rows = (len(metric_specs) + cols - 1) // cols
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[label for _, label in metric_specs],
                # Each subplot sorts its own 3 models independently (best model
                # differs per metric), so "Random Forest" can sit in row 2 of
                # this subplot while "Logistic Regression" sits in row 2 of the
                # next one -- same y-pixel, adjacent subplots. The x-range
                # headroom below gives each bar's outside-end value label room
                # within its own subplot, but at the old 0.08 spacing that
                # label still reached far enough right to collide with the
                # neighbouring subplot's own left-side category label at that
                # shared row height (e.g. "0.641" running into "Logistic
                # Regression"). Wider spacing keeps the two apart.
                horizontal_spacing=0.14,
                vertical_spacing=0.16,
            )
            for idx, (column, label) in enumerate(metric_specs, start=1):
                row = (idx - 1) // cols + 1
                col = (idx - 1) % cols + 1
                ordered = plot_data.sort_values(column, ascending=(column == "brier_score")).copy()
                ordered["model_display"] = ordered["model_name"].map(self._display_name)
                fig.add_trace(
                    go.Bar(
                        x=ordered[column],
                        y=ordered["model_display"],
                        orientation="h",
                        marker_color=ordered["model_name"].map(self._model_color),
                        text=ordered[column].map(lambda value: f"{value:.3g}"),
                        textposition="outside",
                        showlegend=False,
                        hovertemplate=f"{label}=%{{x:.4e}}<extra>%{{y}}</extra>",
                    ),
                    row=row,
                    col=col,
                )
                # cliponaxis=False (the previous setting) lets "outside" bar-end
                # text escape this subplot's own boundary entirely -- in a 3-wide
                # grid that means every row's rightmost label spills into the
                # next column and overlaps its y-axis category labels (e.g.
                # "0.799" merging into "Logistic Regression"). Padding each
                # subplot's own x-range gives the label room within its own
                # boundary instead, so cliponaxis can stay at its default
                # (True) and nothing needs to cross into the next subplot.
                axis_max = float(ordered[column].max())
                axis_min = min(0.0, float(ordered[column].min()))
                headroom = max((axis_max - axis_min) * 0.22, 0.03)
                fig.update_xaxes(tickformat=".3g", range=[axis_min, axis_max + headroom], row=row, col=col)
            prevalence = float(plot_data["prevalence"].iloc[0]) if "prevalence" in plot_data.columns and not plot_data.empty else None
            self._apply_standard_layout(
                fig,
                title=f"Model Performance Comparison ({split.title()} Split)",
                height=max(self._height, 420 * rows + 140),
                margin={"l": 150, "r": 80, "t": 150, "b": 60},
            )
            if prevalence is not None and not np.isnan(prevalence):
                fig.add_annotation(
                    text=f"Observed event prevalence on the {split} split: {prevalence:.6f}",
                    x=0,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    showarrow=False,
                    font={"size": 13, "color": "#475467"},
                )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_metrics_static(plot_data, fmt, path))

    def export_precision_recall(
        self,
        predictions: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        split: str = "test",
    ) -> dict[str, Path]:
        plot_data = predictions.loc[predictions["split"] == split].copy()
        if plot_data.empty:
            fig = self._empty_figure(f"No prediction data available for split='{split}'")
        else:
            fig = go.Figure()
            prevalence = float(plot_data["target"].mean()) if not plot_data.empty else 0.0
            fig.add_hline(y=prevalence, line_dash="dash", line_color="#7F8C8D")
            for model_name, subset in plot_data.groupby("model_name", sort=False):
                precision, recall, _ = precision_recall_curve(subset["target"], subset["predicted_probability"])
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=self._display_name(model_name),
                        line={"width": 3, "color": self._model_color(model_name)},
                        hovertemplate="Recall=%{x:.4f}<br>Precision=%{y:.4f}<extra>%{fullData.name}</extra>",
                    )
                )
            self._apply_standard_layout(
                fig,
                title=f"Precision-Recall Curves ({split.title()} Split)",
                xaxis_title="Recall",
                yaxis_title="Precision",
            )
            fig.add_annotation(
                text=f"Dashed line shows positive-class prevalence ({prevalence:.6f}).",
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="left",
                showarrow=False,
                font={"size": 13, "color": "#475467"},
            )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_curve_static(plot_data, fmt, path, curve_type="pr"))

    def export_roc(
        self,
        predictions: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        split: str = "test",
    ) -> dict[str, Path]:
        plot_data = predictions.loc[predictions["split"] == split].copy()
        if plot_data.empty:
            fig = self._empty_figure(f"No prediction data available for split='{split}'")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#7F8C8D"}, name="Chance"))
            for model_name, subset in plot_data.groupby("model_name", sort=False):
                fpr, tpr, _ = roc_curve(subset["target"], subset["predicted_probability"])
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=self._display_name(model_name),
                        line={"width": 3, "color": self._model_color(model_name)},
                        hovertemplate="FPR=%{x:.4f}<br>TPR=%{y:.4f}<extra>%{fullData.name}</extra>",
                    )
                )
            self._apply_standard_layout(
                fig,
                title=f"ROC Curves ({split.title()} Split)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_curve_static(plot_data, fmt, path, curve_type="roc"))

    def export_calibration(
        self,
        predictions: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        split: str = "test",
    ) -> dict[str, Path]:
        plot_data = predictions.loc[predictions["split"] == split].copy()
        if plot_data.empty:
            fig = self._empty_figure(f"No prediction data available for split='{split}'")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#7F8C8D"}, name="Ideal"))
            for model_name, subset in plot_data.groupby("model_name", sort=False):
                frac_pos, mean_pred = calibration_curve(
                    subset["target"].astype(int),
                    subset["predicted_probability"],
                    n_bins=10,
                    strategy="quantile",
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_pred,
                        y=frac_pos,
                        mode="lines+markers",
                        name=self._display_name(model_name),
                        line={"width": 2.5, "color": self._model_color(model_name)},
                        marker={"size": 8},
                        hovertemplate="Mean predicted=%{x:.4f}<br>Observed=%{y:.4f}<extra>%{fullData.name}</extra>",
                    )
                )
            self._apply_standard_layout(
                fig,
                title=f"Calibration Curves ({split.title()} Split)",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Observed Event Rate",
            )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_calibration_static(plot_data, fmt, path))

    def export_threshold_analysis(
        self,
        threshold_metrics: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        split: str = "test",
    ) -> dict[str, Path]:
        plot_data = threshold_metrics.loc[threshold_metrics["split"] == split].copy()
        if plot_data.empty:
            fig = self._empty_figure(f"No threshold metrics available for split='{split}'")
        else:
            metric_specs = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC")]
            cols = 2
            rows = (len(metric_specs) + cols - 1) // cols
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[label for _, label in metric_specs],
                horizontal_spacing=0.045,
                vertical_spacing=0.12,
            )
            for idx, (metric, label) in enumerate(metric_specs, start=1):
                row = (idx - 1) // cols + 1
                col = (idx - 1) % cols + 1
                for model_name, subset in plot_data.groupby("model_name", sort=False):
                    subset = subset.sort_values("threshold")
                    fig.add_trace(
                        go.Scatter(
                            x=subset["threshold"],
                            y=subset[metric],
                            mode="lines+markers",
                            name=self._display_name(model_name),
                            legendgroup=model_name,
                            showlegend=idx == 1,
                            line={"width": 2.5, "color": self._model_color(model_name)},
                            marker={"size": 7},
                            hovertemplate="Threshold=%{x:.2f}<br>" + f"{label}=%{{y:.4f}}<extra>%{{fullData.name}}</extra>",
                        ),
                        row=row,
                        col=col,
                    )
                    selected = subset.loc[subset["selected_for_decision"].fillna(False)]
                    if not selected.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=selected["threshold"],
                                y=selected[metric],
                                mode="markers",
                                name=f"{self._display_name(model_name)} selected",
                                legendgroup=model_name,
                                showlegend=False,
                                marker={
                                    "size": 12,
                                    "color": self._model_color(model_name),
                                    "line": {"width": 2, "color": "white"},
                                    "symbol": "diamond",
                                },
                                hovertemplate=(
                                    "Selected threshold=%{x:.2f}<br>"
                                    + f"{label}=%{{y:.4f}}<extra>{self._display_name(model_name)}</extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )
            self._apply_standard_layout(
                fig,
                title=f"Threshold Sensitivity by Model ({split.title()} Split)",
                width=1450,
                height=max(self._height, 360 * rows + 120),
                margin={"l": 70, "r": 40, "t": 100, "b": 85},
            )
            for idx, (_, label) in enumerate(metric_specs, start=1):
                row = (idx - 1) // cols + 1
                col = (idx - 1) % cols + 1
                fig.update_xaxes(title_text="Threshold", row=row, col=col, range=[0.04, 0.91])
                fig.update_yaxes(title_text=label, row=row, col=col, tickformat=".3g")
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_threshold_static(plot_data, fmt, path))

    def _apply_standard_layout(
        self,
        fig: go.Figure,
        title: str,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        margin: dict[str, int] | None = None,
    ) -> None:
        fig.update_layout(
            template=self._template,
            width=width or self._width,
            height=height or self._height,
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 20}},
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend={
                "title": {"text": "Model"},
                "orientation": "h",
                "yanchor": "top",
                "y": -0.12,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 13},
            },
            margin=margin or {"l": 80, "r": 40, "t": 90, "b": 90},
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"size": 16, "family": "Arial"},
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E8EEF5", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#F2F4F7", zeroline=False)

    def _empty_figure(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    def _model_color(self, model_name: str) -> str:
        return self.MODEL_COLORS.get(model_name, "#4C78A8")

    def _display_name(self, model_name: str) -> str:
        return self.MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())

    def _write_metrics_static(self, plot_data: pd.DataFrame, fmt: str, path: Path) -> None:
        metric_specs = [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("brier_score", "Brier"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("mcc", "MCC"),
        ]
        cols = 3
        rows = (len(metric_specs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 4.8 * rows))
        axes_array = np.atleast_1d(axes).ravel()
        for ax, (metric, label) in zip(axes_array, metric_specs):
            if plot_data.empty:
                ax.axis("off")
                continue
            ordered = plot_data.sort_values(metric, ascending=(metric == "brier_score"))
            colors = ordered["model_name"].map(self._model_color).tolist()
            labels = ordered["model_name"].map(self._display_name)
            values = ordered[metric]
            ax.barh(labels, values, color=colors)
            ax.set_title(label)
            ax.grid(axis="x", color="#E8EEF5", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            for index, value in enumerate(values):
                ax.text(value, index, f" {value:.3g}", va="center", ha="left", fontsize=12, color="#344054")
        for ax in axes_array[len(metric_specs):]:
            ax.axis("off")
        fig.suptitle("Model Performance Comparison", fontsize=20, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_curve_static(self, plot_data: pd.DataFrame, fmt: str, path: Path, curve_type: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D")
        else:
            prevalence = float(plot_data["target"].mean()) if not plot_data.empty else 0.0
            ax.axhline(prevalence, linestyle="--", color="#7F8C8D")
        for model_name, subset in plot_data.groupby("model_name", sort=False):
            if curve_type == "roc":
                x, y, _ = roc_curve(subset["target"], subset["predicted_probability"])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves")
            else:
                y, x, _ = precision_recall_curve(subset["target"], subset["predicted_probability"])
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curves")
            ax.plot(x, y, linewidth=2.5, color=self._model_color(model_name), label=self._display_name(model_name))
        ax.grid(color="#E8EEF5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_calibration_static(self, plot_data: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D")
        for model_name, subset in plot_data.groupby("model_name", sort=False):
            frac_pos, mean_pred = calibration_curve(
                subset["target"].astype(int),
                subset["predicted_probability"],
                n_bins=10,
                strategy="quantile",
            )
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color=self._model_color(model_name), label=self._display_name(model_name))
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Event Rate")
        ax.set_title("Calibration Curves")
        ax.grid(color="#E8EEF5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_threshold_static(self, plot_data: pd.DataFrame, fmt: str, path: Path) -> None:
        metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC")]
        cols = 2
        rows = (len(metrics) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6.0 * rows), sharex=True)
        axes_array = np.atleast_1d(axes).ravel()
        for ax, (metric, label) in zip(axes_array, metrics):
            for model_name, subset in plot_data.groupby("model_name", sort=False):
                subset = subset.sort_values("threshold")
                ax.plot(
                    subset["threshold"],
                    subset[metric],
                    marker="o",
                    markersize=6,
                    linewidth=2.4,
                    color=self._model_color(model_name),
                    label=self._display_name(model_name),
                )
                selected = subset.loc[subset["selected_for_decision"].fillna(False)]
                if not selected.empty:
                    ax.scatter(
                        selected["threshold"],
                        selected[metric],
                        s=85,
                        marker="D",
                        color=self._model_color(model_name),
                        edgecolor="white",
                        linewidth=1.5,
                        zorder=4,
                    )
            ax.set_title(label)
            ax.set_xlabel("Threshold")
            ax.set_ylabel(label)
            ax.grid(color="#E8EEF5", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0.04, 0.91)
        for ax in axes_array[len(metrics):]:
            ax.axis("off")
        handles, labels = axes_array[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=13,
        )
        fig.suptitle("Threshold Sensitivity by Model", fontsize=18, y=0.98)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.22, wspace=0.22)
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
