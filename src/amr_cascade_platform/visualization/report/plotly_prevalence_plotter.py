"""Plot selective-testing prevalence-shift summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from amr_cascade_platform.visualization.report.organism_labels import format_organism_label
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyPrevalencePlotter:
    """Render publication-ready prevalence-shift plots."""

    _SHIFT_DIRECTION_LABELS = {
        "naive_overestimates": "MNAR estimate (naive overstates)",
        "naive_underestimates": "MNAR estimate (naive understates)",
        "no_difference": "MNAR estimate (no shift)",
        "unavailable": "MNAR estimate (not estimable)",
    }

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export_forest(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        figure = self._build_figure(results)
        return self._exporter.write(
            figure,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(results, fmt, path),
        )

    def _build_figure(self, results: pd.DataFrame) -> go.Figure:
        if results.empty:
            fig = go.Figure()
            fig.update_layout(
                template=self._template,
                title="Prevalence Shift Under Selective Testing",
                annotations=[
                    {
                        "text": "No organism-drug pairs met the prevalence-shift support thresholds.",
                        "showarrow": False,
                        "x": 0.5,
                        "y": 0.5,
                        "xref": "paper",
                        "yref": "paper",
                    }
                ],
            )
            return fig

        plot_data = results.copy()
        organisms = plot_data["organism"].dropna().unique()
        single_organism = organisms[0] if len(organisms) == 1 else None
        plot_data["label"] = (
            plot_data["drug"] if single_organism else plot_data["organism"] + " | " + plot_data["drug"]
        )
        plot_data = plot_data.sort_values("mnar_lambda0_absolute_shift_pct", ascending=True).reset_index(drop=True)
        plot_data["shift_direction"] = plot_data["mnar_lambda0_shift_from_naive"].map(
            self._shift_direction
        )
        plot_data["shift_color"] = plot_data["shift_direction"].map(
            {
                "naive_overestimates": "#C0392B",
                "naive_underestimates": "#1F77B4",
                "no_difference": "#7F8C8D",
                "unavailable": "#98A2B3",
            }
        ).fillna("#98A2B3")

        hover = [
            "<br>".join(
                [
                    f"{row['organism']} | {row['drug']}",
                    f"Naive prevalence: {row['naive_prevalence_pct']:.2f}%",
                    f"MNAR prevalence (lambda=0): {row['mnar_lambda0_prevalence_pct']:.2f}%"
                    if pd.notna(row["mnar_lambda0_prevalence_pct"])
                    else "MNAR prevalence (lambda=0): NA",
                    f"Lower bound: {row['prevalence_lower_bound_pct']:.2f}%",
                    f"Upper bound: {row['prevalence_upper_bound_pct']:.2f}%",
                    f"Shift: {row['mnar_lambda0_shift_from_naive_pct']:.2f} percentage points"
                    if pd.notna(row["mnar_lambda0_shift_from_naive_pct"])
                    else "Shift: NA",
                    f"rho independent/cascade: {row['rho_independent_vs_cascade']:.2f}" if pd.notna(row["rho_independent_vs_cascade"]) else "rho independent/cascade: NA",
                    f"Cascade trigger fraction: {row['cascade_trigger_fraction']:.2%}" if pd.notna(row["cascade_trigger_fraction"]) else "Cascade trigger fraction: NA",
                ]
            )
            for _, row in plot_data.iterrows()
        ]

        fig = go.Figure()
        for _, row in plot_data.iterrows():
            if pd.notna(row["prevalence_lower_bound_pct"]) and pd.notna(row["prevalence_upper_bound_pct"]):
                fig.add_trace(
                    go.Scatter(
                        x=[row["prevalence_lower_bound_pct"], row["prevalence_upper_bound_pct"]],
                        y=[row["label"], row["label"]],
                        mode="lines",
                        line={"color": "#98A2B3", "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=plot_data["naive_prevalence_pct"],
                y=plot_data["label"],
                mode="markers",
                marker={"size": 9, "color": "#344054", "symbol": "diamond"},
                name="Naive prevalence",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        for direction, group_name in self._SHIFT_DIRECTION_LABELS.items():
            group = plot_data[plot_data["shift_direction"] == direction]
            if group.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group["mnar_lambda0_prevalence_pct"],
                    y=group["label"],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": group["shift_color"].iloc[0],
                        "line": {"width": 1, "color": "#344054"},
                    },
                    name=group_name,
                    text=[hover[i] for i in group.index],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        fig.update_layout(
            template=self._template,
            title={
                "text": (
                    f"Prevalence Shift Under Selective Testing — {format_organism_label(single_organism)}"
                    if single_organism
                    else "Prevalence Shift Under Selective Testing"
                ),
                "font": {"size": 20},
            },
            xaxis_title={"text": "Resistance prevalence (%)", "font": {"size": 15}},
            yaxis_title={"text": "Drug" if single_organism else "Organism | Drug", "font": {"size": 15}},
            xaxis={"tickfont": {"size": 13}},
            yaxis={"tickfont": {"size": 13}},
            legend={"font": {"size": 13}},
            width=self._width,
            height=max(self._height, 55 * len(plot_data) + 280),
            margin={"l": 430, "r": 100, "t": 130, "b": 220},
            annotations=[
                {
                    "text": "Grey line = model-free eligible-denominator bounds. Dark diamond = naive tested-row prevalence. Circle = MNAR sensitivity estimate at lambda=0, colored by direction of shift from naive (see legend).",
                    "showarrow": False,
                    "x": 0.0,
                    "y": -0.12,
                    "xref": "paper",
                    "yref": "paper",
                    "xanchor": "left",
                    "font": {"size": 12, "color": "#475467"},
                }
            ],
        )
        return fig

    def _write_static(self, results: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(20, max(8, 0.55 * max(len(results), 1) + 4.0)))
        if results.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No organism-drug pairs met the prevalence-shift support thresholds.",
                ha="center",
                va="center",
                fontsize=12,
            )
        else:
            plot_data = results.copy()
            organisms = plot_data["organism"].dropna().unique()
            single_organism = organisms[0] if len(organisms) == 1 else None
            plot_data["label"] = (
                plot_data["drug"] if single_organism else plot_data["organism"] + " | " + plot_data["drug"]
            )
            plot_data = plot_data.sort_values("mnar_lambda0_absolute_shift_pct", ascending=True).reset_index(drop=True)
            y_positions = list(range(len(plot_data)))
            plot_data["shift_direction"] = plot_data["mnar_lambda0_shift_from_naive"].map(self._shift_direction)
            direction_colors = {
                "naive_overestimates": "#C0392B",
                "naive_underestimates": "#1F77B4",
                "no_difference": "#7F8C8D",
                "unavailable": "#98A2B3",
            }
            for y, lo, hi in zip(
                y_positions,
                pd.to_numeric(plot_data["prevalence_lower_bound_pct"], errors="coerce"),
                pd.to_numeric(plot_data["prevalence_upper_bound_pct"], errors="coerce"),
                strict=False,
            ):
                if pd.notna(lo) and pd.notna(hi):
                    ax.hlines(y, lo, hi, color="#98A2B3", linewidth=2.5)
            ax.scatter(
                pd.to_numeric(plot_data["naive_prevalence_pct"], errors="coerce"),
                y_positions,
                s=80,
                c="#344054",
                marker="D",
                zorder=3,
                label="Naive prevalence",
            )
            for direction, group_name in self._SHIFT_DIRECTION_LABELS.items():
                group = plot_data[plot_data["shift_direction"] == direction]
                if group.empty:
                    continue
                ax.scatter(
                    pd.to_numeric(group["mnar_lambda0_prevalence_pct"], errors="coerce"),
                    group.index.tolist(),
                    s=100,
                    c=direction_colors[direction],
                    edgecolors="#344054",
                    linewidths=1.2,
                    zorder=4,
                    label=group_name,
                )
            ax.set_yticks(y_positions)
            ax.set_yticklabels(plot_data["label"], fontsize=13)
            ax.set_xlabel("Resistance prevalence (%)", fontsize=15)
            ax.set_ylabel("Drug" if single_organism else "Organism | Drug", fontsize=15)
            ax.set_title(
                f"Prevalence Shift Under Selective Testing — {format_organism_label(single_organism)}"
                if single_organism
                else "Prevalence Shift Under Selective Testing",
                fontsize=20,
                fontweight="bold",
            )
            ax.tick_params(axis="x", labelsize=13)
            ax.grid(axis="x", color="#E8EEF5", linewidth=0.9)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="lower right", fontsize=13)
            fig.text(
                0.02,
                0.03,
                "Grey line = model-free eligible-denominator bounds; dark diamond = naive prevalence; circle = MNAR sensitivity estimate at lambda=0, colored by direction of shift from naive (see legend).",
                fontsize=12,
                color="#475467",
                wrap=True,
            )
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _shift_direction(value: object) -> str:
        if pd.isna(value):
            return "unavailable"
        value = float(value)
        if value > 0:
            return "naive_overestimates"
        if value < 0:
            return "naive_underestimates"
        return "no_difference"
