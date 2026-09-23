"""Plot selective-testing prevalence-shift summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
        # Sorted by the MNAR point estimate itself (not by shift magnitude): rows
        # then form one visual gradient top-to-bottom instead of jumping between
        # unrelated prevalence levels from row to row.
        plot_data = plot_data.sort_values("mnar_lambda0_prevalence_pct", ascending=True).reset_index(drop=True)
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

        # Classic point-estimate + CI forest plot: the MNAR estimate is the point
        # (square = a direction with a shift, so it reads next to its CI at a
        # glance), the model-free eligible-denominator bounds are its whisker,
        # and naive prevalence rides alongside as a small open reference marker
        # on the same row -- not a second, disconnected point with no CI of its
        # own, and not a dumbbell (there is no meaningful "before/after" pairing
        # here since the naive value has no uncertainty interval to pair with).
        fig = go.Figure()
        for _, row in plot_data.iterrows():
            lo, hi = row["prevalence_lower_bound_pct"], row["prevalence_upper_bound_pct"]
            if pd.notna(lo) and pd.notna(hi):
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[row["label"], row["label"]],
                        mode="lines",
                        line={"color": row["shift_color"], "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=plot_data["naive_prevalence_pct"],
                y=plot_data["label"],
                mode="markers",
                marker={"size": 8, "symbol": "circle-open", "color": "#667085", "line": {"width": 1.5}},
                name="Naive prevalence (reference)",
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
                        "size": 11,
                        "symbol": "square",
                        "color": group["shift_color"].iloc[0],
                        "line": {"width": 1, "color": "#344054"},
                    },
                    name=group_name,
                    text=[hover[i] for i in group.index],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        # Font/margin values below are pre-print-scale: PlotlyFigureExporter
        # multiplies every font size AND every margin value by PRINT_FONT_SCALE
        # (2.6x) for static exports, so these must be sized for the scaled
        # result, not the on-screen appearance -- see plotly_model_plotters.py
        # _apply_standard_layout for the same convention.
        fig.update_layout(
            template=self._template,
            title={
                "text": (
                    f"Prevalence Shift Under Selective Testing — {format_organism_label(single_organism)}"
                    if single_organism
                    else "Prevalence Shift Under Selective Testing"
                ),
                "font": {"size": 16},
            },
            xaxis_title={"text": "Resistance prevalence (%)", "font": {"size": 13}},
            yaxis_title={"text": "Drug" if single_organism else "Organism | Drug", "font": {"size": 13}},
            xaxis={"tickfont": {"size": 12}},
            yaxis={"tickfont": {"size": 12}},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.12,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 12},
            },
            width=self._width,
            height=max(self._height, 75 * len(plot_data) + 560),
            margin={"l": 170, "r": 100, "t": 90, "b": 290},
            annotations=[
                {
                    "text": "Square = MNAR estimate at lambda=0; horizontal bar = model-free<br>eligible-denominator bounds. Open circle = naive tested-row prevalence<br>(reference, no interval of its own). Colour = direction of shift from<br>naive (see legend). n = eligible episode-drug pairs.",
                    "showarrow": False,
                    "x": 0.0,
                    "y": -0.55,
                    "xref": "paper",
                    "yref": "paper",
                    "xanchor": "left",
                    "align": "left",
                    "font": {"size": 10.5, "color": "#475467"},
                }
            ],
        )
        for _, row in plot_data.iterrows():
            n_val = row.get("eligible_n")
            if pd.notna(n_val):
                x_end = row["prevalence_upper_bound_pct"] if pd.notna(row["prevalence_upper_bound_pct"]) else row["mnar_lambda0_prevalence_pct"]
                fig.add_annotation(
                    x=x_end, y=row["label"], text=f"n={int(n_val):,}",
                    showarrow=False, xanchor="left", xshift=8,
                    font={"size": 11, "color": "#667085"},
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
            plot_data = plot_data.sort_values("mnar_lambda0_prevalence_pct", ascending=True).reset_index(drop=True)
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
                    ax.hlines(y, lo, hi, color="#DDE2E8", linewidth=2.5, zorder=1)
            # Dumbbell: one naive-to-MNAR connector per row, colored by shift
            # direction, matching the interactive figure's encoding.
            seen_directions: set[str] = set()
            for y, row in zip(y_positions, plot_data.itertuples(), strict=False):
                direction = row.shift_direction
                color = direction_colors.get(direction, "#98A2B3")
                naive_v = pd.to_numeric(pd.Series([row.naive_prevalence_pct]), errors="coerce").iloc[0]
                mnar_v = pd.to_numeric(pd.Series([row.mnar_lambda0_prevalence_pct]), errors="coerce").iloc[0]
                if pd.notna(naive_v) and pd.notna(mnar_v):
                    ax.plot([naive_v, mnar_v], [y, y], color=color, linewidth=2.5, zorder=2)
                label = self._SHIFT_DIRECTION_LABELS.get(direction, direction) if direction not in seen_directions else None
                seen_directions.add(direction)
                if pd.notna(naive_v):
                    ax.scatter([naive_v], [y], s=70, facecolors="none", edgecolors="#667085", linewidths=1.5, zorder=3)
                if pd.notna(mnar_v):
                    ax.scatter([mnar_v], [y], s=110, c=color, edgecolors="#344054", linewidths=1.0, zorder=4, label=label)
            ax.scatter([], [], s=70, facecolors="none", edgecolors="#667085", linewidths=1.5, label="Naive prevalence")
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
                "Open circle = naive prevalence; filled circle = MNAR estimate at lambda=0, connected by a line colored by shift direction (see legend). Pale line = model-free eligible-denominator bounds.",
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

    def export_kappa_ranked(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        """Ranked bar: cascade-trigger fraction (kappa) per drug.

        How concentrated each drug's observed denominator is in episodes with
        a validated upstream resistant trigger, vs. independently-tested ones.
        """
        required = {"drug", "cascade_trigger_fraction"}
        if results.empty or not required.issubset(results.columns):
            return self._exporter.write(self._empty_figure("Cascade-trigger fraction unavailable"), output_stem, formats)

        data = results.dropna(subset=["cascade_trigger_fraction"]).copy()
        data = data.sort_values("cascade_trigger_fraction", ascending=True)
        fig = go.Figure(go.Bar(
            x=data["cascade_trigger_fraction"] * 100, y=data["drug"], orientation="h",
            marker={"color": data["cascade_trigger_fraction"], "colorscale": [[0, "#CBD5E1"], [1, "#1F4E9C"]], "cmin": 0, "cmax": 100 * data["cascade_trigger_fraction"].max()},
            text=[f"{v:.0%}" for v in data["cascade_trigger_fraction"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>kappa = %{x:.1f}% of observed tests followed a validated upstream trigger<extra></extra>",
        ))
        fig.update_layout(
            template=self._template, width=self._width, height=max(self._height, 32 * len(data) + 200),
            title={
                "text": "<b>Cascade-trigger concentration (κ) by drug</b>"
                        "<br><sup>Share of this drug's observed tests that followed a validated upstream resistant trigger</sup>",
                "font": {"size": 15},
            },
            xaxis={"title": {"text": "Cascade-trigger fraction, κ (%)"}, "range": [0, 100]},
            yaxis={"title": ""}, margin={"l": 200, "r": 60, "t": 100, "b": 60},
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        return self._exporter.write(fig, output_stem, formats)

    def export_cascade_vs_independent_dumbbell(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        """Dumbbell: resistance prevalence among cascade-triggered vs. independently-tested episodes, per drug."""
        required = {"drug", "cascade_prevalence_pct", "independent_prevalence_pct"}
        if results.empty or not required.issubset(results.columns):
            return self._exporter.write(self._empty_figure("Cascade-vs-independent prevalence unavailable"), output_stem, formats)

        data = results.dropna(subset=["cascade_prevalence_pct", "independent_prevalence_pct"]).copy()
        data["_gap"] = (data["cascade_prevalence_pct"] - data["independent_prevalence_pct"]).abs()
        data = data.sort_values("_gap", ascending=True)
        fig = go.Figure()
        for _, row in data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["independent_prevalence_pct"], row["cascade_prevalence_pct"]], y=[row["drug"], row["drug"]],
                mode="lines", line={"color": "#CBD5E1", "width": 3}, hoverinfo="skip", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=data["independent_prevalence_pct"], y=data["drug"], mode="markers", name="Independently tested",
            marker={"size": 11, "color": "#27AE60"},
            hovertemplate="<b>%{y}</b><br>Independent: %{x:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=data["cascade_prevalence_pct"], y=data["drug"], mode="markers", name="Cascade-triggered",
            marker={"size": 11, "color": "#1F4E9C"},
            hovertemplate="<b>%{y}</b><br>Cascade-triggered: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            template=self._template, width=self._width, height=max(self._height, 32 * len(data) + 220),
            title={
                "text": "<b>Resistance prevalence: cascade-triggered vs. independently tested</b>"
                        "<br><sup>Among observed binary tests for this drug, split by whether a validated upstream resistant trigger was active</sup>",
                "font": {"size": 15},
            },
            xaxis={"title": {"text": "Resistance prevalence (%)"}},
            yaxis={"title": ""}, margin={"l": 200, "r": 60, "t": 100, "b": 90},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return self._exporter.write(fig, output_stem, formats)

    def export_surveillance_sensitivity_map(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        """Scatter: kappa vs. cascade/independent prevalence gap, sized by identification-bound width.

        Not a composite risk score -- three separate, individually-interpretable
        diagnostics shown on one map. Top-right, large points are the drugs
        where routine surveillance most warrants caution.
        """
        required = {"drug", "cascade_trigger_fraction", "cascade_prevalence_pct", "independent_prevalence_pct",
                    "prevalence_lower_bound_pct", "prevalence_upper_bound_pct"}
        if results.empty or not required.issubset(results.columns):
            return self._exporter.write(self._empty_figure("Surveillance-sensitivity map unavailable"), output_stem, formats)

        data = results.dropna(subset=list(required)).copy()
        data["_enrichment"] = data["cascade_prevalence_pct"] - data["independent_prevalence_pct"]
        data["_bound_width"] = data["prevalence_upper_bound_pct"] - data["prevalence_lower_bound_pct"]
        # Markers only, not markers+text: with 25-30 drugs several cluster
        # tightly in kappa/enrichment space, and a label on every point makes
        # that cluster an unreadable pile-up. Identity comes from hover (HTML)
        # and the labelled table this figure accompanies, not an always-on label.
        fig = go.Figure(go.Scatter(
            x=data["cascade_trigger_fraction"] * 100, y=data["_enrichment"], mode="markers",
            marker={
                "size": np.clip(data["_bound_width"] * 1.6, 8, 55),
                "color": data["_bound_width"], "colorscale": [[0, "#27AE60"], [0.5, "#F5CBA7"], [1, "#C0392B"]],
                "showscale": True, "colorbar": {"title": {"text": "Bound<br>width (pp)"}}, "line": {"width": 1, "color": "white"},
            },
            hovertemplate="<b>%{text}</b><br>kappa %{x:.1f}%<br>Enrichment %{y:.1f}pp<extra></extra>",
        ))
        fig.add_hline(y=0, line={"color": "#94A3B8", "width": 1})
        fig.update_layout(
            template=self._template, width=self._width, height=max(self._height, 700),
            title={
                "text": "<b>Surveillance-sensitivity map</b>"
                        "<br><sup>x = cascade-trigger concentration (κ); y = cascade-triggered minus independent prevalence; "
                        "point size/colour = eligible-denominator bound width. Top-right, large points warrant the most caution reading naive prevalence.</sup>",
                "font": {"size": 15},
            },
            xaxis={"title": {"text": "Cascade-trigger fraction, κ (%)"}},
            yaxis={"title": {"text": "Cascade-triggered − independent prevalence (percentage points)"}},
            margin={"l": 90, "r": 40, "t": 120, "b": 70}, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        return self._exporter.write(fig, output_stem, formats)

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        return fig
