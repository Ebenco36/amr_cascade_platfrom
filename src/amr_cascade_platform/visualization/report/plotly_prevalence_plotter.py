"""Plot selective-testing prevalence-shift summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from amr_cascade_platform.visualization.report.organism_labels import format_organism_label
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyPrevalencePlotter:
    """Render publication-ready prevalence-shift plots."""

    _SHIFT_DIRECTION_LABELS = {
        "naive_overestimates": "Reference estimate (naive higher)",
        "naive_underestimates": "Reference estimate (naive lower)",
        "no_difference": "Reference estimate (no difference)",
        "unavailable": "Reference estimate (not estimable)",
    }
    _SHIFT_COLORS = {
        "naive_overestimates": "#C0392B",
        "naive_underestimates": "#1F77B4",
        "no_difference": "#7F8C8D",
        "unavailable": "#98A2B3",
    }
    _BOUNDS_COLOR = "#B8C0CC"

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
            prefer_static_fallback=True,
        )

    def _build_figure(self, results: pd.DataFrame) -> go.Figure:
        if results.empty:
            fig = go.Figure()
            fig.update_layout(
                template=self._template,
                title="Eligible-denominator resistance summaries",
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
        # Sorted by the lambda=0 reference-model estimate (not by shift magnitude): rows
        # then form one visual gradient top-to-bottom instead of jumping between
        # unrelated prevalence levels from row to row.
        plot_data = plot_data.sort_values("mnar_lambda0_prevalence_pct", ascending=True).reset_index(drop=True)
        plot_data["shift_direction"] = plot_data["mnar_lambda0_shift_from_naive"].map(
            self._shift_direction
        )
        plot_data["shift_color"] = plot_data["shift_direction"].map(self._SHIFT_COLORS).fillna("#98A2B3")

        hover = [
            "<br>".join(
                [
                    f"{row['organism']} | {row['drug']}",
                    f"Naive prevalence: {row['naive_prevalence_pct']:.2f}%",
                    f"Reference-model prevalence (λ=0): {row['mnar_lambda0_prevalence_pct']:.2f}%"
                    if pd.notna(row["mnar_lambda0_prevalence_pct"])
                    else "Reference-model prevalence (λ=0): NA",
                    f"Lower bound: {row['prevalence_lower_bound_pct']:.2f}%",
                    f"Upper bound: {row['prevalence_upper_bound_pct']:.2f}%",
                    f"Naive minus reference estimate: {row['mnar_lambda0_shift_from_naive_pct']:.2f} percentage points"
                    if pd.notna(row["mnar_lambda0_shift_from_naive_pct"])
                    else "Shift: NA",
                    f"rho independent/cascade: {row['rho_independent_vs_cascade']:.2f}" if pd.notna(row["rho_independent_vs_cascade"]) else "rho independent/cascade: NA",
                    f"Cascade trigger fraction: {row['cascade_trigger_fraction']:.2%}" if pd.notna(row["cascade_trigger_fraction"]) else "Cascade trigger fraction: NA",
                ]
            )
            for _, row in plot_data.iterrows()
        ]

        # Neutral horizontal lines are model-free bounds. The coloured segment
        # links tested-row prevalence to the lambda=0 reference-model estimate;
        # colour encodes the sign of naive minus reference, not uncertainty.
        fig = go.Figure()
        for _, row in plot_data.iterrows():
            lo, hi = row["prevalence_lower_bound_pct"], row["prevalence_upper_bound_pct"]
            if pd.notna(lo) and pd.notna(hi):
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[row["label"], row["label"]],
                        mode="lines",
                        line={"color": self._BOUNDS_COLOR, "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            naive = row.get("naive_prevalence_pct")
            reference = row.get("mnar_lambda0_prevalence_pct")
            if pd.notna(naive) and pd.notna(reference):
                fig.add_trace(
                    go.Scatter(
                        x=[naive, reference],
                        y=[row["label"], row["label"]],
                        mode="lines",
                        line={"color": row["shift_color"], "width": 3},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": self._BOUNDS_COLOR, "width": 2},
                name="Eligible-denominator bounds",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data["naive_prevalence_pct"],
                y=plot_data["label"],
                mode="markers",
                marker={"size": 8, "symbol": "circle-open", "color": "#667085", "line": {"width": 1.5}},
                name="Naive tested-row prevalence",
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
        fig.update_layout(
            template=self._template,
            title={
                "text": (
                    f"<b>Eligible-denominator resistance summaries — {format_organism_label(single_organism)}</b>"
                    if single_organism
                    else "<b>Eligible-denominator resistance summaries</b>"
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
                "y": -0.08,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 10},
            },
            width=self._width,
            height=max(self._height, 58 * len(plot_data) + 260),
            margin={"l": 190, "r": 120, "t": 100, "b": 120},
            plot_bgcolor="white",
            paper_bgcolor="white",
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
        fig, ax = plt.subplots(figsize=(12, max(6.8, 0.48 * max(len(results), 1) + 2.6)))
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
            for y, lo, hi in zip(
                y_positions,
                pd.to_numeric(plot_data["prevalence_lower_bound_pct"], errors="coerce"),
                pd.to_numeric(plot_data["prevalence_upper_bound_pct"], errors="coerce"),
                strict=False,
            ):
                if pd.notna(lo) and pd.notna(hi):
                    ax.hlines(y, lo, hi, color=self._BOUNDS_COLOR, linewidth=2.2, zorder=1)
            for y, row in zip(y_positions, plot_data.itertuples(), strict=False):
                direction = row.shift_direction
                color = self._SHIFT_COLORS.get(direction, "#98A2B3")
                naive_v = pd.to_numeric(pd.Series([row.naive_prevalence_pct]), errors="coerce").iloc[0]
                reference_v = pd.to_numeric(pd.Series([row.mnar_lambda0_prevalence_pct]), errors="coerce").iloc[0]
                if pd.notna(naive_v) and pd.notna(reference_v):
                    ax.plot([naive_v, reference_v], [y, y], color=color, linewidth=2.5, zorder=2)
                if pd.notna(naive_v):
                    ax.scatter([naive_v], [y], s=70, facecolors="none", edgecolors="#667085", linewidths=1.5, zorder=3)
                if pd.notna(reference_v):
                    ax.scatter([reference_v], [y], marker="s", s=80, c=color, edgecolors="#344054", linewidths=0.8, zorder=4)
                n_val = getattr(row, "eligible_n", np.nan)
                bound_end = getattr(row, "prevalence_upper_bound_pct", np.nan)
                if pd.notna(n_val) and pd.notna(bound_end):
                    ax.annotate(
                        f"n={int(n_val):,}",
                        (bound_end, y),
                        xytext=(7, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8.5,
                        color="#667085",
                    )
            ax.set_yticks(y_positions)
            ax.set_yticklabels(plot_data["label"], fontsize=13)
            ax.set_xlabel("Resistance prevalence (%)", fontsize=15)
            ax.set_ylabel("Drug" if single_organism else "Organism | Drug", fontsize=15)
            ax.set_title(
                f"Eligible-denominator resistance summaries — {format_organism_label(single_organism)}"
                if single_organism
                else "Eligible-denominator resistance summaries",
                fontsize=16,
                fontweight="bold",
                pad=14,
            )
            ax.tick_params(axis="x", labelsize=13)
            ax.grid(axis="x", color="#E8EEF5", linewidth=0.9)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            present_directions = set(plot_data["shift_direction"].dropna().astype(str))
            legend_handles = [
                Line2D([0], [0], color=self._BOUNDS_COLOR, linewidth=2.2, label="Eligible-denominator bounds"),
                Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="#667085",
                       markersize=7, label="Naive tested-row prevalence"),
            ]
            for direction in self._SHIFT_DIRECTION_LABELS:
                if direction not in present_directions:
                    continue
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="s",
                        linestyle="None",
                        markerfacecolor=self._SHIFT_COLORS[direction],
                        markeredgecolor="#344054",
                        markersize=7,
                        label=self._SHIFT_DIRECTION_LABELS[direction],
                    )
                )
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.025),
                ncol=2,
                frameon=False,
                fontsize=9.5,
            )
        fig.subplots_adjust(left=0.25, right=0.94, top=0.88, bottom=0.20)
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
            marker={"color": "#1F4E9C"},
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
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_kappa_static(data, fmt, path),
            prefer_static_fallback=True,
        )

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
            x=data["cascade_trigger_fraction"] * 100, y=data["_enrichment"], mode="markers", text=data["drug"],
            marker={
                "size": np.clip(data["_bound_width"] * 1.6, 8, 55),
                "color": data["_bound_width"], "colorscale": [[0, "#27AE60"], [0.5, "#F5CBA7"], [1, "#C0392B"]],
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Bound width<br>(percentage points)"},
                    "thickness": 18,
                    "len": 0.72,
                },
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate="<b>%{text}</b><br>kappa %{x:.1f}%<br>Enrichment %{y:.1f}pp<extra></extra>",
        ))
        fig.add_hline(y=0, line={"color": "#94A3B8", "width": 1})
        fig.update_layout(
            template=self._template, width=self._width, height=max(self._height, 700),
            title={
                "text": (
                    "<b>Surveillance-sensitivity map</b>"
                    "<br><sup>Selective observation, resistance enrichment, "
                    "and eligible-denominator uncertainty by drug</sup>"
                ),
                "font": {"size": 15},
            },
            xaxis={
                "title": {"text": "Cascade-trigger fraction, κ (%)"}
            },
            yaxis={
                "title": {"text": "Resistance enrichment (percentage points)"}
            },
            margin={"l": 110, "r": 100, "t": 110, "b": 80},
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_surveillance_sensitivity_map_static(data, fmt, path),
            prefer_static_fallback=True,
        )

    @staticmethod
    def _write_kappa_static(data: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, max(6.2, 0.40 * max(len(data), 1) + 1.8)))
        values = pd.to_numeric(data["cascade_trigger_fraction"], errors="coerce").to_numpy() * 100
        positions = np.arange(len(data))
        ax.barh(positions, values, color="#1F4E9C", alpha=0.92)
        ax.set_yticks(positions)
        ax.set_yticklabels(data["drug"], fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Cascade-trigger fraction, κ (%)", fontsize=11)
        ax.set_title("Cascade-trigger concentration by drug", fontsize=15, fontweight="bold", pad=22)
        ax.text(
            0.0,
            1.01,
            "Share of observed tests occurring after a validated upstream resistant trigger",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#475467",
        )
        for position, value in zip(positions, values, strict=True):
            ax.text(min(value + 1.0, 102), position, f"{value:.0f}%", va="center", fontsize=9, color="#344054")
        ax.grid(axis="x", color="#E8EEF5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _write_surveillance_sensitivity_map_static(data: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 7))
        x_values = pd.to_numeric(data["cascade_trigger_fraction"], errors="coerce").to_numpy() * 100
        y_values = pd.to_numeric(data["_enrichment"], errors="coerce").to_numpy()
        bound_width = pd.to_numeric(data["_bound_width"], errors="coerce").to_numpy()
        sizes = np.clip(bound_width * 10.0, 70, 560)
        color_map = LinearSegmentedColormap.from_list(
            "surveillance_sensitivity",
            ["#27AE60", "#F5CBA7", "#C0392B"],
        )
        color_min = float(np.nanmin(bound_width))
        color_max = float(np.nanmax(bound_width))
        if np.isclose(color_min, color_max):
            color_max = color_min + 1.0
        scatter = ax.scatter(
            x_values,
            y_values,
            s=sizes,
            c=bound_width,
            cmap=color_map,
            norm=Normalize(vmin=color_min, vmax=color_max),
            edgecolors="white",
            linewidths=1.0,
            alpha=0.90,
            zorder=3,
        )
        ax.axhline(0, color="#94A3B8", linewidth=1.0, zorder=1)
        ax.grid(color="#E8EEF5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cascade-trigger fraction, κ (%)", fontsize=11)
        ax.set_ylabel("Resistance enrichment (percentage points)", fontsize=11)
        ax.set_title("Surveillance-sensitivity map", fontsize=16, fontweight="bold", pad=24)
        ax.text(
            0.0,
            1.01,
            "Selective observation, resistance enrichment, and eligible-denominator uncertainty by drug",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#475467",
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
        colorbar.set_label("Bound width (percentage points)", fontsize=9.5)

        if len(data) <= 12:
            label_indices = list(range(len(data)))
        else:
            label_indices_set: set[int] = set()
            for values in (x_values, y_values, bound_width):
                label_indices_set.update(np.argsort(values)[-4:].tolist())
            label_indices = sorted(label_indices_set)
        offsets = ((6, 6), (6, -11), (-6, 7), (-6, -12))
        for order, index in enumerate(label_indices):
            offset_x, offset_y = offsets[order % len(offsets)]
            ax.annotate(
                str(data.iloc[index]["drug"]),
                (x_values[index], y_values[index]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left" if offset_x > 0 else "right",
                fontsize=8,
                color="#1F2937",
                bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
            )
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        return fig
