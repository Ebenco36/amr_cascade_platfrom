"""Plotly plotters for non-forest cascade figures."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyThresholdSensitivityPlotter:
    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(self, sensitivity_table: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        if sensitivity_table.empty:
            fig = _empty_figure(self._template, self._width, self._height, "No threshold sensitivity results available")
        else:
            plot_data = sensitivity_table.copy()
            fig = self._build_figure(plot_data)
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(sensitivity_table, fmt, path))

    def _build_figure(self, plot_data: pd.DataFrame) -> go.Figure:
        collapsed = (
            plot_data.groupby(
                [
                    "min_total_support",
                    "retained_edges_n",
                    "support_passing_edges_n",
                    "mean_retained_escalation_ratio",
                    "mean_retained_adjusted_odds_ratio",
                ],
                dropna=False,
                as_index=False,
            )
            .size()
            .rename(columns={"size": "threshold_combination_count"})
            .sort_values(["min_total_support", "retained_edges_n"])
        )
        unique_retained_counts = collapsed["retained_edges_n"].nunique()
        if unique_retained_counts == 1:
            retained = int(collapsed["retained_edges_n"].iloc[0])
            support_min = int(collapsed["support_passing_edges_n"].min())
            support_max = int(collapsed["support_passing_edges_n"].max())
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.68,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=str(retained),
                font={"size": 70, "color": "#355C7D"},
            )
            fig.add_annotation(
                x=0.5,
                y=0.54,
                xref="paper",
                yref="paper",
                showarrow=False,
                text="retained edges across all tested threshold combinations",
                font={"size": 17, "color": "#344054"},
            )
            fig.add_annotation(
                x=0.5,
                y=0.38,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                text=(
                    f"The retained edge count was invariant across {len(plot_data)} evaluated threshold combinations.<br>"
                    f"Support-passing candidate pairs ranged from {support_min:,} to {support_max:,}."
                ),
                font={"size": 15, "color": "#475467"},
            )
        else:
            plot_data["series"] = (
                "min_result="
                + plot_data["min_result_support"].astype(str)
                + " | min_ER="
                + plot_data["retained_min_escalation_ratio"].astype(str)
            )
            fig = px.line(
                plot_data,
                x="min_total_support",
                y="retained_edges_n",
                color="series",
                markers=True,
                hover_data={
                    "support_passing_edges_n": True,
                    "mean_retained_escalation_ratio": ":.3f",
                    "mean_retained_adjusted_odds_ratio": ":.3f",
                },
                template=self._template,
                title="Cascade Threshold Sensitivity",
            )
        fig.update_layout(
            template=self._template,
            title={"text": "Cascade Threshold Sensitivity", "x": 0.5, "xanchor": "center"},
            width=self._width,
            height=self._height,
            xaxis_title="Minimum Total Support" if unique_retained_counts != 1 else None,
            yaxis_title="Retained Edge Count" if unique_retained_counts != 1 else None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"size": 15, "family": "Arial"},
            margin={"l": 70, "r": 40, "t": 95, "b": 55},
        )
        if unique_retained_counts == 1:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
        else:
            fig.update_xaxes(showgrid=True, gridcolor="#E8EEF5")
            fig.update_yaxes(showgrid=True, gridcolor="#E8EEF5")
        return fig

    def _write_static(self, sensitivity_table: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 7))
        if sensitivity_table.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No threshold sensitivity results available", ha="center", va="center")
        else:
            collapsed = (
                sensitivity_table.groupby(
                    [
                        "min_total_support",
                        "retained_edges_n",
                        "support_passing_edges_n",
                        "mean_retained_escalation_ratio",
                        "mean_retained_adjusted_odds_ratio",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "threshold_combination_count"})
            )
            if collapsed["retained_edges_n"].nunique() == 1:
                retained = int(collapsed["retained_edges_n"].iloc[0])
                support_min = int(collapsed["support_passing_edges_n"].min())
                support_max = int(collapsed["support_passing_edges_n"].max())
                ax.axis("off")
                ax.text(0.5, 0.66, str(retained), ha="center", va="center", fontsize=42, color="#355C7D")
                ax.text(
                    0.5,
                    0.52,
                    "retained edges across all tested threshold combinations",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="#344054",
                )
                ax.text(
                    0.5,
                    0.37,
                    (
                        f"The retained edge count was invariant across {len(sensitivity_table)} evaluated threshold combinations.\n"
                        f"Support-passing candidate pairs ranged from {support_min:,} to {support_max:,}."
                    ),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#475467",
                )
            else:
                ax.plot(
                    collapsed["min_total_support"],
                    collapsed["retained_edges_n"],
                    marker="o",
                    linewidth=2.0,
                    color="#355C7D",
                )
                ax.set_xlabel("Minimum Total Support")
                ax.set_ylabel("Retained Edge Count")
                ax.set_title("Cascade Threshold Sensitivity")
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


class PlotlySiteComparisonPlotter:
    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(self, comparison_summary: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        if comparison_summary.empty:
            fig = _empty_figure(self._template, self._width, self._height, "No site comparison summary available")
        else:
            fig = px.bar(
                comparison_summary,
                x="site",
                y="edge_count",
                color="edge_presence",
                barmode="stack",
                template=self._template,
                category_orders={"edge_presence": ["shared", "site_only", "combined_only"]},
                color_discrete_map={"shared": "#355c7d", "site_only": "#6c5b7b", "combined_only": "#c06c84"},
                title="Site vs Combined Retained Edge Comparison",
            )
            fig.update_layout(
                width=self._width,
                height=self._height,
                title={"text": "Site vs Combined Retained Edge Comparison", "x": 0.5, "xanchor": "center"},
                xaxis_title="Site",
                yaxis_title="Edge Count",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font={"size": 15, "family": "Arial"},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1.0,
                    "title": {"text": "Edge presence"},
                },
                margin={"l": 70, "r": 40, "t": 95, "b": 55},
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E8EEF5")
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(comparison_summary, fmt, path))

    def _write_static(self, comparison_summary: pd.DataFrame, fmt: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 7))
        if comparison_summary.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No site comparison summary available", ha="center", va="center")
        else:
            pivot = comparison_summary.pivot(index="site", columns="edge_presence", values="edge_count").fillna(0)
            pivot = pivot.reindex(columns=["shared", "site_only", "combined_only"], fill_value=0)
            pivot.plot(kind="bar", stacked=True, ax=ax, color=["#355c7d", "#6c5b7b", "#c06c84"])
            ax.set_xlabel("Site")
            ax.set_ylabel("Edge Count")
            ax.set_title("Site vs Combined Retained Edge Comparison")
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


class PlotlySankeyPlotter:
    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        classification_resolver: AntibioticClassificationResolver,
        top_n: int = 20,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._classification = classification_resolver
        self._top_n = top_n

    def export(
        self,
        pathway_flows: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        title_prefix = f"{tier_label} " if tier_label else ""
        if pathway_flows.empty:
            message = f"No {tier_label.lower()} pathway flow data available" if tier_label else "No pathway flow data available"
            fig = _empty_figure(self._template, self._width, self._height, message)
            return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(pathway_flows, fmt, path, tier_label))

        flows = pathway_flows.copy()
        flows["source"] = flows["source"].map(self._classification.canonical_display_label)
        flows["target"] = flows["target"].map(self._classification.canonical_display_label)
        sankey_data = (
            flows.groupby(["source", "target"], dropna=False, observed=True)
            .agg(
                value=("path_min_support_n", "sum"),
                mean_edge_escalation_ratio=("edge_escalation_ratio", "mean"),
                max_path_escalation_ratio=("path_escalation_ratio", "max"),
            )
            .reset_index()
        )
        sankey_data = sankey_data.sort_values("value", ascending=False).head(self._top_n).reset_index(drop=True)
        labels = pd.Index(pd.unique(pd.concat([sankey_data["source"], sankey_data["target"]], ignore_index=True)))
        node_metadata = pd.DataFrame({"label": labels})
        node_metadata = self._classification.add_metadata(node_metadata, "label", "antibiotic")
        node_metadata["sort_key"] = node_metadata["antibiotic_aware_category"].map(self._classification.category_sort_key)
        node_metadata = node_metadata.sort_values(by=["sort_key", "label"]).reset_index(drop=True)
        labels = pd.Index(node_metadata["label"])
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        sankey_data = self._classification.add_metadata(sankey_data, "source", "source")
        sankey_data = self._classification.add_metadata(sankey_data, "target", "target")
        fig = go.Figure(
            go.Sankey(
                arrangement="fixed",
                node={
                    "label": labels.tolist(),
                    "pad": 12,
                    "thickness": 16,
                    "color": node_metadata["antibiotic_aware_color"].tolist(),
                    "line": {"color": "#22313F", "width": 0.8},
                    "customdata": node_metadata[["antibiotic_aware_category", "antibiotic_broad_class"]].values,
                    "hovertemplate": (
                        "<b>%{label}</b><br>"
                        "AWaRe=%{customdata[0]}<br>"
                        "Broad class=%{customdata[1]}<extra></extra>"
                    ),
                },
                link={
                    "source": sankey_data["source"].map(label_to_idx),
                    "target": sankey_data["target"].map(label_to_idx),
                    "value": sankey_data["value"],
                    "color": sankey_data["source_aware_color"].map(
                        lambda color: AntibioticClassificationResolver.with_alpha_rgba(color, 0.35)
                    ),
                    "customdata": sankey_data[
                        [
                            "mean_edge_escalation_ratio",
                            "max_path_escalation_ratio",
                            "source_aware_category",
                            "target_aware_category",
                        ]
                    ],
                    "hovertemplate": (
                        "%{source.label} -> %{target.label}<br>"
                        "Source AWaRe=%{customdata[2]}<br>"
                        "Target AWaRe=%{customdata[3]}<br>"
                        "Aggregated support=%{value:.0f}<br>"
                        "Mean edge ER=%{customdata[0]:.3f}<br>"
                        "Max path ER=%{customdata[1]:.3f}<extra></extra>"
                    ),
                },
            )
        )
        top_flow = sankey_data.iloc[0]
        headline = (
            f"Largest flow: {top_flow['source']} → {top_flow['target']} "
            f"(support n={top_flow['value']:.0f}, mean edge ER={top_flow['mean_edge_escalation_ratio']:.2f})"
        )
        for tier in self._classification.CATEGORY_ORDER[:3]:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker={"size": 12, "color": self._classification.CATEGORY_COLORS[tier]},
                    name=tier, showlegend=True,
                )
            )
        fig.update_layout(
            template=self._template,
            title={
                "text": f"Top {len(sankey_data)} {title_prefix}Cascade Pathways by Support<br><sub>{headline}</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            width=self._width,
            height=self._height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.03},
            margin={"l": 28, "r": 28, "t": 110, "b": 40},
            font={"size": 16, "family": "Arial"},
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        if len(sankey_data) <= 3:
            fig.add_annotation(
                text=f"Only {len(sankey_data)} pathway link(s) survived the current cascade pathway filter.",
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                xanchor="left",
                showarrow=False,
                font={"size": 13, "color": "#475467"},
            )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(pathway_flows, fmt, path, tier_label))

    def _write_static(self, pathway_flows: pd.DataFrame, fmt: str, path: Path, tier_label: str = "") -> None:
        fig, ax = plt.subplots(figsize=(13, 8))
        if pathway_flows.empty:
            ax.axis("off")
            message = f"No {tier_label.lower()} pathway flow data available" if tier_label else "No pathway flow data available"
            ax.text(0.5, 0.5, message, ha="center", va="center")
        else:
            flows = pathway_flows.copy()
            flows["source"] = flows["source"].map(self._classification.canonical_display_label)
            flows["target"] = flows["target"].map(self._classification.canonical_display_label)
            sankey_data = (
                flows.groupby(["source", "target"], dropna=False, observed=True)
                .agg(value=("path_min_support_n", "sum"))
                .reset_index()
                .sort_values("value", ascending=False)
                .head(10)
            )
            left_nodes = sankey_data["source"].drop_duplicates().tolist()
            right_nodes = sankey_data["target"].drop_duplicates().tolist()
            left_y = {node: idx for idx, node in enumerate(left_nodes)}
            right_y = {node: idx for idx, node in enumerate(right_nodes)}
            max_value = max(float(sankey_data["value"].max()), 1.0)
            source_meta = self._classification.add_metadata(pd.DataFrame({"label": left_nodes}), "label", "left")
            target_meta = self._classification.add_metadata(pd.DataFrame({"label": right_nodes}), "label", "right")
            source_colors = dict(zip(source_meta["label"], source_meta["left_aware_color"]))
            target_colors = dict(zip(target_meta["label"], target_meta["right_aware_color"]))
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, max(len(left_nodes), len(right_nodes)))
            for node, y in left_y.items():
                ax.text(
                    0.05,
                    y,
                    node,
                    va="center",
                    ha="left",
                    fontsize=13,
                    bbox={"facecolor": source_colors.get(node, "#D0D5DD"), "edgecolor": "#1F2937", "boxstyle": "round,pad=0.25"},
                )
            for node, y in right_y.items():
                ax.text(
                    0.95,
                    y,
                    node,
                    va="center",
                    ha="right",
                    fontsize=13,
                    bbox={"facecolor": target_colors.get(node, "#D0D5DD"), "edgecolor": "#1F2937", "boxstyle": "round,pad=0.25"},
                )
            for row in sankey_data.itertuples(index=False):
                y0 = left_y[row.source]
                y1 = right_y[row.target]
                thickness = 1.0 + 8.0 * (float(row.value) / max_value)
                verts = [(0.18, y0), (0.40, y0), (0.60, y1), (0.82, y1)]
                codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
                patch = PathPatch(
                    MplPath(verts, codes),
                    facecolor="none",
                    edgecolor=source_colors.get(row.source, "#4c78a8"),
                    lw=thickness,
                    alpha=0.45,
                )
                ax.add_patch(patch)
            ax.set_title(f"{tier_label} Cascade Pathway Sankey" if tier_label else "Cascade Pathway Sankey")
            if len(sankey_data) <= 3:
                ax.text(
                    0.5,
                    1.02,
                    f"Sparse pathway result: only {len(sankey_data)} pathway link(s) survived current filtering.",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    transform=ax.transAxes,
                    color="#475467",
                )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _declutter_positions(
    positions: dict[str, tuple[float, float]],
    min_dist: float = 0.16,
    iterations: int = 300,
) -> dict[str, tuple[float, float]]:
    """Push node pairs apart until no two are closer than ``min_dist``.

    Force-directed layouts with a few high-degree hub nodes (e.g. ECIM, MCIM
    here) pull many of their neighbours to similar positions no matter how the
    spring/repulsion parameters are tuned -- the physics wants those nodes
    close together. That produces stacked, unreadable text labels regardless
    of `k`/`iterations`. This decouples label readability from layout physics:
    spring_layout still decides overall structure, this just guarantees a
    floor on pairwise spacing afterwards.
    """
    positions = {node: list(xy) for node, xy in positions.items()}
    nodes = list(positions.keys())
    for _ in range(iterations):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                dx = positions[a][0] - positions[b][0]
                dy = positions[a][1] - positions[b][1]
                dist = float(np.hypot(dx, dy)) or 1e-6
                if dist < min_dist:
                    moved = True
                    push = (min_dist - dist) / 2
                    ux, uy = dx / dist, dy / dist
                    positions[a][0] += ux * push
                    positions[a][1] += uy * push
                    positions[b][0] -= ux * push
                    positions[b][1] -= uy * push
        if not moved:
            break
    return {node: (xy[0], xy[1]) for node, xy in positions.items()}


class PlotlyNetworkPlotter:
    """Headline-pathways network: top escalation-ratio edges only.

    Showing every retained edge in a force-directed layout produces a hairball
    once the graph passes a few dozen edges (this platform routinely retains
    hundreds) -- nodes and labels overlap and no individual pathway is
    traceable. The complete pairwise data lives in the cascade matrix heatmap
    (which scales correctly to that size); this figure's job is different --
    surface the strongest, most novel pathways as a story, not repeat the full
    dataset in a form that can't display it. Defaults to the top 40 edges by
    escalation ratio.
    """

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        classification_resolver: AntibioticClassificationResolver,
        top_n_edges: int = 40,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._classification = classification_resolver
        self._top_n_edges = top_n_edges

    def export(
        self,
        network_edges: pd.DataFrame,
        network_nodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        title_prefix = f"{tier_label} " if tier_label else ""
        edge_scope = f"{tier_label.lower()} edges" if tier_label else "network edges"
        if network_edges.empty or network_nodes.empty:
            message = f"No {tier_label.lower()} cascade network data available" if tier_label else "No cascade network data available"
            fig = _empty_figure(self._template, self._width, self._height, message)
            return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(network_edges, network_nodes, fmt, path, tier_label))

        total_edge_n = len(network_edges)
        ranked_edges = network_edges.copy()
        ranked_edges["source"] = ranked_edges["source"].map(self._classification.canonical_display_label)
        ranked_edges["target"] = ranked_edges["target"].map(self._classification.canonical_display_label)
        ranked_edges["_effect_strength"] = np.abs(
            np.log10(np.clip(pd.to_numeric(ranked_edges["escalation_ratio"], errors="coerce"), 1e-9, None))
        )
        top_edges = ranked_edges.sort_values("_effect_strength", ascending=False).head(self._top_n_edges)
        kept_nodes = set(top_edges["source"]) | set(top_edges["target"])
        node_data = network_nodes.copy()
        node_data["antibiotic"] = node_data["antibiotic"].map(self._classification.canonical_display_label)
        top_nodes = node_data[node_data["antibiotic"].isin(kept_nodes)].drop_duplicates("antibiotic", keep="first")

        graph = nx.DiGraph()
        annotated_nodes = self._classification.add_metadata(top_nodes.copy(), "antibiotic", "antibiotic")
        for row in annotated_nodes.itertuples(index=False):
            graph.add_node(row.antibiotic, out_degree=row.out_degree, in_degree=row.in_degree, pagerank=row.pagerank, betweenness=row.betweenness)
        for row in top_edges.itertuples(index=False):
            graph.add_edge(row.source, row.target, escalation_ratio=row.escalation_ratio, adjusted_odds_ratio=row.adjusted_odds_ratio, edge_weight=row.edge_weight)
        positions = nx.spring_layout(graph, seed=42, weight="edge_weight", k=1.6, iterations=300)
        positions = _declutter_positions(positions)

        # Edge line width/opacity scales with absolute log-distance from ER=1, so
        # strong suppression edges (ER << 1) are visible, not flattened as weak.
        er_values = top_edges["escalation_ratio"].to_numpy()
        effect_strength = top_edges["_effect_strength"].to_numpy()
        strength_range = (effect_strength.max() - effect_strength.min()) or 1.0
        edge_traces = []
        edge_arrows = []
        for (source, target), er, strength in zip(
            zip(top_edges["source"], top_edges["target"]), er_values, effect_strength
        ):
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            weight = (strength - effect_strength.min()) / strength_range
            direction = "escalation" if er >= 1.0 else "suppression"
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"width": 1 + 5 * weight, "color": f"rgba(30,64,175,{0.25 + 0.55 * weight:.2f})"},
                    hovertemplate=f"{source} → {target}<br>ER: {er:.2f}<br>Direction: {direction}<extra></extra>",
                    showlegend=False,
                )
            )
            # A plain line trace carries no directional information -- this is a
            # directed cascade graph (upstream resistance -> downstream testing),
            # so every edge needs a visible arrowhead or the direction the figure
            # is titled "pathways" around is simply not shown. Plotly line traces
            # can't carry arrowheads themselves; draw a short arrow annotation
            # over the last stretch of each edge instead, pulled back from the
            # exact target point so it doesn't vanish under the node marker.
            edge_arrows.append(
                {
                    "x": x0 + 0.92 * (x1 - x0),
                    "y": y0 + 0.92 * (y1 - y0),
                    "ax": x0 + 0.72 * (x1 - x0),
                    "ay": y0 + 0.72 * (y1 - y0),
                    "xref": "x",
                    "yref": "y",
                    "axref": "x",
                    "ayref": "y",
                    "showarrow": True,
                    "arrowhead": 3,
                    "arrowsize": 1.3,
                    "arrowwidth": 1 + 5 * weight,
                    "arrowcolor": f"rgba(30,64,175,{0.45 + 0.55 * weight:.2f})",
                }
            )

        node_order = list(graph.nodes())
        node_x = [positions[node][0] for node in node_order]
        node_y = [positions[node][1] for node in node_order]
        pagerank = [graph.nodes[node]["pagerank"] for node in node_order]
        out_degree = [graph.nodes[node]["out_degree"] for node in node_order]
        in_degree = [graph.nodes[node]["in_degree"] for node in node_order]
        betweenness = [graph.nodes[node]["betweenness"] for node in node_order]
        categories = [self._classification.resolve(node).aware_category for node in node_order]
        colors = [self._classification.resolve(node).color for node in node_order]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_order,
            marker={
                "size": [14 + 70 * value for value in pagerank],
                "color": colors,
                "line": {"width": 0.8, "color": "#17324d"},
            },
            customdata=list(zip(out_degree, in_degree, pagerank, betweenness, categories)),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "AWaRe=%{customdata[4]}<br>"
                "Out-degree=%{customdata[0]}<br>"
                "In-degree=%{customdata[1]}<br>"
                "PageRank=%{customdata[2]:.4f}<br>"
                "Betweenness=%{customdata[3]:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
        # All nodes get labels now: the top-N edge filter keeps this small enough
        # (typically well under 30 nodes) that every label fits without collision,
        # unlike the old "top 20 by pagerank out of everything" partial labeling.
        label_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="text",
            text=node_order,
            textposition="top center",
            textfont={"size": 14, "color": "#243B53"},
            hoverinfo="skip",
            showlegend=False,
        )
        legend_traces = [
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"size": 12, "color": self._classification.CATEGORY_COLORS[tier]},
                name=tier, showlegend=True,
            )
            for tier in ("Access", "Watch", "Reserve")
        ]
        fig = go.Figure(data=[*edge_traces, node_trace, label_trace, *legend_traces])
        max_row = top_edges.iloc[0]
        headline = f"Strongest: {max_row['source']} → {max_row['target']} (ER={max_row['escalation_ratio']:.2f})"
        fig.update_layout(
            template=self._template,
            title={
                "text": f"Top {len(top_edges)} {title_prefix}Cascade Observation Edges (of {total_edge_n} {edge_scope})<br>"
                f"<sub>{headline} · node size = PageRank · edge weight = |log10(ER)|</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            width=self._width,
            height=self._height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.02},
            margin={"l": 20, "r": 20, "t": 100, "b": 40},
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"size": 16, "family": "Arial"},
            annotations=edge_arrows,
        )
        return self._exporter.write(fig, output_stem, formats, static_fallback=lambda fmt, path: self._write_static(network_edges, network_nodes, fmt, path, tier_label))

    def _write_static(
        self,
        network_edges: pd.DataFrame,
        network_nodes: pd.DataFrame,
        fmt: str,
        path: Path,
        tier_label: str = "",
    ) -> None:
        fig, ax = plt.subplots(figsize=(16, 12))
        if network_edges.empty or network_nodes.empty:
            ax.axis("off")
            message = f"No {tier_label.lower()} cascade network data available" if tier_label else "No cascade network data available"
            ax.text(0.5, 0.5, message, ha="center", va="center")
        else:
            total_edge_n = len(network_edges)
            ranked_edges = network_edges.copy()
            ranked_edges["source"] = ranked_edges["source"].map(self._classification.canonical_display_label)
            ranked_edges["target"] = ranked_edges["target"].map(self._classification.canonical_display_label)
            ranked_edges["_effect_strength"] = np.abs(
                np.log10(np.clip(pd.to_numeric(ranked_edges["escalation_ratio"], errors="coerce"), 1e-9, None))
            )
            top_edges = ranked_edges.sort_values("_effect_strength", ascending=False).head(self._top_n_edges)
            kept_nodes = set(top_edges["source"]) | set(top_edges["target"])
            node_data = network_nodes.copy()
            node_data["antibiotic"] = node_data["antibiotic"].map(self._classification.canonical_display_label)
            top_nodes = node_data[node_data["antibiotic"].isin(kept_nodes)].drop_duplicates("antibiotic", keep="first")

            graph = nx.DiGraph()
            for row in top_nodes.itertuples(index=False):
                graph.add_node(row.antibiotic, pagerank=row.pagerank)
            for row in top_edges.itertuples(index=False):
                graph.add_edge(row.source, row.target, weight=row.edge_weight, escalation_ratio=row.escalation_ratio)
            positions = nx.spring_layout(graph, seed=42, weight="weight", k=1.6, iterations=300)
            positions = _declutter_positions(positions)
            node_sizes = [180 + 4200 * graph.nodes[node]["pagerank"] for node in graph.nodes()]

            effect_strength = top_edges["_effect_strength"].to_numpy()
            strength_range = (effect_strength.max() - effect_strength.min()) or 1.0
            edge_widths = 0.6 + 4.0 * (effect_strength - effect_strength.min()) / strength_range
            nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#3B5A8A", width=list(edge_widths), arrows=True, arrowsize=14)
            node_colors = [self._classification.resolve(node).color for node in graph.nodes()]
            nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors, edgecolors="#17324d", node_size=node_sizes)
            labels = {node: node for node in graph.nodes()}
            nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=11,
                                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
            title_prefix = f"{tier_label} " if tier_label else ""
            edge_scope = f"{tier_label.lower()} edges" if tier_label else "network edges"
            ax.set_title(f"Top {len(top_edges)} {title_prefix}Cascade Observation Edges (of {total_edge_n} {edge_scope})")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)


class PlotlyCascadeMatrixPlotter:
    """Full-data escalation matrix: upstream x downstream antibiotic heatmap.

    Replaces the chord diagram, which becomes an unreadable hairball once node
    count passes ~15-20 -- this platform routinely has 40+ antibiotics with
    hundreds of retained edges, so every chord-diagram export was all-ribbons,
    no signal. A heatmap is the textbook-correct form for many x many magnitude
    data: it scales cleanly to this size, includes every retained pair (not a
    filtered top-N), and a sequential color scale plus hover detail carries the
    same information the chord diagram tried to encode with arc width and
    crossing ribbons -- just legibly. Rows/columns are grouped by AWaRe tier
    (Access/Watch/Reserve) with divider lines, so cross-tier escalation
    clusters (e.g. Watch -> Reserve pressure) are visible as a block pattern
    instead of buried in visual noise.
    """

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

    def export(
        self,
        retained_edges: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        if retained_edges.empty:
            message = f"No {tier_label.lower()} retained cascade edges available" if tier_label else "No retained cascade edges available"
            fig = _empty_figure(self._template, self._width, self._height, message)
            return self._exporter.write(fig, output_stem, formats)

        data = retained_edges.copy()
        antibiotic_labels = sorted(
            set(data["upstream_antibiotic"].dropna().astype(str))
            | set(data["downstream_antibiotic"].dropna().astype(str))
        )
        axis_label_map = self._classification.abbreviation_axis_label_map(antibiotic_labels)
        data["upstream_abbrev"] = data["upstream_antibiotic"].astype(str).map(axis_label_map)
        data["downstream_abbrev"] = data["downstream_antibiotic"].astype(str).map(axis_label_map)
        # Synonym labels can share an abbreviation (e.g. two source-system spellings
        # of the same drug). Where two raw rows collapse onto the same cell, keep
        # the higher-escalation-ratio one for display -- deterministic, and matches
        # the "strongest observed signal" framing of the rest of the figure suite.
        data = data.sort_values("escalation_ratio", ascending=False).drop_duplicates(
            subset=["upstream_abbrev", "downstream_abbrev"], keep="first"
        )

        label_of: dict[str, str] = {}
        for row in data.itertuples(index=False):
            label_of.setdefault(row.upstream_abbrev, row.upstream_antibiotic)
            label_of.setdefault(row.downstream_abbrev, row.downstream_antibiotic)
        all_abbrevs = sorted(label_of)
        categories = {abbrev: self._classification.resolve(label_of[abbrev]).aware_category for abbrev in all_abbrevs}
        order = sorted(
            all_abbrevs,
            key=lambda abbrev: (self._classification.category_sort_key(categories[abbrev]), abbrev),
        )
        index_of = {abbrev: i for i, abbrev in enumerate(order)}

        n = len(order)
        z_raw = np.full((n, n), np.nan)
        support = np.full((n, n), "", dtype=object)
        adjusted_or = np.full((n, n), "", dtype=object)
        validation = np.full((n, n), "", dtype=object)
        upstream_full = np.full((n, n), "", dtype=object)
        downstream_full = np.full((n, n), "", dtype=object)
        for row in data.itertuples(index=False):
            i = index_of[row.upstream_abbrev]
            j = index_of[row.downstream_abbrev]
            z_raw[i, j] = row.escalation_ratio
            support[i, j] = f"{int(row.total_support_n):,}" if pd.notna(row.total_support_n) else "N/A"
            adj_or = getattr(row, "adjusted_odds_ratio", None)
            adjusted_or[i, j] = f"{adj_or:.2f}" if pd.notna(adj_or) else "N/A"
            validation[i, j] = str(getattr(row, "validation_status", "") or "not evaluated")
            upstream_full[i, j] = self._classification.resolve(row.upstream_antibiotic).display_label
            downstream_full[i, j] = self._classification.resolve(row.downstream_antibiotic).display_label

        # Escalation ratio spans multiple orders of magnitude (roughly 1 to several
        # thousand) -- a linear color scale lets one or two extreme outliers wash
        # out every other cell into an indistinguishable pale wash. Color on log10
        # instead (matching the log-scale x-axis every forest plot in this suite
        # already uses for the same reason), with the colorbar re-labeled back to
        # raw ratio values so it still reads as "escalation ratio," not "log of."
        z_color = np.log10(z_raw, where=~np.isnan(z_raw), out=np.full_like(z_raw, np.nan))
        finite_log = z_color[~np.isnan(z_color)]
        customdata = np.dstack([z_raw, support, adjusted_or, validation, upstream_full, downstream_full])
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
                x=order,
                y=order,
                customdata=customdata,
                colorscale="Blues",
                colorbar={"title": {"text": "Escalation<br>Ratio"}, "x": 1.015, **colorbar_ticks},
                hoverongaps=False,
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Upstream: %{y} (%{customdata[4]})<br>"
                    "Downstream: %{x} (%{customdata[5]})<br>"
                    "Escalation ratio: %{customdata[0]:.2f}<br>"
                    "Adjusted OR: %{customdata[2]}<br>"
                    "Total support N: %{customdata[1]}<br>"
                    "Validation: %{customdata[3]}<extra></extra>"
                ),
            )
        )

        # AWaRe tier block dividers + labels, replacing the per-node color legend
        # a chord diagram would need (Plotly can't color individual tick labels on
        # a shared axis, so grouped blocks + a caption carry the same information).
        boundaries: list[float] = []
        tier_midpoints: list[tuple[str, float]] = []
        start = 0
        for tier in self._classification.CATEGORY_ORDER:
            members = [abbrev for abbrev in order if categories[abbrev] == tier]
            if not members:
                continue
            end = start + len(members)
            tier_midpoints.append((tier, (start + end - 1) / 2))
            if start > 0:
                boundaries.append(start - 0.5)
            start = end
        for boundary in boundaries:
            fig.add_shape(type="line", x0=boundary, x1=boundary, y0=-0.5, y1=n - 0.5, line={"color": "#1E293B", "width": 1})
            fig.add_shape(type="line", x0=-0.5, x1=n - 0.5, y0=boundary, y1=boundary, line={"color": "#1E293B", "width": 1})
        for tier, midpoint in tier_midpoints:
            color = self._classification.CATEGORY_COLORS.get(tier, "#94A3B8")
            fig.add_annotation(
                x=midpoint, y=1.03, xref="x", yref="paper", text=f"<b>{tier}</b>", showarrow=False,
                font={"size": 14, "color": color}, yanchor="bottom",
            )
            fig.add_annotation(
                x=-0.09, y=midpoint, xref="paper", yref="y", text=f"<b>{tier}</b>", showarrow=False,
                font={"size": 14, "color": color}, textangle=-90, xanchor="right",
            )
        fig.add_annotation(
            x=0.5,
            y=-0.18,
            xref="paper",
            yref="paper",
            text=(
                "Axis labels use standard antibiotic abbreviations; true abbreviation collisions are disambiguated in-place. "
                "Full names, source labels, and synonym mappings are listed in table_supplementary_antibiotic_abbreviations.csv."
            ),
            showarrow=False,
            align="center",
            font={"size": 12, "color": "#475467"},
        )

        n_edges = len(data)
        max_row = data.loc[data["escalation_ratio"].idxmax()]
        max_upstream = self._classification.resolve(max_row["upstream_antibiotic"]).display_label
        max_downstream = self._classification.resolve(max_row["downstream_antibiotic"]).display_label
        headline = (
            f"Strongest: {max_upstream} -> {max_downstream} "
            f"(ER={max_row['escalation_ratio']:.1f}, n={int(max_row['total_support_n']):,})"
        )
        title_prefix = f"{tier_label} " if tier_label else ""
        fig.update_layout(
            template=self._template,
            title={
                "text": f"{title_prefix}Escalation Cascade Matrix — {n_edges} retained pathways<br>"
                f"<sub>{headline}</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            width=max(self._width, 700 + 14 * n),
            height=max(self._height, 700 + 14 * n),
            xaxis={
                "tickangle": 90,
                "tickfont": {"size": 13},
                "side": "bottom",
                "constrain": "domain",
                "title": {"text": "Downstream antibiotic abbreviation"},
            },
            yaxis={
                "tickfont": {"size": 13},
                "autorange": "reversed",
                "scaleanchor": "x",
                "title": {"text": "Upstream antibiotic abbreviation"},
            },
            margin={"l": 120, "r": 90, "t": 110, "b": 150},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return self._exporter.write(fig, output_stem, formats)


def _empty_figure(template: str, width: int, height: int, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_layout(template=template, width=width, height=height, xaxis={"visible": False}, yaxis={"visible": False})
    return fig
