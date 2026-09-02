"""Publication-safe network plotting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class NetworkPlotter:
    """Plot retained cascade edges as a directed network figure."""

    def plot(self, retained_edges: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "network_plot.png"
        if retained_edges.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")
            ax.text(0.5, 0.5, "No retained cascade edges", ha="center", va="center")
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return {"network_plot_path": png_path}

        edges = retained_edges.loc[
            :,
            ["upstream_antibiotic", "downstream_antibiotic", "escalation_ratio"],
        ].copy()
        graph = nx.DiGraph()
        for row in edges.itertuples(index=False):
            graph.add_edge(row.upstream_antibiotic, row.downstream_antibiotic, weight=row.escalation_ratio)

        pos = nx.spring_layout(graph, seed=42, k=1.2)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title("Cascade Network (Retained Empirical Edges)")
        ax.axis("off")

        degrees = dict(graph.degree())
        node_sizes = [600 + 250 * degrees[node] for node in graph.nodes]
        edge_widths = [1.0 + 0.75 * graph[u][v]["weight"] for u, v in graph.edges]

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color="#c7dcef",
            edgecolors="#355c7d",
            linewidths=1.2,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            width=edge_widths,
            edge_color="#355c7d",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=8,
            font_family="DejaVu Sans",
            ax=ax,
        )

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return {"network_plot_path": png_path}
