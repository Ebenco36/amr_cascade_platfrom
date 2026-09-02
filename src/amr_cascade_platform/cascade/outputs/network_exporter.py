"""Cascade network exports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd


class NetworkExporter:
    """Export retained cascade edges and node summaries."""

    def export(self, retained_edges: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        edge_path = output_dir / "network_edges.parquet"
        node_path = output_dir / "network_nodes.parquet"
        json_path = output_dir / "network_graph.json"

        if retained_edges.empty:
            empty_edges = pd.DataFrame(
                columns=["source", "target", "escalation_ratio", "adjusted_odds_ratio", "edge_weight"]
            )
            empty_nodes = pd.DataFrame(columns=["antibiotic", "out_degree", "in_degree"])
            empty_edges.to_parquet(edge_path, index=False)
            empty_nodes.to_parquet(node_path, index=False)
            json_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2), encoding="utf-8")
            return {"edge_path": edge_path, "node_path": node_path, "json_path": json_path}

        edges = retained_edges.loc[
            :,
            ["upstream_antibiotic", "downstream_antibiotic", "escalation_ratio", "adjusted_odds_ratio"],
        ].copy()
        edges = edges.rename(
            columns={"upstream_antibiotic": "source", "downstream_antibiotic": "target"}
        )
        edges["edge_weight"] = edges["escalation_ratio"].map(
            lambda value: math.log(value) if pd.notna(value) and value > 0 else None
        )
        edges.to_parquet(edge_path, index=False)

        graph = nx.DiGraph()
        for row in edges.itertuples(index=False):
            centrality_weight = (
                abs(row.edge_weight)
                if row.edge_weight is not None and pd.notna(row.edge_weight) and row.edge_weight != 0
                else 1.0
            )
            graph.add_edge(
                row.source,
                row.target,
                weight=centrality_weight,
                signed_log_er=row.edge_weight,
                escalation_ratio=row.escalation_ratio,
            )

        out_counts = edges.groupby("source", observed=True).size().rename("out_degree")
        in_counts = edges.groupby("target", observed=True).size().rename("in_degree")
        nodes = pd.concat([out_counts, in_counts], axis=1).fillna(0).reset_index()
        nodes = nodes.rename(columns={"index": "antibiotic", "source": "antibiotic", "target": "antibiotic"})
        if "antibiotic" not in nodes.columns:
            nodes = nodes.rename(columns={nodes.columns[0]: "antibiotic"})
        nodes["out_degree"] = nodes["out_degree"].astype(int)
        nodes["in_degree"] = nodes["in_degree"].astype(int)
        # groupby("source"/"target").reset_index() preserves the Categorical dtype of
        # the grouping key. Series.map() on a Categorical can itself return a
        # Categorical, and .fillna(0.0) then fails because 0.0 isn't an existing
        # category. Cast to plain strings before mapping to avoid that.
        nodes["antibiotic"] = nodes["antibiotic"].astype(str)
        pagerank = self._safe_pagerank(graph)
        betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() > 0 else {}
        nodes["pagerank"] = nodes["antibiotic"].map(pagerank).fillna(0.0)
        nodes["betweenness"] = nodes["antibiotic"].map(betweenness).fillna(0.0)
        nodes.to_parquet(node_path, index=False)

        json_payload = {
            "nodes": nodes.to_dict(orient="records"),
            "edges": edges.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        return {"edge_path": edge_path, "node_path": node_path, "json_path": json_path}

    @staticmethod
    def _safe_pagerank(graph: nx.DiGraph) -> dict[str, float]:
        """Compute PageRank without allowing optional centrality to fail the pipeline."""
        if graph.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(graph, max_iter=1000, tol=1.0e-8, weight="weight")
        except nx.PowerIterationFailedConvergence:
            uniform = 1.0 / graph.number_of_nodes()
            return {node: uniform for node in graph.nodes()}
