"""Integration test for the network diagram's per-tier topology recomputation.

Unlike Sankey/Chord tiering (a direct filter of an already-built table), Network
tiering must recompute pagerank/betweenness fresh for each tier's own smaller
graph -- see report_export_workflow.py's cascade_network block. This test
exercises that real path: ReportExportWorkflow._tiered_retained_edges() ->
NetworkExporter().export() -> read back -> the same shape PlotlyNetworkPlotter
consumes. It intentionally does not go through the full ReportExportWorkflow.run()
(which needs a real Settings/PathManager/DatasetStore), just the two pieces that
have to agree with each other: the tiering helper and NetworkExporter.
"""

import tempfile
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.network_exporter import NetworkExporter
from amr_cascade_platform.reporting.workflows.report_export_workflow import ReportExportWorkflow


def test_network_tier_topology_is_recomputed_per_tier() -> None:
    # A -> B -> C is fully robust; D -> E is robust but has no other edges, so it
    # should have a *different* pagerank distribution than if it were mixed in
    # with the full graph -- this is exactly what filtering the precomputed
    # network_nodes (instead of recomputing) would get wrong.
    retained_edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "escalation_ratio": 2.0, "adjusted_odds_ratio": 1.4},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "C", "escalation_ratio": 1.8, "adjusted_odds_ratio": 1.3},
            {"upstream_antibiotic": "D", "downstream_antibiotic": "E", "escalation_ratio": 1.5, "adjusted_odds_ratio": 1.2},
            {"upstream_antibiotic": "F", "downstream_antibiotic": "G", "escalation_ratio": 3.0, "adjusted_odds_ratio": 1.9},
        ]
    )
    validation_results = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "validation_status": "robust"},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "C", "validation_status": "robust"},
            {"upstream_antibiotic": "D", "downstream_antibiotic": "E", "validation_status": "robust"},
            {"upstream_antibiotic": "F", "downstream_antibiotic": "G", "validation_status": "mixed"},
        ]
    )

    tiered = ReportExportWorkflow._tiered_retained_edges(retained_edges, validation_results)
    robust_edges = tiered["robust"]
    assert set(zip(robust_edges["upstream_antibiotic"], robust_edges["downstream_antibiotic"])) == {
        ("A", "B"),
        ("B", "C"),
        ("D", "E"),
    }
    # F -> G is mixed, must not leak into the robust tier's graph.
    assert "F" not in set(robust_edges["upstream_antibiotic"]) | set(robust_edges["downstream_antibiotic"])

    with tempfile.TemporaryDirectory() as tmp:
        outputs = NetworkExporter().export(robust_edges, Path(tmp))
        nodes = pd.read_parquet(outputs["node_path"])
        edges = pd.read_parquet(outputs["edge_path"])

    # Topology was computed on exactly the 3-edge robust subgraph: B (the only
    # node with both in- and out-degree) should outrank the endpoint-only nodes,
    # and F/G must be entirely absent from the recomputed node set.
    assert set(nodes["antibiotic"]) == {"A", "B", "C", "D", "E"}
    assert len(edges) == 3
    pagerank_by_node = dict(zip(nodes["antibiotic"], nodes["pagerank"]))
    assert pagerank_by_node["B"] > pagerank_by_node["A"]
