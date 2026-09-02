from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.network_exporter import NetworkExporter


def test_network_exporter_adds_node_metrics(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.4,
            },
            {
                "upstream_antibiotic": "B",
                "downstream_antibiotic": "C",
                "escalation_ratio": 1.5,
                "adjusted_odds_ratio": 1.2,
            },
        ]
    )

    outputs = NetworkExporter().export(retained_edges, tmp_path)
    nodes = pd.read_parquet(outputs["node_path"])
    assert "pagerank" in nodes.columns
    assert "betweenness" in nodes.columns


def test_network_exporter_handles_suppression_edges(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 0.4,
                "adjusted_odds_ratio": 0.7,
            },
            {
                "upstream_antibiotic": "B",
                "downstream_antibiotic": "A",
                "escalation_ratio": 1.8,
                "adjusted_odds_ratio": 1.3,
            },
        ]
    )

    outputs = NetworkExporter().export(retained_edges, tmp_path)
    edges = pd.read_parquet(outputs["edge_path"])
    nodes = pd.read_parquet(outputs["node_path"])

    assert edges["edge_weight"].lt(0).any()
    assert nodes["pagerank"].notna().all()


def test_network_exporter_handles_categorical_antibiotic_columns(tmp_path: Path) -> None:
    """Real retained_edges arrive with Categorical-dtype antibiotic columns
    (dictionary-encoded for memory efficiency). groupby("source"/"target")
    .reset_index() preserves that dtype into nodes["antibiotic"], and
    Series.map() on a Categorical can itself return a Categorical -- so
    .fillna(0.0) previously raised TypeError: Cannot setitem on a Categorical
    with a new category (0.0). Plain-dict DataFrame construction (as in the
    other tests above) never produces Categorical columns, so it can't catch
    this regression; this test builds the columns explicitly as category
    dtype to match the real shape.
    """
    retained_edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "escalation_ratio": 2.0, "adjusted_odds_ratio": 1.4},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "C", "escalation_ratio": 1.5, "adjusted_odds_ratio": 1.2},
        ]
    )
    retained_edges["upstream_antibiotic"] = retained_edges["upstream_antibiotic"].astype("category")
    retained_edges["downstream_antibiotic"] = retained_edges["downstream_antibiotic"].astype("category")

    outputs = NetworkExporter().export(retained_edges, tmp_path)
    nodes = pd.read_parquet(outputs["node_path"])
    assert nodes["pagerank"].notna().all()
    assert nodes["betweenness"].notna().all()
