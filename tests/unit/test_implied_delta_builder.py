from pathlib import Path

import json
import pandas as pd

from amr_cascade_platform.cascade.outputs.implied_delta_builder import ImpliedDeltaBuilder


def test_implied_delta_builder_writes_summary_and_edges(tmp_path: Path) -> None:
    primary_dir = tmp_path / "armd"
    sensitivity_dir = tmp_path / "armd_sensitivity_implied"
    primary_dir.mkdir()
    sensitivity_dir.mkdir()
    pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
            "escalation_ratio": [2.0],
            "adjusted_odds_ratio": [1.2],
            "rank_by_escalation_ratio": [1],
        }
    ).to_parquet(primary_dir / "edge_report.parquet", index=False)
    pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "C"],
            "downstream_antibiotic": ["B", "D"],
            "escalation_ratio": [3.0, 4.0],
            "adjusted_odds_ratio": [1.5, 1.6],
            "rank_by_escalation_ratio": [1, 2],
        }
    ).to_parquet(sensitivity_dir / "edge_report.parquet", index=False)

    outputs = ImpliedDeltaBuilder().export(primary_dir, sensitivity_dir)
    delta = pd.read_parquet(outputs["implied_delta_edges_path"])
    summary = json.loads((sensitivity_dir / "implied_delta_summary.json").read_text())
    assert len(delta) == 2
    assert summary["sensitivity_only_edge_count"] == 1
