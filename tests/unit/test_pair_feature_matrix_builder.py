from pathlib import Path
import math

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import PairFeatureMatrixBuilder


def test_pair_feature_matrix_builder_uses_training_site_cascade_artifacts_only(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    gold_dir = paths.paths.gold / "combined"
    gold_dir.mkdir(parents=True, exist_ok=True)
    pair = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-01-10T10:00:00Z",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd_ecuh",
        "upstream_antibiotic": "CIPROFLOXACIN",
        "downstream_antibiotic": "MEROPENEM",
        "upstream_susceptibility": "RESISTANT",
        "downstream_tested": 1,
        "downstream_eligible": 1,
        "downstream_intrinsic_resistance": 0,
        "pair_direction": "CIPROFLOXACIN -> MEROPENEM",
    }
    episode = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-01-10T10:00:00Z",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd_ecuh",
        "ordering_mode": "clinical",
        "culture_description": "urine",
        "was_positive": 1,
    }
    pd.DataFrame([pair]).to_parquet(gold_dir / "drug_pair_episodes.parquet", index=False)
    pd.DataFrame([episode]).to_parquet(gold_dir / "culture_episodes.parquet", index=False)

    train_artifact_dir = paths.paths.artifacts / settings.cascade.outputs.result_dir / "armd"
    train_artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 7.5,
                "adjusted_odds_ratio": 1.8,
                "total_support_n": 42,
                "resistant_downstream_test_probability": 0.4,
                "susceptible_downstream_test_probability": 0.1,
                "retained_edge": True,
            }
        ]
    ).to_parquet(train_artifact_dir / "edge_report.parquet", index=False)
    pd.DataFrame(
        [
            {"antibiotic": "CIPROFLOXACIN", "out_degree": 2, "in_degree": 1, "pagerank": 0.2, "betweenness": 0.1},
            {"antibiotic": "MEROPENEM", "out_degree": 1, "in_degree": 3, "pagerank": 0.3, "betweenness": 0.2},
        ]
    ).to_parquet(train_artifact_dir / "network_nodes.parquet", index=False)

    validation_artifact_dir = paths.paths.artifacts / settings.cascade.outputs.result_dir / "armd_ecuh"
    validation_artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 999.0,
                "adjusted_odds_ratio": 999.0,
                "total_support_n": 999,
                "resistant_downstream_test_probability": 0.9,
                "susceptible_downstream_test_probability": 0.01,
                "retained_edge": True,
            }
        ]
    ).to_parquet(validation_artifact_dir / "edge_report.parquet", index=False)

    bundle = PairFeatureMatrixBuilder(settings, paths).build(scope="combined")

    assert len(bundle.dataframe) == 1
    row = bundle.dataframe.iloc[0]
    assert float(row["train_pair_escalation_ratio"]) == 7.5
    assert math.isclose(float(row["train_pair_adjusted_odds_ratio"]), 1.8, rel_tol=0.0, abs_tol=1e-6)
    assert float(row["train_pair_total_support_n"]) == 42.0
