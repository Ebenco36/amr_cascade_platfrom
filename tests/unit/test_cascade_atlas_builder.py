from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.cascade_atlas_builder import CascadeAtlasBuilder


def test_cascade_atlas_builder_exports_scope_metadata(tmp_path: Path) -> None:
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "resistant_support_n": 10,
                "susceptible_support_n": 20,
                "resistant_tested_n": 4,
                "susceptible_tested_n": 1,
                "resistant_downstream_test_probability": 0.4,
                "susceptible_downstream_test_probability": 0.05,
                "escalation_ratio": 8.0,
                "er_ci_lower": 1.5,
                "er_ci_upper": 40.0,
                "adjusted_odds_ratio": 1.8,
                "adjusted_log_odds": 0.59,
                "modeled_n": 30,
                "passes_support_threshold": True,
                "supports_adjusted_model": True,
                "retained_edge": True,
                "raw_effect_direction": "positive",
                "adjusted_effect_direction": "positive",
                "raw_adjusted_direction_agreement": "agree",
            }
        ]
    )
    outputs = CascadeAtlasBuilder().export(
        edge_report=edge_report,
        output_dir=tmp_path,
        scope="site",
        site="armd",
        organism="ESCHERICHIA COLI",
    )
    atlas = pd.read_parquet(outputs["cascade_atlas_path"])
    assert len(atlas) == 1
    row = atlas.iloc[0]
    assert row["scope"] == "site"
    assert row["site"] == "armd"
    assert row["organism_scope"] == "ESCHERICHIA COLI"
