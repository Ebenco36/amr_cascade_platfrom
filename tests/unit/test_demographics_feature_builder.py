from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.features.builders.demographics_feature_builder import DemographicsFeatureBuilder


def _settings():
    project_root = Path(__file__).resolve().parents[2]
    return ConfigLoader(project_root).load("mac")


def _culture_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
            },
            {
                "anon_id": "A2",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "20",
                "source_site": "armd",
            },
        ]
    )


def test_adi_scores_merge_into_baseline_features() -> None:
    builder = DemographicsFeatureBuilder(_settings())
    demographics = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "age": 55,
                "gender": "F",
            }
        ]
    )
    adi = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "adi_score": 87,
                "adi_state_rank": 9,
            }
        ]
    )

    def load_site_table(site: str, canonical_table: str) -> pd.DataFrame:
        if site != "armd":
            return pd.DataFrame()
        if canonical_table == "demographics":
            return demographics
        if canonical_table == "adi_scores":
            return adi
        return pd.DataFrame()

    result = builder.build(_culture_episodes(), load_site_table)
    a1 = result.loc[result["anon_id"] == "A1"].iloc[0]
    a2 = result.loc[result["anon_id"] == "A2"].iloc[0]

    assert a1["baseline_adi_available"] == 1
    assert a1["demo_adi_score"] == 87
    assert a1["demo_adi_state_rank"] == 9
    assert a2["baseline_adi_available"] == 0
    assert a2["demo_adi_score"] == 0.0
