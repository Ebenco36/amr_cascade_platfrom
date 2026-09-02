import pandas as pd
import pytest

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.features.builders.feature_merger import FeatureMerger


def test_feature_merger_preserves_row_count_and_fills_missing() -> None:
    base = pd.DataFrame(
        {
            "anon_id": ["a", "b"],
            "pat_enc_csn_id_coded": ["1", "2"],
            "order_proc_id_coded": ["11", "22"],
            "source_site": ["armd", "armd"],
            "target": [0, 1],
        }
    )
    features = pd.DataFrame(
        {
            "anon_id": ["a"],
            "pat_enc_csn_id_coded": ["1"],
            "order_proc_id_coded": ["11"],
            "source_site": ["armd"],
            "comorbidity_count": [3],
            "comorb_htn": [1],
        }
    )
    result = FeatureMerger().merge_left(
        base=base,
        features=features,
        join_keys=("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "source_site"),
    )
    assert len(result.dataframe) == 2
    assert result.matched_rows == 1
    assert result.unmatched_rows == 1
    assert result.dataframe["comorbidity_count"].tolist() == [3, 0]
    assert result.dataframe["comorb_htn"].tolist() == [1, 0]


def test_feature_merger_rejects_duplicate_feature_keys() -> None:
    base = pd.DataFrame({"anon_id": ["a"], "source_site": ["armd"]})
    features = pd.DataFrame({"anon_id": ["a", "a"], "source_site": ["armd", "armd"], "comorb_htn": [1, 0]})
    with pytest.raises(ValidationError):
        FeatureMerger().merge_left(
            base=base,
            features=features,
            join_keys=("anon_id", "source_site"),
        )
