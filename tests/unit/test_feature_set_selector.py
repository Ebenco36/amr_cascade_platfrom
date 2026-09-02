from pathlib import Path

import pandas as pd

from amr_cascade_platform.modeling.datasets.feature_set_selector import ModelingFeatureSetSelector
from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import ModelingDatasetBundle


def test_feature_set_selector_builds_expected_ablation_sets(tmp_path: Path) -> None:
    metadata_path = tmp_path / "feature_build_summary.json"
    metadata_path.write_text(
        """
        {
          "baseline_feature_columns": ["demo_age"],
          "acute_feature_columns": ["acute_labs_available"],
          "history_feature_columns": ["history_abx_any_30d"],
          "comorbidity_feature_columns": ["comorbidity_count"]
        }
        """,
        encoding="utf-8",
    )
    dataframe = pd.DataFrame(
        {
            "organism": ["E. COLI"],
            "ordering_mode": ["ROUTINE"],
            "culture_description": ["URINE"],
            "upstream_antibiotic": ["CEFEPIME"],
            "downstream_antibiotic": ["LEVOFLOXACIN"],
            "upstream_susceptibility": ["R"],
            "upstream_positive": [1],
            "was_positive_numeric": [1],
            "downstream_intrinsic_resistance": [0],
            "train_pair_escalation_ratio": [11.0],
            "train_upstream_pagerank": [0.1],
            "train_downstream_pagerank": [0.2],
            "demo_age": [63],
            "acute_labs_available": [1],
            "history_abx_any_30d": [1],
            "comorbidity_count": [4],
            "target": [1],
        }
    )
    base_dataset = ModelingDatasetBundle(
        dataframe=dataframe,
        feature_columns=(
            "organism",
            "ordering_mode",
            "culture_description",
            "upstream_antibiotic",
            "downstream_antibiotic",
            "upstream_susceptibility",
            "upstream_positive",
            "was_positive_numeric",
            "downstream_intrinsic_resistance",
            "train_pair_escalation_ratio",
            "train_upstream_pagerank",
            "train_downstream_pagerank",
        ),
        categorical_columns=(
            "organism",
            "ordering_mode",
            "culture_description",
            "upstream_antibiotic",
            "downstream_antibiotic",
            "upstream_susceptibility",
        ),
        numeric_columns=(
            "upstream_positive",
            "was_positive_numeric",
            "downstream_intrinsic_resistance",
            "train_pair_escalation_ratio",
            "train_upstream_pagerank",
            "train_downstream_pagerank",
        ),
        id_columns=("target",),
    )
    full_dataset = ModelingDatasetBundle(
        dataframe=dataframe,
        feature_columns=base_dataset.feature_columns
        + ("demo_age", "acute_labs_available", "history_abx_any_30d", "comorbidity_count"),
        categorical_columns=base_dataset.categorical_columns,
        numeric_columns=base_dataset.numeric_columns
        + ("demo_age", "acute_labs_available", "history_abx_any_30d", "comorbidity_count"),
        id_columns=base_dataset.id_columns,
    )

    selector = ModelingFeatureSetSelector(metadata_path)
    feature_sets = selector.select(base_dataset=base_dataset, full_dataset=full_dataset, ablation=True)

    by_name = {feature_set.name: feature_set for feature_set in feature_sets}
    assert tuple(by_name) == ("observed_only", "cascade_aware", "clinical_no_cascade", "full")
    assert "train_pair_escalation_ratio" not in by_name["observed_only"].feature_columns
    assert "train_pair_escalation_ratio" in by_name["cascade_aware"].feature_columns
    assert "demo_age" in by_name["clinical_no_cascade"].feature_columns
    assert "train_pair_escalation_ratio" not in by_name["clinical_no_cascade"].feature_columns
    assert "demo_age" in by_name["full"].feature_columns
    assert "train_pair_escalation_ratio" in by_name["full"].feature_columns
