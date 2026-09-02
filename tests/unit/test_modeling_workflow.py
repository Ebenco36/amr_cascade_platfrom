from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.modeling.datasets.feature_set_selector import ModelingFeatureSet
from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import ModelingDatasetBundle
from amr_cascade_platform.modeling.workflows.modeling_workflow import (
    ModelingDatasetSelection,
    ModelingRequest,
    ModelingWorkflow,
)


class _SpyEstimator:
    name = "spy_estimator"

    def __init__(self) -> None:
        self.fit_features: pd.DataFrame | None = None
        self.fit_target: pd.Series | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.fit_features = features.copy()
        self.fit_target = target.copy()

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(features["probability_hint"].astype(float), index=features.index, name="predicted_probability")


def test_modeling_workflow_fits_estimators_on_train_rows_only(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    workflow = ModelingWorkflow(settings, paths)

    dataframe = pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2024-01-01T10:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "pair_direction": "A -> B",
                "target": 1,
                "ordering_mode": "clinical",
                "culture_description": "urine",
                "was_positive": 1,
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "upstream_susceptibility": "RESISTANT",
                "upstream_positive": 1.0,
                "was_positive_numeric": 1.0,
                "downstream_intrinsic_resistance": 0.0,
                "probability_hint": 0.8,
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e1b",
                "order_proc_id_coded": "o1b",
                "order_time_jittered": "2024-01-01T12:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "pair_direction": "A -> B",
                "target": 0,
                "ordering_mode": "clinical",
                "culture_description": "urine",
                "was_positive": 1,
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "upstream_susceptibility": "SUSCEPTIBLE",
                "upstream_positive": 0.0,
                "was_positive_numeric": 1.0,
                "downstream_intrinsic_resistance": 0.0,
                "probability_hint": 0.2,
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e2",
                "order_proc_id_coded": "o2",
                "order_time_jittered": "2024-01-02T10:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd_ecuh",
                "pair_direction": "A -> B",
                "target": 1,
                "ordering_mode": "clinical",
                "culture_description": "urine",
                "was_positive": 1,
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "upstream_susceptibility": "SUSCEPTIBLE",
                "upstream_positive": 0.0,
                "was_positive_numeric": 1.0,
                "downstream_intrinsic_resistance": 0.0,
                "probability_hint": 0.9,
            },
            {
                "anon_id": "p2b",
                "pat_enc_csn_id_coded": "e2b",
                "order_proc_id_coded": "o2b",
                "order_time_jittered": "2024-01-02T12:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd_ecuh",
                "pair_direction": "A -> B",
                "target": 0,
                "ordering_mode": "clinical",
                "culture_description": "urine",
                "was_positive": 1,
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "upstream_susceptibility": "SUSCEPTIBLE",
                "upstream_positive": 0.0,
                "was_positive_numeric": 1.0,
                "downstream_intrinsic_resistance": 0.0,
                "probability_hint": 0.2,
            },
            {
                "anon_id": "p3",
                "pat_enc_csn_id_coded": "e3",
                "order_proc_id_coded": "o3",
                "order_time_jittered": "2024-01-03T10:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd_utsw",
                "pair_direction": "A -> B",
                "target": 1,
                "ordering_mode": "clinical",
                "culture_description": "urine",
                "was_positive": 1,
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "upstream_susceptibility": "RESISTANT",
                "upstream_positive": 1.0,
                "was_positive_numeric": 1.0,
                "downstream_intrinsic_resistance": 0.0,
                "probability_hint": 0.35,
            },
        ]
    )
    bundle = ModelingDatasetBundle(
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
            "probability_hint",
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
            "probability_hint",
        ),
        id_columns=(
            "anon_id",
            "pat_enc_csn_id_coded",
            "order_proc_id_coded",
            "order_time_jittered",
            "organism",
            "source_site",
            "pair_direction",
            "target",
        ),
    )
    selection = ModelingDatasetSelection(
        dataset=bundle,
        base_dataset=bundle,
        source_path=paths.paths.features / "combined" / "pair_testing_feature_matrix.parquet",
        persisted=False,
    )
    feature_set = ModelingFeatureSet(
        name="full",
        description="full",
        feature_columns=bundle.feature_columns,
        categorical_columns=bundle.categorical_columns,
        numeric_columns=bundle.numeric_columns,
    )
    estimator = _SpyEstimator()

    workflow._load_modeling_dataset = lambda request: selection  # type: ignore[method-assign]
    workflow._feature_set_selector.select = lambda **kwargs: (feature_set,)  # type: ignore[method-assign]
    workflow._build_estimators = lambda *args, **kwargs: [estimator]  # type: ignore[method-assign]

    outputs = workflow.run(ModelingRequest(scope="combined", estimators=("logistic_regression",)))

    assert "metrics" in outputs
    assert estimator.fit_features is not None
    assert estimator.fit_features["organism"].tolist() == ["ESCHERICHIA COLI", "ESCHERICHIA COLI"]
    assert estimator.fit_target is not None
    assert estimator.fit_target.tolist() == [1, 0]
    selected_thresholds = pd.read_parquet(outputs["selected_thresholds"])
    assert selected_thresholds.iloc[0]["selected_threshold"] == 0.3
    threshold_recommendations = pd.read_parquet(outputs["threshold_recommendations"])
    assert {"f1", "balanced_accuracy", "precision", "recall"}.issubset(
        set(threshold_recommendations["metric_name"])
    )
    predictions = pd.read_parquet(outputs["full__spy_estimator_predictions"])
    test_row = predictions.loc[predictions["split"] == "test"].iloc[0]
    assert test_row["decision_threshold"] == 0.3
    assert int(test_row["predicted_label"]) == 1
