import json
from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
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


def test_modeling_workflow_refuses_to_silently_downgrade_full_model(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    workflow = ModelingWorkflow(settings, paths)

    with pytest.raises(DataDiscoveryError, match="requires the enriched model-ready feature matrix"):
        workflow._load_modeling_dataset(  # noqa: SLF001
            ModelingRequest(
                scope="combined",
                organism="ESCHERICHIA COLI",
                feature_sets=("full",),
            )
        )


def test_modeling_workflow_uses_organism_scoped_metadata_and_preserves_feature_types(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    workflow = ModelingWorkflow(settings, paths)
    organism = "ESCHERICHIA COLI"
    features_dir = scoped_output_dir(paths.paths.features, "combined", organism=organism)
    features_dir.mkdir(parents=True, exist_ok=True)

    layout = workflow._feature_builder.feature_layout()  # noqa: SLF001
    row: dict[str, object] = {
        column: 0.0 for column in layout["numeric_columns"]
    }
    row.update({column: "value" for column in layout["categorical_columns"]})
    row.update(
        {
            "demo_gender_category": "female",
            "comorb_diabetes": 1,
        }
    )
    pd.DataFrame([row]).to_parquet(
        features_dir / settings.feature_build.combined_output_filename,
        index=False,
    )
    (features_dir / settings.feature_build.metadata_filename).write_text(
        json.dumps(
            {
                "baseline_feature_columns": ["demo_gender_category"],
                "acute_feature_columns": [],
                "history_feature_columns": [],
                "comorbidity_feature_columns": ["comorb_diabetes"],
            }
        ),
        encoding="utf-8",
    )

    selection = workflow._load_modeling_dataset(  # noqa: SLF001
        ModelingRequest(scope="combined", organism=organism)
    )
    feature_set = workflow._feature_set_selector.select(  # noqa: SLF001
        base_dataset=selection.base_dataset,
        full_dataset=selection.dataset,
        requested_sets=("full",),
    )[0]

    assert workflow._feature_set_selector._metadata_path == (  # noqa: SLF001
        features_dir / settings.feature_build.metadata_filename
    )
    assert "demo_gender_category" in feature_set.categorical_columns
    assert "comorb_diabetes" in feature_set.numeric_columns
    assert {"demo_gender_category", "comorb_diabetes"}.issubset(feature_set.feature_columns)


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
    # Validation rows (0.9 positive, 0.2 negative) are separated by every candidate
    # above 0.2; ties go to the lowest threshold, 0.25 on the 0.05-step grid.
    assert selected_thresholds.iloc[0]["selected_threshold"] == 0.25
    assert selected_thresholds.iloc[0]["selection_metric"] == "balanced_accuracy"
    assert not selected_thresholds.iloc[0]["at_grid_boundary"]
    threshold_recommendations = pd.read_parquet(outputs["threshold_recommendations"])
    assert {"f1", "balanced_accuracy", "precision", "recall"}.issubset(
        set(threshold_recommendations["metric_name"])
    )
    predictions = pd.read_parquet(outputs["full__spy_estimator_predictions"])
    test_row = predictions.loc[predictions["split"] == "test"].iloc[0]
    assert test_row["decision_threshold"] == 0.25
    assert int(test_row["predicted_label"]) == 1
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert "best_test_model" not in summary
    assert summary["modeling_protocol"]["primary_estimator"] == "logistic_regression"
    assert summary["modeling_protocol"]["split_roles"]["test"]["role"] == (
        "sealed_final_evaluation_not_model_or_threshold_selection"
    )
    assert summary["test_performance_leader_descriptive"]["role"] == (
        "descriptive_only_not_model_selection"
    )


def test_restricted_estimator_run_does_not_replace_canonical_outputs() -> None:
    feature_set = ModelingFeatureSet(
        name="full",
        description="full",
        feature_columns=("x",),
        categorical_columns=(),
        numeric_columns=("x",),
    )

    assert ModelingWorkflow._should_publish_to_task_root(  # noqa: SLF001
        ModelingRequest(feature_sets=("full",)),
        (feature_set,),
    )
    assert not ModelingWorkflow._should_publish_to_task_root(  # noqa: SLF001
        ModelingRequest(feature_sets=("full",), estimators=("logistic_regression",)),
        (feature_set,),
    )


def test_grouped_threshold_runs_use_a_distinct_artifact_label() -> None:
    feature_set = ModelingFeatureSet(
        name="clinical_no_cascade",
        description="clinical",
        feature_columns=("x",),
        categorical_columns=(),
        numeric_columns=("x",),
    )

    legacy = ModelingWorkflow._run_label(  # noqa: SLF001
        ModelingRequest(estimators=("logistic_regression",)),
        (feature_set,),
        "site",
        "validation_holdout",
    )
    grouped = ModelingWorkflow._run_label(  # noqa: SLF001
        ModelingRequest(estimators=("logistic_regression",)),
        (feature_set,),
        "site",
        "grouped_oof",
    )

    assert legacy == "site__clinical_no_cascade__logistic_regression"
    assert grouped == "site__clinical_no_cascade__logistic_regression__threshold_grouped_oof"


def test_grouped_oof_rejects_cascade_feature_sets_before_any_model_is_fitted(tmp_path: Path) -> None:
    from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError

    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    workflow = ModelingWorkflow(settings, PathManager(tmp_path, settings))
    bundle = ModelingDatasetBundle(
        dataframe=pd.DataFrame({"anon_id": ["p1"], "target": [1], "x": [0.5], "train_pair_rate": [0.2]}),
        feature_columns=("x", "train_pair_rate"),
        categorical_columns=(),
        numeric_columns=("x", "train_pair_rate"),
        id_columns=("anon_id", "target"),
    )
    selection = ModelingDatasetSelection(
        dataset=bundle, base_dataset=bundle, source_path=tmp_path / "unused.parquet", persisted=False
    )
    leak_safe = ModelingFeatureSet(
        name="clinical_no_cascade", description="clinical", feature_columns=("x",),
        categorical_columns=(), numeric_columns=("x",),
    )
    contaminated = ModelingFeatureSet(
        name="full", description="full", feature_columns=("x", "train_pair_rate"),
        categorical_columns=(), numeric_columns=("x", "train_pair_rate"),
    )
    estimator = _SpyEstimator()
    workflow._load_modeling_dataset = lambda request: selection  # type: ignore[method-assign]
    workflow._feature_set_selector.select = lambda **kwargs: (leak_safe, contaminated)  # type: ignore[method-assign]
    workflow._build_estimators = lambda *args, **kwargs: [estimator]  # type: ignore[method-assign]

    with pytest.raises(ValidationError, match="train_pair_rate"):
        workflow.run(ModelingRequest(scope="combined", ablation=True, threshold_selection_strategy="grouped_oof"))
    # The leak-safe feature set listed first must not have been trained before the refusal.
    assert estimator.fit_features is None
