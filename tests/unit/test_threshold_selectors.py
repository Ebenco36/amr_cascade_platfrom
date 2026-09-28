from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics
from amr_cascade_platform.modeling.thresholding.factory import build_threshold_selector
from amr_cascade_platform.modeling.thresholding.grouped_oof import GroupedOOFThresholdSelector
from amr_cascade_platform.modeling.thresholding.validation_holdout import (
    ValidationHoldoutThresholdSelector,
)


class _HintEstimator:
    name = "hint"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        del features, target

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(
            features["probability_hint"].astype(float).to_numpy(),
            index=features.index,
            name="predicted_probability",
        )


def _settings(*, folds: int = 5):
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    return replace(
        settings,
        evaluation=replace(
            settings.evaluation,
            threshold_cv_folds=folds,
            threshold_cv_group_column="anon_id",
        ),
    )


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_number in range(20):
        target = group_number % 2
        for pair_number in range(2):
            rows.append(
                {
                    "anon_id": f"patient-{group_number}",
                    "pair_direction": f"A -> B-{pair_number}",
                    "target": target,
                    "probability_hint": 0.8 if target else 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_oof_selector_predicts_every_row_once_without_group_leakage() -> None:
    settings = _settings()
    selector = GroupedOOFThresholdSelector(settings, ClassificationMetrics())
    frame = _training_frame()

    result = selector.select(
        predictions=pd.DataFrame(),
        train_frame=frame,
        feature_columns=("probability_hint",),
        id_columns=("anon_id", "pair_direction", "target"),
        estimator_factory=_HintEstimator,
        model_name="hint",
        feature_set_name="full",
    )

    assert result.strategy == "grouped_oof"
    assert result.selection_source == "training_oof"
    assert result.selection_rows == len(frame)
    assert result.selection_groups == frame["anon_id"].nunique()
    assert result.cv_folds == 5
    assert result.selection_predictions["selection_row_position"].nunique() == len(frame)
    assert result.selection_predictions.groupby("anon_id")["selection_fold"].nunique().max() == 1
    assert set(result.selection_predictions["selection_fold"]) == {1, 2, 3, 4, 5}
    assert result.selected_threshold in settings.evaluation.threshold_candidates


def test_grouped_oof_selector_fails_closed_when_patient_support_is_too_small() -> None:
    settings = _settings(folds=5)
    selector = GroupedOOFThresholdSelector(settings, ClassificationMetrics())
    frame = _training_frame().loc[lambda value: value["anon_id"].isin(["patient-0", "patient-1"])]

    with pytest.raises(ValidationError, match="at least 5 distinct"):
        selector.select(
            predictions=pd.DataFrame(),
            train_frame=frame,
            feature_columns=("probability_hint",),
            id_columns=("anon_id", "pair_direction", "target"),
            estimator_factory=_HintEstimator,
            model_name="hint",
            feature_set_name="full",
        )


def test_grouped_oof_selector_rejects_full_training_target_derived_features() -> None:
    settings = _settings()
    selector = GroupedOOFThresholdSelector(settings, ClassificationMetrics())

    with pytest.raises(ValidationError, match="held-out fold influenced those features"):
        selector.select(
            predictions=pd.DataFrame(),
            train_frame=_training_frame().assign(train_pair_escalation_ratio=2.0),
            feature_columns=("probability_hint", "train_pair_escalation_ratio"),
            id_columns=("anon_id", "pair_direction", "target"),
            estimator_factory=_HintEstimator,
            model_name="hint",
            feature_set_name="full",
        )


def test_validation_holdout_selector_uses_validation_rows_only() -> None:
    settings = _settings()
    selector = ValidationHoldoutThresholdSelector(settings, ClassificationMetrics())
    predictions = pd.DataFrame(
        {
            "split": ["train", "validation", "validation", "test"],
            "target": [1, 1, 0, 0],
            "predicted_probability": [0.99, 0.8, 0.2, 0.01],
        }
    )

    result = selector.select(
        predictions=predictions,
        train_frame=pd.DataFrame(),
        feature_columns=(),
        id_columns=(),
        estimator_factory=_HintEstimator,
        model_name="hint",
        feature_set_name="full",
    )

    assert result.strategy == "validation_holdout"
    assert result.selection_source == "validation"
    assert len(result.selection_predictions) == 2
    assert set(result.selection_predictions["split"]) == {"validation"}


def test_threshold_selector_factory_switches_strategies() -> None:
    settings = _settings()
    metrics = ClassificationMetrics()

    assert isinstance(
        build_threshold_selector(settings, metrics, "validation_holdout"),
        ValidationHoldoutThresholdSelector,
    )
    assert isinstance(
        build_threshold_selector(settings, metrics, "grouped_oof"),
        GroupedOOFThresholdSelector,
    )
    with pytest.raises(ValueError, match="Unsupported threshold-selection strategy"):
        build_threshold_selector(settings, metrics, "not-a-strategy")


def test_grouped_oof_folds_hold_both_classes_even_for_single_class_patients() -> None:
    # Every patient carries one outcome: the case in which scikit-learn's shuffled
    # StratifiedGroupKFold (seed 42) put one class entirely into one fold.
    settings = _settings()
    selector = GroupedOOFThresholdSelector(settings, ClassificationMetrics())
    frame = _training_frame()
    folds, assignment = selector._patient_folds(frame["target"], frame["anon_id"], 5)
    assert assignment == "stratified_group_kfold"
    for fit, holdout in folds:
        assert frame["target"].iloc[holdout].nunique() == 2
        assert not set(frame["anon_id"].iloc[fit]) & set(frame["anon_id"].iloc[holdout])


def test_grouped_oof_fallback_puts_both_classes_in_every_fold(monkeypatch) -> None:
    import numpy as np

    settings = _settings()
    selector = GroupedOOFThresholdSelector(settings, ClassificationMetrics())
    frame = _training_frame()

    class _Unstratified:
        def __init__(self, n_splits, shuffle):
            self._n = n_splits

        def split(self, features, target, groups):
            positions = np.arange(len(target))
            order = np.argsort(target, kind="stable")  # one class per fold: the failure being guarded against
            for chunk in np.array_split(order, self._n):
                yield np.setdiff1d(positions, chunk), chunk

    monkeypatch.setattr("amr_cascade_platform.modeling.thresholding.grouped_oof.StratifiedGroupKFold", _Unstratified)
    folds, assignment = selector._patient_folds(frame["target"], frame["anon_id"], 5)
    assert assignment == "rate_ranked_round_robin"
    covered = np.concatenate([holdout for _, holdout in folds])
    assert sorted(covered.tolist()) == list(range(len(frame)))
    for fit, holdout in folds:
        assert frame["target"].iloc[holdout].nunique() == 2
        assert not set(frame["anon_id"].iloc[fit]) & set(frame["anon_id"].iloc[holdout])


def test_threshold_selection_refuses_an_unknown_objective() -> None:
    with pytest.raises(ValueError, match="Unknown threshold_selection_metric"):
        ClassificationMetrics().select_threshold(
            y_true=pd.Series([0, 1, 0, 1]),
            y_probability=pd.Series([0.1, 0.9, 0.2, 0.8]),
            objective="f1_scor",
            thresholds=(0.3, 0.5),
            default_threshold=0.5,
        )


def test_balanced_accuracy_does_not_collapse_to_predicting_everything_positive() -> None:
    # 70% positives. At 0.5 the model has recall 0.7 and specificity 0.9 (balanced
    # accuracy 0.80), yet its F1 (0.803) is below that of calling every row positive
    # (2p/(1+p) = 0.824): F1 picks the degenerate cutoff, balanced accuracy does not.
    y = pd.Series([1] * 70 + [0] * 30)
    p = pd.Series([0.6] * 49 + [0.4] * 21 + [0.6] * 3 + [0.4] * 27)
    metrics = ClassificationMetrics()
    grid = (0.05, 0.5, 0.65)
    f1_choice, _ = metrics.select_threshold(y, p, objective="f1", thresholds=grid, default_threshold=0.5)
    ba_choice, _ = metrics.select_threshold(y, p, objective="balanced_accuracy", thresholds=grid, default_threshold=0.5)
    assert f1_choice == 0.05
    assert ba_choice == 0.5
