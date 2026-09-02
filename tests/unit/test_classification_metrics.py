import math

import pandas as pd

from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics


def test_classification_metrics_returns_expected_keys() -> None:
    metrics = ClassificationMetrics().evaluate(
        y_true=pd.Series([0, 1, 0, 1]),
        y_probability=pd.Series([0.1, 0.8, 0.2, 0.7]),
        threshold=0.5,
    )
    assert set(metrics) == {
        "prevalence",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
    }
    assert metrics["accuracy"] == 1.0


def test_classification_metrics_handles_single_class_auc() -> None:
    metrics = ClassificationMetrics().evaluate(
        y_true=pd.Series([0, 0, 0]),
        y_probability=pd.Series([0.1, 0.2, 0.3]),
        threshold=0.5,
    )
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])


def test_classification_metrics_selects_threshold_on_prespecified_objective() -> None:
    selected_threshold, threshold_table = ClassificationMetrics().select_threshold(
        y_true=pd.Series([1, 0]),
        y_probability=pd.Series([0.9, 0.2]),
        objective="f1",
        thresholds=(0.05, 0.1, 0.2, 0.3, 0.5, 0.9),
        default_threshold=0.5,
    )
    assert not threshold_table.empty
    assert selected_threshold == 0.3
