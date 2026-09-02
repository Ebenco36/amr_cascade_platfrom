"""Classification metrics for downstream-testing prediction."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationMetrics:
    """Compute common discrimination and calibration metrics."""

    def evaluate(
        self,
        y_true: pd.Series,
        y_probability: pd.Series,
        threshold: float,
    ) -> dict[str, float]:
        y_true = pd.Series(y_true).astype(int)
        y_probability = pd.Series(y_probability).astype(float)
        y_pred = (y_probability >= threshold).astype(int)
        return {
            "prevalence": float(y_true.mean()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "roc_auc": self._safe_metric(roc_auc_score, y_true, y_probability),
            "pr_auc": self._safe_metric(average_precision_score, y_true, y_probability),
            "brier_score": float(brier_score_loss(y_true, y_probability)),
        }

    def threshold_sweep(
        self,
        y_true: pd.Series,
        y_probability: pd.Series,
        thresholds: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for threshold in thresholds:
            metrics = self.evaluate(y_true, y_probability, threshold)
            metrics["threshold"] = threshold
            rows.append(metrics)
        return pd.DataFrame(rows)

    def select_threshold(
        self,
        y_true: pd.Series,
        y_probability: pd.Series,
        objective: str,
        thresholds: tuple[float, ...],
        default_threshold: float,
    ) -> tuple[float, pd.DataFrame]:
        """Choose a fixed operating threshold using validation data only.

        The selected threshold is frozen before test evaluation. Ties are
        resolved deterministically in favor of stronger recall, then stronger
        precision, then lower thresholds.
        """

        threshold_table = self.threshold_sweep(y_true=y_true, y_probability=y_probability, thresholds=thresholds)
        if threshold_table.empty or objective not in threshold_table.columns:
            return default_threshold, threshold_table
        ranked = threshold_table.sort_values(
            by=[objective, "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        return float(ranked.iloc[0]["threshold"]), threshold_table

    @staticmethod
    def _safe_metric(metric_func, y_true: pd.Series, y_probability: pd.Series) -> float:
        if pd.Series(y_true).nunique() < 2:
            return math.nan
        return float(metric_func(y_true, y_probability))
