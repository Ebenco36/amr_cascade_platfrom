"""Threshold selection on a prespecified validation partition."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics
from amr_cascade_platform.modeling.thresholding.base import (
    EstimatorFactory,
    ThresholdSelectionResult,
)


class ValidationHoldoutThresholdSelector:
    """Preserve the legacy validation-site threshold-selection behavior."""

    name = "validation_holdout"
    uses_external_validation = True

    def __init__(self, settings: Settings, metrics: ClassificationMetrics) -> None:
        self._settings = settings
        self._metrics = metrics

    def validate_feature_columns(self, feature_columns: tuple[str, ...]) -> None:
        del feature_columns

    def select(
        self,
        *,
        predictions: pd.DataFrame,
        train_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        id_columns: tuple[str, ...],
        estimator_factory: EstimatorFactory,
        model_name: str,
        feature_set_name: str,
    ) -> ThresholdSelectionResult:
        del train_frame, feature_columns, id_columns, estimator_factory
        validation = predictions.loc[predictions["split"] == "validation"].copy()
        if validation.empty:
            return ThresholdSelectionResult(
                selected_threshold=float(self._settings.evaluation.probability_threshold),
                threshold_table=pd.DataFrame(),
                selection_predictions=validation,
                strategy=self.name,
                selection_source="validation",
            )

        selected_threshold, threshold_table = self._metrics.select_threshold(
            y_true=validation["target"],
            y_probability=validation["predicted_probability"],
            objective=self._settings.evaluation.threshold_selection_metric,
            thresholds=self._settings.evaluation.threshold_candidates,
            default_threshold=self._settings.evaluation.probability_threshold,
        )
        validation["selection_source"] = "validation"
        validation["selection_fold"] = pd.NA
        return ThresholdSelectionResult(
            selected_threshold=selected_threshold,
            threshold_table=threshold_table,
            selection_predictions=validation,
            strategy=self.name,
            selection_source="validation",
            selection_rows=len(validation),
        )
