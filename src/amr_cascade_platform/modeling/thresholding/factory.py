"""Factory for configurable threshold-selection strategies."""

from __future__ import annotations

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics
from amr_cascade_platform.modeling.thresholding.base import ThresholdSelector
from amr_cascade_platform.modeling.thresholding.grouped_oof import GroupedOOFThresholdSelector
from amr_cascade_platform.modeling.thresholding.validation_holdout import (
    ValidationHoldoutThresholdSelector,
)


def build_threshold_selector(
    settings: Settings,
    metrics: ClassificationMetrics,
    strategy: str | None = None,
) -> ThresholdSelector:
    """Construct the configured threshold strategy without workflow branching."""

    strategy_name = strategy or settings.evaluation.threshold_selection_strategy
    if strategy_name == ValidationHoldoutThresholdSelector.name:
        return ValidationHoldoutThresholdSelector(settings, metrics)
    if strategy_name == GroupedOOFThresholdSelector.name:
        return GroupedOOFThresholdSelector(settings, metrics)
    raise ValueError(
        "Unsupported threshold-selection strategy: "
        f"{strategy_name!r}. Expected 'validation_holdout' or 'grouped_oof'."
    )
