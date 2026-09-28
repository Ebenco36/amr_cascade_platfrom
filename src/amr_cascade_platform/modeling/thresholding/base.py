"""Contracts shared by decision-threshold selection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from amr_cascade_platform.modeling.estimators.base import ProbabilityEstimator


EstimatorFactory = Callable[[], ProbabilityEstimator]


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Auditable output from one threshold-selection strategy."""

    selected_threshold: float
    threshold_table: pd.DataFrame
    selection_predictions: pd.DataFrame
    strategy: str
    selection_source: str
    group_column: str | None = None
    cv_folds: int | None = None
    selection_rows: int = 0
    selection_groups: int | None = None
    # How patients were assigned to folds (grouped OOF only).
    fold_assignment: str | None = None


class ThresholdSelector(Protocol):
    """Strategy interface used by the modeling workflow."""

    name: str
    uses_external_validation: bool

    def validate_feature_columns(self, feature_columns: tuple[str, ...]) -> None:
        """Fail before model fitting when the strategy and features are incompatible."""

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
        """Select and document a fixed probability threshold."""
