"""Base estimator protocol for modeling workflows."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ProbabilityEstimator(Protocol):
    name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the estimator."""

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Predict positive-class probabilities."""
