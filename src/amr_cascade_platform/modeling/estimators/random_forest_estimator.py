"""Sklearn random-forest estimator wrapper."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.estimators.preprocessing import build_tabular_preprocessor


class RandomForestEstimator:
    """Random forest baseline for pair-level downstream testing."""

    name = "random_forest"

    def __init__(
        self,
        settings: Settings,
        categorical_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
    ) -> None:
        self._settings = settings
        self._categorical_columns = list(categorical_columns)
        self._numeric_columns = list(numeric_columns)
        self._pipeline = self._build_pipeline()
        self._calibrator: IsotonicRegression | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._pipeline.fit(features, target)

    def calibrate(self, val_features: pd.DataFrame, val_target: pd.Series) -> None:
        """Fit isotonic regression calibrator on held-out validation data.

        Random forests are known to produce over-confident probability estimates,
        particularly at the high end of the predicted range.  Isotonic regression
        post-hoc calibration corrects this using a held-out validation set — the
        calibrator is fit *after* the forest is trained and *never* on training
        data, so there is no leakage into test evaluation.
        """
        raw_probs = self._pipeline.predict_proba(val_features)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, val_target.to_numpy())

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        probabilities = self._pipeline.predict_proba(features)[:, 1]
        if self._calibrator is not None:
            probabilities = self._calibrator.predict(probabilities)
        return pd.Series(probabilities, index=features.index, name="predicted_probability")

    def _build_pipeline(self) -> Pipeline:
        preprocess = build_tabular_preprocessor(
            categorical_columns=tuple(self._categorical_columns),
            numeric_columns=tuple(self._numeric_columns),
        )
        model = RandomForestClassifier(
            n_estimators=self._settings.modeling.random_forest.n_estimators,
            max_depth=self._settings.modeling.random_forest.max_depth,
            min_samples_leaf=self._settings.modeling.random_forest.min_samples_leaf,
            class_weight=self._settings.modeling.random_forest.class_weight,
            random_state=self._settings.modeling.random_state,
            n_jobs=-1,
        )
        return Pipeline(steps=[("preprocess", preprocess), ("model", model)])
