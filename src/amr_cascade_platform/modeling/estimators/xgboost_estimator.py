"""XGBoost estimator wrapper."""

from __future__ import annotations

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.estimators.preprocessing import build_tabular_preprocessor


class XGBoostEstimator:
    """Gradient-boosted tree estimator for sparse, interaction-heavy tabular data."""

    name = "xgboost"

    def __init__(
        self,
        settings: Settings,
        categorical_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
    ) -> None:
        self._settings = settings
        self._preprocess = build_tabular_preprocessor(categorical_columns, numeric_columns)
        self._model: XGBClassifier | None = None
        self._calibrator: IsotonicRegression | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        transformed = self._preprocess.fit_transform(features)
        positives = int(pd.Series(target).sum())
        negatives = int(len(target) - positives)
        scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0
        self._model = XGBClassifier(
            n_estimators=self._settings.modeling.xgboost.n_estimators,
            max_depth=self._settings.modeling.xgboost.max_depth,
            learning_rate=self._settings.modeling.xgboost.learning_rate,
            subsample=self._settings.modeling.xgboost.subsample,
            colsample_bytree=self._settings.modeling.xgboost.colsample_bytree,
            random_state=self._settings.modeling.random_state,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=4,
        )
        self._model.fit(transformed, target)

    def calibrate(self, val_features: pd.DataFrame, val_target: pd.Series) -> None:
        """Fit isotonic regression calibrator on held-out validation data.

        XGBoost probabilities can be over-confident, especially for minority
        classes with class imbalance compensation (scale_pos_weight).  Isotonic
        calibration is fitted once on the validation split (never on training
        data) and applied transparently during predict_proba at inference time.
        """
        if self._model is None:
            raise RuntimeError("XGBoostEstimator.calibrate called before fit().")
        transformed = self._preprocess.transform(val_features)
        raw_probs = self._model.predict_proba(transformed)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, val_target.to_numpy())

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("XGBoostEstimator.predict_proba called before fit().")
        transformed = self._preprocess.transform(features)
        probabilities = self._model.predict_proba(transformed)[:, 1]
        if self._calibrator is not None:
            probabilities = self._calibrator.predict(probabilities)
        return pd.Series(probabilities, index=features.index, name="predicted_probability")
