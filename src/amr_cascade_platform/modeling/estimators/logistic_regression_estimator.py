"""Sklearn logistic-regression estimator wrapper."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.estimators.preprocessing import build_tabular_preprocessor


class LogisticRegressionEstimator:
    """Leak-safe logistic regression with one-hot encoded categorical features.

    A MaxAbsScaler is inserted between the ColumnTransformer and the model.
    It scales each feature by its maximum absolute value, which:
      - preserves sparsity (no centering, so sparse matrices stay sparse)
      - brings all numeric features into [-1, 1] so lbfgs converges reliably
      - is fit only on training data, applied to val/test — no leakage
    """

    name = "logistic_regression"

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

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._pipeline.fit(features, target)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        probabilities = self._pipeline.predict_proba(features)[:, 1]
        return pd.Series(probabilities, index=features.index, name="predicted_probability")

    def coefficients(self) -> pd.DataFrame:
        """Export the fitted model, not a second fit with a different design matrix.

        Coefficients refer to the MaxAbs-scaled design used for prediction.
        The raw-unit coefficient is also retained to make the scale explicit.
        """
        preprocessor = self._pipeline.named_steps["preprocess"]
        names = []
        if self._categorical_columns:
            categorical = preprocessor.named_transformers_["categorical"]
            retained_names = categorical.named_steps["impute"].get_feature_names_out(self._categorical_columns)
            names.extend(categorical.named_steps["encode"].get_feature_names_out(retained_names))
        if self._numeric_columns:
            imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
            names.extend(imputer.get_feature_names_out(self._numeric_columns))
        model = self._pipeline.named_steps["model"]
        coefficients = model.coef_[0]
        scales = self._pipeline.named_steps["scale"].scale_
        return pd.DataFrame({
            "feature_raw": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "feature_scale": scales,
            "raw_unit_coefficient": coefficients / scales,
            "intercept": float(model.intercept_[0]),
            "coefficient_scale": "MaxAbs-scaled fitted design",
        })

    def _build_pipeline(self) -> Pipeline:
        preprocess = build_tabular_preprocessor(
            categorical_columns=tuple(self._categorical_columns),
            numeric_columns=tuple(self._numeric_columns),
        )
        model = LogisticRegression(
            max_iter=self._settings.modeling.logistic_regression.max_iter,
            solver=self._settings.modeling.logistic_regression.solver,
            class_weight=self._settings.modeling.logistic_regression.class_weight,
            random_state=self._settings.modeling.random_state,
        )
        return Pipeline(steps=[
            ("preprocess", preprocess),
            ("scale", MaxAbsScaler()),
            ("model", model),
        ])
