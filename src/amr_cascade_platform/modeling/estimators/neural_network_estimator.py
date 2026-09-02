"""Simple neural-network estimator wrapper."""

from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPClassifier

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.modeling.estimators.preprocessing import build_tabular_preprocessor


class NeuralNetworkEstimator:
    """Small multilayer perceptron baseline for tabular prediction."""

    name = "neural_network"

    def __init__(
        self,
        settings: Settings,
        categorical_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
    ) -> None:
        if settings.modeling.neural_network is None:
            raise ValueError(
                "NeuralNetworkEstimator requires a 'modeling.neural_network' config section, "
                "but it is absent (neural_network is not in modeling.estimators). "
                "Add the neural_network block to configs/base/modeling.yaml or remove "
                "'neural_network' from the estimators list."
            )
        self._settings = settings
        self._preprocess = build_tabular_preprocessor(categorical_columns, numeric_columns)
        self._model = MLPClassifier(
            hidden_layer_sizes=self._settings.modeling.neural_network.hidden_layer_sizes,
            max_iter=self._settings.modeling.neural_network.max_iter,
            alpha=self._settings.modeling.neural_network.alpha,
            learning_rate_init=self._settings.modeling.neural_network.learning_rate_init,
            early_stopping=self._settings.modeling.neural_network.early_stopping,
            batch_size=self._settings.modeling.neural_network.batch_size,
            random_state=self._settings.modeling.random_state,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        transformed = self._preprocess.fit_transform(features)
        self._model.fit(transformed, target)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        probabilities = self._model.predict_proba(self._preprocess.transform(features))[:, 1]
        return pd.Series(probabilities, index=features.index, name="predicted_probability")
