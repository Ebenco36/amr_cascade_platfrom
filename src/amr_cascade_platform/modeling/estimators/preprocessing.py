"""Shared preprocessing for tabular estimators."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def _to_float64(X):
    if hasattr(X, "astype"):
        return X.astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _replace_nonfinite(X):
    """Replace inf/-inf and extreme finite values (>1e15) with NaN.

    sklearn's liblinear (and other solvers) error or freeze when any feature
    value exceeds 1e30. Source data may contain large finite sentinels (e.g.
    9999999999) that survive np.isfinite but still trigger the check.
    """
    arr = np.asarray(X, dtype=np.float64)
    arr[~np.isfinite(arr) | (np.abs(arr) > 1e15)] = np.nan
    return arr


def build_tabular_preprocessor(
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> ColumnTransformer:
    """Create a consistent preprocessing stack for all tabular estimators."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        # Keep the categorical design sparse. Dense one-hot matrices are the
                        # main memory bottleneck on large pair-level datasets.
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                list(categorical_columns),
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("to_float", FunctionTransformer(_to_float64)),
                        ("clip_nonfinite", FunctionTransformer(_replace_nonfinite)),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ]
                ),
                list(numeric_columns),
            ),
        ],
        sparse_threshold=1.0,
    )
