"""Null normalization helpers."""

from __future__ import annotations

import pandas as pd


class NullHandler:
    """Normalize configured missing tokens to pandas NA."""

    def __init__(self, missing_tokens: tuple[str, ...]) -> None:
        self._missing_tokens = tuple(token for token in missing_tokens if token != "")

    def normalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = dataframe.copy()
        object_like = normalized.select_dtypes(include=["object", "string"]).columns
        for column in object_like:
            normalized[column] = normalized[column].replace(list(self._missing_tokens), pd.NA)
        return normalized
