"""Key normalization helpers."""

from __future__ import annotations

import pandas as pd


class KeyNormalizer:
    """Normalize identifiers and common categorical fields."""

    def __init__(
        self,
        id_columns: tuple[str, ...],
        trim_strings: bool,
        uppercase_standard_columns: tuple[str, ...],
    ) -> None:
        self._id_columns = id_columns
        self._trim_strings = trim_strings
        self._uppercase_standard_columns = uppercase_standard_columns

    def normalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = dataframe.copy()
        string_columns = normalized.select_dtypes(include=["object", "string"]).columns
        if self._trim_strings:
            for column in string_columns:
                normalized[column] = normalized[column].astype("string").str.strip()

        for column in self._id_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype("string")

        for column in self._uppercase_standard_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype("string").str.upper()

        return normalized
