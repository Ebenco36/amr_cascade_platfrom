"""Key-integrity validation."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError


class KeyIntegrityValidator:
    """Validate required identifiers are present and non-blank."""

    def __init__(self, id_columns: tuple[str, ...], exempt_tables: tuple[str, ...] = ()) -> None:
        self._id_columns = id_columns
        self._exempt_tables = set(exempt_tables)

    def validate(self, dataframe: pd.DataFrame, canonical_table: str) -> None:
        if canonical_table in self._exempt_tables:
            return
        for column in self._id_columns:
            if column not in dataframe.columns:
                raise ValidationError(
                    f"Identifier column '{column}' missing from table '{canonical_table}'."
                )
            blank_mask = dataframe[column].astype("string").str.strip().eq("")
            if bool(blank_mask.fillna(False).any()):
                raise ValidationError(
                    f"Blank identifier values found in '{column}' for table '{canonical_table}'."
                )
