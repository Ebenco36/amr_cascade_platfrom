"""Schema alignment for cross-site harmonization."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError


class SchemaAligner:
    """Rename, add, and order columns to match explicit harmonization contracts."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def align(self, dataframe: pd.DataFrame, canonical_table: str) -> pd.DataFrame:
        contract = self._settings.harmonization.contracts.get(canonical_table)
        if contract is None:
            raise ValidationError(f"No harmonization contract configured for table '{canonical_table}'.")

        aligned = dataframe.copy()
        for source_column, target_column in self._settings.harmonization.rename_columns.items():
            if source_column in aligned.columns:
                if target_column in aligned.columns:
                    aligned[target_column] = aligned[target_column].combine_first(aligned[source_column])
                    aligned = aligned.drop(columns=[source_column])
                else:
                    aligned = aligned.rename(columns={source_column: target_column})
        for column in contract.columns:
            if column not in aligned.columns:
                aligned[column] = pd.NA

        missing_after_alignment = [column for column in contract.columns if column not in aligned.columns]
        if missing_after_alignment:
            raise ValidationError(
                f"Failed to align required harmonized columns for '{canonical_table}': "
                f"{', '.join(sorted(missing_after_alignment))}"
            )
        return aligned.loc[:, list(contract.columns)]
