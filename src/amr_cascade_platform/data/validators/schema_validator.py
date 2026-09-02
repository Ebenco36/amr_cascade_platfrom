"""Schema validation for processed tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.data.schemas.processed_schema_registry import ProcessedSchemaRegistry


class SchemaValidator:
    """Validate required columns and non-null key fields."""

    def __init__(self, schema_registry: ProcessedSchemaRegistry) -> None:
        self._schema_registry = schema_registry

    def validate(self, dataframe: pd.DataFrame, canonical_table: str) -> None:
        required_columns = self._schema_registry.required_columns_for(canonical_table)
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise ValidationError(
                f"Missing required columns for table '{canonical_table}': {', '.join(sorted(missing))}"
            )

        required_non_null = self._schema_registry.required_non_null_columns_for(canonical_table)
        null_failures = [column for column in required_non_null if dataframe[column].isna().any()]
        if null_failures:
            raise ValidationError(
                f"Null values found in required key columns for table '{canonical_table}': "
                f"{', '.join(sorted(null_failures))}"
            )
