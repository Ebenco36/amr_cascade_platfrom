"""Type coercion and bronze-stage normalization."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class TypeCoercionService:
    """Apply conservative bronze-stage typing rules."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def normalize_pandas(self, dataframe: pd.DataFrame, source_site: str) -> pd.DataFrame:
        normalized = dataframe.copy()
        normalized.columns = [column.strip() for column in normalized.columns]
        for column in self._settings.platform.id_columns + self._settings.platform.time_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype("string")
        normalized["source_site"] = source_site
        return normalized

    def normalize_dask(self, dataframe, source_site: str):
        normalized = dataframe.rename(columns=lambda column: column.strip())
        for column in self._settings.platform.id_columns + self._settings.platform.time_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].astype("string")
        normalized["source_site"] = source_site
        return normalized
