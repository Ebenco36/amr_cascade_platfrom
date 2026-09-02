"""Parquet writing helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings

try:
    import dask.dataframe as dd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dd = None


class ParquetStore:
    """Persist pandas or Dask dataframes to Parquet."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def write_pandas(self, dataframe: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            shutil.rmtree(output_path)
        dataframe.to_parquet(
            output_path,
            index=self._settings.ingestion.bronze.write_index,
            compression=self._settings.parquet.compression,
            engine=self._settings.parquet.engine,
        )

    def write_dask(self, dataframe, output_path: Path) -> None:
        if dd is None:
            raise RuntimeError("Dask write requested but dask is not installed.")
        if output_path.exists() and not output_path.is_dir():
            output_path.unlink()
        output_path.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(
            output_path,
            write_index=self._settings.ingestion.bronze.write_index,
            compression=self._settings.parquet.compression,
            engine=self._settings.parquet.engine,
            overwrite=True,
        )
