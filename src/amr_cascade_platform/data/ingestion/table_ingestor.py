"""CSV-to-Parquet bronze ingestion."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.hashing import md5_file
from amr_cascade_platform.data.ingestion.type_coercion_service import TypeCoercionService
from amr_cascade_platform.data.sampling.sampling_policy import FileSizeSamplingPolicy
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore

try:
    import dask.dataframe as dd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dd = None


@dataclass(frozen=True)
class IngestionResult:
    site: str
    canonical_table: str
    input_path: Path
    output_path: Path
    row_count: int
    column_count: int
    backend: str
    checksum_md5: str


class TableIngestor:
    """Ingest one canonical source table into the bronze layer."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._type_coercion = TypeCoercionService(settings)
        self._parquet_store = ParquetStore(settings)
        self._policy = FileSizeSamplingPolicy(settings.ingestion.csv_read.use_dask_threshold_mb)

    def ingest(self, site: str, canonical_table: str, input_path: Path, output_dir: Path) -> IngestionResult:
        decision = self._policy.decide(input_path, dask_available=dd is not None)
        checksum = md5_file(input_path)
        output_path = output_dir / canonical_table
        self._clear_stale_alternate_format(output_path, use_dask=decision.use_dask)
        if decision.use_dask:
            result = self._ingest_with_dask(site, canonical_table, input_path, output_path, checksum)
        else:
            result = self._ingest_with_pandas(site, canonical_table, input_path, output_path, checksum)
        self._write_metadata(result)
        return result

    def _clear_stale_alternate_format(self, output_path: Path, *, use_dask: bool) -> None:
        """Remove whatever the *other* backend would have written for this table.

        FileSizeSamplingPolicy picks pandas (single `<table>.parquet` file) or dask
        (a `<table>/` directory of partitions) based on the CURRENT input size, so
        the same canonical table can flip backends between runs as the sample
        fraction changes. Downstream readers resolve a table name by scanning the
        directory and keying a dict by stem/name -- if both a stale file and a
        fresh directory exist for the same table, sort order silently decides which
        one wins (a short name like "x" sorts before "x.parquet", so the file was
        clobbering the directory here), with no error either way. Deleting the
        other format before writing keeps exactly one representation on disk per
        table, which removes the ambiguity instead of relying on a reader to
        resolve it correctly.
        """
        if use_dask:
            stale_file = output_path.with_suffix(".parquet")
            if stale_file.is_file():
                stale_file.unlink()
                self._logger.info("Removed stale single-file bronze artifact %s (this table now uses dask)", stale_file)
        else:
            stale_dir = output_path
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
                self._logger.info("Removed stale dask-partitioned bronze artifact %s (this table now uses pandas)", stale_dir)

    def _ingest_with_pandas(
        self,
        site: str,
        canonical_table: str,
        input_path: Path,
        output_path: Path,
        checksum: str,
    ) -> IngestionResult:
        dataframe = pd.read_csv(
            input_path,
            dtype="string",
            low_memory=False,
            na_values=list(self._settings.platform.missing_tokens),
            encoding=self._settings.ingestion.csv_read.encoding,
        )
        normalized = self._type_coercion.normalize_pandas(dataframe, source_site=site)
        parquet_path = output_path.with_suffix(".parquet")
        self._parquet_store.write_pandas(normalized, parquet_path)
        self._logger.info("Bronze wrote %s via pandas", parquet_path)
        return IngestionResult(
            site=site,
            canonical_table=canonical_table,
            input_path=input_path,
            output_path=parquet_path,
            row_count=len(normalized),
            column_count=len(normalized.columns),
            backend="pandas",
            checksum_md5=checksum,
        )

    def _ingest_with_dask(
        self,
        site: str,
        canonical_table: str,
        input_path: Path,
        output_path: Path,
        checksum: str,
    ) -> IngestionResult:
        if dd is None:
            raise RuntimeError("Dask ingestion requested but dask is not installed.")
        dataframe = dd.read_csv(
            input_path,
            dtype=str,
            assume_missing=True,
            blocksize=self._settings.ingestion.csv_read.dask_blocksize,
            na_values=list(self._settings.platform.missing_tokens),
            encoding=self._settings.ingestion.csv_read.encoding,
        )
        normalized = self._type_coercion.normalize_dask(dataframe, source_site=site)
        self._parquet_store.write_dask(normalized, output_path)
        row_count = int(normalized.map_partitions(len).sum().compute())
        column_count = len(normalized.columns)
        self._logger.info("Bronze wrote %s via dask", output_path)
        return IngestionResult(
            site=site,
            canonical_table=canonical_table,
            input_path=input_path,
            output_path=output_path,
            row_count=row_count,
            column_count=column_count,
            backend="dask",
            checksum_md5=checksum,
        )

    def _write_metadata(self, result: IngestionResult) -> None:
        metadata_dir = (
            self._paths.paths.metadata
            / "datasets"
            / self._settings.ingestion.bronze.metadata_subdir
            / result.site
        )
        metadata_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "site": result.site,
            "canonical_table": result.canonical_table,
            "input_path": str(result.input_path),
            "output_path": str(result.output_path),
            "row_count": result.row_count,
            "column_count": result.column_count,
            "backend": result.backend,
            "checksum_md5": result.checksum_md5,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        metadata_path = metadata_dir / f"{result.canonical_table}.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
