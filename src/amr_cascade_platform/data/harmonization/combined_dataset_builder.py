"""Build combined harmonized datasets across sites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class CombinedDatasetArtifact:
    canonical_table: str
    input_paths: tuple[Path, ...]
    output_path: Path
    row_count: int
    column_count: int
    contributing_sites: tuple[str, ...]


class CombinedDatasetBuilder:
    """Concatenate aligned site tables into cross-site combined datasets."""

    def __init__(self, dataset_store: DatasetStore, parquet_store: ParquetStore) -> None:
        self._dataset_store = dataset_store
        self._parquet_store = parquet_store
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

    def build(
        self,
        canonical_table: str,
        aligned_paths: tuple[Path, ...],
        output_path: Path,
    ) -> CombinedDatasetArtifact:
        frames = [self._dataset_store.read_pandas(path) for path in aligned_paths]
        non_empty_frames = [frame for frame in frames if not frame.empty]
        if non_empty_frames:
            combined = pd.concat(non_empty_frames, axis=0, ignore_index=True)
        else:
            combined = frames[0].copy() if frames else pd.DataFrame()
        contributing_sites = tuple(sorted(set(combined["source_site"].dropna().astype(str)))) if "source_site" in combined.columns else tuple()
        self._parquet_store.write_pandas(combined, output_path)
        self._logger.info("Built combined harmonized table %s into %s", canonical_table, output_path)
        return CombinedDatasetArtifact(
            canonical_table=canonical_table,
            input_paths=aligned_paths,
            output_path=output_path,
            row_count=len(combined),
            column_count=len(combined.columns),
            contributing_sites=contributing_sites,
        )
