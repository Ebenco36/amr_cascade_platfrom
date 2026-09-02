"""Site-level harmonized dataset creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.data.harmonization.schema_aligner import SchemaAligner
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class HarmonizedSiteTable:
    site: str
    canonical_table: str
    input_path: Path
    output_path: Path
    row_count: int
    column_count: int
    available: bool


class SiteHarmonizer:
    """Harmonize one site's silver tables into site-aligned outputs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._dataset_store = DatasetStore(settings)
        self._parquet_store = ParquetStore(settings)
        self._schema_aligner = SchemaAligner(settings)

    def harmonize_site(
        self,
        site: str,
        silver_dir: Path,
        output_dir: Path,
        tables: tuple[str, ...],
    ) -> list[HarmonizedSiteTable]:
        if not silver_dir.exists():
            raise DataDiscoveryError(f"Silver directory does not exist for site '{site}': {silver_dir}")

        artifacts: list[HarmonizedSiteTable] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for canonical_table in tables:
            input_path = silver_dir / f"{canonical_table}.parquet"
            available = input_path.exists()
            if not available and canonical_table not in self._settings.harmonization.optional_shared_tables:
                raise DataDiscoveryError(
                    f"Silver table '{canonical_table}' missing for site '{site}': {input_path}"
                )
            dataframe = (
                self._dataset_store.read_pandas(input_path)
                if available
                else pd.DataFrame()
            )
            aligned = self._schema_aligner.align(dataframe, canonical_table)
            output_path = output_dir / f"{canonical_table}.parquet"
            self._parquet_store.write_pandas(aligned, output_path)
            if available:
                self._logger.info("Harmonized %s for %s into %s", canonical_table, site, output_path)
            else:
                self._logger.info(
                    "Created empty harmonized placeholder for optional table %s at %s",
                    canonical_table,
                    output_path,
                )
            artifacts.append(
                HarmonizedSiteTable(
                    site=site,
                    canonical_table=canonical_table,
                    input_path=input_path,
                    output_path=output_path,
                    row_count=len(aligned),
                    column_count=len(aligned.columns),
                    available=available,
                )
            )
        return artifacts
