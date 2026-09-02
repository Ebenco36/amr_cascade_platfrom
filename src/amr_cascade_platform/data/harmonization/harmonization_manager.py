"""Orchestrate cross-site harmonization outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.harmonization.combined_dataset_builder import (
    CombinedDatasetArtifact,
    CombinedDatasetBuilder,
)
from amr_cascade_platform.data.harmonization.site_harmonizer import (
    HarmonizedSiteTable,
    SiteHarmonizer,
)
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class HarmonizationRequest:
    sites: tuple[str, ...] | None = None
    tables: tuple[str, ...] | None = None
    source_layer: str = "silver"


@dataclass(frozen=True)
class HarmonizationRunResult:
    site_artifacts: tuple[HarmonizedSiteTable, ...]
    combined_artifacts: tuple[CombinedDatasetArtifact, ...]


class HarmonizationManager:
    """Create site-aligned and combined harmonized datasets."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._site_harmonizer = SiteHarmonizer(settings)
        self._combined_builder = CombinedDatasetBuilder(
            dataset_store=DatasetStore(settings),
            parquet_store=ParquetStore(settings),
        )

    def run(self, request: HarmonizationRequest) -> HarmonizationRunResult:
        sites = request.sites or self._settings.platform.sites
        tables = request.tables or self._settings.harmonization.shared_tables

        site_artifacts: list[HarmonizedSiteTable] = []
        for site in sites:
            silver_dir = self._paths.site_dir(request.source_layer, site)
            site_output_dir = self._paths.paths.harmonized / "site_aligned" / site
            artifacts = self._site_harmonizer.harmonize_site(site, silver_dir, site_output_dir, tables)
            site_artifacts.extend(artifacts)

        combined_output_dir = self._paths.paths.harmonized / "combined"
        combined_output_dir.mkdir(parents=True, exist_ok=True)
        combined_artifacts: list[CombinedDatasetArtifact] = []
        for table in tables:
            aligned_paths = tuple(
                self._paths.paths.harmonized / "site_aligned" / site / f"{table}.parquet"
                for site in sites
            )
            output_path = combined_output_dir / f"{table}.parquet"
            artifact = self._combined_builder.build(table, aligned_paths, output_path)
            combined_artifacts.append(artifact)

        self._write_metadata(site_artifacts, combined_artifacts)
        return HarmonizationRunResult(
            site_artifacts=tuple(site_artifacts),
            combined_artifacts=tuple(combined_artifacts),
        )

    def _write_metadata(
        self,
        site_artifacts: list[HarmonizedSiteTable],
        combined_artifacts: list[CombinedDatasetArtifact],
    ) -> None:
        metadata_dir = self._paths.paths.metadata / "datasets" / "harmonized"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "site_artifacts": [
                {
                    "site": artifact.site,
                    "canonical_table": artifact.canonical_table,
                    "input_path": str(artifact.input_path),
                    "output_path": str(artifact.output_path),
                    "row_count": artifact.row_count,
                    "column_count": artifact.column_count,
                    "available": artifact.available,
                }
                for artifact in site_artifacts
            ],
            "combined_artifacts": [
                {
                    "canonical_table": artifact.canonical_table,
                    "input_paths": [str(path) for path in artifact.input_paths],
                    "output_path": str(artifact.output_path),
                    "row_count": artifact.row_count,
                    "column_count": artifact.column_count,
                    "contributing_sites": list(artifact.contributing_sites),
                }
                for artifact in combined_artifacts
            ],
        }
        metadata_path = metadata_dir / "harmonization_run.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
