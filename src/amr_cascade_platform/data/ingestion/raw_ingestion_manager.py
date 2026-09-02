"""Bronze ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.ingestion.canonical_table_mapper import CanonicalTableMapper
from amr_cascade_platform.data.ingestion.table_ingestor import IngestionResult, TableIngestor


@dataclass(frozen=True)
class RawIngestionRequest:
    site: str
    source_layer: str
    target_layer: str = "bronze"
    tables: tuple[str, ...] | None = None
    allow_raw_on_local: bool = False


class RawIngestionManager:
    """Coordinate site ingestion from source CSVs into bronze Parquet."""

    def __init__(
        self,
        settings: Settings,
        path_manager: PathManager,
        canonical_mapper: CanonicalTableMapper,
    ) -> None:
        self._settings = settings
        self._paths = path_manager
        self._mapper = canonical_mapper
        self._ingestor = TableIngestor(settings, path_manager)
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

    def ingest_site(self, request: RawIngestionRequest) -> list[IngestionResult]:
        if (
            request.source_layer == "raw"
            and self._settings.environment.execution_mode == "local"
            and not self._settings.environment.allow_raw_processing
            and not request.allow_raw_on_local
        ):
            raise DataDiscoveryError(
                "Raw-layer ingestion is blocked in the mac/local environment. "
                "Use sampled inputs by default or pass an explicit override."
            )

        tables = self._resolve_source_tables(request.site, request.source_layer)
        if request.tables:
            wanted = set(request.tables)
            tables = {name: path for name, path in tables.items() if name in wanted}
            missing = wanted - set(tables)
            if missing:
                raise DataDiscoveryError(
                    f"Requested tables not available for site '{request.site}': {', '.join(sorted(missing))}"
                )

        output_dir = self._paths.site_dir(request.target_layer, request.site)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[IngestionResult] = []
        for canonical_table, input_path in tables.items():
            self._logger.info(
                "Ingesting %s for %s from %s into %s",
                canonical_table,
                request.site,
                request.source_layer,
                request.target_layer,
            )
            results.append(
                self._ingestor.ingest(
                    site=request.site,
                    canonical_table=canonical_table,
                    input_path=input_path,
                    output_dir=output_dir,
                )
            )
        return results

    def _resolve_source_tables(self, site: str, source_layer: str) -> dict[str, Path]:
        raw_tables = self._mapper.site_tables(site)
        if source_layer == "raw":
            return raw_tables
        if source_layer == "sample":
            sample_dir = self._paths.site_dir("sample", site)
            resolved: dict[str, Path] = {}
            missing_required: list[str] = []
            for canonical_table, raw_path in raw_tables.items():
                sample_path = sample_dir / raw_path.name
                if sample_path.exists():
                    resolved[canonical_table] = sample_path
                elif canonical_table not in self._settings.platform.optional_tables:
                    missing_required.append(canonical_table)
            if missing_required:
                raise DataDiscoveryError(
                    f"Missing sampled tables for site '{site}': {', '.join(sorted(missing_required))}. "
                    f"Create sampled inputs first with scripts/create_sample_dataset.py."
                )
            return resolved
        raise DataDiscoveryError(f"Unsupported source layer for ingestion: {source_layer}")
