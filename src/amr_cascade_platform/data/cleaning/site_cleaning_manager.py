"""Bronze-to-silver cleaning orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.cleaning.table_cleaner import CleaningResult, TableCleaner
from amr_cascade_platform.data.schemas.processed_schema_registry import ProcessedSchemaRegistry
from amr_cascade_platform.data.validators.key_integrity_validator import KeyIntegrityValidator
from amr_cascade_platform.data.validators.schema_validator import SchemaValidator
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore

try:
    import dask.dataframe as dd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dd = None


@dataclass(frozen=True)
class SilverCleaningRequest:
    site: str
    source_layer: str = "bronze"
    target_layer: str = "silver"
    tables: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SilverCleaningArtifact:
    site: str
    canonical_table: str
    input_path: Path
    output_path: Path
    rows_before: int
    rows_after: int
    duplicates_removed: int
    discordant_susceptibility_group_n: int = 0
    discordant_susceptibility_row_n: int = 0
    shard_coalesced_group_n: int = 0
    shard_conflict_group_n: int = 0
    shard_conflict_cell_n: int = 0


class SiteCleaningManager:
    """Read bronze datasets, clean them, validate them, and write silver parquet."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._dataset_store = DatasetStore(settings)
        self._parquet_store = ParquetStore(settings)
        self._table_cleaner = TableCleaner(settings)
        schema_registry = ProcessedSchemaRegistry(settings)
        self._schema_validator = SchemaValidator(schema_registry)
        self._key_validator = KeyIntegrityValidator(
            settings.platform.id_columns,
            exempt_tables=settings.schema.default_exempt_tables,
        )

    def clean_site(self, request: SilverCleaningRequest) -> list[SilverCleaningArtifact]:
        source_dir = self._paths.site_dir(request.source_layer, request.site)
        if not source_dir.exists():
            raise DataDiscoveryError(f"Source layer does not exist: {source_dir}")

        dataset_paths = self._resolve_dataset_paths(source_dir, request.tables)
        output_dir = self._paths.site_dir(request.target_layer, request.site)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[SilverCleaningArtifact] = []
        for canonical_table, input_path in dataset_paths.items():
            output_path = output_dir / f"{canonical_table}.parquet"
            if input_path.is_dir() and dd is not None:
                artifact = self._clean_large_table_with_dask(
                    site=request.site,
                    canonical_table=canonical_table,
                    input_path=input_path,
                    output_path=output_path,
                )
            else:
                dataframe = self._dataset_store.read_pandas(input_path)
                cleaning = self._table_cleaner.clean(dataframe, canonical_table)
                self._schema_validator.validate(cleaning.dataframe, canonical_table)
                self._key_validator.validate(cleaning.dataframe, canonical_table)
                self._parquet_store.write_pandas(cleaning.dataframe, output_path)
                artifact = SilverCleaningArtifact(
                    site=request.site,
                    canonical_table=canonical_table,
                    input_path=input_path,
                    output_path=output_path,
                    rows_before=cleaning.rows_before,
                    rows_after=cleaning.rows_after,
                    duplicates_removed=cleaning.duplicates_removed,
                    discordant_susceptibility_group_n=cleaning.discordant_susceptibility_group_n,
                    discordant_susceptibility_row_n=cleaning.discordant_susceptibility_row_n,
                    shard_coalesced_group_n=cleaning.shard_coalesced_group_n,
                    shard_conflict_group_n=cleaning.shard_conflict_group_n,
                    shard_conflict_cell_n=cleaning.shard_conflict_cell_n,
                )
            self._write_metadata(artifact)
            artifacts.append(artifact)
        return artifacts

    # Tables whose silver-layer drop_duplicates is too memory-heavy for local Mac
    # runs. Their deduplication happens in the downstream aggregator instead, so
    # skipping it at the silver layer is analytically safe: bronze still holds
    # every row, and the aggregator deduplicates before producing features.
    _DEDUP_SKIP_FOR_DASK = {"comorbidity"}

    def _clean_large_table_with_dask(
        self,
        *,
        site: str,
        canonical_table: str,
        input_path: Path,
        output_path: Path,
    ) -> SilverCleaningArtifact:
        dataframe = self._dataset_store.read_dask(input_path)
        rule = self._settings.cleaning.duplicate_resolution.per_table.get(canonical_table)
        if rule and rule.strategy == "shard_coalesce":
            self._logger.info(
                "Materializing %s for audited shard coalescing; correctness is preferred over partition-local deduplication.",
                canonical_table,
            )
            pandas_frame = dataframe.compute()
            cleaning = self._table_cleaner.clean(pandas_frame, canonical_table)
            self._schema_validator.validate(cleaning.dataframe, canonical_table)
            self._key_validator.validate(cleaning.dataframe, canonical_table)
            self._parquet_store.write_pandas(cleaning.dataframe, output_path)
            return SilverCleaningArtifact(
                site=site,
                canonical_table=canonical_table,
                input_path=input_path,
                output_path=output_path,
                rows_before=cleaning.rows_before,
                rows_after=cleaning.rows_after,
                duplicates_removed=cleaning.duplicates_removed,
                discordant_susceptibility_group_n=cleaning.discordant_susceptibility_group_n,
                discordant_susceptibility_row_n=cleaning.discordant_susceptibility_row_n,
                shard_coalesced_group_n=cleaning.shard_coalesced_group_n,
                shard_conflict_group_n=cleaning.shard_conflict_group_n,
                shard_conflict_cell_n=cleaning.shard_conflict_cell_n,
            )
        rows_before = int(dataframe.map_partitions(len).sum().compute())
        meta = self._table_cleaner.clean_partition(dataframe._meta_nonempty.copy(), canonical_table).iloc[0:0]
        cleaned = dataframe.map_partitions(
            self._table_cleaner.clean_partition,
            canonical_table=canonical_table,
            meta=meta,
        )
        rule = self._settings.cleaning.duplicate_resolution.per_table.get(canonical_table)
        subset = None
        if rule:
            available_subset = [column for column in rule.subset if column in cleaned.columns]
            subset = available_subset or None
        skip_dedup = canonical_table in self._DEDUP_SKIP_FOR_DASK
        if skip_dedup:
            self._logger.info(
                "Skipping global drop_duplicates for %s at silver layer "
                "(memory-heavy on small machines; downstream aggregator deduplicates).",
                canonical_table,
            )
            discordant_group_n, discordant_row_n = 0, 0
        else:
            discordant_group_n, discordant_row_n = self._compute_dask_discordant_susceptibility_audit(
                cleaned,
                canonical_table,
                subset,
            )
            cleaned = cleaned.drop_duplicates(subset=subset)
        self._validate_dask_dataframe(cleaned, canonical_table)
        self._parquet_store.write_dask(cleaned, output_path)
        rows_after = int(cleaned.map_partitions(len).sum().compute())
        duplicates_removed = rows_before - rows_after
        self._logger.info(
            "Cleaned %s via dask | rows_before=%s rows_after=%s duplicates_removed=%s discordant_susceptibility_groups=%s discordant_susceptibility_rows=%s",
            canonical_table,
            rows_before,
            rows_after,
            duplicates_removed,
            discordant_group_n,
            discordant_row_n,
        )
        return SilverCleaningArtifact(
            site=site,
            canonical_table=canonical_table,
            input_path=input_path,
            output_path=output_path,
            rows_before=rows_before,
            rows_after=rows_after,
            duplicates_removed=duplicates_removed,
            discordant_susceptibility_group_n=discordant_group_n,
            discordant_susceptibility_row_n=discordant_row_n,
        )

    def _validate_dask_dataframe(self, dataframe, canonical_table: str) -> None:
        required_columns = self._schema_validator._schema_registry.required_columns_for(canonical_table)
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise DataDiscoveryError(
                f"Missing required columns for table '{canonical_table}': {', '.join(sorted(missing))}"
            )

        required_non_null = self._schema_validator._schema_registry.required_non_null_columns_for(canonical_table)
        for column in required_non_null:
            if bool(dataframe[column].isna().any().compute()):
                raise DataDiscoveryError(
                    f"Null values found in required key columns for table '{canonical_table}': {column}"
                )

        if canonical_table not in self._key_validator._exempt_tables:
            for column in self._key_validator._id_columns:
                if column not in dataframe.columns:
                    raise DataDiscoveryError(
                        f"Identifier column '{column}' missing from table '{canonical_table}'."
                    )
                blank_mask = dataframe[column].astype("string").str.strip().eq("")
                if bool(blank_mask.fillna(False).any().compute()):
                    raise DataDiscoveryError(
                        f"Blank identifier values found in '{column}' for table '{canonical_table}'."
                    )

    def _resolve_dataset_paths(
        self,
        source_dir: Path,
        tables: tuple[str, ...] | None,
    ) -> dict[str, Path]:
        # A table can be written as either a single `<name>.parquet` file (pandas
        # backend) or a `<name>/` directory of partitions (dask backend), decided
        # per-run by input size (see TableIngestor). Both map to the same table
        # name, so if a stale artifact from a prior run in the other format is
        # still on disk (TableIngestor now cleans this up going forward, but
        # pre-existing conflicts can still be present), naively keying a dict by
        # name/stem lets whichever one is visited last in iteration order win
        # silently -- and since "x" sorts before "x.parquet", that was the file
        # clobbering the directory. Break ties by mtime instead so the artifact
        # that was actually written most recently wins, and say so out loud
        # rather than resolving it invisibly.
        candidates: dict[str, list[Path]] = {}
        for path in sorted(source_dir.iterdir()):
            if path.is_file() and path.suffix == ".parquet":
                candidates.setdefault(path.stem, []).append(path)
            elif path.is_dir():
                candidates.setdefault(path.name, []).append(path)
        available: dict[str, Path] = {}
        for name, paths in candidates.items():
            if len(paths) > 1:
                paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                self._logger.warning(
                    "Bronze table '%s' has both a file and directory artifact (%s); "
                    "using the most recently written one: %s",
                    name, [str(p) for p in paths], paths[0],
                )
            available[name] = paths[0]
        if tables is None:
            return available
        filtered = {table: available[table] for table in tables if table in available}
        missing = [table for table in tables if table not in available]
        if missing:
            raise DataDiscoveryError(
                f"Requested bronze tables are missing: {', '.join(sorted(missing))}"
            )
        return filtered

    def _write_metadata(self, artifact: SilverCleaningArtifact) -> None:
        metadata_dir = self._paths.paths.metadata / "datasets" / "silver" / artifact.site
        metadata_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "site": artifact.site,
            "canonical_table": artifact.canonical_table,
            "input_path": str(artifact.input_path),
            "output_path": str(artifact.output_path),
            "rows_before": artifact.rows_before,
            "rows_after": artifact.rows_after,
            "duplicates_removed": artifact.duplicates_removed,
            "discordant_susceptibility_group_n": artifact.discordant_susceptibility_group_n,
            "discordant_susceptibility_row_n": artifact.discordant_susceptibility_row_n,
            "shard_coalesced_group_n": artifact.shard_coalesced_group_n,
            "shard_conflict_group_n": artifact.shard_conflict_group_n,
            "shard_conflict_cell_n": artifact.shard_conflict_cell_n,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        metadata_path = metadata_dir / f"{artifact.canonical_table}.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _compute_dask_discordant_susceptibility_audit(
        dataframe,
        canonical_table: str,
        subset: list[str] | None,
    ) -> tuple[int, int]:
        if canonical_table != "microbial_resistance" or "susceptibility" not in dataframe.columns or not subset:
            return 0, 0

        audit_frame = dataframe.loc[:, subset + ["susceptibility"]].copy()
        audit_frame["susceptibility"] = audit_frame["susceptibility"].astype("string").str.strip()
        audit_frame = audit_frame.loc[audit_frame["susceptibility"].notna() & audit_frame["susceptibility"].ne("")]

        susceptibility_counts = (
            audit_frame.groupby(subset, dropna=False)["susceptibility"]
            .nunique(dropna=True)
            .reset_index(name="susceptibility_nunique")
        )
        discordant_keys = susceptibility_counts.loc[
            susceptibility_counts["susceptibility_nunique"] > 1,
            subset,
        ]
        discordant_group_n = int(discordant_keys.map_partitions(len).sum().compute())
        if discordant_group_n == 0:
            return 0, 0
        discordant_rows = audit_frame.merge(discordant_keys, on=subset, how="inner")
        discordant_row_n = int(discordant_rows.map_partitions(len).sum().compute())
        return discordant_group_n, discordant_row_n
