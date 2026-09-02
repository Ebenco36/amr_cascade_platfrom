"""Build model-ready feature matrices with lossless enrichment."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as _pq
except Exception:
    _pq = None

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.data.cleaning.site_cleaning_manager import (
    SilverCleaningRequest,
    SiteCleaningManager,
)
from amr_cascade_platform.data.harmonization.site_harmonizer import SiteHarmonizer
from amr_cascade_platform.data.ingestion.raw_ingestion_manager import (
    RawIngestionManager,
    RawIngestionRequest,
)
from amr_cascade_platform.features.builders.acute_feature_builder import AcuteFeatureBuilder
from amr_cascade_platform.features.builders.demographics_feature_builder import DemographicsFeatureBuilder
from amr_cascade_platform.features.builders.feature_merger import FeatureMerger
from amr_cascade_platform.features.builders.history_feature_builder import HistoryFeatureBuilder
from amr_cascade_platform.features.preprocessors.comorbidity_aggregator import (
    ComorbidityAggregationRequest,
    ComorbidityAggregator,
)
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore
from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import PairFeatureMatrixBuilder


@dataclass(frozen=True)
class FeatureBuildRequest:
    scope: str = "combined"
    organism: str | None = None


class FeatureBuildWorkflow:
    """Create combined model-ready pair features with validated left joins."""

    def __init__(
        self,
        settings: Settings,
        path_manager: PathManager,
        canonical_mapper,
    ) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._dataset_store = DatasetStore(settings)
        self._parquet_store = ParquetStore(settings)
        self._pair_builder = PairFeatureMatrixBuilder(settings, path_manager)
        self._raw_ingestion_manager = RawIngestionManager(settings, path_manager, canonical_mapper)
        self._site_cleaning_manager = SiteCleaningManager(settings, path_manager)
        self._site_harmonizer = SiteHarmonizer(settings)
        self._canonical_mapper = canonical_mapper
        self._comorbidity_aggregator = ComorbidityAggregator(settings, path_manager, canonical_mapper)
        self._feature_merger = FeatureMerger()
        self._baseline_builder = DemographicsFeatureBuilder(settings)
        self._acute_builder = AcuteFeatureBuilder(settings)
        self._history_builder = HistoryFeatureBuilder(settings)

    def run(self, request: FeatureBuildRequest) -> dict[str, Path]:
        if request.scope != "combined":
            raise ValueError("Current feature build workflow supports scope='combined' only.")

        base_bundle = self._pair_builder.build(scope=request.scope, organism=request.organism)
        # The pair matrix can already be several gigabytes on Mac sample runs.
        # Avoid duplicating it before enrichment; we only need one mutable frame.
        base_frame = base_bundle.dataframe
        base_row_count = len(base_frame)
        culture_episodes = self._load_required_culture_episodes(request)

        id_keys = list(self._settings.platform.id_columns)
        site_episode_keys = {
            site: culture_episodes.loc[culture_episodes["source_site"] == site, id_keys].drop_duplicates()
            for site in self._settings.platform.sites
        }

        def filtered_loader(site: str, canonical_table: str) -> pd.DataFrame:
            return self._load_site_silver_table_filtered(
                site, canonical_table, site_episode_keys.get(site, pd.DataFrame())
            )

        baseline_features = self._baseline_builder.build(culture_episodes, filtered_loader)
        acute_features = self._acute_builder.build(culture_episodes, filtered_loader)
        history_features = self._history_builder.build(culture_episodes, filtered_loader)
        comorbidity_features = self._load_or_build_combined_comorbidity(culture_episodes)
        join_keys = tuple(self._settings.platform.id_columns) + ("source_site",)
        domain_results = []
        episode_feature_base = culture_episodes.loc[:, join_keys].drop_duplicates().reset_index(drop=True)
        episode_features = episode_feature_base
        for features in (
            baseline_features,
            acute_features,
            history_features,
            comorbidity_features,
        ):
            result = self._feature_merger.merge_left(
                base=episode_features,
                features=features,
                join_keys=join_keys,
            )
            episode_features = result.dataframe
            domain_results.append(result)

        if len(episode_features) != len(episode_feature_base):
            raise DataDiscoveryError("Episode-level feature workflow violated row-count preservation.")

        self._logger.info(
            "base_frame: %s rows x %s cols (%.1f GB) | episode_features: %s rows x %s cols (%.1f GB)",
            len(base_frame), len(base_frame.columns),
            base_frame.memory_usage(deep=False).sum() / 1e9,
            len(episode_features), len(episode_features.columns),
            episode_features.memory_usage(deep=False).sum() / 1e9,
        )

        output_dir = scoped_output_dir(
            root=self._paths.paths.features,
            scope=request.scope,
            organism=request.organism,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        interim_dir = scoped_output_dir(
            root=self._paths.paths.interim,
            scope=request.scope,
            organism=request.organism,
        ) / "feature_matrices"
        interim_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = interim_dir / self._settings.feature_build.baseline_output_filename
        acute_path = interim_dir / self._settings.feature_build.acute_output_filename
        history_path = interim_dir / self._settings.feature_build.history_output_filename
        feature_path = output_dir / self._settings.feature_build.combined_output_filename
        self._parquet_store.write_pandas(baseline_features, baseline_path)
        self._parquet_store.write_pandas(acute_features, acute_path)
        self._parquet_store.write_pandas(history_features, history_path)

        baseline_feature_cols = [c for c in baseline_features.columns if c not in join_keys]
        acute_feature_cols = [c for c in acute_features.columns if c not in join_keys]
        history_feature_cols = [c for c in history_features.columns if c not in join_keys]
        comorbidity_feature_cols = [c for c in comorbidity_features.columns if c not in join_keys]
        merge_summary_list = [
            {
                "added_feature_columns": list(result.added_feature_columns),
                "matched_rows": result.matched_rows,
                "unmatched_rows": result.unmatched_rows,
            }
            for result in domain_results
        ]

        del baseline_features, acute_features, history_features, comorbidity_features, domain_results
        gc.collect()

        enriched_rows = self._write_enriched_chunked(base_frame, episode_features, join_keys, feature_path)

        if enriched_rows != base_row_count:
            raise DataDiscoveryError("Feature workflow violated row-count preservation.")

        metadata_path = output_dir / self._settings.feature_build.metadata_filename
        payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": request.scope,
            "organism": request.organism,
            "base_row_count": base_row_count,
            "output_row_count": enriched_rows,
            "baseline_feature_columns": baseline_feature_cols,
            "acute_feature_columns": acute_feature_cols,
            "history_feature_columns": history_feature_cols,
            "comorbidity_feature_columns": comorbidity_feature_cols,
            "merge_summaries": merge_summary_list,
            "outputs": {
                "baseline": str(baseline_path),
                "acute": str(acute_path),
                "history": str(history_path),
                "feature_matrix": str(feature_path),
            },
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return {
            "baseline": baseline_path,
            "acute": acute_path,
            "history": history_path,
            "feature_matrix": feature_path,
            "summary": metadata_path,
        }

    def _write_enriched_chunked(
        self,
        base_frame: pd.DataFrame,
        episode_features: pd.DataFrame,
        join_keys: tuple[str, ...],
        feature_path: Path,
        chunk_size: int = 2_000_000,
    ) -> int:
        """Merge base_frame with episode_features in row chunks and stream to parquet.

        Avoids materialising the full enriched DataFrame (can be hundreds of GB)
        by merging and writing chunk_size rows at a time.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        join_key_list = list(join_keys)
        writer = None
        total_rows = 0
        n_chunks = (len(base_frame) + chunk_size - 1) // chunk_size
        try:
            for i, start in enumerate(range(0, len(base_frame), chunk_size)):
                chunk = base_frame.iloc[start:start + chunk_size].copy()
                merged = chunk.merge(episode_features, on=join_key_list, how="left")
                del chunk
                table = pa.Table.from_pandas(merged, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(feature_path), table.schema)
                writer.write_table(table)
                total_rows += len(merged)
                self._logger.info(
                    "Enrichment chunk %s/%s written (%s rows, %.1f GB)",
                    i + 1, n_chunks, len(merged),
                    merged.memory_usage(deep=False).sum() / 1e9,
                )
                del merged, table
                gc.collect()
        finally:
            if writer is not None:
                writer.close()
        return total_rows

    def _load_required_culture_episodes(self, request: FeatureBuildRequest) -> pd.DataFrame:
        path = scoped_output_dir(
            root=self._paths.paths.gold,
            scope=request.scope,
            organism=request.organism,
        ) / "culture_episodes.parquet"
        if not path.exists():
            raise DataDiscoveryError(f"Gold culture episodes not found: {path}")
        return self._dataset_store.read_pandas(path)

    def _load_site_silver_table_filtered(
        self, site: str, canonical_table: str, episode_keys: pd.DataFrame
    ) -> pd.DataFrame:
        harmonized_path = self._paths.paths.harmonized / "site_aligned" / site / f"{canonical_table}.parquet"
        if not harmonized_path.exists():
            return self._load_site_silver_table(site, canonical_table)
        if episode_keys.empty:
            return pd.DataFrame()
        id_keys = list(self._settings.platform.id_columns)
        filter_sets = {col: set(episode_keys[col].tolist()) for col in id_keys if col in episode_keys.columns}
        if not filter_sets or _pq is None:
            df = self._dataset_store.read_pandas(harmonized_path)
            return df.merge(episode_keys, on=id_keys, how="inner") if not df.empty else df
        chunks: list[pd.DataFrame] = []
        for batch in _pq.ParquetFile(str(harmonized_path)).iter_batches(batch_size=500_000):
            chunk = batch.to_pandas()
            mask = pd.Series(True, index=chunk.index)
            for col, values in filter_sets.items():
                if col in chunk.columns:
                    mask &= chunk[col].isin(values)
            filtered = chunk.loc[mask]
            if not filtered.empty:
                chunks.append(filtered)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def _load_site_silver_table(self, site: str, canonical_table: str) -> pd.DataFrame:
        harmonized_path = self._paths.paths.harmonized / "site_aligned" / site / f"{canonical_table}.parquet"
        if harmonized_path.exists():
            return self._dataset_store.read_pandas(harmonized_path)

        silver_path = self._paths.site_dir("silver", site) / f"{canonical_table}.parquet"

        if not self._settings.feature_build.auto_prepare_silver_feature_sources:
            return pd.DataFrame()

        mapped_tables = self._canonical_mapper.site_tables(site)
        if canonical_table not in mapped_tables:
            return pd.DataFrame()

        self._logger.info("Preparing missing silver feature source %s for %s", canonical_table, site)
        self._raw_ingestion_manager.ingest_site(
            RawIngestionRequest(
                site=site,
                source_layer=self._settings.environment.default_data_layer,
                tables=(canonical_table,),
            )
        )
        self._site_cleaning_manager.clean_site(
            SilverCleaningRequest(site=site, tables=(canonical_table,))
        )
        self._site_harmonizer.harmonize_site(
            site=site,
            silver_dir=self._paths.site_dir("silver", site),
            output_dir=self._paths.paths.harmonized / "site_aligned" / site,
            tables=(canonical_table,),
        )
        if harmonized_path.exists():
            return self._dataset_store.read_pandas(harmonized_path)
        return pd.DataFrame()

    def _load_or_build_combined_comorbidity(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        id_keys = list(self._settings.platform.id_columns)
        frames: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            feature_path = (
                self._paths.site_dir("interim", site)
                / "feature_matrices"
                / self._settings.feature_build.comorbidity_output_filename
            )
            if not feature_path.exists():
                if not self._settings.feature_build.auto_build_comorbidity:
                    raise DataDiscoveryError(f"Missing comorbidity feature matrix for site '{site}': {feature_path}")
                self._logger.info("Building missing comorbidity feature matrix for %s", site)
                self._comorbidity_aggregator.run(
                    ComorbidityAggregationRequest(
                        site=site,
                        data_layer=self._settings.environment.default_data_layer,
                        min_coverage=0.9,
                    )
                )
            if not feature_path.exists():
                raise DataDiscoveryError(f"Failed to create comorbidity feature matrix for site '{site}'.")
            frame = self._dataset_store.read_pandas(feature_path)
            site_episode_keys = (
                culture_episodes.loc[culture_episodes["source_site"] == site, id_keys]
                .drop_duplicates()
            )
            frame = frame.merge(site_episode_keys, on=id_keys, how="inner").copy()
            frame["source_site"] = site
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
