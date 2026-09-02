"""Comorbidity pre-aggregation for feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.utils.text import safe_feature_name
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.ingestion.canonical_table_mapper import CanonicalTableMapper

try:
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dd = None
    ProgressBar = None


@dataclass(frozen=True)
class ComorbidityAggregationRequest:
    site: str
    data_layer: str = "raw"
    top_k: int | None = None
    min_count: int | None = None
    min_coverage: float | None = None
    analyze_only: bool = False
    allow_raw_on_local: bool = False

    def validate(self) -> None:
        selected = [self.top_k is not None, self.min_count is not None, self.min_coverage is not None]
        if sum(selected) > 1:
            raise ValueError("Only one of top_k, min_count, or min_coverage may be provided.")


class ComorbiditySelector:
    """Select retained comorbidities from frequency summaries."""

    def __init__(self, default_top_k: int) -> None:
        self._default_top_k = default_top_k

    def select(self, frequency: pd.Series, request: ComorbidityAggregationRequest) -> list[str]:
        request.validate()
        if request.min_count is not None:
            return frequency[frequency >= request.min_count].index.tolist()
        if request.min_coverage is not None:
            total = frequency.sum()
            cumulative = frequency.cumsum() / total
            selected_count = int((cumulative < request.min_coverage).sum()) + 1
            selected_count = min(selected_count, len(frequency))
            return frequency.iloc[:selected_count].index.tolist()
        keep = request.top_k if request.top_k is not None else self._default_top_k
        return frequency.head(keep).index.tolist()


class ComorbidityAggregator:
    """Aggregate a long comorbidity table into a flat culture-level feature matrix."""

    def __init__(
        self,
        settings: Settings,
        path_manager: PathManager,
        canonical_mapper: CanonicalTableMapper,
    ) -> None:
        self._settings = settings
        self._paths = path_manager
        self._mapper = canonical_mapper
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._selector = ComorbiditySelector(settings.comorbidity.default_top_k)

    def run(self, request: ComorbidityAggregationRequest) -> Path | None:
        request.validate()
        if (
            request.data_layer == "raw"
            and self._settings.environment.execution_mode == "local"
            and not self._settings.environment.allow_raw_processing
            and not request.allow_raw_on_local
        ):
            raise DataDiscoveryError(
                "Raw-layer comorbidity aggregation is blocked in the mac/local environment. "
                "Create sampled inputs first or pass an explicit override."
            )
        raw_path = self._resolve_source_file(request.site, request.data_layer)
        self._logger.info("Processing comorbidity table: %s", raw_path)

        if dd is not None:
            frequency, active_unique, all_cultures = self._prepare_with_dask(raw_path)
        else:
            frequency, active_unique, all_cultures = self._prepare_with_pandas(raw_path)

        self._log_frequency_diagnostics(frequency)
        if request.analyze_only:
            self._logger.info("Analyze-only mode enabled; skipping parquet output.")
            return None

        selected = self._selector.select(frequency, request)
        result = self._build_feature_matrix(
            active_unique=active_unique,
            all_cultures=all_cultures,
            selected=selected,
        )

        output_dir = self._paths.site_dir("interim", request.site) / "feature_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self._settings.comorbidity.output_filename
        result.to_parquet(output_path, index=False)
        self._logger.info("Saved %s with shape %s", output_path, result.shape)
        return output_path

    def _resolve_source_file(self, site: str, data_layer: str) -> Path:
        if data_layer == "raw":
            return self._mapper.site_tables(site)["comorbidity"]

        if data_layer == "sample":
            site_root = self._paths.site_dir("sample", site)
            canonical_name = self._mapper.site_tables(site)["comorbidity"].name
            sample_path = site_root / canonical_name
            if not sample_path.exists():
                raise DataDiscoveryError(
                    f"Sample comorbidity file not found for site '{site}': {sample_path}. "
                    f"Create sampled inputs first with scripts/create_sample_dataset.py."
                )
            return sample_path

        raise DataDiscoveryError(f"Unsupported data layer for comorbidity aggregation: {data_layer}")

    def _prepare_with_dask(
        self, path: Path
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        if dd is None:
            raise ImportError(
                "dask is required for comorbidity pre-aggregation on large files but is not installed. "
                "Install it with: pip install dask[dataframe]"
            )
        dataframe = dd.read_csv(
            path,
            assume_missing=True,
            dtype=str,
            blocksize=self._settings.comorbidity.dask_blocksize,
            na_values=list(self._settings.platform.missing_tokens),
        )
        normalized = self._normalize_numeric_columns(dataframe)
        active_unique = self._active_unique_rows(normalized).drop_duplicates()
        with ProgressBar():
            frequency = active_unique["comorbidity_component"].value_counts().compute()
            active_unique_pd = active_unique.compute()
            all_cultures = dataframe[list(self._settings.platform.id_columns)].drop_duplicates().compute()
        return frequency, active_unique_pd, all_cultures

    def _prepare_with_pandas(
        self, path: Path
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        dataframe = pd.read_csv(
            path,
            low_memory=False,
            dtype=str,
            na_values=list(self._settings.platform.missing_tokens),
        )
        normalized = self._normalize_numeric_columns(dataframe)
        active_unique = self._active_unique_rows(normalized).drop_duplicates()
        frequency = active_unique["comorbidity_component"].value_counts()
        all_cultures = dataframe[list(self._settings.platform.id_columns)].drop_duplicates()
        return frequency, active_unique, all_cultures

    def _normalize_numeric_columns(self, dataframe: pd.DataFrame):
        start_col = "comorbidity_component_start_days_culture"
        end_col = "comorbidity_component_end_days_culture"
        dataframe[start_col] = dataframe[start_col].astype("float64")
        dataframe[end_col] = dataframe[end_col].astype("float64")
        return dataframe

    def _active_unique_rows(self, dataframe):
        keys = list(self._settings.platform.id_columns)
        start_col = "comorbidity_component_start_days_culture"
        end_col = "comorbidity_component_end_days_culture"
        # The ARMD day-offset convention is culture_time - event_time:
        # positive values occurred before culture; negative values occurred after culture.
        # Active-at-culture comorbidities therefore started on/before culture and
        # had not ended before culture.
        active = dataframe[
            ((dataframe[start_col] >= 0) | (dataframe[start_col].isna()))
            & ((dataframe[end_col].isna()) | (dataframe[end_col] <= 0))
        ]
        return active[keys + ["comorbidity_component"]]

    def _build_feature_matrix(
        self,
        active_unique: pd.DataFrame,
        all_cultures: pd.DataFrame,
        selected: list[str],
    ) -> pd.DataFrame:
        keys = list(self._settings.platform.id_columns)
        counts = (
            active_unique.groupby(keys)
            .size()
            .rename("comorbidity_count")
            .reset_index()
        )

        selected_rows = active_unique[active_unique["comorbidity_component"].isin(selected)].copy()
        if not selected_rows.empty:
            selected_rows["value"] = 1
            pivot = selected_rows.pivot_table(
                index=keys,
                columns="comorbidity_component",
                values="value",
                aggfunc="first",
                fill_value=0,
            ).reset_index()
            pivot.columns = [
                column if column in keys else self.normalize_column_name(str(column))
                for column in pivot.columns
            ]
        else:
            pivot = pd.DataFrame(columns=keys)

        result = all_cultures.merge(counts, on=keys, how="left")
        if not pivot.empty:
            result = result.merge(pivot, on=keys, how="left")

        feature_columns = [column for column in result.columns if column not in keys]
        if feature_columns:
            result[feature_columns] = result[feature_columns].fillna(0).astype("int64")
        return result

    def _log_frequency_diagnostics(self, frequency: pd.Series) -> None:
        total = frequency.sum()
        self._logger.info("Computed frequencies for %s distinct comorbidities.", len(frequency))
        for k in self._settings.comorbidity.coverage_report_k_values:
            if k <= len(frequency):
                coverage = frequency.iloc[:k].sum() / total
                self._logger.info("Top %s comorbidities cover %.1f%% of active records.", k, coverage * 100)
        rarest = frequency.tail(self._settings.comorbidity.rarest_count).sort_values()
        for name, count in rarest.items():
            self._logger.info("Rare comorbidity | %s | %s occurrence(s)", name, count)

    @staticmethod
    def normalize_column_name(name: str) -> str:
        return f"comorb_{safe_feature_name(name)}"
