"""Production-ready sample dataset creation."""

from __future__ import annotations

import abc
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.eskape import ESKAPE_TARGETS
from amr_cascade_platform.data.ingestion.canonical_table_mapper import CanonicalTableMapper
from amr_cascade_platform.data.sampling.sampling_policy import FileSizeSamplingPolicy

try:
    import dask.dataframe as dd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dd = None


@dataclass(frozen=True)
class SamplingRequest:
    fraction: float
    seed: int
    source_layer: str = "raw"
    target_layer: str = "sample"
    sites: tuple[str, ...] | None = None
    ensure_eskape_coverage: bool = False
    eskape_min_rows_per_target: int = 25


class _CsvSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, input_path: Path, output_path: Path, fraction: float, seed: int) -> None:
        ...


class PandasCsvSampler(_CsvSampler):
    def __init__(self, large_file_threshold_mb: float, chunk_rows: int = 200_000) -> None:
        self._large_file_threshold_mb = large_file_threshold_mb
        self._chunk_rows = chunk_rows

    def sample(self, input_path: Path, output_path: Path, fraction: float, seed: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._should_stream(input_path):
            self._sample_large_csv(input_path, output_path, fraction, seed)
            return
        dataframe = pd.read_csv(input_path, low_memory=False, dtype=str)
        if dataframe.empty:
            shutil.copy2(input_path, output_path)
            return
        sample_size = max(1, int(len(dataframe) * fraction))
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        dataframe.sample(n=sample_size, random_state=seed).to_csv(temp_path, index=False)
        temp_path.replace(output_path)

    def _should_stream(self, input_path: Path) -> bool:
        size_mb = input_path.stat().st_size / (1024 * 1024)
        return size_mb > self._large_file_threshold_mb

    def _sample_large_csv(self, input_path: Path, output_path: Path, fraction: float, seed: int) -> None:
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        wrote_any_rows = False
        wrote_header = False
        for chunk_index, chunk in enumerate(
            pd.read_csv(input_path, low_memory=False, dtype=str, chunksize=self._chunk_rows)
        ):
            if chunk.empty:
                continue
            sampled_chunk = chunk.sample(frac=fraction, random_state=seed + chunk_index)
            if sampled_chunk.empty:
                continue
            sampled_chunk.to_csv(
                temp_path,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                index=False,
            )
            wrote_any_rows = True
            wrote_header = True

        if not wrote_any_rows:
            header_only = pd.read_csv(input_path, nrows=0)
            header_only.to_csv(temp_path, index=False)
        temp_path.replace(output_path)


class DaskCsvSampler(_CsvSampler):
    def __init__(self, blocksize: str, missing_tokens: tuple[str, ...]) -> None:
        self._blocksize = blocksize
        self._missing_tokens = list(missing_tokens)

    def sample(self, input_path: Path, output_path: Path, fraction: float, seed: int) -> None:
        if dd is None:
            raise RuntimeError("Dask sampler requested but dask is not installed.")
        dataframe = dd.read_csv(
            input_path,
            dtype=str,
            assume_missing=True,
            blocksize=self._blocksize,
            na_values=self._missing_tokens,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sampled = dataframe.sample(frac=fraction, random_state=seed)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        sampled.to_csv(
            str(temp_path),
            index=False,
            single_file=True,
        )
        temp_path.replace(output_path)


class SampleDatasetCreator:
    """Create local-development sample datasets from raw site CSVs."""

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
        self._policy = FileSizeSamplingPolicy(settings.sampling.use_dask_threshold_mb)
        self._pandas_sampler = PandasCsvSampler(
            large_file_threshold_mb=settings.sampling.use_dask_threshold_mb,
        )
        self._dask_sampler = DaskCsvSampler(
            blocksize=settings.sampling.dask_blocksize,
            missing_tokens=settings.platform.missing_tokens,
        )

    def create(self, request: SamplingRequest) -> None:
        if not 0 < request.fraction <= 1:
            raise ValueError("Sampling fraction must be in the interval (0, 1].")

        sites = request.sites or self._settings.platform.sites
        for site in sites:
            self._sample_site(site=site, request=request)

        if self._settings.sampling.copy_reference_files_to_output:
            self._copy_reference_files()

    def _sample_site(self, site: str, request: SamplingRequest) -> None:
        try:
            table_map = self._mapper.site_tables(site)
        except DataDiscoveryError as exc:
            self._logger.warning("%s", exc)
            return

        output_dir = self._paths.site_dir(request.target_layer, site)
        output_dir.mkdir(parents=True, exist_ok=True)
        sampled_cohort_keys: dict[str, set[str]] = {}
        coverage_summary: pd.DataFrame | None = None

        if request.ensure_eskape_coverage and "cohort" in table_map:
            cohort_output_path = output_dir / table_map["cohort"].name
            sampled_cohort, coverage_summary = self._sample_cohort_with_eskape_guard(
                input_path=table_map["cohort"],
                output_path=cohort_output_path,
                fraction=request.fraction,
                seed=request.seed,
                min_rows_per_target=request.eskape_min_rows_per_target,
            )
            sampled_cohort_keys = self._build_sampled_key_sets(sampled_cohort)
            self._logger.info(
                "Applied ESKAPE-aware cohort sampling for %s; sampled %s rows with %s available targets preserved.",
                site,
                len(sampled_cohort),
                int((coverage_summary["raw_count"] > 0).sum()),
            )
            coverage_output = output_dir / "eskape_coverage_summary.csv"
            coverage_summary.to_csv(coverage_output, index=False)

        for canonical_name, input_path in tqdm(
            table_map.items(),
            desc=f"Sampling {site}",
            unit="file",
        ):
            output_path = output_dir / input_path.name
            if request.ensure_eskape_coverage and canonical_name == "cohort":
                continue
            if sampled_cohort_keys and canonical_name != "cohort":
                linked_column = self._select_link_column(input_path)
                if linked_column is not None and sampled_cohort_keys.get(linked_column):
                    self._logger.info(
                        "Filtering %s -> %s by sampled cohort key %s (%s values)",
                        input_path.name,
                        output_path,
                        linked_column,
                        len(sampled_cohort_keys[linked_column]),
                    )
                    self._filter_to_sampled_keys(
                        input_path=input_path,
                        output_path=output_path,
                        key_column=linked_column,
                        keys=sampled_cohort_keys[linked_column],
                    )
                    continue
            sampler = self._choose_sampler(input_path)
            self._logger.info(
                "Sampling %s -> %s using %s at fraction=%s",
                input_path.name,
                output_path,
                sampler.__class__.__name__,
                request.fraction,
            )
            sampler.sample(input_path, output_path, request.fraction, request.seed)

    def _choose_sampler(self, input_path: Path) -> _CsvSampler:
        decision = self._policy.decide(input_path, dask_available=dd is not None)
        if decision.use_dask:
            return self._dask_sampler
        return self._pandas_sampler

    def _sample_cohort_with_eskape_guard(
        self,
        *,
        input_path: Path,
        output_path: Path,
        fraction: float,
        seed: int,
        min_rows_per_target: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataframe = pd.read_csv(input_path, low_memory=False, dtype=str)
        sampled, coverage = self._sample_cohort_dataframe_with_eskape_floor(
            dataframe=dataframe,
            fraction=fraction,
            seed=seed,
            min_rows_per_target=min_rows_per_target,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        sampled.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
        return sampled, coverage

    @staticmethod
    def _sample_cohort_dataframe_with_eskape_floor(
        *,
        dataframe: pd.DataFrame,
        fraction: float,
        seed: int,
        min_rows_per_target: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if dataframe.empty:
            return dataframe.copy(), SampleDatasetCreator._build_eskape_coverage_summary(
                dataframe=dataframe,
                sampled_dataframe=dataframe.copy(),
                min_rows_per_target=min_rows_per_target,
            )

        base_sample_size = max(1, int(len(dataframe) * fraction))
        organism = dataframe.get("organism", pd.Series("", index=dataframe.index, dtype="string")).fillna("").astype(str).str.upper()

        selected_index: set[object] = set()
        for target in ESKAPE_TARGETS:
            matching_index = dataframe.index[organism.str.contains(target.armd_match_pattern, regex=True, na=False)]
            available_count = len(matching_index)
            if available_count == 0:
                continue
            target_sample_size = min(
                available_count,
                max(int(math.ceil(available_count * fraction)), min_rows_per_target),
            )
            chosen = dataframe.loc[matching_index].sample(n=target_sample_size, random_state=seed).index
            selected_index.update(chosen.tolist())

        selected_positions = list(selected_index)
        remaining_needed = max(0, base_sample_size - len(selected_positions))
        if remaining_needed > 0:
            remaining_frame = dataframe.drop(index=selected_positions, errors="ignore")
            if not remaining_frame.empty:
                extra_n = min(remaining_needed, len(remaining_frame))
                extra_index = remaining_frame.sample(n=extra_n, random_state=seed).index
                selected_positions.extend(extra_index.tolist())

        sampled = dataframe.loc[sorted(selected_positions)].copy()
        coverage = SampleDatasetCreator._build_eskape_coverage_summary(
            dataframe=dataframe,
            sampled_dataframe=sampled,
            min_rows_per_target=min_rows_per_target,
        )
        return sampled, coverage

    @staticmethod
    def _build_eskape_coverage_summary(
        *,
        dataframe: pd.DataFrame,
        sampled_dataframe: pd.DataFrame,
        min_rows_per_target: int,
    ) -> pd.DataFrame:
        raw_organism = dataframe.get("organism", pd.Series("", index=dataframe.index, dtype="string")).fillna("").astype(str).str.upper()
        sampled_organism = sampled_dataframe.get("organism", pd.Series("", index=sampled_dataframe.index, dtype="string")).fillna("").astype(str).str.upper()
        rows: list[dict[str, object]] = []
        for target in ESKAPE_TARGETS:
            raw_count = int(raw_organism.str.contains(target.armd_match_pattern, regex=True, na=False).sum())
            sampled_count = int(sampled_organism.str.contains(target.armd_match_pattern, regex=True, na=False).sum())
            rows.append(
                {
                    "eskape_target": target.display_name,
                    "scientific_focus": target.scientific_focus,
                    "raw_count": raw_count,
                    "sampled_count": sampled_count,
                    "min_rows_floor": min_rows_per_target,
                    "present_in_raw": raw_count > 0,
                    "present_in_sample": sampled_count > 0,
                }
            )
        return pd.DataFrame(rows)

    def _build_sampled_key_sets(self, sampled_cohort: pd.DataFrame) -> dict[str, set[str]]:
        key_sets: dict[str, set[str]] = {}
        for column in self._settings.platform.id_columns:
            if column not in sampled_cohort.columns:
                continue
            values = (
                sampled_cohort[column]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            values = values[values.ne("")]
            key_sets[column] = set(values.tolist())
        return key_sets

    def _select_link_column(self, input_path: Path) -> str | None:
        try:
            header = pd.read_csv(input_path, nrows=0)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.warning("Failed to inspect columns for %s: %s", input_path, exc)
            return None
        for column in ("order_proc_id_coded", "pat_enc_csn_id_coded", "anon_id"):
            if column in header.columns:
                return column
        return None

    def _filter_to_sampled_keys(
        self,
        *,
        input_path: Path,
        output_path: Path,
        key_column: str,
        keys: set[str],
    ) -> None:
        if not keys:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return

        decision = self._policy.decide(input_path, dask_available=dd is not None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        key_values = sorted(keys)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        if decision.use_dask and dd is not None:
            dataframe = dd.read_csv(
                input_path,
                dtype=str,
                assume_missing=True,
                blocksize=self._settings.sampling.dask_blocksize,
                na_values=list(self._settings.platform.missing_tokens),
            )
            filtered = dataframe[dataframe[key_column].fillna("").astype(str).str.strip().isin(key_values)]
            filtered.to_csv(str(temp_path), index=False, single_file=True)
            temp_path.replace(output_path)
            return

        if self._pandas_sampler._should_stream(input_path):
            self._filter_large_csv_to_sampled_keys(
                input_path=input_path,
                output_path=output_path,
                key_column=key_column,
                keys=key_values,
            )
            return

        dataframe = pd.read_csv(input_path, low_memory=False, dtype=str)
        filtered = dataframe[dataframe[key_column].fillna("").astype(str).map(str.strip).isin(key_values)]
        filtered.to_csv(temp_path, index=False)
        temp_path.replace(output_path)

    def _filter_large_csv_to_sampled_keys(
        self,
        *,
        input_path: Path,
        output_path: Path,
        key_column: str,
        keys: list[str],
    ) -> None:
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        wrote_any_rows = False
        wrote_header = False
        key_lookup = set(keys)
        for chunk in pd.read_csv(
            input_path,
            low_memory=False,
            dtype=str,
            chunksize=self._pandas_sampler._chunk_rows,
        ):
            filtered = chunk[chunk[key_column].fillna("").astype(str).map(str.strip).isin(key_lookup)]
            if filtered.empty:
                continue
            filtered.to_csv(
                temp_path,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                index=False,
            )
            wrote_any_rows = True
            wrote_header = True

        if not wrote_any_rows:
            header_only = pd.read_csv(input_path, nrows=0)
            header_only.to_csv(temp_path, index=False)
        temp_path.replace(output_path)

    def _copy_reference_files(self) -> None:
        for filename in self._settings.platform.reference_files.values():
            source = self._paths.paths.reference / filename
            target = self._paths.paths.sample / filename
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
