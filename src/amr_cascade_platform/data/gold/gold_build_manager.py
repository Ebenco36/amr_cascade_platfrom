"""Construct gold analytical datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.core.utils.organism_names import matches_requested_organism
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.core.utils.text import normalize_label, safe_feature_name
from amr_cascade_platform.data.transformations.culture_episode_builder import CultureEpisodeBuilder
from amr_cascade_platform.data.transformations.observation_space_builder import ObservationSpaceBuilder
from amr_cascade_platform.data.transformations.pair_generator import PairGenerator
from amr_cascade_platform.data.transformations.testing_matrix_builder import TestingMatrixBuilder
from amr_cascade_platform.domain.services.eligibility_service import EligibilityService
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class GoldBuildRequest:
    source_scope: str = "combined"
    site: str | None = None
    organism: str | None = None


class GoldBuildManager:
    """Build gold tables from harmonized cohort datasets."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._dataset_store = DatasetStore(settings)
        self._parquet_store = ParquetStore(settings)
        self._culture_episode_builder = CultureEpisodeBuilder(settings)
        self._observation_space_builder = ObservationSpaceBuilder(settings)
        self._testing_matrix_builder = TestingMatrixBuilder(settings)
        self._pair_generator = PairGenerator(settings)
        self._eligibility_service = EligibilityService(settings, path_manager.paths.reference)

    def build(self, request: GoldBuildRequest) -> dict[str, Path]:
        cohort_path = self._resolve_cohort_path(request)
        cohort = self._dataset_store.read_pandas(cohort_path)
        cohort_rows = cohort["source_site"].astype("string").fillna("unknown").value_counts()
        organism_list = None
        organism_labels: dict[str, int] = {}
        if request.organism:
            cohort, organism_labels = self._select_organism(cohort, request.organism)
            organism_list = (normalize_label(request.organism),)

        culture_episodes = self._culture_episode_builder.build(cohort, organism_list=organism_list)
        culture_drug_episodes, ledger = self._observation_space_builder.build_with_ledger(cohort, organism_list=organism_list)
        # Raw site data records the same antibiotic under different spellings
        # (site-specific order-set abbreviations, alternate salts). Normalizing
        # here, at the single earliest point shared by every downstream gold
        # table (eligibility, pair generation, testing matrix), guarantees
        # every later groupby/join keyed on "antibiotic" sees canonical names
        # rather than silently splitting one drug into several.
        culture_drug_episodes["antibiotic"] = culture_drug_episodes["antibiotic"].map(normalize_antibiotic_label)
        testing_matrix = self._testing_matrix_builder.build(culture_drug_episodes)
        eligible_pairs = self._eligibility_service.build_episode_eligibility(
            culture_episodes=culture_episodes,
            culture_drug_episodes=culture_drug_episodes,
        )
        drug_pair_episodes = self._pair_generator.build(
            culture_drug_episodes=culture_drug_episodes,
            eligibility_space=eligible_pairs,
        )
        observation_ledger = self._observation_ledger(ledger, cohort_rows, culture_episodes, culture_drug_episodes)

        output_dir = self._resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "culture_episodes": output_dir / "culture_episodes.parquet",
            "culture_drug_episodes": output_dir / "culture_drug_episodes.parquet",
            "drug_pair_episodes": output_dir / "drug_pair_episodes.parquet",
            "testing_matrix": output_dir / "testing_matrix.parquet",
            "eligible_pairs": output_dir / "eligible_pairs.parquet",
        }
        self._parquet_store.write_pandas(culture_episodes, outputs["culture_episodes"])
        self._parquet_store.write_pandas(culture_drug_episodes, outputs["culture_drug_episodes"])
        self._parquet_store.write_pandas(drug_pair_episodes, outputs["drug_pair_episodes"])
        self._parquet_store.write_pandas(testing_matrix, outputs["testing_matrix"])
        self._parquet_store.write_pandas(eligible_pairs, outputs["eligible_pairs"])
        self._write_metadata(
            request=request,
            outputs=outputs,
            organism_labels=organism_labels,
            observation_ledger=observation_ledger,
            culture_episodes=culture_episodes,
            culture_drug_episodes=culture_drug_episodes,
            drug_pair_episodes=drug_pair_episodes,
            testing_matrix=testing_matrix,
            eligible_pairs=eligible_pairs,
        )
        return outputs

    def _observation_ledger(
        self,
        ledger: pd.DataFrame,
        cohort_rows: pd.Series,
        culture_episodes: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
    ) -> dict[str, dict[str, int]]:
        """Per site, how many rows each rule removed between the harmonised cohort and the observed results.

        Extends the observation-space builder's ledger with the rows present
        before organism selection and with what antibiotic-name normalisation
        exposed: rows repeating an episode, antibiotic and result, and episode
        x antibiotic pairs holding conflicting results (counted once as
        observed by the eligibility service; excluded from the resistant-versus-
        susceptible upstream contrast by the pair generator). The report step
        turns this into the exclusion-flow table, which must reconcile row for row.
        """
        keys = list(self._settings.gold.episode_key_columns)
        results = culture_drug_episodes.loc[:, keys + ["antibiotic", "susceptibility"]]
        distinct = results.drop_duplicates()
        per_pair = distinct.groupby(keys + ["antibiotic"], observed=True, dropna=False).size().rename("results").reset_index()
        conflicting = per_pair.loc[per_pair["results"].gt(1)]
        observed_episodes = per_pair.loc[:, keys].drop_duplicates()
        without_result = culture_episodes.loc[:, keys].drop_duplicates().merge(observed_episodes, on=keys, how="left", indicator=True)
        without_result = without_result.loc[without_result["_merge"].eq("left_only")]

        def by_site(frame: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
            sites = frame["source_site"].astype("string").fillna("unknown")
            if weights is None:
                return sites.value_counts()
            return weights.groupby(sites.to_numpy()).sum()

        extra = pd.DataFrame(
            {
                "cohort_rows": cohort_rows,
                "same_result_duplicate_rows": by_site(results).sub(by_site(distinct), fill_value=0),
                "conflicting_result_rows": by_site(conflicting, conflicting["results"]),
                "conflicting_episode_drugs": by_site(conflicting),
                "observed_episode_drugs": by_site(per_pair),
                "culture_episodes": by_site(culture_episodes.loc[:, keys].drop_duplicates()),
                "episodes_without_result": by_site(without_result),
            }
        )
        combined = ledger.join(extra, how="outer").fillna(0).astype("int64")
        combined = combined.loc[combined["organism_rows"].gt(0) | combined["culture_episodes"].gt(0)]
        return {str(site): {name: int(value) for name, value in row.items()} for site, row in combined.iterrows()}

    def _select_organism(self, cohort, requested: str):
        """Rows of the requested organism, relabelled to the requested name.

        Raw labels that resolve to the requested species -- e.g. "ESBL
        ESCHERICHIA COLI" or "ESCHERICHIA COLI (CARBAPENEM RESISTANT)" for
        "ESCHERICHIA COLI" -- are kept and written under the requested name, so
        every gold table and every later groupby treats them as one organism
        (see matches_requested_organism). Returns the rows and the number of
        culture episodes contributed by each raw label.
        """
        labels = cohort["organism"].astype("string")
        kept = {label for label in labels.dropna().unique() if matches_requested_organism(label, requested)}
        selected = cohort.loc[labels.isin(kept).fillna(False)].copy()
        episode_columns = list(self._settings.gold.episode_key_columns)
        counts = selected.drop_duplicates(episode_columns)["organism"].astype("string").value_counts()
        organism_labels = {str(label): int(count) for label, count in counts.items()}
        if len(organism_labels) > 1:
            self._logger.info("Pooled organism labels into %s: %s", normalize_label(requested), organism_labels)
        selected["organism"] = normalize_label(requested)
        return selected, organism_labels

    def _resolve_cohort_path(self, request: GoldBuildRequest) -> Path:
        if request.source_scope == "combined":
            path = self._paths.paths.harmonized / "combined" / "cohort.parquet"
        elif request.source_scope == "site" and request.site:
            path = self._paths.paths.harmonized / "site_aligned" / request.site / "cohort.parquet"
        else:
            raise DataDiscoveryError(
                "Gold build requires source_scope='combined' or source_scope='site' with site set."
            )
        if not path.exists():
            raise DataDiscoveryError(f"Harmonized cohort dataset not found: {path}")
        return path

    def _resolve_output_dir(self, request: GoldBuildRequest) -> Path:
        return scoped_output_dir(
            root=self._paths.paths.gold,
            scope=request.source_scope,
            site=request.site,
            organism=request.organism,
        )

    def _write_metadata(
        self,
        request: GoldBuildRequest,
        outputs: dict[str, Path],
        organism_labels: dict[str, int],
        observation_ledger: dict[str, dict[str, int]],
        culture_episodes,
        culture_drug_episodes,
        drug_pair_episodes,
        testing_matrix,
        eligible_pairs,
    ) -> None:
        metadata_dir = self._paths.paths.metadata / "datasets" / "gold"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_scope": request.source_scope,
            "site": request.site,
            "organism": request.organism,
            "outputs": {name: str(path) for name, path in outputs.items()},
            # Culture episodes per raw organism label pooled into this organism.
            "organism_labels": organism_labels,
            # Per site: rows removed by each rule from the harmonised cohort to the
            # observed episode x antibiotic results (see _observation_ledger).
            "observation_ledger": observation_ledger,
            "row_counts": {
                "culture_episodes": len(culture_episodes),
                "culture_drug_episodes": len(culture_drug_episodes),
                "drug_pair_episodes": len(drug_pair_episodes),
                "testing_matrix": len(testing_matrix),
                "eligible_pairs": len(eligible_pairs),
            },
        }
        suffix = request.site if request.source_scope == "site" and request.site else "combined"
        if request.organism:
            suffix = f"{suffix}__{safe_feature_name(request.organism)}"
        metadata_path = metadata_dir / f"gold_build_{suffix}.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
