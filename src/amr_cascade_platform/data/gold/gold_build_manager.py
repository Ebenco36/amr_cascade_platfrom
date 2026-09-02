"""Construct gold analytical datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
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
        organism_list = (normalize_label(request.organism),) if request.organism else None

        culture_episodes = self._culture_episode_builder.build(cohort, organism_list=organism_list)
        culture_drug_episodes = self._observation_space_builder.build(cohort, organism_list=organism_list)
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
            culture_episodes=culture_episodes,
            culture_drug_episodes=culture_drug_episodes,
            drug_pair_episodes=drug_pair_episodes,
            testing_matrix=testing_matrix,
            eligible_pairs=eligible_pairs,
        )
        return outputs

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
