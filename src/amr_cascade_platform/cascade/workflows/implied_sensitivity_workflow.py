"""ARMD-only implied-susceptibility sensitivity workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.implied_delta_builder import ImpliedDeltaBuilder
from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import CascadeAnalysisWorkflow
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.transformations.pair_generator import PairGenerator
from amr_cascade_platform.data.transformations.testing_matrix_builder import TestingMatrixBuilder
from amr_cascade_platform.domain.services.eligibility_service import EligibilityService
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class ImpliedSensitivityRequest:
    site: str = "armd"


class ImpliedSensitivityWorkflow:
    """Build a combined observed-plus-implied observation layer for armd sensitivity analysis."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)
        self._parquet_store = ParquetStore(settings)
        self._eligibility_service = EligibilityService(settings, path_manager.paths.reference)
        self._pair_generator = PairGenerator(settings)
        self._testing_matrix_builder = TestingMatrixBuilder(settings)
        self._primary_workflow = CascadeAnalysisWorkflow(settings, path_manager)
        self._delta_builder = ImpliedDeltaBuilder()

    def run(self, request: ImpliedSensitivityRequest) -> dict[str, Path]:
        if request.site != "armd":
            raise DataDiscoveryError("Implied-susceptibility sensitivity workflow is supported only for site 'armd'.")

        gold_dir = self._paths.paths.gold / request.site
        culture_episodes = self._dataset_store.read_pandas(gold_dir / "culture_episodes.parquet")
        observed = self._dataset_store.read_pandas(gold_dir / "culture_drug_episodes.parquet")
        implied = self._dataset_store.read_pandas(self._paths.paths.silver / request.site / "implied_susceptibility.parquet")

        merged = implied.merge(
            culture_episodes.loc[:, list(self._settings.gold.episode_key_columns)].drop_duplicates(),
            on=["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "organism", "source_site"],
            how="inner",
        )
        merged["effective_susceptibility"] = merged["implied_susceptibility"].combine_first(merged["susceptibility"])
        merged = merged[merged["effective_susceptibility"].notna()].copy()
        implied_rows = merged.loc[
            :,
            [
                "anon_id",
                "pat_enc_csn_id_coded",
                "order_proc_id_coded",
                "order_time_jittered",
                "organism",
                "source_site",
                "antibiotic",
            ],
        ].copy()
        implied_rows["susceptibility"] = merged["effective_susceptibility"]
        implied_rows["was_tested"] = 1
        implied_rows["observation_layer"] = "observed_plus_implied"

        join_keys = list(self._settings.gold.episode_key_columns) + ["antibiotic"]
        implied_only = implied_rows.merge(observed.loc[:, join_keys], on=join_keys, how="left", indicator=True)
        implied_only = implied_only[implied_only["_merge"] == "left_only"].drop(columns="_merge")
        combined_observation = pd.concat([observed, implied_only], ignore_index=True)

        eligibility = self._eligibility_service.build_episode_eligibility(culture_episodes, combined_observation)
        pair_table = self._pair_generator.build(combined_observation, eligibility)
        testing_matrix = self._testing_matrix_builder.build(combined_observation)

        sensitivity_dir = self._paths.paths.gold / f"{request.site}_sensitivity_implied"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        self._write_gold_like_outputs(
            sensitivity_dir=sensitivity_dir,
            culture_episodes=culture_episodes,
            combined_observation=combined_observation,
            testing_matrix=testing_matrix,
            eligibility=eligibility,
            pair_table=pair_table,
        )

        outputs = self._primary_workflow.run(
            request=type(
                "Req",
                (),
                {
                    "gold_scope": "site",
                    "site": f"{request.site}_sensitivity_implied",
                    "organism": None,
                },
            )()
        )
        outputs.update(
            self._delta_builder.export(
                primary_dir=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir / request.site,
                sensitivity_dir=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir / f"{request.site}_sensitivity_implied",
            )
        )
        return outputs

    def _write_gold_like_outputs(
        self,
        sensitivity_dir: Path,
        culture_episodes: pd.DataFrame,
        combined_observation: pd.DataFrame,
        testing_matrix: pd.DataFrame,
        eligibility: pd.DataFrame,
        pair_table: pd.DataFrame,
    ) -> None:
        outputs = {
            "culture_episodes": sensitivity_dir / "culture_episodes.parquet",
            "culture_drug_episodes": sensitivity_dir / "culture_drug_episodes.parquet",
            "testing_matrix": sensitivity_dir / "testing_matrix.parquet",
            "eligible_pairs": sensitivity_dir / "eligible_pairs.parquet",
            "drug_pair_episodes": sensitivity_dir / "drug_pair_episodes.parquet",
        }
        self._parquet_store.write_pandas(culture_episodes, outputs["culture_episodes"])
        self._parquet_store.write_pandas(combined_observation, outputs["culture_drug_episodes"])
        self._parquet_store.write_pandas(testing_matrix, outputs["testing_matrix"])
        self._parquet_store.write_pandas(eligibility, outputs["eligible_pairs"])
        self._parquet_store.write_pandas(pair_table, outputs["drug_pair_episodes"])

        metadata = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "site": "armd",
            "observation_layer": "observed_plus_implied",
            "description": (
                "Sensitivity-only gold namespace combining observed AST with implied susceptibility rows that "
                "were not directly observed for the same episode-antibiotic pair."
            ),
            "row_counts": {
                "culture_episodes": int(len(culture_episodes)),
                "culture_drug_episodes": int(len(combined_observation)),
                "testing_matrix": int(len(testing_matrix)),
                "eligible_pairs": int(len(eligibility)),
                "drug_pair_episodes": int(len(pair_table)),
            },
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        metadata_path = sensitivity_dir / "sensitivity_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
