"""ESKAPE-family validation workflows for ARMD-style cohort data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import ConditionalProbabilityAnalyzer
from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.cascade.analyzers.cascade_pair_dependence_analyzer import CascadePairDependenceAnalyzer
from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import EscalationRatioAnalyzer
from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
from amr_cascade_platform.cascade.analyzers.guideline_concordance_analyzer import GuidelineConcordanceAnalyzer
from amr_cascade_platform.cascade.outputs.cascade_report_builder import CascadeReportBuilder
from amr_cascade_platform.cascade.outputs.cascade_result_writer import CascadeResultWriter
from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.eskape import EskapeTarget
from amr_cascade_platform.core.utils.text import normalize_label, safe_feature_name
from amr_cascade_platform.data.transformations.culture_episode_builder import CultureEpisodeBuilder
from amr_cascade_platform.data.transformations.observation_space_builder import ObservationSpaceBuilder
from amr_cascade_platform.data.transformations.pair_generator import PairGenerator
from amr_cascade_platform.data.transformations.testing_matrix_builder import TestingMatrixBuilder
from amr_cascade_platform.domain.services.eligibility_service import EligibilityService
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


@dataclass(frozen=True)
class ArmdEskapeRequest:
    target: EskapeTarget
    source_scope: str = "site"
    site: str | None = "armd"
    output_root: str | Path = "./data/artifacts/eskape_validation/armd"


class ArmdEskapeCascadeRunner:
    """Run cascade validation on ARMD-style cohort data filtered to one ESKAPE family."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)
        self._culture_episode_builder = CultureEpisodeBuilder(settings)
        self._observation_space_builder = ObservationSpaceBuilder(settings)
        self._testing_matrix_builder = TestingMatrixBuilder(settings)
        self._pair_generator = PairGenerator(settings)
        self._eligibility_service = EligibilityService(settings, path_manager.paths.reference)

        self._cotesting_filter = CoTestingFilterAnalyzer(settings)
        self._conditional_analyzer = ConditionalProbabilityAnalyzer(settings)
        self._escalation_analyzer = EscalationRatioAnalyzer(settings)
        self._regression_analyzer = DownstreamTestingRegression(settings, path_manager)
        self._retained_edge_analyzer = RetainedEdgeAnalyzer(settings)
        self._validation_analyzer = CascadeValidationAnalyzer(settings)
        self._pair_dependence_analyzer = CascadePairDependenceAnalyzer(settings)
        self._guideline_concordance_analyzer = GuidelineConcordanceAnalyzer(settings, path_manager)
        self._writer = CascadeResultWriter(settings)
        self._report_builder = CascadeReportBuilder(settings.cascade.continuity_correction)

    def run(self, request: ArmdEskapeRequest) -> dict[str, Path]:
        cohort = self._load_cohort(request)
        filtered = self._filter_cohort(cohort, request.target)
        output_dir = self._resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)

        filter_path = output_dir / "filtered_input_cohort.parquet"
        filtered.to_parquet(filter_path, index=False)
        matched_labels = (
            filtered["organism"].astype("string").value_counts(dropna=False).rename_axis("armd_organism_label").reset_index(name="row_count")
            if not filtered.empty
            else pd.DataFrame(columns=["armd_organism_label", "row_count"])
        )
        matched_labels_path = output_dir / "matched_labels.csv"
        matched_labels.to_csv(matched_labels_path, index=False)

        if filtered.empty:
            raise ValidationError(
                f"ARMD ESKAPE target '{request.target.display_name}' has no usable rows after applying "
                f"the family filter '{request.target.armd_match_pattern}'."
            )

        culture_episodes = self._culture_episode_builder.build(filtered, organism_list=None)
        culture_drug_episodes = self._observation_space_builder.build(filtered, organism_list=None)
        testing_matrix = self._testing_matrix_builder.build(culture_drug_episodes)
        eligible_pairs = self._eligibility_service.build_episode_eligibility(
            culture_episodes=culture_episodes,
            culture_drug_episodes=culture_drug_episodes,
        )
        drug_pair_episodes = self._pair_generator.build(
            culture_drug_episodes=culture_drug_episodes,
            eligibility_space=eligible_pairs,
        )

        gold_dir = output_dir / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        culture_episodes.to_parquet(gold_dir / "culture_episodes.parquet", index=False)
        culture_drug_episodes.to_parquet(gold_dir / "culture_drug_episodes.parquet", index=False)
        testing_matrix.to_parquet(gold_dir / "testing_matrix.parquet", index=False)
        eligible_pairs.to_parquet(gold_dir / "eligible_pairs.parquet", index=False)
        drug_pair_episodes.to_parquet(gold_dir / "drug_pair_episodes.parquet", index=False)

        filtered_pairs, cotesting_pairs = self._cotesting_filter.filter(drug_pair_episodes)

        # ESKAPE family analysis pools organism variants only for the conditional
        # edge-estimation step. Keep filtered_pairs itself on the original organism
        # labels so downstream adjusted regression can merge episode covariates on
        # the configured episode key, which includes organism.
        family_name = request.target.display_name
        conditional_pairs = (
            filtered_pairs.assign(organism=family_name)
            if "organism" in filtered_pairs.columns
            else filtered_pairs
        )

        conditional_probabilities = self._conditional_analyzer.analyze(conditional_pairs)
        escalation_results = self._escalation_analyzer.analyze(conditional_probabilities)
        adjusted_results = self._regression_analyzer._empty_results()
        retained_edges = self._retained_edge_analyzer.analyze(escalation_results, adjusted_results)
        validation_results = self._validation_analyzer.analyze(filtered_pairs, retained_edges)
        validated_edge_results = self._validated_edge_results(
            escalation_results,
            validation_results,
        )
        adjusted_results = self._regression_analyzer.analyze(
            filtered_pairs,
            validated_edge_results,
            culture_episodes,
        )
        retained_edges = self._retained_edge_analyzer.analyze(escalation_results, adjusted_results)
        dependence_results = self._pair_dependence_analyzer.analyze(filtered_pairs, retained_edges)
        rule_concordance = self._guideline_concordance_analyzer.analyze(filtered_pairs, retained_edges)

        outputs = self._writer.write(
            conditional_probabilities=conditional_probabilities,
            escalation_results=escalation_results,
            adjusted_results=adjusted_results,
            retained_edges=retained_edges,
            validation_results=validation_results,
            dependence_results=dependence_results,
            rule_concordance=rule_concordance,
            cotesting_pairs=cotesting_pairs,
            output_dir=output_dir,
        )
        outputs.update(
            self._report_builder.export(
                retained_edges=retained_edges,
                adjusted_results=adjusted_results,
                validation_results=validation_results,
                dependence_results=dependence_results,
                output_dir=output_dir,
            )
        )
        outputs.update(
            self._write_existence_summary(
                dataset="armd",
                target=request.target,
                output_dir=output_dir,
                retained_edges=retained_edges,
                validation_results=validation_results,
                filtered_rows=len(filtered),
                culture_episode_rows=len(culture_episodes),
                drug_pair_rows=len(drug_pair_episodes),
                source_scope=request.source_scope,
                site=request.site,
            )
        )
        outputs["filtered_input"] = filter_path
        outputs["matched_labels"] = matched_labels_path
        return outputs

    def _load_cohort(self, request: ArmdEskapeRequest) -> pd.DataFrame:
        if request.source_scope == "combined":
            cohort_path = self._paths.paths.harmonized / "combined" / "cohort.parquet"
        elif request.source_scope == "site" and request.site:
            cohort_path = self._paths.paths.harmonized / "site_aligned" / request.site / "cohort.parquet"
        else:
            raise ValueError("ARMD ESKAPE validation requires source_scope='combined' or source_scope='site' with site set.")
        return self._dataset_store.read_pandas(cohort_path)

    def _filter_cohort(self, cohort: pd.DataFrame, target: EskapeTarget) -> pd.DataFrame:
        normalized = cohort["organism"].map(normalize_label)
        mask = normalized.str.contains(target.armd_match_pattern, regex=True, na=False)
        filtered = cohort.loc[mask].copy()
        filtered["eskape_target"] = target.display_name
        filtered["eskape_focus"] = target.scientific_focus
        return filtered.reset_index(drop=True)

    def _resolve_output_dir(self, request: ArmdEskapeRequest) -> Path:
        root = Path(request.output_root)
        scope_dir = "combined" if request.source_scope == "combined" else str(request.site)
        return root / scope_dir / "organisms" / safe_feature_name(request.target.display_name)

    @staticmethod
    def _validated_edge_results(
        escalation_results: pd.DataFrame,
        validation_results: pd.DataFrame,
    ) -> pd.DataFrame:
        keys = ["upstream_antibiotic", "downstream_antibiotic"]
        if escalation_results.empty:
            return escalation_results
        if validation_results.empty:
            return escalation_results.head(0).copy()
        if not set(keys + ["validation_status"]).issubset(validation_results.columns):
            return escalation_results.head(0).copy()
        validated = validation_results.loc[
            validation_results["validation_status"].isin(["robust", "supported"]),
            keys,
        ].drop_duplicates()
        if validated.empty:
            return escalation_results.head(0).copy()
        return escalation_results.merge(validated, on=keys, how="inner")

    def _write_existence_summary(
        self,
        dataset: str,
        target: EskapeTarget,
        output_dir: Path,
        retained_edges: pd.DataFrame,
        validation_results: pd.DataFrame,
        filtered_rows: int,
        culture_episode_rows: int,
        drug_pair_rows: int,
        source_scope: str,
        site: str | None,
    ) -> dict[str, Path]:
        status_series = validation_results.get("validation_status", pd.Series(dtype="string"))
        robust = int((status_series == "robust").sum())
        supported = int((status_series == "supported").sum())
        mixed = int((status_series == "mixed").sum())
        insufficient = int((status_series == "insufficient").sum())
        summary = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "source_scope": source_scope,
                    "site": site or "combined",
                    "eskape_target": target.display_name,
                    "scientific_focus": target.scientific_focus,
                    "filtered_rows": int(filtered_rows),
                    "culture_episode_rows": int(culture_episode_rows),
                    "drug_pair_rows": int(drug_pair_rows),
                    "retained_edges": int(len(retained_edges)),
                    "robust_edges": robust,
                    "supported_edges": supported,
                    "mixed_edges": mixed,
                    "insufficient_edges": insufficient,
                    "validated_edges": robust + supported,
                    "cascade_exists": bool((robust + supported) > 0),
                    "decision_rule": (
                        "Validated robust/supported edges present"
                        if (robust + supported) > 0
                        else "No validated robust/supported edges present"
                    ),
                }
            ]
        )
        csv_path = output_dir / "eskape_existence_summary.csv"
        json_path = output_dir / "eskape_existence_summary.json"
        summary.to_csv(csv_path, index=False)
        json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
        return {
            "existence_summary_csv": csv_path,
            "existence_summary_json": json_path,
        }
