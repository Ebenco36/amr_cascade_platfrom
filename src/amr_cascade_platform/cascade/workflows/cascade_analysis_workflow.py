"""Primary cascade analysis workflow."""

from __future__ import annotations

import gc
import logging
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _log_stage(label: str, start_time: float) -> None:
    if not os.environ.get("AMR_CASCADE_STAGE_PROFILE"):
        return
    # ru_maxrss is bytes on Darwin, KB on Linux.
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    elapsed = time.monotonic() - start_time
    print(f"[stage-profile] {label}: peak_rss={peak_rss_mb:.0f}MB elapsed={elapsed:.1f}s", flush=True)

from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import ConditionalProbabilityAnalyzer
from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.cascade.analyzers.cascade_pair_dependence_analyzer import CascadePairDependenceAnalyzer
from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import EscalationRatioAnalyzer
from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
from amr_cascade_platform.cascade.analyzers.guideline_concordance_analyzer import GuidelineConcordanceAnalyzer
from amr_cascade_platform.cascade.outputs.cascade_comparison_builder import CascadeComparisonBuilder
from amr_cascade_platform.cascade.outputs.cascade_atlas_builder import CascadeAtlasBuilder
from amr_cascade_platform.cascade.outputs.cascade_report_builder import CascadeReportBuilder
from amr_cascade_platform.cascade.outputs.network_exporter import NetworkExporter
from amr_cascade_platform.cascade.outputs.pathway_flow_builder import PathwayFlowBuilder
from amr_cascade_platform.cascade.outputs.threshold_sensitivity_builder import ThresholdSensitivityBuilder
from amr_cascade_platform.cascade.outputs.cascade_result_writer import CascadeResultWriter
from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.visualization.plotting.network_plotter import NetworkPlotter


@dataclass(frozen=True)
class CascadeAnalysisRequest:
    gold_scope: str = "combined"
    site: str | None = None
    organism: str | None = None
    ignore_precomputed_validation: bool = False


# drug_pair_episodes.parquet is an organism-level fan-out: every row belongs to one of a
# few hundred thousand episodes, repeated ~1000x for every ordered upstream/downstream
# antibiotic pair. That means anon_id/pat_enc_csn_id_coded/order_proc_id_coded/
# order_time_jittered are NOT "effectively unique per row" -- empirically verified via
# .nunique() on a real sample, each repeats ~970-1000x, same order as the antibiotic
# columns. At sample scale these 4 columns alone accounted for ~98% of this table's
# memory as object dtype (~1.1GB of ~1.14GB); dictionary-encoding them is the dominant
# lever for this table's footprint, well beyond the antibiotic/site columns below.
_PAIR_TABLE_CATEGORICAL_COLUMNS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "order_time_jittered",
    "organism",
    "source_site",
    "upstream_antibiotic",
    "upstream_susceptibility",
    "downstream_antibiotic",
    "pair_direction",
]


class CascadeAnalysisWorkflow:
    """Run primary cascade analyses from gold datasets."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)
        self._cotesting_filter = CoTestingFilterAnalyzer(settings)
        self._conditional_analyzer = ConditionalProbabilityAnalyzer(settings)
        self._escalation_analyzer = EscalationRatioAnalyzer(settings)
        self._regression_analyzer = DownstreamTestingRegression(settings, path_manager)
        self._retained_edge_analyzer = RetainedEdgeAnalyzer(settings)
        self._guideline_concordance_analyzer = GuidelineConcordanceAnalyzer(settings, path_manager)
        self._validation_analyzer = CascadeValidationAnalyzer(settings)
        self._pair_dependence_analyzer = CascadePairDependenceAnalyzer(settings)
        self._writer = CascadeResultWriter(settings)
        self._report_builder = CascadeReportBuilder(settings.cascade.continuity_correction)
        self._network_exporter = NetworkExporter()
        self._network_plotter = NetworkPlotter()
        self._pathway_flow_builder = PathwayFlowBuilder()
        self._threshold_sensitivity_builder = ThresholdSensitivityBuilder(settings)
        self._comparison_builder = CascadeComparisonBuilder()
        self._atlas_builder = CascadeAtlasBuilder()

    def run(self, request: CascadeAnalysisRequest) -> dict[str, Path]:
        start_time = time.monotonic()
        gold_dir = self._resolve_gold_dir(request)
        pair_path = gold_dir / "drug_pair_episodes.parquet"
        if not pair_path.exists():
            raise DataDiscoveryError(f"Gold drug-pair dataset not found: {pair_path}")
        culture_episode_path = gold_dir / "culture_episodes.parquet"
        if not culture_episode_path.exists():
            raise DataDiscoveryError(f"Gold culture-episode dataset not found: {culture_episode_path}")
        eligible_pairs_path = gold_dir / "eligible_pairs.parquet"
        if not eligible_pairs_path.exists():
            raise DataDiscoveryError(f"Gold eligible-pairs dataset not found: {eligible_pairs_path}")

        drug_pairs = self._dataset_store.read_pandas(
            pair_path, categorical_columns=_PAIR_TABLE_CATEGORICAL_COLUMNS
        )
        culture_episodes = self._dataset_store.read_pandas(culture_episode_path)
        _log_stage("load drug_pairs + culture_episodes", start_time)
        # eligible_pairs is intentionally not loaded here: DownstreamTestingRegression.analyze()
        # never needed it -- it builds its own covariates from culture_episodes. Loading the
        # full multi-million-row file just to discard it would waste tens of GB of peak memory.
        filtered_pairs, cotesting_pairs = self._cotesting_filter.filter(drug_pairs)
        del drug_pairs
        gc.collect()
        _log_stage("cotesting_filter", start_time)
        conditional_probabilities = self._conditional_analyzer.analyze(filtered_pairs)
        _log_stage("conditional_probability_analyzer", start_time)
        escalation_results = self._escalation_analyzer.analyze(conditional_probabilities)
        _log_stage("escalation_ratio_analyzer", start_time)
        # Retention is intentionally based on the unadjusted observation pattern.
        # The adjusted model is a post-validation directional coherence check, not
        # a gate into the validation set.
        adjusted_results = self._regression_analyzer._empty_results()
        retained_edges = self._retained_edge_analyzer.analyze(escalation_results, adjusted_results)
        _log_stage("retained_edge_analyzer", start_time)
        del culture_episodes
        gc.collect()

        output_dir = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope=request.gold_scope,
            site=request.site,
            organism=request.organism,
        )

        # Load pre-computed validation results if available (written by
        # merge_cascade_validation_shards.py after all SLURM shard jobs complete).
        # This lets the main cascade job skip the ~800-hour validation loop when
        # sharded parallelism was used.
        precomputed_validation_path = output_dir / "validation_results.parquet"
        if precomputed_validation_path.exists() and not request.ignore_precomputed_validation:
            logger.info(
                "Loading pre-computed validation results from shard merge: %s",
                precomputed_validation_path,
            )
            validation_results = self._dataset_store.read_pandas(precomputed_validation_path)
            self._validate_precomputed_validation_matches_retained_edges(
                validation_results=validation_results,
                retained_edges=retained_edges,
                validation_path=precomputed_validation_path,
            )
        else:
            validation_results = self._validation_analyzer.analyze(filtered_pairs, retained_edges)
        _log_stage("cascade_validation_analyzer", start_time)

        validated_edge_results = self._validated_edge_results(
            escalation_results,
            validation_results,
        )
        culture_episodes = self._dataset_store.read_pandas(culture_episode_path)
        adjusted_results = self._regression_analyzer.analyze(
            filtered_pairs,
            validated_edge_results,
            culture_episodes,
        )
        _log_stage("downstream_testing_regression", start_time)
        del culture_episodes
        gc.collect()
        retained_edges = self._retained_edge_analyzer.analyze(escalation_results, adjusted_results)
        _log_stage("retained_edge_analyzer_adjusted_annotation", start_time)

        dependence_results = self._pair_dependence_analyzer.analyze(filtered_pairs, retained_edges)
        _log_stage("cascade_pair_dependence_analyzer", start_time)
        rule_concordance = self._guideline_concordance_analyzer.analyze(filtered_pairs, retained_edges)
        _log_stage("guideline_concordance_analyzer", start_time)
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
        _log_stage("result_writer", start_time)
        report_outputs = self._report_builder.export(
            retained_edges=retained_edges,
            adjusted_results=adjusted_results,
            validation_results=validation_results,
            dependence_results=dependence_results,
            escalation_results=escalation_results,
            output_dir=output_dir,
        )
        outputs.update(report_outputs)
        _log_stage("report_builder", start_time)
        edge_report = self._dataset_store.read_pandas(report_outputs["edge_report_path"])
        validated_edges_for_primary_outputs = self._validated_edge_report_rows(edge_report)
        outputs.update(
            self._atlas_builder.export(
                edge_report=edge_report,
                output_dir=output_dir,
                scope=request.gold_scope,
                site=request.site,
                organism=request.organism,
            )
        )
        _log_stage("atlas_builder", start_time)
        top_paths = self._dataset_store.read_pandas(report_outputs["top_paths_path"])
        outputs.update(
            self._pathway_flow_builder.export(
                top_paths=top_paths,
                retained_edges=retained_edges,
                output_dir=output_dir,
                top_n=self._settings.cascade.pathway_top_n,
            )
        )
        _log_stage("pathway_flow_builder", start_time)
        outputs.update(
            self._threshold_sensitivity_builder.export(
                escalation_results=escalation_results,
                adjusted_results=adjusted_results,
                output_dir=output_dir,
            )
        )
        _log_stage("threshold_sensitivity_builder", start_time)
        outputs.update(self._network_exporter.export(validated_edges_for_primary_outputs, output_dir))
        _log_stage("network_exporter", start_time)
        outputs.update(self._network_plotter.plot(validated_edges_for_primary_outputs, output_dir))
        _log_stage("network_plotter", start_time)
        if request.gold_scope == "combined" and request.site is None:
            outputs.update(
                self._comparison_builder.export(
                    artifact_root=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
                    scope_name=output_dir.relative_to(
                        self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir
                    ).as_posix(),
                )
            )
        return outputs

    def _resolve_gold_dir(self, request: CascadeAnalysisRequest) -> Path:
        if request.gold_scope not in {"combined", "site"} or (request.gold_scope == "site" and not request.site):
            raise DataDiscoveryError("Cascade analysis requires gold_scope='combined' or gold_scope='site' with site set.")
        return scoped_output_dir(
            root=self._paths.paths.gold,
            scope=request.gold_scope,
            site=request.site,
            organism=request.organism,
        )

    @staticmethod
    def _validated_edge_results(escalation_results, validation_results):
        """Restrict adjusted modeling to robust/supported validated pairs.

        DownstreamTestingRegression uses the rows it receives as its model
        universe. Passing all support-screened edges here would make RQ2 partly
        exploratory. Passing only robust/supported validation results makes RQ2
        exactly: among validated cascade edges, do adjusted associations remain
        directionally coherent?
        """
        keys = ["upstream_antibiotic", "downstream_antibiotic"]
        required_validation = set(keys + ["validation_status"])
        required_escalation = set(keys)
        if escalation_results is None or escalation_results.empty:
            return escalation_results
        if validation_results is None or validation_results.empty:
            return escalation_results.head(0).copy()
        if not required_validation.issubset(validation_results.columns):
            return escalation_results.head(0).copy()
        if not required_escalation.issubset(escalation_results.columns):
            return escalation_results.head(0).copy()

        validated = validation_results.loc[
            validation_results["validation_status"].isin(CascadeValidationAnalyzer.VALIDATED_STATUSES),
            keys,
        ].drop_duplicates()
        if validated.empty:
            return escalation_results.head(0).copy()
        return escalation_results.merge(validated, on=keys, how="inner")

    @staticmethod
    def _validated_edge_report_rows(edge_report):
        """Return robust/supported rows for primary network/pathway artifacts."""
        if edge_report is None or edge_report.empty or "validation_status" not in edge_report.columns:
            return edge_report.head(0).copy() if edge_report is not None else edge_report
        return edge_report.loc[
            edge_report["validation_status"].isin(CascadeValidationAnalyzer.VALIDATED_STATUSES)
        ].reset_index(drop=True)

    @staticmethod
    def _validate_precomputed_validation_matches_retained_edges(
        *,
        validation_results,
        retained_edges,
        validation_path: Path,
    ) -> None:
        """Fail closed when a shard-merged validation file is stale or mismatched."""
        keys = ["upstream_antibiotic", "downstream_antibiotic"]
        if validation_results is None or validation_results.empty:
            if retained_edges.empty:
                return
            raise ValueError(
                f"Precomputed validation file is empty but {len(retained_edges)} retained edges "
                f"need validation: {validation_path}"
            )
        required = set(keys + ["validation_status"])
        missing = required - set(validation_results.columns)
        if missing:
            raise ValueError(
                f"Precomputed validation file is missing required columns {sorted(missing)}: "
                f"{validation_path}"
            )
        retained_keys = retained_edges.loc[:, keys].drop_duplicates()
        validation_keys = validation_results.loc[:, keys].drop_duplicates()
        merged = retained_keys.merge(validation_keys, on=keys, how="outer", indicator=True)
        mismatch = merged[merged["_merge"] != "both"]
        if not mismatch.empty:
            examples = mismatch.head(10).to_dict(orient="records")
            raise ValueError(
                "Precomputed validation results do not match the current retained-edge set. "
                "This usually means validation_results.parquet is stale relative to the current "
                "code/config/gold artifacts. Delete and regenerate validation shards before "
                f"running cascade analysis. Path: {validation_path}. "
                f"Retained edges={len(retained_keys)}, validation edges={len(validation_keys)}, "
                f"mismatched keys={len(mismatch)}. Examples: {examples}"
            )
