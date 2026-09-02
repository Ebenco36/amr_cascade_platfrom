"""Export manuscript-ready tables and figures."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.cascade.outputs.cascade_report_builder import CascadeReportBuilder
from amr_cascade_platform.cascade.outputs.network_exporter import NetworkExporter
from amr_cascade_platform.cascade.outputs.pathway_flow_builder import PathwayFlowBuilder
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.reporting.builders.manuscript_table_builder import ManuscriptTableBuilder
from amr_cascade_platform.surveillance.workflows.prevalence_shift_workflow import PrevalenceShiftWorkflow
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.figure_manager import FigureManager
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


@dataclass(frozen=True)
class ReportExportRequest:
    scope: str = "combined"
    site: str | None = None
    organism: str | None = None
    figure_formats: tuple[str, ...] | None = None
    figures: tuple[str, ...] | None = None
    fail_on_missing_figures: bool = False


class ReportExportWorkflow:
    """Create manuscript tables and figures in outputs/ for a given scope."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._table_builder = ManuscriptTableBuilder(settings, path_manager)
        self._figure_manager = FigureManager(settings, path_manager)
        self._dataset_store = DatasetStore(settings)

    def run(self, request: ReportExportRequest) -> dict[str, Path]:
        scope_dir = scoped_output_dir(
            root=Path("."),
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        figure_formats = request.figure_formats or self._settings.reporting.default_figure_formats
        selected_figures = request.figures or self._settings.reporting.default_figures
        tables_dir = self._paths.project_root / self._settings.reporting.tables_dir / scope_dir
        figures_dir = self._paths.project_root / self._settings.reporting.figures_dir / scope_dir
        reports_dir = self._paths.project_root / self._settings.reporting.reports_dir / scope_dir
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        self._archive_existing_figure_exports(figures_dir)

        outputs: dict[str, Path] = {}
        table_a = self._table_builder.build_provenance_table(request.scope, request.site, request.organism)
        table_a_flow = self._table_builder.build_data_quality_flow_table(request.scope, request.site, request.organism)
        table_b = self._table_builder.build_eligibility_table(request.scope, request.site, request.organism)
        table_c = self._table_builder.build_primary_cascade_table(request.scope, request.site, request.organism)
        table_c_validated = self._table_builder.build_validated_primary_cascade_table(request.scope, request.site, request.organism)
        table_c_aware = self._table_builder.build_aware_transition_summary_table(request.scope, request.site, request.organism)
        table_c_aware_hypothesis = self._table_builder.build_aware_direction_hypothesis_table(request.scope, request.site, request.organism)
        table_c_asymmetry = self._table_builder.build_asymmetry_sensitivity_table(request.scope, request.site, request.organism)
        table_d_summary = self._table_builder.build_concordance_summary_table(request.scope, request.site, request.organism)
        table_d_examples = self._table_builder.build_concordance_examples_table(request.scope, request.site, request.organism)
        table_e = self._table_builder.build_threshold_sensitivity_table(request.scope, request.site, request.organism)
        table_f = self._table_builder.build_pathway_flow_table(request.scope, request.site, request.organism)
        comparison = self._table_builder.build_site_vs_combined_comparison(request.scope, request.site, request.organism)
        comparison_summary = self._table_builder.build_site_vs_combined_summary(request.scope, request.site, request.organism)
        implied_delta = self._table_builder.build_implied_delta_table(request.scope, request.site, request.organism)
        cascade_existence = self._table_builder.build_cascade_existence_summary(request.scope, request.site, request.organism)
        downstream_trigger_availability = self._table_builder.build_downstream_trigger_availability_table(
            request.scope, request.site, request.organism
        )
        upstream_selection_balance = self._table_builder.build_upstream_selection_balance_table(
            request.scope, request.site, request.organism
        )
        covariate_correlation = self._table_builder.build_covariate_correlation_table(
            request.scope, request.site, request.organism
        )
        prevalence_shift_table = self._table_builder.build_prevalence_shift_table(
            request.scope, request.site, request.organism
        )
        prevalence_shift_diagnostics = self._table_builder.build_prevalence_shift_diagnostics_table(
            request.scope, request.site, request.organism
        )
        correction_sensitivity = self._table_builder.build_correction_sensitivity_table(
            request.scope, request.site, request.organism
        )

        table_map = {
            "table_a_provenance.csv": table_a,
            "table_a_data_quality_flow.csv": table_a_flow,
            "table_b_eligibility.csv": table_b,
            "table_c_primary_cascade.csv": table_c,
            "table_l_validated_primary_cascade.csv": table_c_validated,
            "table_m_aware_transition_summary.csv": table_c_aware,
            "table_n_aware_direction_hypothesis.csv": table_c_aware_hypothesis,
            "table_o_asymmetry_sensitivity.csv": table_c_asymmetry,
            "table_d_concordance_summary.csv": table_d_summary,
            "table_d_concordance_examples.csv": table_d_examples,
            "table_e_threshold_sensitivity.csv": table_e,
            "table_f_pathway_flows.csv": table_f,
            "site_vs_combined_edge_comparison.csv": comparison,
            "site_vs_combined_summary.csv": comparison_summary,
        }
        if not implied_delta.empty:
            table_map["table_g_implied_delta.csv"] = implied_delta
        if not cascade_existence.empty:
            table_map["table_p_cascade_existence_summary.csv"] = cascade_existence
        if not downstream_trigger_availability.empty:
            table_map["table_j_downstream_trigger_availability.csv"] = downstream_trigger_availability
        if not upstream_selection_balance.empty:
            table_map["table_q_upstream_selection_balance.csv"] = upstream_selection_balance
        if not covariate_correlation.empty:
            table_map["table_s_covariate_correlation.csv"] = covariate_correlation
        if not prevalence_shift_table.empty:
            table_map["table_k_prevalence_shift.csv"] = prevalence_shift_table
        if not prevalence_shift_diagnostics.empty:
            table_map["table_k_prevalence_shift_diagnostics.csv"] = prevalence_shift_diagnostics
        if not correction_sensitivity.empty:
            table_map["table_r_correction_sensitivity.csv"] = correction_sensitivity
        required_schema_tables = {
            "table_c_primary_cascade.csv",
            "table_l_validated_primary_cascade.csv",
            "table_k_prevalence_shift.csv",
        }
        for filename, dataframe in table_map.items():
            path = tables_dir / filename
            self._write_table_if_nonempty(
                dataframe,
                path,
                outputs,
                write_empty_schema=filename in required_schema_tables,
            )

        scope_artifact_dir = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        prevalence_artifact_dir = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.prevalence.output_dir,
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        edge_report = self._dataset_store.read_pandas(scope_artifact_dir / "edge_report.parquet")
        escalation_results = self._dataset_store.read_pandas(scope_artifact_dir / "escalation_results.parquet")
        threshold_sensitivity = self._dataset_store.read_pandas(scope_artifact_dir / "threshold_sensitivity.parquet")
        pathway_flows = self._dataset_store.read_pandas(scope_artifact_dir / "pathway_flows.parquet")
        retained_edges = self._dataset_store.read_pandas(scope_artifact_dir / "retained_edges.parquet")
        validation_results = self._dataset_store.read_pandas(scope_artifact_dir / "validation_results.parquet")
        network_nodes = self._dataset_store.read_pandas(scope_artifact_dir / "network_nodes.parquet")
        network_edges = self._dataset_store.read_pandas(scope_artifact_dir / "network_edges.parquet")
        validated_edge_report = self._validated_edge_report(edge_report)
        prevalence_shift = self._dataset_store.read_pandas(prevalence_artifact_dir / "prevalence_shift.parquet")
        mnar_curve_path = prevalence_artifact_dir / "prevalence_mnar_sensitivity_curves.parquet"
        prevalence_mnar_curves = (
            self._dataset_store.read_pandas(mnar_curve_path) if mnar_curve_path.exists() else pd.DataFrame()
        )
        tipping_point_path = prevalence_artifact_dir / "prevalence_mnar_tipping_points.parquet"
        prevalence_mnar_tipping_points = (
            self._dataset_store.read_pandas(tipping_point_path) if tipping_point_path.exists() else pd.DataFrame()
        )
        if not prevalence_mnar_curves.empty:
            path = tables_dir / "table_k_mnar_sensitivity_curve_points.csv"
            PrevalenceShiftWorkflow._format_mnar_curve_table(prevalence_mnar_curves).to_csv(path, index=False)
            outputs[path.name] = path
        if not prevalence_mnar_tipping_points.empty:
            path = tables_dir / "table_k_mnar_tipping_points.csv"
            PrevalenceShiftWorkflow._format_mnar_tipping_point_table(prevalence_mnar_tipping_points).to_csv(path, index=False)
            outputs[path.name] = path

        if not retained_edges.empty:
            classification = AntibioticClassificationResolver(
                self._paths.project_root / "data" / "antibiotic_classification_complete.csv"
            )
            antibiotic_labels: list[str] = []
            for col in ("upstream_antibiotic", "downstream_antibiotic"):
                if col in retained_edges.columns:
                    antibiotic_labels.extend(retained_edges[col].dropna().unique().tolist())
            if edge_report is not None and not edge_report.empty:
                for col in ("upstream_antibiotic", "downstream_antibiotic"):
                    if col in edge_report.columns:
                        antibiotic_labels.extend(edge_report[col].dropna().unique().tolist())
            antibiotic_labels = list(dict.fromkeys(antibiotic_labels))
            abbrev_table = classification.abbreviation_table(antibiotic_labels)
            if not abbrev_table.empty:
                abbrev_path = tables_dir / "table_supplementary_antibiotic_abbreviations.csv"
                abbrev_table.to_csv(abbrev_path, index=False)
                outputs[abbrev_path.name] = abbrev_path
        modeling_dir = self._paths.paths.artifacts / self._settings.modeling.output_dir / self._settings.modeling.task_name
        if request.organism:
            modeling_dir = scoped_output_dir(modeling_dir, request.scope, site=request.site, organism=request.organism)
        model_metrics = None
        model_threshold_metrics = None
        model_predictions = None
        if modeling_dir.exists():
            metrics_path = modeling_dir / "metrics.parquet"
            threshold_path = modeling_dir / "threshold_metrics.parquet"
            if metrics_path.exists():
                model_metrics = self._dataset_store.read_pandas(metrics_path)
                path = tables_dir / "table_h_model_metrics.csv"
                model_metrics.to_csv(path, index=False)
                outputs[path.name] = path
            if threshold_path.exists():
                model_threshold_metrics = self._dataset_store.read_pandas(threshold_path)
                path = tables_dir / "table_i_model_threshold_metrics.csv"
                model_threshold_metrics.to_csv(path, index=False)
                outputs[path.name] = path
            prediction_frames = []
            for estimator in self._settings.modeling.estimators:
                prediction_path = modeling_dir / f"{estimator}_predictions.parquet"
                if prediction_path.exists():
                    prediction_frames.append(
                        self._dataset_store.read_pandas(
                            prediction_path,
                            columns=["model_name", "split", "target", "predicted_probability"],
                        )
                    )
            if prediction_frames:
                model_predictions = pd.concat(prediction_frames, ignore_index=True)

        figure_exports: dict[str, Path] = {}
        if "primary_cascade_forest" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_primary_forest(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_primary_cascade_forest",
                    formats=figure_formats,
                )
            )
        if "validated_primary_cascade_forest" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_validated_primary_forest(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_validated_primary_cascade_forest",
                    formats=figure_formats,
                )
            )
        if "aware_transition_heatmap" in selected_figures and not table_c_aware.empty:
            figure_exports.update(
                self._figure_manager.export_aware_transition_heatmap(
                    transition_summary=table_c_aware,
                    output_stem=figures_dir / "figure_aware_transition_heatmap",
                    formats=figure_formats,
                )
            )
        if "prevalence_shift_forest" in selected_figures and not prevalence_shift.empty:
            sort_columns = ["mnar_lambda0_absolute_shift_pct", "eligible_n", "tested_n"]
            plot_prevalence_shift = (
                prevalence_shift.sort_values(
                    by=sort_columns,
                    ascending=[False, False, False],
                ).head(self._settings.prevalence.default_top_n)
                if set(sort_columns).issubset(prevalence_shift.columns)
                else prevalence_shift.head(0)
            )
            figure_exports.update(
                self._figure_manager.export_prevalence_shift_forest(
                    results=plot_prevalence_shift,
                    output_stem=figures_dir / "figure_prevalence_shift_forest",
                    formats=figure_formats,
                )
            )
        if "prevalence_shift_curves" in selected_figures and not prevalence_mnar_curves.empty:
            figure_exports.update(
                self._figure_manager.export_prevalence_delta_curves(
                    curves=prevalence_mnar_curves,
                    prevalence_summary=prevalence_shift,
                    output_stem=figures_dir / "figure_prevalence_mnar_sensitivity_curves",
                    formats=figure_formats,
                    organism=request.organism or "",
                )
            )
        if (
            "mnar_tipping_point" in selected_figures
            and not prevalence_mnar_curves.empty
            and not prevalence_mnar_tipping_points.empty
        ):
            figure_exports.update(
                self._figure_manager.export_mnar_tipping_point(
                    curves=prevalence_mnar_curves,
                    tipping_points=prevalence_mnar_tipping_points,
                    prevalence_summary=prevalence_shift,
                    output_stem=figures_dir / "figure_mnar_tipping_point",
                    formats=figure_formats,
                    organism=request.organism or "",
                )
            )
        if any(key in selected_figures for key in ("upstream_contribution_forest", "downstream_trigger_forest")) and not validated_edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_all_downstream_trigger_forests(
                    escalation_results=validated_edge_report,
                    output_dir=figures_dir,
                    formats=figure_formats,
                )
            )
        if "threshold_sensitivity" in selected_figures and not threshold_sensitivity.empty:
            figure_exports.update(
                self._figure_manager.export_threshold_sensitivity(
                    sensitivity_table=threshold_sensitivity,
                    output_stem=figures_dir / "figure_threshold_sensitivity",
                    formats=figure_formats,
                )
            )
        if "site_vs_combined_summary" in selected_figures and not comparison_summary.empty:
            figure_exports.update(
                self._figure_manager.export_site_comparison_summary(
                    comparison_summary=comparison_summary,
                    output_stem=figures_dir / "figure_site_vs_combined_summary",
                    formats=figure_formats,
                )
            )
        if "cascade_sankey" in selected_figures and not pathway_flows.empty:
            tiered_flows = self._tiered_pathway_flows(pathway_flows, validation_results)
            for tier, (suffix, label) in (
                ("robust", ("robust", "Robust")),
                ("supported", ("supported", "Supported")),
                ("validated", ("validated", "Robust + Supported")),
            ):
                figure_exports.update(
                    self._figure_manager.export_sankey(
                        pathway_flows=tiered_flows[tier],
                        output_stem=figures_dir / f"figure_cascade_sankey_{suffix}",
                        formats=figure_formats,
                        tier_label=label,
                    )
                )
            if not edge_report.empty:
                figure_exports.update(
                    self._figure_manager.export_sankey(
                        pathway_flows=self._adjusted_or_confirmed_pathway_flows(
                            edge_report, self._settings.cascade.pathway_top_n
                        ),
                        output_stem=figures_dir / "figure_cascade_sankey_adjusted_or_confirmed",
                        formats=figure_formats,
                        tier_label="Adjusted-OR Confirmed",
                    )
                )
        if "cascade_chord" in selected_figures and not retained_edges.empty:
            tiered_edges = self._tiered_retained_edges(retained_edges, validation_results)
            for tier, (suffix, label) in (
                ("robust", ("robust", "Robust")),
                ("supported", ("supported", "Supported")),
                ("validated", ("validated", "Robust + Supported")),
            ):
                figure_exports.update(
                    self._figure_manager.export_chord(
                        retained_edges=tiered_edges[tier],
                        output_stem=figures_dir / f"figure_cascade_chord_{suffix}",
                        formats=figure_formats,
                        tier_label=label,
                    )
                )
        if "cascade_network" in selected_figures and not network_edges.empty and not network_nodes.empty:
            # Node-level topology (pagerank, betweenness) is graph-dependent: filtering the
            # already-computed network_edges/network_nodes by tier would leave pagerank scores
            # from the full 858-edge graph attached to a much smaller robust/supported subgraph,
            # which is not a meaningful quantity. Recompute the graph (and its topology metrics)
            # fresh for each tier's own edge set instead of just filtering the precomputed output.
            tiered_edges = self._tiered_retained_edges(retained_edges, validation_results)
            for tier, (suffix, label) in (
                ("robust", ("robust", "Robust")),
                ("supported", ("supported", "Supported")),
                ("validated", ("validated", "Robust + Supported")),
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    tier_paths = NetworkExporter().export(tiered_edges[tier], Path(tmp))
                    tier_network_edges = self._dataset_store.read_pandas(tier_paths["edge_path"])
                    tier_network_nodes = self._dataset_store.read_pandas(tier_paths["node_path"])
                figure_exports.update(
                    self._figure_manager.export_network(
                        network_edges=tier_network_edges,
                        network_nodes=tier_network_nodes,
                        output_stem=figures_dir / f"figure_cascade_network_{suffix}",
                        formats=figure_formats,
                        tier_label=label,
                    )
                )
        if "model_metrics_comparison" in selected_figures and model_metrics is not None and not model_metrics.empty:
            figure_exports.update(
                self._figure_manager.export_model_metrics_comparison(
                    metrics=model_metrics,
                    output_stem=figures_dir / "figure_model_metrics_comparison",
                    formats=figure_formats,
                )
            )
        if "model_pr_curve" in selected_figures and model_predictions is not None and not model_predictions.empty:
            figure_exports.update(
                self._figure_manager.export_model_precision_recall(
                    predictions=model_predictions,
                    output_stem=figures_dir / "figure_model_precision_recall",
                    formats=figure_formats,
                )
            )
        if "model_roc_curve" in selected_figures and model_predictions is not None and not model_predictions.empty:
            figure_exports.update(
                self._figure_manager.export_model_roc(
                    predictions=model_predictions,
                    output_stem=figures_dir / "figure_model_roc",
                    formats=figure_formats,
                )
            )
        if "model_calibration" in selected_figures and model_predictions is not None and not model_predictions.empty:
            figure_exports.update(
                self._figure_manager.export_model_calibration(
                    predictions=model_predictions,
                    output_stem=figures_dir / "figure_model_calibration",
                    formats=figure_formats,
                )
            )
        if "model_threshold_analysis" in selected_figures and model_threshold_metrics is not None and not model_threshold_metrics.empty:
            figure_exports.update(
                self._figure_manager.export_model_threshold_analysis(
                    threshold_metrics=model_threshold_metrics,
                    output_stem=figures_dir / "figure_model_threshold_analysis",
                    formats=figure_formats,
                )
            )
        if "validation_diagnostics" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_validation_diagnostics(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_validation_diagnostics",
                    formats=figure_formats,
                )
            )
        if "er_landscape" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_er_landscape(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_er_landscape",
                    formats=figure_formats,
                )
            )
        if "mnar_prevalence_shift_distribution" in selected_figures and not prevalence_shift.empty:
            figure_exports.update(
                self._figure_manager.export_mnar_prevalence_shift_distribution(
                    prevalence_shift=prevalence_shift,
                    output_stem=figures_dir / "figure_mnar_prevalence_shift_distribution",
                    formats=figure_formats,
                )
            )
        if "validation_funnel" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_validation_funnel(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_validation_funnel",
                    formats=figure_formats,
                )
            )
        if "panel_bundling" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_panel_bundling(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_panel_bundling",
                    formats=figure_formats,
                )
            )
        if "temporal_stability" in selected_figures and not edge_report.empty:
            figure_exports.update(
                self._figure_manager.export_temporal_stability(
                    edge_report=edge_report,
                    output_stem=figures_dir / "figure_temporal_stability",
                    formats=figure_formats,
                )
            )
        if "cross_site_concordance" in selected_figures and not comparison_summary.empty:
            figure_exports.update(
                self._figure_manager.export_cross_site_concordance(
                    comparison_summary=comparison_summary,
                    output_stem=figures_dir / "figure_cross_site_concordance",
                    formats=figure_formats,
                )
            )
        if "consort_diagram" in selected_figures:
            figure_exports.update(
                self._figure_manager.export_consort_diagram(
                    flow_table=table_a_flow,
                    edge_report=edge_report if not edge_report.empty else pd.DataFrame(),
                    output_stem=figures_dir / "figure_consort_diagram",
                    formats=figure_formats,
                )
            )
        if "dataset_characterization" in selected_figures:
            gold_dir = scoped_output_dir(
                root=self._paths.paths.gold,
                scope=request.scope,
                site=request.site,
                organism=request.organism,
            )
            eligible_pairs = self._dataset_store.read_pandas(gold_dir / "eligible_pairs.parquet")
            culture_episodes = self._dataset_store.read_pandas(gold_dir / "culture_episodes.parquet")
            drug_pair_episodes = self._dataset_store.read_pandas(gold_dir / "drug_pair_episodes.parquet")
            char_dir = figures_dir / "dataset_characterization"
            char_dir.mkdir(parents=True, exist_ok=True)
            figure_exports.update(
                self._figure_manager.export_dataset_characterization(
                    eligible_pairs=eligible_pairs if eligible_pairs is not None else pd.DataFrame(),
                    culture_episodes=culture_episodes if culture_episodes is not None else pd.DataFrame(),
                    drug_pair_episodes=drug_pair_episodes if drug_pair_episodes is not None else pd.DataFrame(),
                    escalation_results=escalation_results if not escalation_results.empty else pd.DataFrame(),
                    upstream_balance_table=upstream_selection_balance if not upstream_selection_balance.empty else pd.DataFrame(),
                    output_dir=char_dir,
                    formats=figure_formats,
                )
            )
        outputs.update(figure_exports)
        missing_figure_groups = self._missing_requested_figure_groups(selected_figures, figure_exports)

        table_outputs = {
            name: str(path)
            for name, path in outputs.items()
            if path.parent == tables_dir
        }
        figure_outputs = {
            name: str(path)
            for name, path in outputs.items()
            if path.parent == figures_dir
        }

        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": request.scope,
            "site": request.site,
            "organism": request.organism,
            "figure_formats": list(figure_formats),
            "figures": list(selected_figures),
            "tables": table_outputs,
            "figure_exports": figure_outputs,
            "missing_figure_groups": missing_figure_groups,
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        summary_path = reports_dir / "report_manifest.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        outputs["report_manifest.json"] = summary_path
        if request.fail_on_missing_figures and missing_figure_groups:
            missing = ", ".join(
                f"{group} (expected prefix: {prefix})"
                for group, prefix in missing_figure_groups.items()
            )
            raise RuntimeError(
                "Requested figure group(s) were not generated. "
                "This usually means the required upstream artefact is missing or empty. "
                f"Missing: {missing}"
            )
        return outputs

    @staticmethod
    def _missing_requested_figure_groups(
        selected_figures: tuple[str, ...],
        figure_exports: dict[str, Path],
    ) -> dict[str, str]:
        """Return requested figure families that produced no current output file.

        Reporting can only draw figures from existing artefacts. A missing model
        prediction parquet, prevalence curve parquet, or cascade network parquet
        should be visible in the manifest instead of silently producing a partial
        publication folder.
        """

        expected_prefixes: dict[str, str | tuple[str, ...]] = {
            "primary_cascade_forest": "figure_primary_cascade_forest",
            "validated_primary_cascade_forest": "figure_validated_primary_cascade_forest",
            "aware_transition_heatmap": "figure_aware_transition_heatmap",
            "prevalence_shift_forest": "figure_prevalence_shift_forest",
            "prevalence_shift_curves": "figure_prevalence_mnar_sensitivity_curves",
            "mnar_tipping_point": "figure_mnar_tipping_point",
            "upstream_contribution_forest": "figure_downstream_trigger_forest__",
            "downstream_trigger_forest": "figure_downstream_trigger",
            "threshold_sensitivity": "figure_threshold_sensitivity",
            "site_vs_combined_summary": "figure_site_vs_combined_summary",
            "cascade_sankey": "figure_cascade_sankey",
            "cascade_chord": "figure_cascade_chord",
            "cascade_network": "figure_cascade_network",
            "model_metrics_comparison": "figure_model_metrics_comparison",
            "model_pr_curve": "figure_model_precision_recall",
            "model_roc_curve": "figure_model_roc",
            "model_calibration": "figure_model_calibration",
            "model_threshold_analysis": "figure_model_threshold_analysis",
            "validation_diagnostics": "figure_validation_diagnostics",
            "er_landscape": "figure_er_landscape",
            "mnar_prevalence_shift_distribution": "figure_mnar_prevalence_shift_distribution",
            "validation_funnel": "figure_validation_funnel",
            "panel_bundling": "figure_panel_bundling",
            "temporal_stability": "figure_temporal_stability",
            "cross_site_concordance": "figure_cross_site_concordance",
            "consort_diagram": "figure_consort_diagram",
            "dataset_characterization": (
                "dataset_eligible_vs_observed",
                "dataset_observation_rate_heatmap",
                "dataset_testing_rate_by_upstream_result",
                "dataset_selection_imbalance_love_plot",
                "dataset_episode_temporal_heatmap",
                "dataset_covariate_comparison",
                "dataset_pair_support_landscape",
                "dataset_missing_data_profile",
            ),
        }
        exported_names = set(figure_exports)
        exported_paths = {str(path.name) for path in figure_exports.values()}
        exported = exported_names | exported_paths
        missing: dict[str, str] = {}
        for group in selected_figures:
            prefix = expected_prefixes.get(group)
            if prefix is None:
                continue
            prefixes = (prefix,) if isinstance(prefix, str) else prefix
            missing_prefixes = [
                expected_prefix
                for expected_prefix in prefixes
                if not any(name.startswith(expected_prefix) for name in exported)
            ]
            if missing_prefixes:
                missing[group] = ", ".join(missing_prefixes)
        return missing

    @staticmethod
    def _validated_edge_report(edge_report: pd.DataFrame) -> pd.DataFrame:
        """Return publication-facing validated edges for downstream trigger forests."""

        if edge_report.empty or "validation_status" not in edge_report.columns:
            return edge_report.iloc[0:0].copy()
        return edge_report.loc[
            edge_report["validation_status"].astype("string").str.lower().isin({"robust", "supported"})
        ].copy()

    @staticmethod
    def _archive_existing_figure_exports(figures_dir: Path) -> None:
        """Move previous top-level figure exports out of the publication surface.

        The report manifest is a contract for the current reporting run. Leaving
        old figure files beside current exports makes the publication package
        ambiguous, so previous top-level figure artifacts are moved to a stale
        archive before new figures are written. Subdirectories are left alone.
        """

        archive_dir = figures_dir / "_stale_figure_archive"
        stale_candidates = [
            path
            for suffix in (".html", ".png", ".pdf", ".svg")
            for path in figures_dir.glob(f"*{suffix}")
            if path.is_file() and path.name != "_contact_sheet.png"
        ]
        if not stale_candidates:
            return
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in stale_candidates:
            target = archive_dir / path.name
            if target.exists():
                target.unlink()
            path.replace(target)

    @staticmethod
    def _tiered_pathway_flows(
        pathway_flows: pd.DataFrame,
        validation_results: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Split pathway_flows into robust / supported / validated (robust+supported) tiers.

        A multi-hop path is only as strong as its weakest stage: every stage of a path
        must independently reach a tier for the whole path to qualify for it. Stages that
        cannot be matched to a validation_results row are treated as unvalidated and
        excluded from all three tiers, so nothing unconfirmed is ever drawn.
        """
        empty = {
            "robust": pathway_flows.head(0),
            "supported": pathway_flows.head(0),
            "validated": pathway_flows.head(0),
        }
        if pathway_flows.empty or validation_results.empty or "path_rank" not in pathway_flows.columns:
            return empty

        status_lookup = validation_results.drop_duplicates(
            subset=["upstream_antibiotic", "downstream_antibiotic"], keep="first"
        )[["upstream_antibiotic", "downstream_antibiotic", "validation_status"]]

        annotated = pathway_flows.merge(
            status_lookup,
            left_on=["source", "target"],
            right_on=["upstream_antibiotic", "downstream_antibiotic"],
            how="left",
        )

        path_status = annotated.groupby("path_rank")["validation_status"]
        all_robust = path_status.apply(lambda s: s.notna().all() and (s == "robust").all())
        all_at_least_supported = path_status.apply(
            lambda s: s.notna().all() and s.isin(CascadeValidationAnalyzer.VALIDATED_STATUSES).all()
        )
        robust_ranks = set(all_robust[all_robust].index)
        validated_ranks = set(all_at_least_supported[all_at_least_supported].index)
        supported_only_ranks = validated_ranks - robust_ranks

        return {
            "robust": pathway_flows[pathway_flows["path_rank"].isin(robust_ranks)].reset_index(drop=True),
            "supported": pathway_flows[pathway_flows["path_rank"].isin(supported_only_ranks)].reset_index(drop=True),
            "validated": pathway_flows[pathway_flows["path_rank"].isin(validated_ranks)].reset_index(drop=True),
        }

    @staticmethod
    def _adjusted_or_confirmed_pathway_flows(
        edge_report: pd.DataFrame,
        pathway_top_n: int,
    ) -> pd.DataFrame:
        """Rebuild 2-hop chains from edges whose adjusted OR confirms the raw escalation ratio.

        _tiered_pathway_flows only splits the already-built, top-30-by-raw-escalation-ratio
        pathway_flows by validation status. That source pool is mechanically biased against
        this tier: quasi-separated edges (the ones with the largest raw escalation ratios,
        near-zero-in-one-branch data) are exactly the edges whose adjusted OR is inestimable,
        so ranking chains by raw ratio first systematically excludes most adjusted-OR-estimable
        edges before this tier could ever see them. Rebuilding chains fresh from a pool
        pre-filtered to validated + adjusted-OR-estimable + direction-consistent edges (the same
        220-of-541 pool the abstract's 88.4% concordance figure is drawn from) avoids that bias.
        """
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return edge_report.head(0)
        qualifying_edges = edge_report.loc[
            edge_report["validation_status"].isin(CascadeValidationAnalyzer.VALIDATED_STATUSES)
            & (edge_report["raw_adjusted_direction_agreement"] == "directionally_aligned")
        ]
        top_paths = CascadeReportBuilder.build_top_paths(qualifying_edges)
        return PathwayFlowBuilder.build_flow_table(top_paths, edge_report.head(0), pathway_top_n)

    @staticmethod
    def _tiered_retained_edges(
        retained_edges: pd.DataFrame,
        validation_results: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Split retained_edges into robust / supported / validated (robust+supported) tiers.

        Unlike pathway_flows, retained_edges is a flat one-row-per-edge table already
        keyed on upstream_antibiotic/downstream_antibiotic, so this is a direct
        join-and-filter rather than the path_rank grouping _tiered_pathway_flows needs.
        """
        empty = {
            "robust": retained_edges.head(0),
            "supported": retained_edges.head(0),
            "validated": retained_edges.head(0),
        }
        if retained_edges.empty or validation_results.empty:
            return empty

        status_lookup = validation_results.drop_duplicates(
            subset=["upstream_antibiotic", "downstream_antibiotic"], keep="first"
        )[["upstream_antibiotic", "downstream_antibiotic", "validation_status"]]

        annotated = retained_edges.merge(
            status_lookup,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="left",
        )

        def _filter(statuses: list[str]) -> pd.DataFrame:
            return (
                annotated[annotated["validation_status"].isin(statuses)]
                .drop(columns=["validation_status"])
                .reset_index(drop=True)
            )

        return {
            "robust": _filter(["robust"]),
            "supported": _filter(["supported"]),
            "validated": _filter(list(CascadeValidationAnalyzer.VALIDATED_STATUSES)),
        }

    @staticmethod
    def _write_table_if_nonempty(
        dataframe: pd.DataFrame,
        path: Path,
        outputs: dict[str, Path],
        *,
        write_empty_schema: bool = False,
    ) -> None:
        if len(dataframe.columns) == 0 or (dataframe.empty and not write_empty_schema):
            if path.exists():
                path.unlink()
            return
        dataframe.to_csv(path, index=False)
        outputs[path.name] = path
