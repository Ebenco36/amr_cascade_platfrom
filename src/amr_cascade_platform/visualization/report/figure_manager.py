"""Orchestrate cascade figure exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_aware_transition_plotter import PlotlyAwareTransitionPlotter
from amr_cascade_platform.visualization.report.plotly_forest_plotter import PlotlyForestPlotter
from amr_cascade_platform.visualization.report.publication_forest_plotter import PublicationForestPlotter
from amr_cascade_platform.visualization.report.plotly_cascade_plotters import (
    PlotlyCascadeMatrixPlotter,
    PlotlyNetworkPlotter,
    PlotlySankeyPlotter,
    PlotlySiteComparisonPlotter,
    PlotlyThresholdSensitivityPlotter,
)
from amr_cascade_platform.visualization.report.plotly_delta_curves_plotter import PrevalenceDeltaCurvePlotter
from amr_cascade_platform.visualization.report.plotly_model_plotters import PlotlyModelEvaluationPlotter
from amr_cascade_platform.visualization.report.plotly_mnar_tipping_point_plotter import PlotlyMNARTippingPointPlotter
from amr_cascade_platform.visualization.report.plotly_prevalence_plotter import PlotlyPrevalencePlotter
from amr_cascade_platform.visualization.report.plotly_validation_diagnostics_plotter import PlotlyValidationDiagnosticsPlotter
from amr_cascade_platform.visualization.report.plotly_er_landscape_plotter import PlotlyERLandscapePlotter
from amr_cascade_platform.visualization.report.plotly_mnar_prevalence_shift_distribution_plotter import (
    PlotlyMNARPrevalenceShiftDistributionPlotter,
)
from amr_cascade_platform.visualization.report.plotly_validation_funnel_plotter import PlotlyValidationFunnelPlotter
from amr_cascade_platform.visualization.report.plotly_panel_bundling_plotter import PlotlyPanelBundlingPlotter
from amr_cascade_platform.visualization.report.plotly_temporal_stability_plotter import PlotlyTemporalStabilityPlotter
from amr_cascade_platform.visualization.report.plotly_cross_site_concordance_plotter import PlotlyCrossSiteConcordancePlotter
from amr_cascade_platform.visualization.report.plotly_consort_plotter import PlotlyConsortPlotter
from amr_cascade_platform.visualization.report.plotly_dataset_characterization_plotter import DatasetCharacterizationPlotter


class FigureManager:
    """High-level facade for manuscript-facing figure families."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        exporter = PlotlyFigureExporter(
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        classification = AntibioticClassificationResolver(
            path_manager.project_root / "data" / "antibiotic_classification_complete.csv"
        )
        self._forest_plotter = PlotlyForestPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            top_n=settings.reporting.forest_top_n,
            classification_resolver=classification,
        )
        self._publication_forest_plotter = PublicationForestPlotter(
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            top_n=settings.reporting.forest_top_n,
            classification_resolver=classification,
        )
        self._threshold_plotter = PlotlyThresholdSensitivityPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._site_comparison_plotter = PlotlySiteComparisonPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._sankey_plotter = PlotlySankeyPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            classification_resolver=classification,
            top_n=settings.reporting.sankey_top_n,
        )
        self._network_plotter = PlotlyNetworkPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            classification_resolver=classification,
        )
        self._chord_plotter = PlotlyCascadeMatrixPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            classification_resolver=classification,
        )
        self._model_plotter = PlotlyModelEvaluationPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._aware_transition_plotter = PlotlyAwareTransitionPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            classification_resolver=classification,
        )
        self._prevalence_plotter = PlotlyPrevalencePlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._delta_curves_plotter = PrevalenceDeltaCurvePlotter()
        self._mnar_tipping_point_plotter = PlotlyMNARTippingPointPlotter()
        self._validation_diagnostics_plotter = PlotlyValidationDiagnosticsPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._er_landscape_plotter = PlotlyERLandscapePlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._mnar_prevalence_shift_distribution_plotter = PlotlyMNARPrevalenceShiftDistributionPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._validation_funnel_plotter = PlotlyValidationFunnelPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
            bootstrap_stability_threshold=getattr(
                getattr(settings, "cascade", None), "bootstrap_sign_stability_threshold", 0.80
            ),
        )
        self._panel_bundling_plotter = PlotlyPanelBundlingPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._temporal_stability_plotter = PlotlyTemporalStabilityPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._cross_site_concordance_plotter = PlotlyCrossSiteConcordancePlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._consort_plotter = PlotlyConsortPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )
        self._dataset_characterization_plotter = DatasetCharacterizationPlotter(
            exporter=exporter,
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )

    def export_primary_forest(self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._publication_forest_plotter.export_primary_forest(edge_report, output_stem, formats)

    def export_validated_primary_forest(self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._publication_forest_plotter.export_validated_primary_forest(edge_report, output_stem, formats)

    def export_all_downstream_trigger_forests(
        self,
        escalation_results: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._forest_plotter.export_downstream_trigger_suite(escalation_results, output_dir, formats)

    def export_all_upstream_forest(self, escalation_results: pd.DataFrame, output_dir: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self.export_all_downstream_trigger_forests(escalation_results, output_dir, formats)

    def export_forest_summary(self, edge_report: pd.DataFrame, output_path: Path) -> Path:
        return self._publication_forest_plotter.export_forest_summary(edge_report, output_path)

    def export_threshold_sensitivity(self, sensitivity_table: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._threshold_plotter.export(sensitivity_table, output_stem, formats)

    def export_site_comparison_summary(self, comparison_summary: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._site_comparison_plotter.export(comparison_summary, output_stem, formats)

    def export_sankey(
        self,
        pathway_flows: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        return self._sankey_plotter.export(pathway_flows, output_stem, formats, tier_label)

    def export_chord(
        self,
        retained_edges: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        return self._chord_plotter.export(retained_edges, output_stem, formats, tier_label)

    def export_network(
        self,
        network_edges: pd.DataFrame,
        network_nodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        return self._network_plotter.export(network_edges, network_nodes, output_stem, formats, tier_label)

    def export_model_metrics_comparison(self, metrics: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._model_plotter.export_metrics_comparison(metrics, output_stem, formats)

    def export_model_precision_recall(self, predictions: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._model_plotter.export_precision_recall(predictions, output_stem, formats)

    def export_model_roc(self, predictions: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._model_plotter.export_roc(predictions, output_stem, formats)

    def export_model_calibration(self, predictions: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._model_plotter.export_calibration(predictions, output_stem, formats)

    def export_model_threshold_analysis(self, threshold_metrics: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._model_plotter.export_threshold_analysis(threshold_metrics, output_stem, formats)

    def export_aware_transition_heatmap(
        self,
        transition_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._aware_transition_plotter.export_transition_heatmap(transition_summary, output_stem, formats)

    def export_prevalence_shift_forest(
        self,
        results: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._prevalence_plotter.export_forest(results, output_stem, formats)

    def export_prevalence_delta_curves(
        self,
        curves: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        organism: str = "",
    ) -> dict[str, Path]:
        return self._delta_curves_plotter.export(curves, prevalence_summary, output_stem, formats, organism=organism)

    def export_mnar_tipping_point(
        self,
        curves: pd.DataFrame,
        tipping_points: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        organism: str = "",
    ) -> dict[str, Path]:
        return self._mnar_tipping_point_plotter.export(
            curves=curves,
            tipping_points=tipping_points,
            prevalence_summary=prevalence_summary,
            output_stem=output_stem,
            formats=formats,
            organism=organism,
        )

    def export_validation_diagnostics(
        self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._validation_diagnostics_plotter.export(edge_report, output_stem, formats)

    def export_er_landscape(
        self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._er_landscape_plotter.export(edge_report, output_stem, formats)

    def export_mnar_prevalence_shift_distribution(
        self, prevalence_shift: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._mnar_prevalence_shift_distribution_plotter.export(prevalence_shift, output_stem, formats)

    def export_validation_funnel(
        self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._validation_funnel_plotter.export(edge_report, output_stem, formats)

    def export_panel_bundling(
        self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._panel_bundling_plotter.export(edge_report, output_stem, formats)

    def export_temporal_stability(
        self, edge_report: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._temporal_stability_plotter.export(edge_report, output_stem, formats)

    def export_cross_site_concordance(
        self, comparison_summary: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]
    ) -> dict[str, Path]:
        return self._cross_site_concordance_plotter.export(comparison_summary, output_stem, formats)

    def export_consort_diagram(
        self,
        flow_table: pd.DataFrame,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._consort_plotter.export(flow_table, edge_report, output_stem, formats)

    # ── Dataset characterisation suite ────────────────────────────────────────

    def export_dataset_characterization(
        self,
        eligible_pairs: pd.DataFrame,
        culture_episodes: pd.DataFrame,
        drug_pair_episodes: pd.DataFrame,
        escalation_results: pd.DataFrame,
        upstream_balance_table: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Export the full dataset characterisation suite (eligible vs. observed space,
        covariate imbalance, temporal distribution, missing data profile, etc.)."""
        return self._dataset_characterization_plotter.export_all(
            eligible_pairs=eligible_pairs,
            culture_episodes=culture_episodes,
            drug_pair_episodes=drug_pair_episodes,
            escalation_results=escalation_results,
            upstream_balance_table=upstream_balance_table,
            output_dir=output_dir,
            formats=formats,
        )

    def export_eligible_vs_observed_by_drug(
        self,
        eligible_pairs: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._dataset_characterization_plotter.export_eligible_vs_observed_by_drug(
            eligible_pairs, output_stem, formats,
        )

    def export_testing_rate_by_upstream_result(
        self,
        drug_pair_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._dataset_characterization_plotter.export_testing_rate_by_upstream_result(
            drug_pair_episodes, output_stem, formats,
        )
