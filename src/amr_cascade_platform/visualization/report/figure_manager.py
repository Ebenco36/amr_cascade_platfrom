"""Orchestrate cascade figure exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.reporting.builders import directional_views as dv
from amr_cascade_platform.visualization.report.directional_plotters import (
    AdjustmentConcordancePlotter,
    AdjustmentConcordanceSummaryPlotter,
    AntiArtefactScatterPlotter,
    DirectionalAwarePlotter,
    DirectionalForestPlotter,
    DirectionalMatrixPlotter,
    DirectionalNetworkPlotter,
    DirectionalSankeyPlotter,
)
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
        self._classification = classification
        self._directional_sankey_plotter = DirectionalSankeyPlotter(
            exporter=exporter, template=settings.reporting.plotly_template, width=settings.reporting.image_width,
            classification=classification,
        )
        self._directional_matrix_plotter = DirectionalMatrixPlotter(
            exporter=exporter, template=settings.reporting.plotly_template, classification=classification
        )
        self._directional_network_plotter = DirectionalNetworkPlotter(classification)
        self._directional_forest_plotter = DirectionalForestPlotter(
            exporter=exporter, template=settings.reporting.plotly_template, classification=classification,
        )
        self._directional_aware_plotter = DirectionalAwarePlotter(classification)
        self._anti_artefact_plotter = AntiArtefactScatterPlotter(
            exporter=exporter, template=settings.reporting.plotly_template,
            width=settings.reporting.image_width, height=settings.reporting.image_height,
        )
        self._adjustment_concordance_plotter = AdjustmentConcordancePlotter(
            exporter=exporter, template=settings.reporting.plotly_template,
            width=settings.reporting.image_width, height=settings.reporting.image_height,
        )
        self._adjustment_concordance_summary_plotter = AdjustmentConcordanceSummaryPlotter(
            exporter=exporter, template=settings.reporting.plotly_template,
            width=settings.reporting.image_width, height=settings.reporting.image_height,
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

    def export_directional_suite(
        self,
        edge_report: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
        *,
        tier: str = "validated",
        min_fold: float = 1.5,
        sankey_top_n: int = 30,
        network_top_n: int = 60,
        forest_top_n: int = 20,
    ) -> dict[str, Path]:
        """Escalation and suppression cascade figures, always as separate files, never pooled.

        Produces, for each of ``escalation``/``suppression``: a Sankey, an
        upstream x downstream matrix, a directed network, a forest of the
        strongest CI-bearing effects, and an AWaRe-tier heatmap -- ten files
        per tier. See `reporting.builders.directional_views` for why pooling
        the two directions into one figure is never done here.
        """
        outputs: dict[str, Path] = {}
        canonical = dv.canonicalize_edges(edge_report, self._classification.canonical_display_label)
        validated = dv.tier_edges(dv.validated_edges(canonical), tier)
        if validated.empty:
            return outputs
        aware_of = lambda label: self._classification.resolve(label).aware_category  # noqa: E731
        order = dv.shared_drug_order(validated, aware_of, self._classification.category_sort_key)
        positions = dv.circular_layout(order, aware_of, self._classification.category_sort_key)
        for direction in dv.DIRECTIONS:
            n_direction = int((validated.direction == direction).sum())
            displayed = dv.display_selection(validated, direction, min_fold)

            links = dv.bipartite_links(displayed, direction, top_n=sankey_top_n)
            coverage = dv.coverage_summary(links, displayed, direction)
            outputs.update(self._directional_sankey_plotter.export(
                links, direction, output_dir / f"figure_{direction}_sankey_{tier}", formats,
                order=order, tier=tier, coverage=coverage, n_eligible=len(displayed), min_fold=min_fold,
            ))

            cells = dv.matrix_frame(validated.loc[validated.direction == direction], direction, order)
            outputs.update(self._directional_matrix_plotter.export(
                cells, direction, output_dir / f"figure_{direction}_matrix_{tier}", formats,
                order=order, tier=tier, n_direction=n_direction,
            ))

            net_pool = displayed.sort_values("effect_magnitude", ascending=False).head(network_top_n) if "effect_magnitude" in displayed.columns else displayed.head(network_top_n)
            g_edges, g_nodes = dv.directional_graph_tables(dv.with_effect_columns(net_pool), direction)
            outputs.update(self._directional_network_plotter.export(
                g_edges, g_nodes, positions, direction, output_dir / f"figure_{direction}_network_{tier}", formats,
                tier=tier, n_eligible=len(displayed), n_direction=n_direction, min_fold=min_fold,
            ))

            outputs.update(self._directional_forest_plotter.export(
                validated, direction, output_dir / f"figure_{direction}_forest_{tier}", formats,
                tier=tier, top_n=forest_top_n, n_direction=n_direction,
            ))

            aware_table = dv.aware_direction_table(validated, direction, aware_of, self._classification.transition_direction)
            outputs.update(self._directional_aware_plotter.export(
                aware_table, direction, output_dir / f"figure_{direction}_aware_{tier}", formats,
                tier=tier, n_direction=n_direction,
            ))
        return outputs

    def export_evidence_scatter_suite(
        self,
        edge_report: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
        *,
        tier: str = "validated",
    ) -> dict[str, Path]:
        """Anti-artefact (DAS vs. PBI) and adjustment-concordance (raw ER vs. adjusted OR) scatters.

        Both directions share one plot in each of these two figures -- see
        AntiArtefactScatterPlotter and AdjustmentConcordancePlotter for why
        that is the correct reading here rather than a violation of the
        never-pool-directions rule the five-plotter suite above enforces.
        """
        outputs: dict[str, Path] = {}
        canonical = dv.canonicalize_edges(edge_report, self._classification.canonical_display_label)
        validated = dv.tier_edges(dv.validated_edges(canonical), tier)
        if validated.empty:
            return outputs
        outputs.update(self._anti_artefact_plotter.export(
            validated, output_dir / f"figure_das_vs_pbi_{tier}", formats, tier=tier,
        ))
        outputs.update(self._adjustment_concordance_plotter.export(
            validated, output_dir / f"figure_raw_vs_adjusted_{tier}", formats, tier=tier,
        ))
        validated_with_concordance = dv.with_adjustment_concordance(validated)
        outputs.update(self._adjustment_concordance_summary_plotter.export(
            validated_with_concordance, output_dir / f"figure_adjustment_concordance_summary_{tier}", formats, tier=tier,
        ))
        return outputs

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

    def export_kappa_ranked(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._prevalence_plotter.export_kappa_ranked(results, output_stem, formats)

    def export_cascade_vs_independent_dumbbell(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._prevalence_plotter.export_cascade_vs_independent_dumbbell(results, output_stem, formats)

    def export_surveillance_sensitivity_map(self, results: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return self._prevalence_plotter.export_surveillance_sensitivity_map(results, output_stem, formats)

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

    def export_operational_availability_matrix(
        self, availability_table: pd.DataFrame, output_stem: Path, formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._dataset_characterization_plotter.export_operational_availability_matrix(
            availability_table, output_stem, formats,
        )

    def export_site_era_availability_timeline(
        self, availability_table: pd.DataFrame, output_stem: Path, formats: tuple[str, ...],
    ) -> dict[str, Path]:
        return self._dataset_characterization_plotter.export_site_era_availability_timeline(
            availability_table, output_stem, formats,
        )
