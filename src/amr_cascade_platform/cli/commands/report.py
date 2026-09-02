"""Reporting commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.reporting.workflows.report_export_workflow import (
    ReportExportRequest,
    ReportExportWorkflow,
)


def register_reporting_command(subparsers, project_root: Path, context_builder) -> None:
    figure_choices = (
        "primary_cascade_forest",
        "validated_primary_cascade_forest",
        "aware_transition_heatmap",
        "prevalence_shift_forest",
        "prevalence_shift_curves",
        "mnar_tipping_point",
        "upstream_contribution_forest",
        "downstream_trigger_forest",
        "threshold_sensitivity",
        "site_vs_combined_summary",
        "cascade_sankey",
        "cascade_chord",
        "cascade_network",
        "model_metrics_comparison",
        "model_pr_curve",
        "model_roc_curve",
        "model_calibration",
        "model_threshold_analysis",
        "validation_diagnostics",
        "er_landscape",
        "mnar_prevalence_shift_distribution",
        "validation_funnel",
        "panel_bundling",
        "temporal_stability",
        "cross_site_concordance",
        "consort_diagram",
        "dataset_characterization",
    )
    parser = subparsers.add_parser(
        "export-report",
        help="Export manuscript-ready tables and figures into outputs/.",
    )
    parser.add_argument(
        "--scope",
        choices=("combined", "site"),
        default="combined",
        help="Export for combined outputs or one site scope.",
    )
    parser.add_argument("--site", default=None, help="Site name when using --scope site.")
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified reporting outputs.")
    parser.add_argument(
        "--figure-format",
        action="append",
        choices=("html", "png", "svg", "pdf"),
        help="Override configured figure output formats. Repeat to request multiple formats.",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=figure_choices,
        help="Restrict exported figures to one or more named figure groups.",
    )
    parser.add_argument(
        "--fail-on-missing-figures",
        action="store_true",
        help="Fail if any requested figure family does not produce an output file.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = ReportExportWorkflow(settings, path_manager)
    workflow.run(
        ReportExportRequest(
            scope=args.scope,
            site=args.site,
            organism=args.organism,
            figure_formats=tuple(args.figure_format) if args.figure_format else None,
            figures=tuple(args.figure) if args.figure else None,
            fail_on_missing_figures=args.fail_on_missing_figures,
        )
    )
