"""Cascade analysis commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import (
    CascadeAnalysisRequest,
    CascadeAnalysisWorkflow,
)


def register_cascade_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "run-cascade",
        help="Run primary cascade analysis from gold datasets.",
    )
    parser.add_argument(
        "--gold-scope",
        choices=("combined", "site"),
        default="combined",
        help="Analyze combined gold data or one site's gold data.",
    )
    parser.add_argument("--site", default=None, help="Site name when using --gold-scope site.")
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified cascade analysis.")
    parser.add_argument(
        "--ignore-precomputed-validation",
        action="store_true",
        help="Recompute validation even if validation_results.parquet already exists.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = CascadeAnalysisWorkflow(settings, path_manager)
    request = CascadeAnalysisRequest(
        gold_scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
        ignore_precomputed_validation=args.ignore_precomputed_validation,
    )
    workflow.run(request)
