"""Prevalence-shift analysis commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.surveillance.workflows.prevalence_shift_workflow import (
    PrevalenceShiftRequest,
    PrevalenceShiftWorkflow,
)


def register_prevalence_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "analyze-prevalence",
        help="Estimate naive vs eligible-scale resistance prevalence under selective testing.",
    )
    parser.add_argument("--scope", choices=("combined", "site"), default="combined")
    parser.add_argument("--site", default=None, help="Site name when using --scope site.")
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified prevalence analysis.")
    parser.add_argument("--drug", action="append", default=None, help="Optionally restrict to one or more antibiotics.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of organism-drug pairs to include in the forest plot.")
    parser.add_argument("--min-eligible-support", type=int, default=None, help="Override the minimum eligible denominator support.")
    parser.add_argument("--min-tested-support", type=int, default=None, help="Override the minimum evaluable tested support.")
    parser.add_argument("--min-resistant-support", type=int, default=None, help="Override the minimum resistant count support.")
    parser.add_argument(
        "--figure-format",
        action="append",
        choices=("html", "png", "svg", "pdf"),
        default=None,
        help="Override default prevalence-shift figure export formats.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = PrevalenceShiftWorkflow(settings, path_manager)
    workflow.run(
        PrevalenceShiftRequest(
            scope=args.scope,
            site=args.site,
            organism=args.organism,
            drugs=tuple(args.drug) if args.drug else None,
            top_n=args.top_n,
            figure_formats=tuple(args.figure_format) if args.figure_format else None,
            min_eligible_support=args.min_eligible_support,
            min_tested_support=args.min_tested_support,
            min_resistant_support=args.min_resistant_support,
        )
    )
