"""Threshold-selection strategy comparison command."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.modeling.workflows.threshold_strategy_comparison_workflow import (
    ThresholdStrategyComparisonWorkflow,
)


def register_threshold_comparison_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "compare-thresholds",
        help=(
            "Compare validation-site holdout and patient-grouped OOF threshold selection "
            "for the same clinical-no-cascade logistic model."
        ),
    )
    parser.add_argument("--scope", default="combined", choices=("combined",))
    parser.add_argument("--organism", default=None)
    parser.add_argument("--split-strategy", default="site", choices=("site",))
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = ThresholdStrategyComparisonWorkflow(settings, path_manager)
    workflow.run(
        scope=args.scope,
        organism=args.organism,
        split_strategy=args.split_strategy,
    )
