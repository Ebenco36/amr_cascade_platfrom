"""Model training commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.modeling.workflows.modeling_workflow import ModelingRequest, ModelingWorkflow


def register_training_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train split-safe downstream-testing prediction models.",
    )
    parser.add_argument(
        "--scope",
        default="combined",
        choices=("combined",),
        help="Modeling scope. Current implementation supports combined only.",
    )
    parser.add_argument(
        "--estimator",
        action="append",
        default=None,
        help="Optionally restrict training to one or more estimators.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=("observed_only", "cascade_aware", "clinical_no_cascade", "full"),
        default=None,
        help="Optionally restrict training to one or more named feature sets.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the full feature-set ablation across observed-only, cascade-aware, clinical-no-cascade, and full.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("site", "temporal"),
        default=None,
        help="Optional modeling split strategy. Defaults to the configured primary strategy.",
    )
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified modeling.")
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = ModelingWorkflow(settings, path_manager)
    workflow.run(
        ModelingRequest(
            scope=args.scope,
            estimators=tuple(args.estimator) if args.estimator else None,
            feature_sets=tuple(args.feature_set) if args.feature_set else None,
            ablation=bool(args.ablation),
            organism=args.organism,
            split_strategy=args.split_strategy,
        )
    )
