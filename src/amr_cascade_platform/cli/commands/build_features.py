"""Feature build commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.features.workflows.feature_build_workflow import (
    FeatureBuildRequest,
    FeatureBuildWorkflow,
)


def register_feature_build_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "build-features",
        help="Build model-ready combined feature matrices with validated left joins.",
    )
    parser.add_argument(
        "--scope",
        default="combined",
        choices=("combined",),
        help="Feature build scope. Current implementation supports combined only.",
    )
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified feature builds.")
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, mapper = context_builder(project_root, args.env)
    workflow = FeatureBuildWorkflow(settings, path_manager, mapper)
    workflow.run(FeatureBuildRequest(scope=args.scope, organism=args.organism))
