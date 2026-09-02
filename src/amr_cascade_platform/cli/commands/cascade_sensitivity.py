"""Cascade sensitivity commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.cascade.workflows.implied_sensitivity_workflow import (
    ImpliedSensitivityRequest,
    ImpliedSensitivityWorkflow,
)


def register_cascade_sensitivity_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "run-cascade-sensitivity",
        help="Run the armd implied-susceptibility sensitivity cascade workflow.",
    )
    parser.add_argument("--site", default="armd", help="Currently only 'armd' is supported.")
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = ImpliedSensitivityWorkflow(settings, path_manager)
    workflow.run(ImpliedSensitivityRequest(site=args.site))
