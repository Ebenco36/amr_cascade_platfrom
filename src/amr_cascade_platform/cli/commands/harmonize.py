"""Harmonization commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.data.harmonization.harmonization_manager import (
    HarmonizationManager,
    HarmonizationRequest,
)
from amr_cascade_platform.data.pipelines.harmonized_pipeline import HarmonizedPipeline


def register_harmonization_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "build-harmonized",
        help="Align silver datasets across sites and build combined harmonized tables.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        help="Restrict harmonization to one or more sites. Repeat for multiple sites.",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        help="Restrict harmonization to one or more canonical shared tables.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    request = HarmonizationRequest(
        sites=tuple(args.site) if args.site else None,
        tables=tuple(args.table) if args.table else None,
    )
    manager = HarmonizationManager(settings, path_manager)
    HarmonizedPipeline(manager).run(request)
