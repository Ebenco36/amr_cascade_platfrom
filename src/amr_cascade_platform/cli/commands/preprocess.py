"""Preprocessing commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.data.cleaning.site_cleaning_manager import (
    SilverCleaningRequest,
    SiteCleaningManager,
)
from amr_cascade_platform.data.pipelines.silver_pipeline import SilverPipeline


def register_preprocessing_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "build-silver",
        help="Clean bronze parquet datasets into validated silver parquet datasets.",
    )
    parser.add_argument("--site", required=True, help="Site name to preprocess.")
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        help="Restrict preprocessing to one or more canonical table names.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    request = SilverCleaningRequest(
        site=args.site,
        tables=tuple(args.table) if args.table else None,
    )
    manager = SiteCleaningManager(settings, path_manager)
    SilverPipeline(manager).run(request)
