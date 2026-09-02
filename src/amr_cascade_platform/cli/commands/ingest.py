"""Ingestion commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.data.ingestion.raw_ingestion_manager import (
    RawIngestionManager,
    RawIngestionRequest,
)
from amr_cascade_platform.data.pipelines.bronze_pipeline import BronzePipeline


def register_ingestion_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "ingest-bronze",
        help="Convert source CSV tables into bronze Parquet datasets.",
    )
    parser.add_argument("--site", required=True, help="Site name to ingest.")
    parser.add_argument(
        "--source-layer",
        default=None,
        choices=("raw", "sample"),
        help="Source layer to ingest from. Defaults to the environment default.",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        help="Restrict ingestion to one or more canonical table names.",
    )
    parser.add_argument(
        "--allow-raw-on-local",
        action="store_true",
        help="Explicitly allow raw-layer ingestion in the local/mac environment.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, mapper = context_builder(project_root, args.env)
    request = RawIngestionRequest(
        site=args.site,
        source_layer=args.source_layer or settings.environment.default_data_layer,
        tables=tuple(args.table) if args.table else None,
        allow_raw_on_local=args.allow_raw_on_local,
    )
    manager = RawIngestionManager(settings, path_manager, mapper)
    BronzePipeline(manager).run(request)
