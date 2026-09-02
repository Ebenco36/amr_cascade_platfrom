"""Gold dataset commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.data.gold.gold_build_manager import GoldBuildManager, GoldBuildRequest
from amr_cascade_platform.data.pipelines.gold_pipeline import GoldPipeline


def register_gold_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "build-gold",
        help="Construct gold analytical datasets from harmonized cohort data.",
    )
    parser.add_argument(
        "--source-scope",
        choices=("combined", "site"),
        default="combined",
        help="Build gold outputs from combined harmonized data or one site-aligned dataset.",
    )
    parser.add_argument("--site", default=None, help="Site name when using --source-scope site.")
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified gold outputs.")
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    request = GoldBuildRequest(source_scope=args.source_scope, site=args.site, organism=args.organism)
    manager = GoldBuildManager(settings, path_manager)
    GoldPipeline(manager).run(request)
