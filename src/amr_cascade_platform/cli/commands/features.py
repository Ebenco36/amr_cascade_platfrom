"""Feature preprocessing commands."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.features.preprocessors.comorbidity_aggregator import (
    ComorbidityAggregationRequest,
    ComorbidityAggregator,
)


def register_comorbidity_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "aggregate-comorbidity",
        help="Pre-aggregate comorbidity data into a culture-level feature matrix.",
    )
    parser.add_argument("--site", required=True, help="Site name, e.g. armd or armd_ecuh.")
    parser.add_argument(
        "--data-layer",
        default=None,
        choices=("raw", "sample"),
        help="Source layer to aggregate from. Defaults to the environment default.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Retain the top K comorbidities.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="Retain comorbidities with at least this many occurrences.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help="Retain comorbidities until cumulative frequency reaches this fraction.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Compute diagnostics without writing parquet output.",
    )
    parser.add_argument(
        "--allow-raw-on-local",
        action="store_true",
        help="Explicitly allow raw-layer processing in the local/mac environment.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, mapper = context_builder(project_root, args.env)
    request = ComorbidityAggregationRequest(
        site=args.site,
        data_layer=args.data_layer or settings.environment.default_data_layer,
        top_k=args.top_k,
        min_count=args.min_count,
        min_coverage=args.min_coverage,
        analyze_only=args.analyze_only,
        allow_raw_on_local=args.allow_raw_on_local,
    )
    aggregator = ComorbidityAggregator(settings, path_manager, mapper)
    aggregator.run(request)
