"""Sampling command."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.data.sampling.sample_dataset_creator import (
    SampleDatasetCreator,
    SamplingRequest,
)


def register_sampling_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser("sample-data", help="Create sample datasets from raw site CSVs.")
    parser.add_argument("--fraction", type=float, default=None, help="Sampling fraction in (0, 1].")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--ensure-eskape-coverage",
        action="store_true",
        help="Guarantee that available ESKAPE targets are preserved in the sampled cohort and linked tables.",
    )
    parser.add_argument(
        "--eskape-min-rows-per-target",
        type=int,
        default=25,
        help="Minimum cohort rows to preserve per available ESKAPE target when coverage guarding is enabled.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        help="Restrict sampling to one or more sites. Repeat the flag for multiple sites.",
    )
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, mapper = context_builder(project_root, args.env)
    request = SamplingRequest(
        fraction=args.fraction if args.fraction is not None else settings.sampling.default_fraction,
        seed=args.seed if args.seed is not None else settings.sampling.random_seed,
        sites=tuple(args.site) if args.site else None,
        ensure_eskape_coverage=args.ensure_eskape_coverage,
        eskape_min_rows_per_target=args.eskape_min_rows_per_target,
    )
    creator = SampleDatasetCreator(settings, path_manager, mapper)
    creator.create(request)
