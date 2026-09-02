#!/usr/bin/env python3
"""Operator preflight checks for Mac and HPC pipeline runners."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.core.utils.pipeline_preflight import (  # noqa: E402
    PipelinePreflightChecker,
    PipelinePreflightRequest,
)


def _csv_tuple(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preflight validation for the AMR pipeline.")
    parser.add_argument("--env", required=True, help="Environment name, e.g. mac or hpc.")
    parser.add_argument("--sites", required=True, help="Comma-separated sites.")
    parser.add_argument("--organisms", default="", help="Comma-separated organisms.")
    parser.add_argument("--estimators", default="", help="Comma-separated estimator names.")
    parser.add_argument("--report-formats", default="", help="Comma-separated report formats.")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-organism-primary", action="store_true")
    parser.add_argument("--skip-pooled-sensitivity", action="store_true")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-harmonization", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-cascade", action="store_true")
    parser.add_argument("--skip-site-cascade", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--skip-reporting", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checker = PipelinePreflightChecker(PROJECT_ROOT, args.env)
    request = PipelinePreflightRequest(
        environment=args.env,
        sites=_csv_tuple(args.sites),
        organisms=_csv_tuple(args.organisms),
        estimators=_csv_tuple(args.estimators),
        report_formats=_csv_tuple(args.report_formats),
        run_sampling=not args.skip_sampling,
        run_organism_primary=not args.skip_organism_primary,
        run_pooled_sensitivity=not args.skip_pooled_sensitivity,
        run_ingestion=not args.skip_ingestion,
        run_preprocessing=not args.skip_preprocessing,
        run_harmonization=not args.skip_harmonization,
        run_gold=not args.skip_gold,
        run_cascade=not args.skip_cascade,
        run_site_cascade=not args.skip_site_cascade,
        run_sensitivity=not args.skip_sensitivity,
        run_features=not args.skip_features,
        run_training=not args.skip_training,
        run_ablation=args.run_ablation,
        run_reporting=not args.skip_reporting,
    )
    result = checker.run(request)
    print(f"Preflight environment: {result.environment}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
        raise ValidationError(f"Preflight failed with {len(result.errors)} error(s).")
    print("Preflight status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
