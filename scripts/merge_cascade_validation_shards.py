#!/usr/bin/env python
"""Merge per-shard validation parquets and apply global BH-FDR.

Run this after all run_cascade_validation_shard.py jobs have completed.
It reads every shard_*.parquet from the validation_shards subdirectory,
concatenates them, applies Benjamini-Hochberg FDR across all edges, and
writes validation_results.parquet to the cascade output directory.

The subsequent run_cascade_analysis.py job detects validation_results.parquet
and loads it instead of re-running the (very expensive) validation loop.

Usage:
  python scripts/merge_cascade_validation_shards.py \\
      --env hpc --gold-scope combined --organism "ESCHERICHIA COLI" \\
      --shard-total 41
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge shard validation parquets and apply global BH-FDR."
    )
    p.add_argument("--env", default=None, help="Runtime environment (e.g. mac, hpc).")
    p.add_argument(
        "--gold-scope",
        choices=("combined", "site"),
        default="combined",
    )
    p.add_argument("--site", default=None)
    p.add_argument("--organism", default=None)
    p.add_argument(
        "--shard-total",
        type=int,
        required=True,
        help="Total number of shards (used to validate all shards are present).",
    )
    return p


def main() -> None:
    _bootstrap()

    import pandas as pd

    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import (
        CascadeValidationAnalyzer,
    )
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
    from amr_cascade_platform.core.utils.scopes import scoped_output_dir

    LoggerFactory.configure()
    logger = logging.getLogger(__name__)

    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)

    cascade_dir = scoped_output_dir(
        root=path_manager.paths.artifacts / settings.cascade.outputs.result_dir,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    shard_dir = cascade_dir / "validation_shards"
    output_path = cascade_dir / "validation_results.parquet"

    if output_path.exists():
        logger.info("Merged validation results already exist: %s", output_path)
        return

    shard_total = args.shard_total
    shard_paths = sorted(shard_dir.glob("shard_*.parquet"))

    # BH-FDR's q-value threshold is a function of the total number of tests, so
    # merging a partial shard set doesn't just omit the missing edges -- it silently
    # shifts the q-value, and therefore the robust/supported label, of every edge
    # that DID merge. A partial merge is never safe to treat as final: fail loudly
    # and require the missing shard jobs to be resubmitted and rerun instead.
    expected = {
        shard_dir / f"shard_{i:04d}_of_{shard_total:04d}.parquet"
        for i in range(shard_total)
    }
    missing = expected - set(shard_paths)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {shard_total} expected shard files are missing from {shard_dir}; "
            f"refusing to merge a partial shard set (BH-FDR q-values depend on the total test "
            f"count, so a partial merge would silently mislabel edges from the shards that DID "
            f"complete). Resubmit and rerun the missing shard jobs, then re-run this merge. "
            f"Missing: {', '.join(p.name for p in sorted(missing))}"
        )

    if not shard_paths:
        raise FileNotFoundError(
            f"No shard parquets found in {shard_dir}. "
            "Ensure all run_cascade_validation_shard.py jobs completed."
        )

    logger.info("Loading %d shard files from %s", len(shard_paths), shard_dir)
    shards = [pd.read_parquet(p) for p in shard_paths]
    non_empty = [s for s in shards if not s.empty]
    logger.info(
        "Loaded %d shards (%d non-empty, %d total edges before FDR)",
        len(shards), len(non_empty), sum(len(s) for s in non_empty),
    )

    analyzer = CascadeValidationAnalyzer(settings)
    merged = analyzer.merge_shards(shards)

    cascade_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(
        "Merged validation results written: %s (%d edges, %d validated)",
        output_path, len(merged),
        int((merged["validation_status"] != "insufficient").sum()) if not merged.empty else 0,
    )


if __name__ == "__main__":
    main()
