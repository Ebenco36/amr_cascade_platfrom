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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Merge again even when a merged result from the same inputs exists.",
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
    from amr_cascade_platform.core.utils.provenance import (
        atomic_write_json,
        atomic_write_parquet,
        read_json,
        sidecar_path,
        validation_fingerprint,
    )
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
    gold_dir = scoped_output_dir(
        root=path_manager.paths.gold,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    fingerprint, components = validation_fingerprint(project_root, settings, gold_dir / "drug_pair_episodes.parquet")

    if output_path.exists() and not args.overwrite:
        recorded = (read_json(sidecar_path(output_path)) or {}).get("fingerprint")
        if recorded == fingerprint:
            logger.info("Merged validation results for these inputs already exist: %s", output_path)
            return
        logger.warning(
            "Existing %s was produced from different gold data, configuration or code "
            "(recorded fingerprint %s, expected %s); merging again.",
            output_path, recorded, fingerprint,
        )

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

    # Every shard must come from the same gold data, configuration and code as
    # this merge: a leftover shard from an earlier run would otherwise be merged
    # silently (its edge keys can match while its seeds or iterations do not).
    stale = [
        path.name for path in sorted(expected)
        if (read_json(sidecar_path(path)) or {}).get("fingerprint") != fingerprint
    ]
    if stale:
        raise RuntimeError(
            f"{len(stale)} of {shard_total} shards in {shard_dir} were produced from different gold data, "
            f"configuration or code than this merge (fingerprint {fingerprint}); refusing to merge them. "
            f"Rerun those shard jobs (run_cascade_validation_shard.py recomputes a mismatched shard): "
            f"{', '.join(stale)}"
        )
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard parquets found in {shard_dir}. "
            "Ensure all run_cascade_validation_shard.py jobs completed."
        )

    # Only this run's shard set: files from an earlier run with a different
    # --shard-total also match shard_*.parquet and must not be merged.
    logger.info("Loading %d shard files from %s", len(expected), shard_dir)
    shards = [pd.read_parquet(p) for p in sorted(expected)]
    non_empty = [s for s in shards if not s.empty]
    logger.info(
        "Loaded %d shards (%d non-empty, %d total edges before FDR)",
        len(shards), len(non_empty), sum(len(s) for s in non_empty),
    )

    analyzer = CascadeValidationAnalyzer(settings)
    merged = analyzer.merge_shards(shards)

    cascade_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(merged, output_path)
    atomic_write_json({"fingerprint": fingerprint, **components, "shard_total": shard_total, "rows": len(merged)}, sidecar_path(output_path))
    logger.info(
        "Merged validation results written: %s (%d edges, %d validated)",
        output_path, len(merged),
        int((merged["validation_status"] != "insufficient").sum()) if not merged.empty else 0,
    )


if __name__ == "__main__":
    main()
