#!/usr/bin/env python
"""Run a sensitivity analysis over the final validated pattern set, optionally in shards.

Both analyses apply one statistic to each pattern of the merged validation result,
so they must run after merge_cascade_validation_shards.py:
  era-stratified   site x era stratified permutation, robust + supported patterns
  patient-cluster  patient-level cluster bootstrap (Supplementary Table S7), robust patterns
Each --shard-index processes one deterministic slice; --merge-shards combines the slices
and refuses to write anything unless they cover the pattern set exactly.

Usage:
  python scripts/run_validated_pattern_sensitivity.py --analysis patient-cluster \\
      --env hpc --gold-scope combined --organism "ESCHERICHIA COLI" --shard-total 16 --shard-index 0
  ... --shard-total 16 --merge-shards
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
    p = argparse.ArgumentParser(description="Run a sensitivity analysis over the validated pattern set.")
    p.add_argument("--analysis", required=True, choices=("era-stratified", "patient-cluster"))
    p.add_argument("--env", default=None, help="Runtime environment (e.g. mac, hpc).")
    p.add_argument("--gold-scope", choices=("combined", "site"), default="combined")
    p.add_argument("--site", default=None)
    p.add_argument("--organism", default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-total", type=int, default=1)
    p.add_argument("--merge-shards", action="store_true", help="Combine finished shards into the final output.")
    return p


def main() -> None:
    _bootstrap()

    import pandas as pd

    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
    from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
    from amr_cascade_platform.cascade.workflows import validated_pattern_sensitivity as vps
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
    from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
    from amr_cascade_platform.core.utils.scopes import scoped_output_dir
    from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore

    LoggerFactory.configure()
    logger = logging.getLogger(__name__)

    args = _build_arg_parser().parse_args()
    if args.shard_total < 1 or not 0 <= args.shard_index < args.shard_total:
        raise SystemExit("--shard-index must satisfy 0 <= index < --shard-total.")
    if args.merge_shards and args.shard_total < 2:
        raise SystemExit("--merge-shards needs --shard-total of at least 2.")
    analysis = vps.ANALYSES[args.analysis]

    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)
    dataset_store = DatasetStore(settings)

    gold_dir = scoped_output_dir(root=path_manager.paths.gold, scope=args.gold_scope, site=args.site, organism=args.organism)
    pair_path = gold_dir / "drug_pair_episodes.parquet"
    if not pair_path.exists():
        raise DataDiscoveryError(f"Gold drug-pair dataset not found: {pair_path}")
    cascade_dir = scoped_output_dir(
        root=path_manager.paths.artifacts / settings.cascade.outputs.result_dir,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    validation_path = cascade_dir / "validation_results.parquet"
    if not validation_path.exists():
        raise DataDiscoveryError(
            f"validation_results.parquet not found at {validation_path}; run merge_cascade_validation_shards.py first."
        )

    output_path = cascade_dir / analysis.output_name
    if output_path.exists():
        logger.info("%s results already exist: %s", args.analysis, output_path)
        return
    edges = vps.select_edges(dataset_store.read_pandas(validation_path), analysis.statuses)
    logger.info("%s: %d patterns selected from %s", args.analysis, len(edges), validation_path)

    shard_dir = cascade_dir / analysis.shard_dir_name
    if args.merge_shards:
        paths = [vps.shard_path(shard_dir, index, args.shard_total) for index in range(args.shard_total)]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise DataDiscoveryError(f"{len(missing)} of {args.shard_total} {args.analysis} shard(s) missing, e.g. {missing[0]}")
        merged = vps.merge_shard_frames([dataset_store.read_pandas(path) for path in paths], vps.expected_keys(edges))
        merged.to_parquet(output_path, index=False)
        logger.info("%s shards merged: %s (%d patterns)", args.analysis, output_path, len(merged))
        return

    sharded = args.shard_total > 1
    selected = vps.shard_slice(edges, args.shard_index, args.shard_total) if sharded else edges
    target = vps.shard_path(shard_dir, args.shard_index, args.shard_total) if sharded else output_path
    if target.exists():
        logger.info("%s shard already exists: %s", args.analysis, target)
        return

    analyzer = CascadeValidationAnalyzer(settings)
    if selected.empty:
        logger.warning("No patterns in this slice -- writing an empty result.")
        result = analysis.run(analyzer, pd.DataFrame(), selected)
    else:
        drug_pairs = dataset_store.read_pandas(pair_path)
        filtered_pairs, _ = CoTestingFilterAnalyzer(settings).filter(drug_pairs)
        del drug_pairs
        result = analysis.run(analyzer, filtered_pairs, selected)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    logger.info("%s results written: %s (%d patterns)", args.analysis, target, len(result))


if __name__ == "__main__":
    main()
