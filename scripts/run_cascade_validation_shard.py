#!/usr/bin/env python
"""Run one shard of the cascade validation loop without applying BH-FDR.

The full 1,637-edge E. coli cascade validation at B=1000 takes ~800 hours
sequentially -- far beyond any single SLURM wall-time limit.  This script
runs one slice of those edges so many such jobs can run in parallel.  After
all shards complete, merge_cascade_validation_shards.py applies global
BH-FDR across the combined result, and the main cascade job (run_cascade_analysis.py)
loads the merged parquet in place of re-running validation.

Usage (SLURM, 41 shards of ~40 edges each):
  python scripts/run_cascade_validation_shard.py \\
      --env hpc --gold-scope combined --organism "ESCHERICHIA COLI" \\
      --shard-index 0 --shard-total 41
"""

from __future__ import annotations

import argparse
import gc
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
        description="Validate one shard of cascade edges (raw p-values; no BH-FDR)."
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
        "--shard-index",
        type=int,
        required=True,
        help="Zero-based shard index (0 … shard-total-1).",
    )
    p.add_argument(
        "--shard-total",
        type=int,
        required=True,
        help="Total number of shards the edge set is split into.",
    )
    return p


_PAIR_TABLE_CATEGORICAL_COLUMNS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "order_time_jittered",
    "organism",
    "source_site",
    "upstream_antibiotic",
    "upstream_susceptibility",
    "downstream_antibiotic",
    "pair_direction",
]


def main() -> None:
    _bootstrap()

    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import (
        CascadeValidationAnalyzer,
    )
    from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import (
        ConditionalProbabilityAnalyzer,
    )
    from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import (
        CoTestingFilterAnalyzer,
    )
    from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import (
        EscalationRatioAnalyzer,
    )
    from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import (
        RetainedEdgeAnalyzer,
    )
    from amr_cascade_platform.cascade.statistics.downstream_testing_regression import (
        DownstreamTestingRegression,
    )
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
    from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
    from amr_cascade_platform.core.utils.scopes import scoped_output_dir
    from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore

    LoggerFactory.configure()
    logger = logging.getLogger(__name__)

    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)
    dataset_store = DatasetStore(settings)

    # ------------------------------------------------------------------ #
    # Locate gold inputs                                                   #
    # ------------------------------------------------------------------ #
    gold_dir = scoped_output_dir(
        root=path_manager.paths.gold,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    pair_path = gold_dir / "drug_pair_episodes.parquet"
    culture_episode_path = gold_dir / "culture_episodes.parquet"
    if not pair_path.exists():
        raise DataDiscoveryError(f"Gold drug-pair dataset not found: {pair_path}")
    if not culture_episode_path.exists():
        raise DataDiscoveryError(f"Gold culture-episode dataset not found: {culture_episode_path}")

    # ------------------------------------------------------------------ #
    # Determine output path and skip if already done                       #
    # ------------------------------------------------------------------ #
    shard_idx = args.shard_index
    shard_total = args.shard_total
    cascade_dir = scoped_output_dir(
        root=path_manager.paths.artifacts / settings.cascade.outputs.result_dir,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    shard_dir = cascade_dir / "validation_shards"
    shard_path = shard_dir / f"shard_{shard_idx:04d}_of_{shard_total:04d}.parquet"

    if shard_path.exists():
        logger.info("Shard already exists, nothing to do: %s", shard_path)
        return

    shard_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Replay the pre-validation workflow steps                             #
    # ------------------------------------------------------------------ #
    logger.info(
        "Cascade validation shard %d/%d | organism=%s scope=%s",
        shard_idx, shard_total, args.organism, args.gold_scope,
    )

    drug_pairs = dataset_store.read_pandas(
        pair_path, categorical_columns=_PAIR_TABLE_CATEGORICAL_COLUMNS
    )
    culture_episodes = dataset_store.read_pandas(culture_episode_path)

    cotesting_filter = CoTestingFilterAnalyzer(settings)
    filtered_pairs, _ = cotesting_filter.filter(drug_pairs)
    del drug_pairs
    gc.collect()

    conditional_analyzer = ConditionalProbabilityAnalyzer(settings)
    conditional_probabilities = conditional_analyzer.analyze(filtered_pairs)

    escalation_analyzer = EscalationRatioAnalyzer(settings)
    escalation_results = escalation_analyzer.analyze(conditional_probabilities)

    regression_analyzer = DownstreamTestingRegression(settings, path_manager)
    adjusted_results = regression_analyzer.analyze(
        filtered_pairs, escalation_results, culture_episodes
    )
    del culture_episodes
    gc.collect()

    retained_edge_analyzer = RetainedEdgeAnalyzer(settings)
    retained_edges = retained_edge_analyzer.analyze(escalation_results, adjusted_results)

    # ------------------------------------------------------------------ #
    # Run this shard's validation slice                                    #
    # ------------------------------------------------------------------ #
    validation_analyzer = CascadeValidationAnalyzer(settings)
    shard_result = validation_analyzer.analyze_shard(
        filtered_pairs, retained_edges, shard_idx, shard_total
    )

    if shard_result.empty:
        logger.warning(
            "Shard %d/%d produced no rows — writing empty parquet to mark completion.",
            shard_idx, shard_total,
        )

    shard_result.to_parquet(shard_path, index=False)
    logger.info("Shard %d/%d written: %s (%d rows)", shard_idx, shard_total, shard_path, len(shard_result))


if __name__ == "__main__":
    main()
