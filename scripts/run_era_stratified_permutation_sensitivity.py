#!/usr/bin/env python
"""Run the site x era-stratified permutation sensitivity check.

CascadeValidationAnalyzer.permutation_summary_era_stratified() tests whether
the primary two-sided permutation result could instead reflect shared
within-site temporal drift (e.g. a panel or reporting-policy change) rather
than a drug-specific result-conditioned association -- see that method's
docstring. It is scoped to the validated (robust/supported) pattern set, not
the full candidate set, so it must run AFTER merge_cascade_validation_shards.py
has produced the final validation_results.parquet for this scope.

Usage:
  python scripts/run_era_stratified_permutation_sensitivity.py \\
      --env hpc --gold-scope combined --organism "ESCHERICHIA COLI"
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
        description="Run the era-stratified permutation sensitivity check on validated edges."
    )
    p.add_argument("--env", default=None, help="Runtime environment (e.g. mac, hpc).")
    p.add_argument(
        "--gold-scope",
        choices=("combined", "site"),
        default="combined",
    )
    p.add_argument("--site", default=None)
    p.add_argument("--organism", default=None)
    return p


def main() -> None:
    _bootstrap()

    import pandas as pd

    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import (
        CascadeValidationAnalyzer,
    )
    from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import (
        CoTestingFilterAnalyzer,
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

    gold_dir = scoped_output_dir(
        root=path_manager.paths.gold,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
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
            f"validation_results.parquet not found at {validation_path}. This sensitivity "
            f"check runs on the FINAL validated pattern set, so run "
            f"merge_cascade_validation_shards.py first."
        )

    output_path = cascade_dir / "era_stratified_permutation_sensitivity.parquet"
    if output_path.exists():
        logger.info("Era-stratified sensitivity results already exist: %s", output_path)
        return

    validation_results = dataset_store.read_pandas(validation_path)
    validated_edges = validation_results.loc[
        validation_results["validation_status"].isin(CascadeValidationAnalyzer.VALIDATED_STATUSES),
        ["upstream_antibiotic", "downstream_antibiotic", "observed_escalation_ratio"],
    ].rename(columns={"observed_escalation_ratio": "escalation_ratio"})
    logger.info(
        "Loaded %d validated (robust/supported) edges from %s",
        len(validated_edges), validation_path,
    )
    if validated_edges.empty:
        logger.warning("No robust/supported edges to check -- writing empty output.")
        pd.DataFrame(
            columns=[
                "upstream_antibiotic",
                "downstream_antibiotic",
                "observed_escalation_ratio",
                "permutation_p_value_two_sided_era_stratified",
            ]
        ).to_parquet(output_path, index=False)
        return

    drug_pairs = dataset_store.read_pandas(pair_path)
    cotesting_filter = CoTestingFilterAnalyzer(settings)
    filtered_pairs, _ = cotesting_filter.filter(drug_pairs)
    del drug_pairs

    analyzer = CascadeValidationAnalyzer(settings)
    result = analyzer.era_stratified_sensitivity_summary(filtered_pairs, validated_edges)

    cascade_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    n_finite = int(result["permutation_p_value_two_sided_era_stratified"].notna().sum())
    logger.info(
        "Era-stratified sensitivity results written: %s (%d edges, %d with a finite p-value)",
        output_path, len(result), n_finite,
    )


if __name__ == "__main__":
    main()
