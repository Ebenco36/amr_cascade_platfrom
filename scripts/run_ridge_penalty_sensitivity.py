#!/usr/bin/env python
"""Refit the adjusted downstream-observation model at several ridge penalties.

Covers the same validated (robust/supported) patterns and covariates as the primary
adjusted model, pooled over all sites, so the refit at the primary penalty reproduces
the primary adjusted odds ratios. Must run after merge_cascade_validation_shards.py.

Usage:
  python scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined \\
      --organism "ESCHERICHIA COLI" --penalties 1,10,100
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
    p = argparse.ArgumentParser(description="Refit the adjusted model at alternative ridge penalties.")
    p.add_argument("--env", default=None, help="Runtime environment (e.g. mac, hpc).")
    p.add_argument("--gold-scope", choices=("combined", "site"), default="combined")
    p.add_argument("--site", default=None)
    p.add_argument("--organism", default=None)
    p.add_argument("--penalties", default="1,10,100", help="Comma-separated ridge penalties.")
    return p


def main() -> None:
    _bootstrap()

    import pandas as pd

    from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import ConditionalProbabilityAnalyzer
    from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
    from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import EscalationRatioAnalyzer
    from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
    from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
    from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import (
        _PAIR_TABLE_CATEGORICAL_COLUMNS,
        CascadeAnalysisWorkflow,
    )
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
    from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
    from amr_cascade_platform.core.utils.scopes import scoped_output_dir
    from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore

    LoggerFactory.configure()
    logger = logging.getLogger(__name__)

    args = _build_arg_parser().parse_args()
    penalties = [float(value) for value in args.penalties.split(",") if value.strip()]
    if not penalties or any(value <= 0 for value in penalties):
        raise SystemExit("--penalties must list one or more positive values.")

    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)
    dataset_store = DatasetStore(settings)

    gold_dir = scoped_output_dir(root=path_manager.paths.gold, scope=args.gold_scope, site=args.site, organism=args.organism)
    pair_path = gold_dir / "drug_pair_episodes.parquet"
    culture_episode_path = gold_dir / "culture_episodes.parquet"
    for path in (pair_path, culture_episode_path):
        if not path.exists():
            raise DataDiscoveryError(f"Gold dataset not found: {path}")
    cascade_dir = scoped_output_dir(
        root=path_manager.paths.artifacts / settings.cascade.outputs.result_dir,
        scope=args.gold_scope,
        site=args.site,
        organism=args.organism,
    )
    validation_path = cascade_dir / "validation_results.parquet"
    if not validation_path.exists():
        raise DataDiscoveryError(f"validation_results.parquet not found at {validation_path}; run the validation merge first.")
    output_path = cascade_dir / "ridge_penalty_sensitivity.parquet"
    if output_path.exists():
        logger.info("Ridge-penalty sensitivity results already exist: %s", output_path)
        return

    drug_pairs = dataset_store.read_pandas(pair_path, categorical_columns=_PAIR_TABLE_CATEGORICAL_COLUMNS)
    filtered_pairs, _ = CoTestingFilterAnalyzer(settings).filter(drug_pairs)
    del drug_pairs
    escalation_results = EscalationRatioAnalyzer(settings).analyze(ConditionalProbabilityAnalyzer(settings).analyze(filtered_pairs))
    validated = CascadeAnalysisWorkflow._validated_edge_results(escalation_results, dataset_store.read_pandas(validation_path))
    logger.info("Refitting %d validated patterns at ridge penalties %s", len(validated), penalties)

    covariates = CascadeCovariateBuilder(settings, path_manager).build(dataset_store.read_pandas(culture_episode_path))
    results = []
    for penalty in penalties:
        fitted = DownstreamTestingRegression(settings, path_manager, ridge_penalty=penalty).analyze_with_covariates(
            filtered_pairs, validated, covariates
        )
        results.append(fitted.assign(ridge_penalty=penalty))
        logger.info("Ridge penalty %g: %d of %d patterns estimable", penalty, int(fitted["adjusted_odds_ratio"].notna().sum()), len(fitted))
    output = pd.concat(results, ignore_index=True).loc[
        :,
        [
            "upstream_antibiotic",
            "downstream_antibiotic",
            "ridge_penalty",
            "adjusted_odds_ratio",
            "adjusted_odds_ratio_ci_lower",
            "adjusted_odds_ratio_ci_upper",
            "non_estimable_reason",
            "modeled_n",
        ],
    ]
    output.to_parquet(output_path, index=False)
    logger.info("Ridge-penalty sensitivity written: %s", output_path)


if __name__ == "__main__":
    main()
