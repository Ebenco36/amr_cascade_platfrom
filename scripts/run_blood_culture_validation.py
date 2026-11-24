"""Blood-culture-only detection and validation of directional observation patterns (paper 1, Supplementary Note
"Blood-culture analysis").

Repeats the primary candidate screening and validation on the episode-pair rows of blood-culture episodes only, with the
pipeline's own classes and the chosen environment's settings (``hpc``: 1,000 permutations and bootstraps, PBI 0.95, at
least 25 resistant-or-susceptible rows, 5 per upstream branch and 5 downstream-observed episodes):

    co-testing screen -> conditional probabilities -> escalation ratios -> retention ->
    validation (within-site permutation, bootstrap, site and temporal replication; Benjamini-Hochberg across the
    blood-culture candidates)

and compares the resulting labels with the primary (all-specimen) validation of the same run.

Inputs (read only), below ``--run-root``:
    data/gold/combined/organisms/escherichia_coli/{culture_episodes,drug_pair_episodes}.parquet
    data/artifacts/cascade/combined/organisms/escherichia_coli/validation_results.parquet

Outputs, in ``--out-dir``:
    cotesting_pairs.parquet, escalation_results.parquet, retained_edges.parquet, validation_results.parquet
    blood_vs_primary_validation.csv   (one row per pair: blood and primary ER and label)

Adjusted models (``--adjust``): the pair-specific covariate-adjusted logistic models of the primary analysis are fitted to
the pairs validated in blood cultures, with the covariates built from the harmonised site tables (so they must be the
final run's: on the HPC). Specimen type is constant within blood cultures and so drops out. With ``--validation-dir`` the
validation is read from an earlier run instead of recomputed (screening is recomputed and must agree with it). Writes
adjusted_results.parquet and adjusted_direction_summary.json (the same summary as the primary 344/380).
    .venv/bin/python scripts/run_blood_culture_validation.py --env hpc --run-root . --adjust \\
        --validation-dir <folder with the blood run> --out-dir <output folder>
The script stops if any site's harmonised covariate tables are missing.

Usage (about 3.5 hours with 12 workers on an Apple M-series laptop):
    .venv/bin/python scripts/run_blood_culture_validation.py --run-root export_archives/hpc_final \\
        --out-dir manuscript_edit_log/2026-11-20/blood_check/blood_run --workers 12
Validation results are seeded per pair, so they do not depend on the number of workers. ``--max-pairs N`` validates only
the first N retained pairs, for a quick check of the set-up (its Benjamini-Hochberg step then covers those N pairs only).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

ORGANISM_DIR = "combined/organisms/escherichia_coli"
PAIR_KEYS = ["upstream_antibiotic", "downstream_antibiotic"]
VALIDATED = ("robust", "supported")
_STATE: dict = {}


def _bootstrap() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blood-culture-only screening and validation of directional observation patterns.")
    p.add_argument("--env", default="hpc", help="Settings environment (default: hpc, the settings of the final run).")
    p.add_argument("--run-root", default="export_archives/hpc_final", help="Folder that contains data/gold and data/artifacts.")
    p.add_argument("--out-dir", required=True, help="Output folder.")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2), help="Parallel worker processes.")
    p.add_argument("--max-pairs", type=int, default=None, help="Validate only the first N retained pairs (quick check).")
    p.add_argument("--adjust", action="store_true", help="Fit the covariate-adjusted models to the blood-validated pairs.")
    p.add_argument("--validation-dir", default=None, help="Read validation_results.parquet from this earlier blood run.")
    p.add_argument("--smoke-test-sites", default=None, help=argparse.SUPPRESS)  # code-path test only: restrict to these sites
    return p


def _validate_shard(task: tuple[int, int]):
    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer

    index, total = task
    return CascadeValidationAnalyzer(_STATE["settings"]).analyze_shard(_STATE["pairs"], _STATE["retained"], index, total)


def main() -> None:
    _bootstrap()
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds

    from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
    from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import ConditionalProbabilityAnalyzer
    from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
    from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import EscalationRatioAnalyzer
    from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
    from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
    from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
    from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import _PAIR_TABLE_CATEGORICAL_COLUMNS
    from amr_cascade_platform.cli.main import build_context

    logging.basicConfig(level=logging.WARNING)
    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_root = (project_root / args.run_root).resolve()
    gold = run_root / "data/gold" / ORGANISM_DIR
    primary = run_root / "data/artifacts/cascade" / ORGANISM_DIR / "validation_results.parquet"
    out = (project_root / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings, path_manager, _ = build_context(project_root, args.env)
    t0 = time.time()

    # Blood-culture episodes, classified exactly as the covariate builder classifies specimen type.
    episodes = pd.read_parquet(gold / "culture_episodes.parquet", columns=["source_site", "order_proc_id_coded", "culture_description"])
    episodes["specimen"] = episodes["culture_description"].map(CascadeCovariateBuilder._specimen_type_category)
    blood = episodes.loc[episodes["specimen"].eq("blood"), ["source_site", "order_proc_id_coded"]]
    if args.smoke_test_sites:
        blood = blood[blood["source_site"].isin(args.smoke_test_sites.split(","))]
    table = ds.dataset(gold / "drug_pair_episodes.parquet").to_table(
        filter=ds.field("order_proc_id_coded").isin(blood["order_proc_id_coded"].unique().tolist()))
    pairs = table.to_pandas().merge(blood, on=["source_site", "order_proc_id_coded"], how="inner")
    for column in _PAIR_TABLE_CATEGORICAL_COLUMNS:
        if column in pairs.columns:
            pairs[column] = pairs[column].astype("category")
    print(f"blood-culture episodes {len(blood):,}; episode-pair rows {len(pairs):,}", flush=True)

    # Screening and retention, as in the primary cascade workflow (retention uses the unadjusted pattern).
    filtered, cotesting, _ = CoTestingFilterAnalyzer(settings).filter_with_probabilities(pairs)
    escalation = EscalationRatioAnalyzer(settings).analyze(ConditionalProbabilityAnalyzer(settings).analyze(filtered))
    no_adjustment = DownstreamTestingRegression(settings, None, explain=False)._empty_results()
    retained = RetainedEdgeAnalyzer(settings).analyze(escalation, no_adjustment)
    if args.max_pairs:
        retained = retained.head(args.max_pairs)
    for name, frame in (("cotesting_pairs", cotesting), ("escalation_results", escalation), ("retained_edges", retained)):
        frame.to_parquet(out / f"{name}.parquet")
    print(f"co-testing screen removed {len(cotesting)} pairs; retained {len(retained)}", flush=True)

    if args.validation_dir:
        # Reuse an earlier validation; it must belong to exactly this screening.
        source = (project_root / args.validation_dir).resolve()
        validation = pd.read_parquet(source / "validation_results.parquet")
        saved = pd.read_parquet(source / "retained_edges.parquet")[PAIR_KEYS]
        same = saved.sort_values(PAIR_KEYS).reset_index(drop=True).equals(retained[PAIR_KEYS].sort_values(PAIR_KEYS).reset_index(drop=True))
        if not same and not args.smoke_test_sites:
            raise SystemExit(f"The retained pairs differ from those of {source}: the validation cannot be reused.")
        print(f"validation read from {source}", flush=True)
    else:
        # Validation in shards across worker processes (fork: the workers share the pair table), then Benjamini-Hochberg.
        _STATE.update(settings=settings, pairs=filtered, retained=retained)
        total = max(1, min(args.workers * 4, len(retained)))
        shards = []
        with mp.get_context("fork").Pool(args.workers) as pool:
            for done, shard in enumerate(pool.imap_unordered(_validate_shard, [(i, total) for i in range(total)]), 1):
                shards.append(shard)
                print(f"  shard {done}/{total} done ({time.time() - t0:.0f} s)", flush=True)
        validation = CascadeValidationAnalyzer(settings).merge_shards(shards)
    validation.to_parquet(out / "validation_results.parquet")
    print(validation["validation_status"].value_counts().to_dict(), flush=True)

    # Comparison with the primary (all-specimen) validation.
    b = validation[PAIR_KEYS + ["observed_escalation_ratio", "validation_status"]].rename(
        columns={"observed_escalation_ratio": "er_blood", "validation_status": "label_blood"})
    p = pd.read_parquet(primary)[PAIR_KEYS + ["observed_escalation_ratio", "validation_status"]].rename(
        columns={"observed_escalation_ratio": "er_primary", "validation_status": "label_primary"})
    both = b.merge(p, on=PAIR_KEYS, how="outer")
    both["validated_blood"] = both["label_blood"].isin(VALIDATED)
    both["validated_primary"] = both["label_primary"].isin(VALIDATED)
    both["same_direction"] = np.sign(np.log(both["er_blood"])) == np.sign(np.log(both["er_primary"]))
    both.to_csv(out / "blood_vs_primary_validation.csv", index=False)
    v = both[both["validated_blood"]]
    print(f"validated in blood {len(v)}: same direction as primary-validated {int((v.validated_primary & v.same_direction).sum())}, "
          f"opposite {int((v.validated_primary & ~v.same_direction).sum())}, primary candidate not validated "
          f"{int((v.label_primary.notna() & ~v.validated_primary).sum())}, not a primary candidate {int(v.label_primary.isna().sum())} "
          f"({time.time() - t0:.0f} s)", flush=True)

    if args.adjust:
        _adjust(args, settings, path_manager, gold, out, filtered, escalation, validation, both, t0)


def _adjust(args, settings, path_manager, gold, out, filtered, escalation, validation, both, t0) -> None:
    """Covariate-adjusted models for the blood-validated pairs, as in the primary cascade workflow."""
    import json
    import pandas as pd
    from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
    from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import CascadeAnalysisWorkflow
    from amr_cascade_platform.reporting.builders.supplementary_table_builder import adjusted_direction_summary

    keys = list(settings.gold.episode_key_columns)
    episodes = pd.read_parquet(gold / "culture_episodes.parquet")
    blood_keys = filtered[["source_site", "order_proc_id_coded"]].drop_duplicates().astype(str)
    episodes = episodes.merge(blood_keys, on=["source_site", "order_proc_id_coded"], how="inner")
    # The covariates come from the harmonised site tables; a missing table would silently become missing covariates.
    needed = ["comorbidity", "antibiotic_class_exposure", "prior_infecting_organism", "labs", "vitals",
              "nursing_home_visits", "prior_procedures"]
    missing = [f"{site}/{table}" for site in sorted(episodes["source_site"].astype(str).unique()) for table in needed
               if not (path_manager.paths.harmonized / "site_aligned" / site / f"{table}.parquet").exists()]
    if missing:
        raise SystemExit("Harmonised covariate tables missing (run on the HPC, where the final run's tables are): " + ", ".join(missing))
    validated_edges = CascadeAnalysisWorkflow._validated_edge_results(escalation, validation)
    adjusted = DownstreamTestingRegression(settings, path_manager, explain=False).analyze(filtered, validated_edges, episodes)
    adjusted.to_parquet(out / "adjusted_results.parquet")
    summary = adjusted_direction_summary(adjusted, validation)
    json.dump(summary, open(out / "adjusted_direction_summary.json", "w"), indent=1, default=float)
    print(f"adjusted models: {summary['consistent']} of {summary['estimable']} estimable blood-validated pairs kept their direction "
          f"({summary['validated']} validated; {time.time() - t0:.0f} s)", flush=True)
    # The pairs validated in opposite directions in blood cultures and in the primary analysis: does adjustment keep the blood direction?
    opposite = both[both["validated_blood"] & both["validated_primary"] & ~both["same_direction"]]
    view = opposite.merge(adjusted[PAIR_KEYS + ["adjusted_odds_ratio", "non_estimable_reason"]], on=PAIR_KEYS, how="left")
    view.to_csv(out / "opposite_direction_adjusted.csv", index=False)
    print(view[PAIR_KEYS + ["er_blood", "er_primary", "adjusted_odds_ratio", "non_estimable_reason"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
