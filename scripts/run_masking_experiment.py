#!/usr/bin/env python
"""Masking experiment for paper 2: do the prevalence estimators recover a known truth?

For a near-universally tested antibiotic D (the "truth" antibiotic), resistance among the
episodes with a resistant or susceptible result for D is known. Results for D are then hidden
in a selective pattern and the paper's own estimators (PrevalenceShiftAnalyzer: tested-only
prevalence, model-free bounds, standardised prevalence and the MNAR odds-tilt curve) are run
on the masked data and compared with the known value.

Two kinds of selective pattern:
  transplanted  D's result is kept only where a selectively tested "donor" antibiotic T was
                observed in the same episode, so D inherits T's real observation pattern
                (including its dependence on other antibiotics' results and on case mix).
                Population: episodes eligible for both D and T with a binary result for D.
  synthetic     D's result is kept with probability q_S if susceptible and q_R if resistant,
                chosen so that the true odds tilt is a known lambda. The MNAR curve at that
                lambda should then recover the truth.

Nothing in the pipeline's outputs is changed; results go to --output-dir, with masking_run.json
recording how they were made. The paper tables read them only from a run on the full data
(SupplementaryTableBuilder ignores results made with --data-root).

    .venv/bin/python scripts/run_masking_experiment.py --env hpc --organism "ESCHERICHIA COLI" \
        --output-dir outputs/tables/combined/organisms/escherichia_coli/masking_experiment
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder  # noqa: E402
from amr_cascade_platform.cli.main import build_context  # noqa: E402
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label  # noqa: E402
from amr_cascade_platform.core.utils.scopes import scoped_output_dir  # noqa: E402
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore  # noqa: E402
from amr_cascade_platform.surveillance.prevalence_shift_analyzer import PrevalenceShiftAnalyzer  # noqa: E402

TRUTH = ("CIPROFLOXACIN", "CEFAZOLIN", "CEFTRIAXONE", "PIPERACILLIN/TAZOBACTAM")  # tested in 86-95% of eligible episodes
DONORS = ("MOXIFLOXACIN", "TIGECYCLINE", "MINOCYCLINE", "CEFOXITIN", "AZTREONAM", "CEFEPIME")
SYNTHETIC_LAMBDAS = (-1.0, 0.0, 1.0)
SYNTHETIC_KEEP_SUSCEPTIBLE = 0.30
# Fine grid so the curve can be read at any lambda in [-3, 3]; includes the synthetic values.
FINE_GRID = tuple(round(x, 2) for x in np.arange(-3.0, 3.0001, 0.05))


def keep_probabilities(lam: float, q_s: float = SYNTHETIC_KEEP_SUSCEPTIBLE) -> tuple[float, float]:
    """q_R such that the observed-versus-unobserved odds ratio of resistance is exp(lambda)."""
    odds = math.exp(lam) * q_s / (1 - q_s)
    return odds / (1 + odds), q_s


def curve_value(curve: pd.DataFrame, lam: float) -> float:
    c = curve.dropna(subset=["mnar_prevalence"]).sort_values("mnar_lambda")
    return float(np.interp(lam, c["mnar_lambda"], c["mnar_prevalence"])) if len(c) else math.nan


def implied_lambda(curve: pd.DataFrame, truth: float) -> float:
    """Lambda at which the MNAR curve equals the truth (curves decrease in lambda); NaN if outside."""
    c = curve.dropna(subset=["mnar_prevalence"]).sort_values("mnar_lambda")
    if c.empty or not (c["mnar_prevalence"].min() <= truth <= c["mnar_prevalence"].max()):
        return math.nan
    return float(np.interp(truth, c["mnar_prevalence"].to_numpy()[::-1], c["mnar_lambda"].to_numpy()[::-1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="hpc")
    ap.add_argument("--organism", default="ESCHERICHIA COLI")
    ap.add_argument("--truth", nargs="*", default=list(TRUTH))
    ap.add_argument("--donors", nargs="*", default=list(DONORS))
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-root", default=None, help="Override the environment's data root (relative to the project), e.g. for a test.")
    args = ap.parse_args()

    settings, paths, _ = build_context(ROOT, args.env)
    if args.data_root:
        from amr_cascade_platform.core.paths.path_manager import PathManager
        settings = replace(settings, environment=replace(settings.environment, data_root=args.data_root))
        paths = PathManager(ROOT, settings)
    settings = replace(settings, prevalence=replace(settings.prevalence, mnar_lambda_grid=list(FINE_GRID)))
    store = DatasetStore(settings)
    gold = scoped_output_dir(paths.paths.gold, "combined", organism=args.organism)
    cascade = scoped_output_dir(paths.paths.artifacts / settings.cascade.outputs.result_dir, "combined", organism=args.organism)
    keys = list(settings.gold.episode_key_columns)

    started = time.time()
    eligible = store.read_pandas(gold / "eligible_pairs.parquet")
    culture_drug = store.read_pandas(gold / "culture_drug_episodes.parquet")
    culture_episodes = store.read_pandas(gold / "culture_episodes.parquet")
    edges = store.read_pandas(cascade / "edge_report.parquet")
    for frame in (eligible, culture_drug):
        frame["antibiotic"] = frame["antibiotic"].map(normalize_antibiotic_label)
    covariates = CascadeCovariateBuilder(settings, paths).build(culture_episodes)
    print(f"inputs loaded in {time.time() - started:.0f}s: {len(eligible):,} eligible rows, "
          f"{len(culture_drug):,} result rows, {len(covariates):,} episodes with covariates", flush=True)

    analyzer = PrevalenceShiftAnalyzer(settings)
    positive, negative = settings.cascade.upstream_result_positive, PrevalenceShiftAnalyzer._EVALUABLE_NEGATIVE
    collapsed = PrevalenceShiftAnalyzer._collapse_observed_results(
        culture_drug.loc[:, keys + ["antibiotic", "susceptibility"]].copy(), keys + ["antibiotic"])
    elig_sets = {a: g.loc[:, keys].drop_duplicates() for a, g in eligible[eligible["is_eligible"] == 1].groupby("antibiotic", observed=True)}
    available = set(elig_sets)

    def run(truth_drug: str, population: pd.DataFrame, kept: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        e = population.assign(antibiotic=truth_drug, is_eligible=1)
        others = culture_drug[culture_drug["antibiotic"] != truth_drug]
        mine = culture_drug[culture_drug["antibiotic"] == truth_drug].merge(kept, on=keys, how="inner")
        bundle = analyzer.analyze(e, pd.concat([others, mine], ignore_index=True), covariates, edges)
        row = bundle.pair_results[bundle.pair_results["drug"] == truth_drug]
        curve = bundle.mnar_curve_results[bundle.mnar_curve_results["drug"] == truth_drug]
        return (row.iloc[0] if len(row) else pd.Series(dtype=object)), curve

    rows, curves = [], []
    rng = np.random.default_rng(args.seed)
    for truth_drug in map(normalize_antibiotic_label, args.truth):
        if truth_drug not in available:
            print(f"skip truth {truth_drug}: not eligible in this run", flush=True); continue
        result = collapsed[collapsed["antibiotic"] == truth_drug]
        binary = result[result["susceptibility"].isin([positive, negative])]
        base = elig_sets[truth_drug].merge(binary.loc[:, keys + ["susceptibility"]], on=keys, how="inner")
        truth_eligible_n = len(elig_sets[truth_drug])
        scenarios = []
        for donor in map(normalize_antibiotic_label, args.donors):
            if donor not in available or donor == truth_drug:
                print(f"skip donor {donor} for {truth_drug}", flush=True); continue
            population = base.merge(elig_sets[donor], on=keys, how="inner")
            donor_observed = collapsed.loc[collapsed["antibiotic"] == donor, keys].drop_duplicates()
            kept = population.loc[:, keys].merge(donor_observed, on=keys, how="inner")
            scenarios.append(("transplanted", donor, math.nan, population, kept))
        for lam in SYNTHETIC_LAMBDAS:
            q_r, q_s = keep_probabilities(lam)
            resistant = base["susceptibility"].eq(positive).to_numpy()
            keep = rng.random(len(base)) < np.where(resistant, q_r, q_s)
            scenarios.append(("synthetic", f"lambda={lam:+.1f}", lam, base, base.loc[keep, keys]))
        for kind, label, lam_true, population, kept in scenarios:
            t0 = time.time()
            truth = float(population["susceptibility"].eq(positive).mean())
            row, curve = run(truth_drug, population.loc[:, keys], kept)
            if row.empty:
                print(f"{truth_drug} / {label}: not estimable (support rules)", flush=True); continue
            lam_star = implied_lambda(curve, truth)
            rows.append({
                "truth_antibiotic": truth_drug, "mechanism": kind, "pattern": label, "true_lambda": lam_true,
                "truth_eligible_n": truth_eligible_n, "truth_binary_share": len(base) / truth_eligible_n,
                "population_n": len(population), "kept_n": len(kept), "kept_share": len(kept) / len(population),
                "truth_pct": 100 * truth, "naive_pct": float(row["naive_prevalence_pct"]),
                "lower_bound_pct": float(row["prevalence_lower_bound_pct"]), "upper_bound_pct": float(row["prevalence_upper_bound_pct"]),
                "standardised_pct": float(row["standardised_prevalence_pct"]) if pd.notna(row["standardised_prevalence_pct"]) else math.nan,
                "mnar_lambda0_pct": float(row["mnar_lambda0_prevalence_pct"]) if pd.notna(row["mnar_lambda0_prevalence_pct"]) else math.nan,
                "mnar_at_true_lambda_pct": 100 * curve_value(curve, lam_true) if not math.isnan(lam_true) else math.nan,
                "mnar_curve_min_pct_grid_pm2": 100 * curve[curve["mnar_lambda"].between(-2, 2)]["mnar_prevalence"].min(),
                "mnar_curve_max_pct_grid_pm2": 100 * curve[curve["mnar_lambda"].between(-2, 2)]["mnar_prevalence"].max(),
                "implied_lambda": lam_star, "cascade_trigger_fraction": float(row["cascade_trigger_fraction"]),
            })
            curves.append(curve.assign(truth_antibiotic=truth_drug, mechanism=kind, pattern=label, truth_pct=100 * truth))
            r = rows[-1]
            print(f"{truth_drug:24s} {label:14s} kept {r['kept_share']:5.1%} truth {r['truth_pct']:5.1f} naive {r['naive_pct']:5.1f} "
                  f"lambda0 {r['mnar_lambda0_pct']:5.1f} bounds {r['lower_bound_pct']:5.1f}-{r['upper_bound_pct']:5.1f} "
                  f"implied lambda {lam_star:+.2f} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.output_dir); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    for col in ("naive", "standardised", "mnar_lambda0", "mnar_at_true_lambda"):
        results[f"{col}_error_pp"] = results[f"{col}_pct"] - results["truth_pct"]
    results["truth_within_bounds"] = results["truth_pct"].between(results["lower_bound_pct"], results["upper_bound_pct"])
    results["truth_within_curve_pm2"] = results["truth_pct"].between(results["mnar_curve_min_pct_grid_pm2"], results["mnar_curve_max_pct_grid_pm2"])
    results.to_csv(out / "masking_results.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out / "masking_curves.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "masking_run.json").write_text(json.dumps({
        "env": args.env, "organism": args.organism, "data_root_override": args.data_root, "seed": args.seed,
        "truth": args.truth, "donors": args.donors, "git_commit": commit, "scenarios": len(results),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} scenarios to {out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
