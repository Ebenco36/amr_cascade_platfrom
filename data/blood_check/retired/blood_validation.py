"""Blood-culture-only detection and validation (post hoc secondary analysis, 20 Nov 2026).
Runs the pipeline's own classes, with the final run's settings (env hpc: B = 1,000 permutations and bootstraps, PBI 0.95,
>= 25 R/S rows, >= 5 per branch, >= 5 observed), on the episode-pair rows of blood-culture episodes only:
co-testing screen -> conditional probabilities -> ER -> retention -> validation (permutation within site, bootstrap, site and
temporal replication, BH across the retained blood candidates). Reads export_archives/hpc_final (read-only).
    .venv/bin/python manuscript_edit_log/2026-11-20/blood_check/blood_validation.py [--workers 12]"""
import multiprocessing as mp, sys, time
from pathlib import Path
import pandas as pd, pyarrow.dataset as ds
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.cascade.workflows import cascade_analysis_workflow as W

OUT = Path(__file__).parent / "blood_run"; OUT.mkdir(exist_ok=True)
G = ROOT / "export_archives/hpc_final/data/gold/combined/organisms/escherichia_coli"
SETTINGS = ConfigLoader(ROOT).load("hpc")
PAIRS = None


def _shard(args):
    i, n, retained = args
    return CascadeValidationAnalyzer(SETTINGS).analyze_shard(PAIRS, retained, i, n)


def main(workers: int):
    global PAIRS
    t0 = time.time()
    ep = pd.read_parquet(G / "culture_episodes.parquet", columns=["source_site", "order_proc_id_coded", "culture_description"])
    ep["specimen"] = ep.culture_description.map(CascadeCovariateBuilder._specimen_type_category)
    blood = ep[ep.specimen.eq("blood")]
    tbl = ds.dataset(G / "drug_pair_episodes.parquet").to_table(filter=ds.field("order_proc_id_coded").isin(blood.order_proc_id_coded.unique().tolist()))
    pairs = tbl.to_pandas().merge(blood[["source_site", "order_proc_id_coded"]], on=["source_site", "order_proc_id_coded"], how="inner")
    for col in W._PAIR_TABLE_CATEGORICAL_COLUMNS:
        if col in pairs.columns: pairs[col] = pairs[col].astype("category")
    print(f"blood episodes {len(blood):,}; episode-pair rows {len(pairs):,} ({time.time()-t0:.0f}s)", flush=True)
    filtered, cotesting, cot_prob = W.CoTestingFilterAnalyzer(SETTINGS).filter_with_probabilities(pairs)
    cond = W.ConditionalProbabilityAnalyzer(SETTINGS).analyze(filtered)
    esc = W.EscalationRatioAnalyzer(SETTINGS).analyze(cond)
    reg = W.DownstreamTestingRegression(SETTINGS, None, explain=False)
    retained = W.RetainedEdgeAnalyzer(SETTINGS).analyze(esc, reg._empty_results())
    for name, frame in (("cotesting_pairs", cotesting), ("escalation_results", esc), ("retained_edges", retained)):
        frame.to_parquet(OUT / f"{name}.parquet")
    print(f"co-testing screen removed {len(cotesting)} pairs; retained candidates {len(retained)} ({time.time()-t0:.0f}s)", flush=True)
    PAIRS = filtered
    n = max(1, min(workers * 4, len(retained)))
    with mp.get_context("fork").Pool(workers) as pool:
        shards = []
        for k, s in enumerate(pool.imap_unordered(_shard, [(i, n, retained) for i in range(n)]), 1):
            shards.append(s); print(f"  shard {k}/{n} done ({time.time()-t0:.0f}s)", flush=True)
    val = CascadeValidationAnalyzer(SETTINGS).merge_shards(shards)
    val.to_parquet(OUT / "validation_results.parquet")
    print(val.validation_status.value_counts().to_dict(), f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12)
