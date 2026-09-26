import json
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.gold.gold_build_manager import GoldBuildManager, GoldBuildRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _row(index: int, organism: str, antibiotic: str, susceptibility: str) -> dict:
    return {
        "anon_id": f"p{index}",
        "pat_enc_csn_id_coded": f"e{index}",
        "order_proc_id_coded": f"o{index}",
        "order_time_jittered": "2021-05-01 10:00:00",
        "ordering_mode": "Inpatient",
        "culture_description": "URINE",
        "was_positive": "1",
        "organism": organism,
        "antibiotic": antibiotic,
        "susceptibility": susceptibility,
        "source_site": "armd",
    }


def test_species_gold_build_pools_phenotype_annotated_labels(tmp_path: Path) -> None:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    paths = PathManager(tmp_path, settings)
    paths.paths.reference.mkdir(parents=True)
    for name in ("intrinsic_resistance.csv", "supplemental_intrinsic_resistance.csv"):
        (paths.paths.reference / name).symlink_to(PROJECT_ROOT / "data" / "reference" / name)
    cohort_dir = paths.paths.harmonized / "combined"
    cohort_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            _row(1, "ESCHERICHIA COLI", "CEFTRIAXONE", "SUSCEPTIBLE"),
            _row(1, "ESCHERICHIA COLI", "MEROPENEM", "SUSCEPTIBLE"),
            _row(2, "ESBL ESCHERICHIA COLI", "CEFTRIAXONE", "RESISTANT"),
            _row(2, "ESBL ESCHERICHIA COLI", "MEROPENEM", "SUSCEPTIBLE"),
            _row(3, "ESCHERICHIA FERGUSONII", "CEFTRIAXONE", "SUSCEPTIBLE"),
        ]
    ).to_parquet(cohort_dir / "cohort.parquet", index=False)

    outputs = GoldBuildManager(settings, paths).build(
        GoldBuildRequest(source_scope="combined", organism="ESCHERICHIA COLI")
    )

    episodes = pd.read_parquet(outputs["culture_episodes"])
    assert sorted(episodes["anon_id"]) == ["p1", "p2"]
    assert set(episodes["organism"]) == {"ESCHERICHIA COLI"}
    observed = pd.read_parquet(outputs["culture_drug_episodes"])
    assert set(observed["organism"]) == {"ESCHERICHIA COLI"}
    assert observed.loc[observed["anon_id"].eq("p2"), "susceptibility"].tolist().count("RESISTANT") == 1
    pairs = pd.read_parquet(outputs["drug_pair_episodes"])
    assert "p2" in set(pairs["anon_id"])
    metadata = json.loads((paths.paths.metadata / "datasets" / "gold" / "gold_build_combined__escherichia_coli.json").read_text())
    assert metadata["organism_labels"] == {"ESCHERICHIA COLI": 1, "ESBL ESCHERICHIA COLI": 1}
    ledger = metadata["observation_ledger"]["armd"]
    # Five cohort rows, one of another species; every E. coli row interpretable and distinct.
    assert (ledger["cohort_rows"], ledger["organism_rows"], ledger["interpretable_rows"]) == (5, 4, 4)
    assert (ledger["observed_episode_drugs"], ledger["conflicting_episode_drugs"]) == (4, 0)
    assert (ledger["culture_episodes"], ledger["episodes_without_result"]) == (2, 0)
