from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.features.builders.acute_feature_builder import AcuteFeatureBuilder
from amr_cascade_platform.features.builders.history_feature_builder import HistoryFeatureBuilder


def test_acute_feature_builder_disables_untimed_labs_and_vitals(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-03-15T10:00:00Z",
        "source_site": "armd",
        "organism": "ESCHERICHIA COLI",
    }
    culture_episodes = pd.DataFrame([episode])

    # `Period_Day` in the real extract is a constant window-size ("labs were recorded within
    # N days of the culture order"), not a per-row signed offset -- so even rows that look like
    # they encode "pre-culture" (-1) and "post-culture" (2) values here cannot actually be
    # trusted to mean that. Labs must be treated as unusable regardless of what Period_Day says.
    pd.DataFrame(
        [
            {**episode, "Period_Day": -1, "first_wbc": 8.0},
            {**episode, "Period_Day": 2, "first_wbc": 50.0},
        ]
    ).to_parquet(site_dir / "labs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "anon_id": episode["anon_id"],
                "pat_enc_csn_id_coded": episode["pat_enc_csn_id_coded"],
                "order_proc_id_coded": episode["order_proc_id_coded"],
                "source_site": episode["source_site"],
                "median_temp": 37.0,
            },
        ]
    ).to_parquet(site_dir / "vitals.parquet", index=False)
    pd.DataFrame(
        [
            {**episode, "order_time_jittered": "2024-03-15T09:00:00Z", "hosp_ward_ICU": 1},
            {**episode, "order_time_jittered": "2024-03-15T12:00:00Z", "hosp_ward_ICU": 0},
        ]
    ).to_parquet(site_dir / "ward_info.parquet", index=False)

    def load_site_table(site: str, table: str) -> pd.DataFrame:
        path = site_dir / f"{table}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    result = AcuteFeatureBuilder(settings).build(culture_episodes, load_site_table)

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["acute_labs_available"]) == 0
    assert "lab_first_wbc" not in result.columns
    assert int(row["ward_hosp_ward_icu"]) == 1
    assert int(row["acute_ward_available"]) == 1
    assert int(row["acute_vitals_available"]) == 0


def test_history_feature_builder_excludes_post_culture_history_rows(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-03-15T10:00:00Z",
        "source_site": "armd",
        "organism": "ESCHERICHIA COLI",
    }
    culture_episodes = pd.DataFrame([episode])

    pd.DataFrame(
        [
            {**episode, "antibiotic_class": "Cephalosporin", "time_to_culturetime": -5},
            {**episode, "antibiotic_class": "Carbapenem", "time_to_culturetime": 14},
        ]
    ).to_parquet(site_dir / "antibiotic_class_exposure.parquet", index=False)
    pd.DataFrame(
        [
            {**episode, "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": -3},
            {**episode, "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": 30},
        ]
    ).to_parquet(site_dir / "prior_infecting_organism.parquet", index=False)

    def load_site_table(site: str, table: str) -> pd.DataFrame:
        path = site_dir / f"{table}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    result = HistoryFeatureBuilder(settings).build(culture_episodes, load_site_table)

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["history_abx_available"]) == 1
    assert int(row["history_abx_exposure_count"]) == 1
    assert int(row["history_abx_unique_class_count"]) == 1
    assert float(row["history_abx_min_days"]) == 14.0
    assert int(row["history_abx_any_30d"]) == 1
    assert int(row["history_prior_organism_available"]) == 1
    assert int(row["history_prior_organism_count"]) == 1
    assert float(row["history_prior_organism_min_days"]) == 30.0
    assert int(row["history_prior_same_organism_any_30d"]) == 1
