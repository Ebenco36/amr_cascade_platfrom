from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.transformations.observation_space_builder import ObservationSpaceBuilder
from amr_cascade_platform.reporting.builders import descriptive_summaries as ds
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITES = ("armd", "armd_ecuh")


@pytest.fixture(scope="module")
def resolver():
    return AntibioticClassificationResolver(PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")


def _grid(rows: list[tuple]) -> pd.DataFrame:
    """(episode_id, site, era, antibiotic, intrinsic, eligible, observed, support)."""
    return pd.DataFrame(
        rows,
        columns=["episode_id", "source_site", "availability_era", "antibiotic", "is_intrinsic_resistance",
                 "is_eligible", "is_observed_tested", "availability_support_n"],
    )


def test_opportunity_space_categories_are_exclusive_and_unobserved_counts_within_eligible() -> None:
    grid = _grid([
        (1, "armd", "2020-2024", "AMPICILLIN", 0, 1, 1, 3),
        (1, "armd", "2020-2024", "MEROPENEM", 0, 1, 0, 3),
        (1, "armd", "2020-2024", "BENZYLPENICILLIN", 1, 0, 1, 1),  # intrinsic, yet a result was recorded
        (1, "armd", "2020-2024", "FOSFOMYCIN", 0, 0, 0, 0),  # not operationally available
        (2, "armd_ecuh", "2020-2024", "AMPICILLIN", 0, 1, 0, 5),
    ])
    table = ds.opportunity_space(grid, SITES).set_index("site")
    pooled = table.loc["all_sites"]
    assert pooled["grid_rows"] == 5
    assert pooled["intrinsic_rows"] + pooled["not_available_rows"] + pooled["eligible_rows"] == pooled["grid_rows"]
    assert pooled["intrinsic_observed_rows"] == 1
    # Eligible minus every observed row would give 3 - 2 = 1; the intrinsic result must not be subtracted.
    assert pooled["eligible_unobserved_rows"] == 2
    assert pooled["observed_ineligible_rows"] == 1
    assert table.loc["armd", "episodes"] == 1 and table.loc["armd_ecuh", "eligible_rows"] == 1


def test_availability_exposure_counts_thin_strata_and_the_observed_window() -> None:
    grid = _grid([
        (1, "armd", "2020-2024", "AMPICILLIN", 0, 1, 0, 2),  # before the first observed result
        (2, "armd", "2020-2024", "AMPICILLIN", 0, 1, 1, 2),
        (3, "armd", "2020-2024", "AMPICILLIN", 0, 1, 1, 2),
        (4, "armd", "2020-2024", "AMPICILLIN", 0, 1, 0, 2),  # after the last observed result
        (5, "armd", "2020-2024", "AMPICILLIN", 0, 1, 0, 2),  # no timestamp
        (2, "armd", "2020-2024", "MEROPENEM", 0, 1, 1, 9),
        (2, "armd", "2020-2024", "FOSFOMYCIN", 0, 0, 0, 0),  # ineligible rows are ignored
    ])
    episodes = pd.DataFrame({
        "episode_id": [1, 2, 3, 4, 5],
        "episode_time": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", None], utc=True),
    })
    summary, strata = ds.availability_exposure(grid, episodes, ("armd",), thin_support=5)
    row = summary.set_index("site").loc["armd"]
    assert row["eligible_rows"] == 6
    assert row["thin_rows"] == 5 and row["thin_strata"] == 1
    assert row["before_first_observed_rows"] == 1 and row["after_last_observed_rows"] == 1
    assert row["time_unknown_rows"] == 1
    assert row["outside_observed_window_rows"] == 2
    ampicillin = strata.set_index("antibiotic").loc["AMPICILLIN"]
    assert bool(ampicillin["thin_support"]) and ampicillin["observed_n"] == 2


def test_panel_breadth_keeps_episodes_without_any_result() -> None:
    grid = _grid([
        (1, "armd", "e", "A", 0, 1, 1, 1), (1, "armd", "e", "B", 0, 1, 1, 1), (1, "armd", "e", "C", 0, 1, 0, 1),
        (2, "armd", "e", "A", 0, 1, 0, 1), (2, "armd", "e", "B", 0, 1, 0, 1), (2, "armd", "e", "C", 0, 0, 0, 0),
    ])
    episodes = pd.DataFrame({"episode_id": [1, 2], "source_site": ["armd", "armd"], "specimen_group": ["Urine", "Blood"]})
    summary, distribution = ds.panel_breadth(grid, episodes, ("armd",))
    pooled = summary.set_index(["site", "specimen_group"]).loc[("all_sites", ds.ALL_SPECIMENS)]
    assert pooled["episodes"] == 2
    assert pooled["episodes_without_result"] == 1
    assert pooled["observed_share_of_eligible"] == pytest.approx(2 / 5)
    shares = distribution.loc[distribution["site"].eq("all_sites") & distribution["specimen_group"].eq(ds.ALL_SPECIMENS)]
    assert dict(zip(shares["observed_antibiotics"], shares["episodes"])) == {0: 1, 2: 1}


def test_coverage_by_era_pools_antibiotics_and_drops_empty_cells(resolver) -> None:
    availability = pd.DataFrame({
        "site": ["armd", "armd", "armd_ecuh", "armd_ecuh"],
        "era": ["2015-2019", "2015-2019", "2015-2019", "2020-2024"],
        "antibiotic": ["AMPICILLIN", "MEROPENEM", "AMPICILLIN", "MEROPENEM"],
        "eligible_n": [10, 10, 4, 0],
        "observed_n": [9, 1, 2, 0],
    })
    table = ds.coverage_by_era(availability, SITES, resolver)
    pooled = table.loc[table["antibiotic"].eq(ds.ALL_ANTIBIOTICS) & table["site"].eq("all_sites")].set_index("era")
    assert pooled.loc["2015-2019", "eligible_n"] == 24 and pooled.loc["2015-2019", "observed_n"] == 12
    assert "2020-2024" not in set(table["era"])
    assert set(table["antibiotic_display"]) >= {"All antibiotics", "Ampicillin"}


def test_antibiogram_uses_the_most_resistant_result_and_first_isolate_per_patient_year(resolver) -> None:
    episodes = pd.DataFrame({
        "episode_id": [1, 2, 3],
        "source_site": ["armd"] * 3,
        "anon_id": ["p1", "p1", "p2"],
        "episode_time": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"], utc=True),
    })
    results = pd.DataFrame({
        "episode_id": [1, 1, 2, 3],
        "antibiotic": ["AMPICILLIN"] * 4,
        "susceptibility": ["SUSCEPTIBLE", "RESISTANT", "RESISTANT", "SUSCEPTIBLE"],
    })
    table = ds.antibiogram(results, episodes, ("armd",), resolver, min_tested=30)
    row = table.set_index("site").loc["all_sites"]
    assert row["tested_n"] == 3
    assert row["resistant_n"] == 2 and row["conflicting_n"] == 1
    # p1's first 2020 episode (1, counted as resistant) and p2's 2021 episode (susceptible).
    assert row["first_isolate_tested_n"] == 2 and row["first_isolate_resistant_n"] == 1
    assert bool(row["sparse"])


def test_antibiogram_sorts_sparse_drugs_last_within_their_group(resolver) -> None:
    episodes = pd.DataFrame({"episode_id": range(40), "source_site": ["armd"] * 40, "anon_id": [f"p{i}" for i in range(40)],
                             "episode_time": pd.to_datetime(["2020-01-01"] * 40, utc=True)})
    results = pd.DataFrame({
        "episode_id": list(range(40)) + [0, 1],
        "antibiotic": ["AMPICILLIN"] * 40 + ["NITROFURANTOIN"] * 2,
        "susceptibility": ["RESISTANT"] * 10 + ["SUSCEPTIBLE"] * 30 + ["RESISTANT"] * 2,
    })
    table = ds.antibiogram(results, episodes, ("armd",), resolver, min_tested=30)
    order = list(dict.fromkeys(table["antibiotic"]))
    assert order == ["AMPICILLIN", "NITROFURANTOIN"]  # 100% resistant on 2 tests does not head the list


def test_episodes_per_patient_counts_patients_per_site() -> None:
    episodes = pd.DataFrame({"episode_id": range(6), "source_site": ["armd"] * 4 + ["armd_ecuh"] * 2,
                             "anon_id": ["p1", "p1", "p1", "p2", "p1", "p3"]})
    table = ds.episodes_per_patient(episodes, SITES).set_index("site")
    pooled = table.loc["all_sites"]
    assert pooled["patients"] == 4  # p1 at two sites counts twice
    assert pooled["patients_with_multiple"] == 1
    assert pooled["episodes_from_patients_with_multiple_share"] == pytest.approx(3 / 6)
    assert pooled["patients_3_5"] == 1 and pooled["patients_1"] == 3


def test_cohort_characteristics_reports_record_based_covariates_and_ignores_zero_adi() -> None:
    covariates = pd.DataFrame({
        "source_site": ["armd", "armd", "armd_ecuh"],
        "anon_id": ["p1", "p2", "p3"],
        "cov_age_bin": ["18_40", "66_plus", "unknown"],
        "cov_sex": ["code_0", "code_1", "code_0"],
        "cov_calendar_year": ["2016", "2021", "unknown"],
        "cov_specimen_type": ["urine", "blood", "urine"],
        "cov_ordering_mode": ["inpatient", "outpatient", "unknown"],
        "cov_er_status": [1, 0, 0], "cov_er_available": [1, 1, 0],
        "cov_icu_status": [0, 0, 0], "cov_icu_available": [1, 1, 0],
        "cov_nursing_home_90d": [1, 0, 0], "cov_nursing_home_available": [1, 0, 0],
        "cov_comorbidity_count": [3, 5, 0], "cov_comorbidity_available": [1, 1, 0],
        "cov_adi_score": [40.0, 60.0, 0.0], "cov_adi_state_rank": [0.0, 7.0, 0.0], "cov_adi_available": [1, 1, 0],
    })
    long, display = ds.cohort_characteristics(covariates, SITES, era_years=5)
    rows = long.loc[long["scope"].eq("all_sites")].set_index("characteristic")
    assert rows.loc["Culture episodes", "n"] == 3
    assert rows.loc["Nursing-home stay", "denominator"] == 1 and rows.loc["Nursing-home stay", "share"] == 1.0
    assert rows.loc["No nursing-home record", "n"] == 2
    assert rows.loc["Area Deprivation Index state rank, median [IQR]", "median"] == 7.0  # the zero is missing, not a rank
    assert rows.loc["18\u201344", "n"] == 1
    age = long.loc[long["scope"].eq("all_sites") & long["section"].eq("Age, years")].set_index("characteristic")
    assert age.loc["Not recorded", "n"] == 1
    assert list(display.columns[:4]) == ["section", "characteristic", "all_sites", "armd"]
    assert display.set_index("characteristic").loc["Emergency-department presentation", "all_sites"] == "1 (50.0)"


def _ledger(**overrides) -> dict[str, int]:
    entry = {"cohort_rows": 100, "organism_rows": 60, "no_result_rows": 5, "other_non_interpretive_rows": 3,
             "excluded_assay_rows": 2, "interpretable_rows": 50, "identical_rows_collapsed": 1,
             "same_result_duplicate_rows": 2, "conflicting_result_rows": 4, "conflicting_episode_drugs": 2,
             "observed_episode_drugs": 45, "culture_episodes": 10, "episodes_without_result": 1}
    entry.update(overrides)
    return entry


def test_exclusion_flow_reconciles_every_row_and_reports_a_gap() -> None:
    cleaning = {"armd": {"rows_before": 110, "rows_after": 100, "duplicates_removed": 10}}
    table, problems = ds.exclusion_flow({"armd": _ledger()}, {"armd": 110}, cleaning, ("armd",))
    assert problems == []
    counts = table.loc[table["site"].eq("armd")].set_index("stage")["count"]
    assert counts["other_organism_rows"] == 40 and counts["repeated_rows"] == 3 and counts["conflicting_extra_rows"] == 2
    _, problems = ds.exclusion_flow({"armd": _ledger(observed_episode_drugs=44)}, {"armd": 110}, cleaning, ("armd",))
    assert problems and "episode-antibiotic results" in problems[0]


def test_observation_ledger_counts_each_rule() -> None:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    base = {"anon_id": "1", "pat_enc_csn_id_coded": "2", "order_proc_id_coded": "3", "order_time_jittered": "2024-01-01",
            "organism": "ESCHERICHIA COLI", "source_site": "armd"}
    cohort = pd.DataFrame([
        {**base, "antibiotic": "CEFTRIAXONE", "susceptibility": "RESISTANT"},
        {**base, "antibiotic": "CEFTRIAXONE", "susceptibility": "RESISTANT"},  # identical row
        {**base, "antibiotic": "CIPROFLOXACIN", "susceptibility": "SUSCEPTIBLE"},
        {**base, "antibiotic": "NITROFURANTOIN", "susceptibility": None},
        {**base, "antibiotic": "COLISTIN", "susceptibility": "INCONCLUSIVE"},
        {**base, "antibiotic": "CEFTAZIDIME+CLAVULANIC ACID", "susceptibility": "SUSCEPTIBLE"},
    ])
    observed, ledger = ObservationSpaceBuilder(settings).build_with_ledger(cohort)
    row = ledger.loc["armd"]
    assert (row["organism_rows"], row["no_result_rows"], row["other_non_interpretive_rows"]) == (6, 1, 1)
    assert (row["excluded_assay_rows"], row["interpretable_rows"], row["identical_rows_collapsed"]) == (1, 3, 1)
    assert len(observed) == 2
