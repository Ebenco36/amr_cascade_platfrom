from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager


def test_cascade_covariate_builder_materializes_priority_covariates(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    join = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-03-15T10:00:00Z",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd",
        "ordering_mode": "Inpatient",
        "culture_description": "Urine culture",
    }
    culture_episodes = pd.DataFrame([join])

    pd.DataFrame([{**join, "age": 72, "gender": "F"}]).to_parquet(site_dir / "demographics.parquet", index=False)
    pd.DataFrame([{**join, "adi_score": 91.5, "adi_state_rank": 8.0}]).to_parquet(
        site_dir / "adi_scores.parquet",
        index=False,
    )
    pd.DataFrame([{**join, "hosp_ward_ICU": 1, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 1}]).to_parquet(
        site_dir / "ward_info.parquet",
        index=False,
    )
    pd.DataFrame([{**join, "antibiotic_class": "Cephalosporin", "time_to_culturetime": 14}]).to_parquet(
        site_dir / "antibiotic_class_exposure.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                **join,
                "prior_organism": "Escherichia coli",
                "prior_infecting_organism_days_to_culture": 30,
            }
        ]
    ).to_parquet(site_dir / "prior_infecting_organism.parquet", index=False)
    pd.DataFrame(
        [
            {
                **join,
                "comorbidity_component": "diabetes",
                "comorbidity_component_start_days_culture": 365,
                "comorbidity_component_end_days_culture": None,
            },
            {
                **join,
                "comorbidity_component": "copd",
                "comorbidity_component_start_days_culture": None,
                "comorbidity_component_end_days_culture": None,
            },
            {
                **join,
                "comorbidity_component": "asthma",
                "comorbidity_component_start_days_culture": 200,
                "comorbidity_component_end_days_culture": 10,
            },
        ]
    ).to_parquet(site_dir / "comorbidity.parquet", index=False)
    pd.DataFrame(
        [
            {
                "anon_id": join["anon_id"],
                "pat_enc_csn_id_coded": join["pat_enc_csn_id_coded"],
                "order_proc_id_coded": join["order_proc_id_coded"],  # labs are keyed by culture order
                "source_site": join["source_site"],
                "median_wbc": 12.0,
                "median_neutrophils": 8.0,
                "median_hgb": 10.5,
                "median_cr": 1.7,
                "median_lactate": 2.4,
                "median_procalcitonin": 0.8,
            }
        ]
    ).to_parquet(site_dir / "labs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "anon_id": join["anon_id"],
                "pat_enc_csn_id_coded": join["pat_enc_csn_id_coded"],
                "order_proc_id_coded": join["order_proc_id_coded"],  # vitals too
                "source_site": join["source_site"],
                "median_heartrate": 110,
                "median_sysbp": 95,
                "median_temp": 38.3,
                "median_resprate": 24,
            }
        ]
    ).to_parquet(site_dir / "vitals.parquet", index=False)
    pd.DataFrame([{**join, "nursing_home_visit_culture": 45}]).to_parquet(
        site_dir / "nursing_home_visits.parquet",
        index=False,
    )
    pd.DataFrame([{**join, "procedure_time_to_culturetime": 30, "procedure_description": "central line placement"}]).to_parquet(
        site_dir / "prior_procedures.parquet",
        index=False,
    )

    result = CascadeCovariateBuilder(settings, paths).build(culture_episodes)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["cov_calendar_year"] == "2024"
    assert row["cov_calendar_month"] == "3"
    assert row["cov_ordering_mode"] == "inpatient"
    assert row["cov_specimen_type"] == "urine"
    assert row["cov_age_bin"] == "66_plus"
    assert row["cov_sex"] == "female"
    assert int(row["cov_icu_status"]) == 1
    assert int(row["cov_er_status"]) == 1
    assert int(row["cov_er_available"]) == 1
    assert int(row["cov_prior_abx_any_90d"]) == 1
    assert int(row["cov_prior_same_organism_any_90d"]) == 1
    assert int(row["cov_comorbidity_count"]) == 1
    assert int(row["cov_comorbidity_available"]) == 1
    assert int(row["comorb_diabetes"]) == 1
    assert "comorb_copd" not in result.columns
    assert "comorb_asthma" not in result.columns
    assert float(row["cov_adi_score"]) == 91.5
    assert float(row["cov_adi_state_rank"]) == 8.0
    assert int(row["cov_adi_available"]) == 1
    assert int(row["cov_labs_available"]) == 1
    assert float(row["cov_lab_lactate"]) == 2.4
    assert int(row["cov_vitals_available"]) == 1
    assert float(row["cov_vital_heartrate"]) == 110.0
    assert int(row["cov_nursing_home_90d"]) == 1
    assert int(row["cov_nursing_home_available"]) == 1
    assert int(row["cov_prior_procedure_90d"]) == 1
    assert int(row["cov_prior_procedure_available"]) == 1
    assert int(row["proc_central_line_placement"]) == 1


def test_cascade_covariate_builder_parses_source_age_bins_and_gender_codes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "anon_id": "p1",
            "pat_enc_csn_id_coded": "e1",
            "order_proc_id_coded": "o1",
            "order_time_jittered": "2024-03-15T10:00:00Z",
            "organism": "ESCHERICHIA COLI",
            "source_site": "armd",
        },
        {
            "anon_id": "p2",
            "pat_enc_csn_id_coded": "e2",
            "order_proc_id_coded": "o2",
            "order_time_jittered": "2024-03-16T10:00:00Z",
            "organism": "ESCHERICHIA COLI",
            "source_site": "armd",
        },
    ]
    culture_episodes = pd.DataFrame(rows)
    pd.DataFrame(
        [
            {**rows[0], "age": "65-74 years", "gender": "0"},
            {**rows[1], "age": "above 90", "gender": "1"},
        ]
    ).to_parquet(site_dir / "demographics.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths).build(culture_episodes)

    assert result["cov_age_bin"].tolist() == ["66_plus", "66_plus"]
    assert result["cov_sex"].tolist() == ["code_0", "code_1"]


def test_cascade_covariate_builder_drops_future_history_rows(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    join = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-03-15T10:00:00Z",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd",
    }
    culture_episodes = pd.DataFrame([join])

    pd.DataFrame([{**join, "age": 72, "gender": "F"}]).to_parquet(site_dir / "demographics.parquet", index=False)
    pd.DataFrame(
        [
            {**join, "order_time_jittered": "2024-03-15T09:00:00Z", "hosp_ward_ICU": 1, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**join, "order_time_jittered": "2024-03-15T12:00:00Z", "hosp_ward_ICU": 0, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
        ]
    ).to_parquet(site_dir / "ward_info.parquet", index=False)
    pd.DataFrame(
        [
            {**join, "antibiotic_class": "Cephalosporin", "time_to_culturetime": -2},
            {**join, "antibiotic_class": "Carbapenem", "time_to_culturetime": 14},
        ]
    ).to_parquet(site_dir / "antibiotic_class_exposure.parquet", index=False)
    pd.DataFrame(
        [
            {**join, "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": -4},
            {**join, "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": 20},
        ]
    ).to_parquet(site_dir / "prior_infecting_organism.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths).build(culture_episodes)

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["cov_icu_status"]) == 1
    assert int(row["cov_prior_abx_any_90d"]) == 1
    assert int(row["cov_prior_same_organism_any_90d"]) == 1


def test_comorbidity_available_is_source_availability_not_positive_count(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    join = {
        "anon_id": "p1",
        "pat_enc_csn_id_coded": "e1",
        "order_proc_id_coded": "o1",
        "order_time_jittered": "2024-03-15T10:00:00Z",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd",
    }
    culture_episodes = pd.DataFrame([join])
    pd.DataFrame(
        [
            {
                **join,
                "comorbidity_component": "resolved_asthma",
                "comorbidity_component_start_days_culture": 200,
                "comorbidity_component_end_days_culture": 10,
            }
        ]
    ).to_parquet(site_dir / "comorbidity.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths)._build_comorbidity_counts(culture_episodes)

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["comorbidity_count"]) == 0
    assert int(row["cov_comorbidity_available"]) == 1


def test_prior_same_organism_history_is_matched_at_genus_level(tmp_path: Path) -> None:
    """The prior-infection extracts record genus ("Escherichia", "CONS"), episodes record species."""
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    def episode(index: int, organism: str) -> dict:
        return {
            "anon_id": f"p{index}",
            "pat_enc_csn_id_coded": f"e{index}",
            "order_proc_id_coded": f"o{index}",
            "order_time_jittered": "2024-03-15T10:00:00Z",
            "organism": organism,
            "source_site": "armd",
        }

    episodes = [
        episode(1, "ESCHERICHIA COLI"),
        episode(2, "ESCHERICHIA COLI"),
        episode(3, "COAG NEGATIVE STAPHYLOCOCCUS"),
        episode(4, "ESCHERICHIA COLI"),
    ]
    pd.DataFrame(
        [
            {**episodes[0], "prior_organism": "Escherichia", "prior_infecting_organism_days_to_culture": 20},
            {**episodes[1], "prior_organism": "Klebsiella", "prior_infecting_organism_days_to_culture": 20},
            {**episodes[2], "prior_organism": "CONS", "prior_infecting_organism_days_to_culture": 10},
            {**episodes[3], "prior_organism": "Escherichia", "prior_infecting_organism_days_to_culture": 200},
        ]
    ).to_parquet(site_dir / "prior_infecting_organism.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths).build(pd.DataFrame(episodes)).set_index("anon_id")

    assert result["cov_prior_same_organism_any_90d"].astype(int).to_dict() == {"p1": 1, "p2": 0, "p3": 1, "p4": 0}
    assert result["cov_prior_organism_available"].astype(int).tolist() == [1, 1, 1, 1]


def _comorbidity_frame(tmp_path: Path, rows_by_order: dict[str, list[dict]]):
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)
    base = {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_time_jittered": "2024-03-15T10:00:00Z",
            "organism": "ESCHERICHIA COLI", "source_site": "armd"}
    culture_episodes = pd.DataFrame([{**base, "order_proc_id_coded": order} for order in rows_by_order])
    rows = [{**base, "order_proc_id_coded": order, **row} for order, rows in rows_by_order.items() for row in rows]
    pd.DataFrame(rows).to_parquet(site_dir / "comorbidity.parquet", index=False)
    result = CascadeCovariateBuilder(settings, paths)._build_comorbidity_counts(culture_episodes)
    return result.set_index("order_proc_id_coded")


def test_comorbidity_placeholder_rows_lag_zero_and_undated_conditions(tmp_path: Path) -> None:
    def row(label, start, end=None):
        return {"comorbidity_component": label, "comorbidity_component_start_days_culture": start,
                "comorbidity_component_end_days_culture": end}

    result = _comorbidity_frame(tmp_path, {
        "placeholder": [row(None, None)],                                      # extract's "nothing found" row
        "same_day": [row("urinary tract infections", 0), row("diabetes", 400)],
        "undated": [row("copd", None), row("asthma", 30)],
        "only_undated": [row("copd", None)],
    })

    # A blank-label row is not evidence of no comorbidity: unknown, like no row at all.
    assert (result.loc["placeholder", "cov_comorbidity_available"], result.loc["placeholder", "comorbidity_count"]) == (0, 0)
    # Conditions recorded on or before the culture day (lag >= 0) are counted.
    assert (result.loc["same_day", "cov_comorbidity_available"], result.loc["same_day", "comorbidity_count"]) == (1, 2)
    assert result.loc["same_day", "cov_comorbidity_start_unrecorded"] == 0
    # A named condition without a start is flagged, not counted.
    assert (result.loc["undated", "comorbidity_count"], result.loc["undated", "cov_comorbidity_start_unrecorded"]) == (1, 1)
    assert tuple(result.loc["only_undated", ["cov_comorbidity_available", "comorbidity_count", "cov_comorbidity_start_unrecorded"]]) == (1, 0, 1)


def test_adi_state_rank_has_its_own_availability(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)
    base = {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_time_jittered": "2024-03-15T10:00:00Z",
            "organism": "ESCHERICHIA COLI", "source_site": "armd"}
    culture_episodes = pd.DataFrame([{**base, "order_proc_id_coded": "ranked"}, {**base, "order_proc_id_coded": "unranked"}])
    pd.DataFrame([{**base, "order_proc_id_coded": "ranked", "age": "45-54", "gender": "0"},
                  {**base, "order_proc_id_coded": "unranked", "age": "45-54", "gender": "1"}]).to_parquet(site_dir / "demographics.parquet", index=False)
    pd.DataFrame([{**base, "order_proc_id_coded": "ranked", "adi_score": 80.0, "adi_state_rank": 9.0},
                  {**base, "order_proc_id_coded": "unranked", "adi_score": 75.0, "adi_state_rank": None}]).to_parquet(site_dir / "adi_scores.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths).build(culture_episodes).set_index("order_proc_id_coded")

    assert tuple(result.loc["ranked", ["cov_adi_available", "cov_adi_state_rank_available", "cov_adi_state_rank"]]) == (1, 1, 9.0)
    # The missing rank is filled with 0 for the model, and its indicator says so.
    assert tuple(result.loc["unranked", ["cov_adi_available", "cov_adi_state_rank_available", "cov_adi_state_rank"]]) == (1, 0, 0.0)


def test_labs_and_vitals_are_matched_by_culture_not_borrowed_within_the_encounter(tmp_path: Path) -> None:
    # Two cultures in one encounter; only o1 has labs and vitals. o2 must not inherit them.
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)
    base = {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_time_jittered": "2024-03-15T10:00:00Z",
            "organism": "ESCHERICHIA COLI", "source_site": "armd", "ordering_mode": "Inpatient", "culture_description": "Urine culture"}
    culture_episodes = pd.DataFrame([{**base, "order_proc_id_coded": "o1"}, {**base, "order_proc_id_coded": "o2"}])
    keys = {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "source_site": "armd"}
    pd.DataFrame([{**keys, "median_wbc": 12.0}]).to_parquet(site_dir / "labs.parquet", index=False)
    pd.DataFrame([{**keys, "median_heartrate": 110}]).to_parquet(site_dir / "vitals.parquet", index=False)

    result = CascadeCovariateBuilder(settings, paths).build(culture_episodes).set_index("order_proc_id_coded")
    assert (int(result.loc["o1", "cov_labs_available"]), int(result.loc["o1", "cov_vitals_available"])) == (1, 1)
    assert (int(result.loc["o2", "cov_labs_available"]), int(result.loc["o2", "cov_vitals_available"])) == (0, 0)
