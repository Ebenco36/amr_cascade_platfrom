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
                "order_proc_id_coded": "lab-order-not-culture-order",
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
                "order_proc_id_coded": "vital-order-not-culture-order",
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
