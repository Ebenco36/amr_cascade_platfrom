from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.cascade.statistics.downstream_testing_regression import (
    DownstreamTestingRegression,
)
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager


def test_downstream_testing_regression_uses_episode_covariates(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)

    site_dir = paths.paths.harmonized / "site_aligned" / "armd"
    site_dir.mkdir(parents=True, exist_ok=True)

    # 6 episodes: 3 RESISTANT (2 tested, 1 not) and 3 SUSCEPTIBLE (1 tested, 2 not).
    # This avoids complete separation in either arm (required for BFGS convergence
    # when upstream_positive is exempt from the ridge penalty under double-selection),
    # while maintaining a clear upstream→downstream association (ER ≈ 1.4 > 1.0).
    # ICU assignments are uncorrelated with RESISTANT/SUSCEPTIBLE status so that
    # cov_icu_status and upstream_positive are not collinear in the design matrix.
    episodes = pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1",
             "order_time_jittered": "2024-01-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2",
             "order_time_jittered": "2024-02-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3",
             "order_time_jittered": "2024-03-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
            {"anon_id": "p4", "pat_enc_csn_id_coded": "e4", "order_proc_id_coded": "o4",
             "order_time_jittered": "2024-04-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
            {"anon_id": "p5", "pat_enc_csn_id_coded": "e5", "order_proc_id_coded": "o5",
             "order_time_jittered": "2024-05-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
            {"anon_id": "p6", "pat_enc_csn_id_coded": "e6", "order_proc_id_coded": "o6",
             "order_time_jittered": "2024-06-10T10:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd"},
        ]
    )

    pd.DataFrame(
        [
            {**episodes.iloc[0].to_dict(), "age": 70, "gender": "F"},
            {**episodes.iloc[1].to_dict(), "age": 66, "gender": "F"},
            {**episodes.iloc[2].to_dict(), "age": 55, "gender": "M"},
            {**episodes.iloc[3].to_dict(), "age": 35, "gender": "M"},
            {**episodes.iloc[4].to_dict(), "age": 28, "gender": "F"},
            {**episodes.iloc[5].to_dict(), "age": 45, "gender": "M"},
        ]
    ).to_parquet(site_dir / "demographics.parquet", index=False)
    # ICU uncorrelated with RESISTANT/SUSCEPTIBLE: mixed within each arm
    pd.DataFrame(
        [
            {**episodes.iloc[0].to_dict(), "hosp_ward_ICU": 1, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**episodes.iloc[1].to_dict(), "hosp_ward_ICU": 0, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**episodes.iloc[2].to_dict(), "hosp_ward_ICU": 1, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**episodes.iloc[3].to_dict(), "hosp_ward_ICU": 0, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**episodes.iloc[4].to_dict(), "hosp_ward_ICU": 1, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
            {**episodes.iloc[5].to_dict(), "hosp_ward_ICU": 0, "hosp_ward_IP": 1, "hosp_ward_OP": 0, "hosp_ward_ER": 0},
        ]
    ).to_parquet(site_dir / "ward_info.parquet", index=False)
    pd.DataFrame(
        [
            {**episodes.iloc[0].to_dict(), "antibiotic_class": "Cephalosporin", "time_to_culturetime": 10},
            {**episodes.iloc[1].to_dict(), "antibiotic_class": "Cephalosporin", "time_to_culturetime": 20},
        ]
    ).to_parquet(site_dir / "antibiotic_class_exposure.parquet", index=False)
    pd.DataFrame(
        [
            {**episodes.iloc[0].to_dict(), "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": 20},
            {**episodes.iloc[1].to_dict(), "prior_organism": "Escherichia coli", "prior_infecting_organism_days_to_culture": 25},
        ]
    ).to_parquet(site_dir / "prior_infecting_organism.parquet", index=False)
    pd.DataFrame(
        [
            {**episodes.iloc[0].to_dict(), "comorbidity_component": "diabetes",
             "comorbidity_component_start_days_culture": 365, "comorbidity_component_end_days_culture": None},
            {**episodes.iloc[1].to_dict(), "comorbidity_component": "copd",
             "comorbidity_component_start_days_culture": 90, "comorbidity_component_end_days_culture": -4},
        ]
    ).to_parquet(site_dir / "comorbidity.parquet", index=False)

    drug_pairs = pd.DataFrame(
        [
            # RESISTANT arm: 2 tested, 1 not (no complete separation)
            {**episodes.iloc[0].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "RESISTANT",
             "downstream_tested": 1, "downstream_eligible": 1},
            {**episodes.iloc[1].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "RESISTANT",
             "downstream_tested": 1, "downstream_eligible": 1},
            {**episodes.iloc[2].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "RESISTANT",
             "downstream_tested": 0, "downstream_eligible": 1},
            # SUSCEPTIBLE arm: 1 tested, 2 not (no complete separation)
            {**episodes.iloc[3].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "SUSCEPTIBLE",
             "downstream_tested": 1, "downstream_eligible": 1},
            {**episodes.iloc[4].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "SUSCEPTIBLE",
             "downstream_tested": 0, "downstream_eligible": 1},
            {**episodes.iloc[5].to_dict(), "upstream_antibiotic": "CIPROFLOXACIN",
             "downstream_antibiotic": "MEROPENEM", "upstream_susceptibility": "SUSCEPTIBLE",
             "downstream_tested": 0, "downstream_eligible": 1},
        ]
    )
    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "passes_support_threshold": True,
            }
        ]
    )
    result = DownstreamTestingRegression(settings, paths).analyze(
        drug_pairs,
        escalation_results,
        episodes,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["upstream_antibiotic"] == "CIPROFLOXACIN"
    assert row["downstream_antibiotic"] == "MEROPENEM"
    assert bool(row["supports_adjusted_model"]) is False
    assert row["non_estimable_reason"] == "outcome_events_below_model_dimension"
    assert pd.isna(row["adjusted_odds_ratio"])


def test_adjustment_covariate_contract_matches_design_matrix(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))

    frame = pd.DataFrame(
        [
            {
                "upstream_positive": 0,
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "cov_ordering_mode": "inpatient",
                "cov_specimen_type": "urine",
                "cov_calendar_year": "2024",
                "cov_calendar_month": "1",
                "cov_age_bin": "18_40",
                "cov_sex": "female",
                "cov_er_status": 0,
                "cov_er_available": 1,
            },
            {
                "upstream_positive": 1,
                "organism": "KLEBSIELLA PNEUMONIAE",
                "source_site": "armd_ecuh",
                "cov_ordering_mode": "outpatient",
                "cov_specimen_type": "blood",
                "cov_calendar_year": "2025",
                "cov_calendar_month": "2",
                "cov_age_bin": "66_plus",
                "cov_sex": "male",
                "cov_er_status": 1,
                "cov_er_available": 1,
            },
            {
                "upstream_positive": 0,
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd_ecuh",
                "cov_ordering_mode": "outpatient",
                "cov_specimen_type": "urine",
                "cov_calendar_year": "2025",
                "cov_calendar_month": "1",
                "cov_age_bin": "66_plus",
                "cov_sex": "female",
                "cov_er_status": 1,
                "cov_er_available": 1,
            },
            {
                "upstream_positive": 1,
                "organism": "KLEBSIELLA PNEUMONIAE",
                "source_site": "armd",
                "cov_ordering_mode": "inpatient",
                "cov_specimen_type": "blood",
                "cov_calendar_year": "2024",
                "cov_calendar_month": "2",
                "cov_age_bin": "18_40",
                "cov_sex": "male",
                "cov_er_status": 0,
                "cov_er_available": 1,
            },
        ]
    )
    for idx, column in enumerate(regression._ADJUSTMENT_COVARIATES):
        if column not in frame.columns:
            frame[column] = [idx, idx + 1, idx, idx + 1]
    for column in regression._TIMING_LIMITED_ACUITY_COVARIATES:
        frame[column] = [idx + 10, idx + 11, idx + 10, idx + 11]
    frame["comorb_diabetes"] = [1, 0, 1, 0]

    covariates = regression._adjustment_covariates_for_frame(frame)
    design = regression._build_design_matrix(frame, include_upstream_positive=True)

    assert "comorb_diabetes" not in covariates
    for column in regression._TIMING_LIMITED_ACUITY_COVARIATES:
        assert column not in covariates
    assert design is not None
    assert "upstream_positive" in design.columns
    assert "_intercept" in design.columns
    assert "cov_ordering_mode" in covariates
    assert "cov_specimen_type" in covariates
    assert "cov_er_status" in covariates
    assert "cov_er_available" in covariates
    assert design.shape[1] == np.linalg.matrix_rank(design.to_numpy(dtype=float))
    for column in regression._TIMING_LIMITED_ACUITY_COVARIATES:
        assert column not in design.columns


def test_supported_comorbidity_components_enter_adjusted_model(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))

    frame = pd.DataFrame(
        {
            "upstream_positive": [0, 1] * 6,
            "downstream_tested": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "organism": ["ESCHERICHIA COLI"] * 12,
            "source_site": ["armd"] * 6 + ["armd_ecuh"] * 6,
            "cov_calendar_year": ["2024"] * 12,
            "cov_calendar_month": ["1"] * 12,
            "cov_age_bin": ["unknown"] * 12,
            "cov_sex": ["unknown"] * 12,
            "comorb_diabetes": [1] * 6 + [0] * 6,
            "comorb_rare": [1] + [0] * 11,
        }
    )
    for idx, column in enumerate(regression._ADJUSTMENT_COVARIATES):
        if column not in frame.columns:
            frame[column] = [idx] * len(frame)

    covariates = regression._adjustment_covariates_for_frame(frame)

    assert "comorb_diabetes" in covariates
    assert "comorb_rare" not in covariates


def test_supported_procedure_components_enter_adjusted_model(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))

    frame = pd.DataFrame(
        {
            "upstream_positive": [0, 1] * 6,
            "downstream_tested": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "organism": ["ESCHERICHIA COLI"] * 12,
            "source_site": ["armd"] * 6 + ["armd_ecuh"] * 6,
            "cov_ordering_mode": ["inpatient", "outpatient"] * 6,
            "cov_specimen_type": ["urine"] * 6 + ["blood"] * 6,
            "cov_calendar_year": ["2024"] * 12,
            "cov_calendar_month": ["1"] * 12,
            "cov_age_bin": ["unknown"] * 12,
            "cov_sex": ["unknown"] * 12,
            "proc_cvc": [1] * 6 + [0] * 6,
            "proc_rare": [1] + [0] * 11,
        }
    )
    for idx, column in enumerate(regression._ADJUSTMENT_COVARIATES):
        if column not in frame.columns:
            frame[column] = [idx] * len(frame)

    covariates = regression._adjustment_covariates_for_frame(frame)

    assert "proc_cvc" in covariates
    assert "proc_rare" not in covariates


def test_downstream_testing_regression_empty_result_keeps_schema(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))

    result = regression.analyze(
        drug_pairs=pd.DataFrame(),
        escalation_results=pd.DataFrame(),
        culture_episodes=pd.DataFrame(),
    )

    assert result.empty
    assert "adjusted_odds_ratio" in result.columns
    assert "quasi_separation_flagged" in result.columns
    assert "adjustment_covariates" in result.columns
