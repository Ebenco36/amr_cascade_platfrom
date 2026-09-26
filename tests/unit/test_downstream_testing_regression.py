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
    # This avoids complete separation in either arm (the unpenalised upstream
    # coefficient would otherwise diverge and be flagged non-estimable),
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


def test_resolve_cluster_ids_prefixes_by_site(tmp_path: Path) -> None:
    """A raw identifier that collides across sites must not collapse into one cluster.

    Multi-site pooled fits are vulnerable to this: anon_id is generated independently
    per site, so the same raw value (e.g. "123") can appear at two different
    institutions without referring to the same patient.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))

    frame = pd.DataFrame(
        {
            "anon_id": ["123", "123", "456"],
            "source_site": ["armd", "armd_ecuh", "armd"],
        }
    )

    cluster_ids = regression._resolve_cluster_ids(frame)

    assert cluster_ids.name == "anon_id"
    assert cluster_ids.nunique() == 3
    assert cluster_ids.iloc[0] != cluster_ids.iloc[1]


def test_resolve_cluster_ids_falls_back_per_row_on_missing_anon_id() -> None:
    """A row missing its primary identifier must not collapse into a shared fake cluster.

    Two different patients can both have a missing anon_id (e.g. a linkage gap).
    Naively casting NaN to the literal string "nan" would merge every such row
    into one cluster, telling the sandwich estimator two unrelated patients were
    the same person. Each row must instead fall through to its own
    pat_enc_csn_id_coded independently.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(Path("/tmp"), settings))

    frame = pd.DataFrame(
        {
            "anon_id": ["123", np.nan, np.nan, "456"],
            "pat_enc_csn_id_coded": ["e123", "e_missing_1", "e_missing_2", "e456"],
            "source_site": ["armd", "armd", "armd", "armd"],
        }
    )

    cluster_ids = regression._resolve_cluster_ids(frame)

    # The two rows with a missing anon_id must NOT be merged into one cluster --
    # each falls back independently to its own pat_enc_csn_id_coded.
    assert cluster_ids.iloc[1] != cluster_ids.iloc[2]
    assert "nan" not in cluster_ids.iloc[1]
    assert "nan" not in cluster_ids.iloc[2]
    assert cluster_ids.nunique() == 4
    # Primary identifier is still anon_id, since it supplied at least one row.
    assert cluster_ids.name == "anon_id"


def test_resolve_cluster_ids_falls_back_to_episode_key_when_all_candidates_missing() -> None:
    """Two rows missing every ID candidate still land in different clusters.

    They must not silently collapse into one cluster just because their
    identifier columns are equally absent -- the remaining episode-key columns
    (here, order_time_jittered) still distinguish genuinely different episodes.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(Path("/tmp"), settings))

    frame = pd.DataFrame(
        {
            "anon_id": [np.nan, np.nan],
            "pat_enc_csn_id_coded": [np.nan, np.nan],
            "order_proc_id_coded": [np.nan, np.nan],
            "source_site": ["armd", "armd"],
            "order_time_jittered": ["2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z"],
            "organism": ["ESCHERICHIA COLI", "ESCHERICHIA COLI"],
        }
    )

    cluster_ids = regression._resolve_cluster_ids(frame)

    assert cluster_ids.name == "episode_key"
    assert cluster_ids.iloc[0] != cluster_ids.iloc[1]
    assert cluster_ids.nunique() == 2
    assert cluster_ids.iloc[0].startswith("armd::")


def test_calendar_year_enters_as_categorical_term_and_month_is_not_adjusted_for(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))
    frame = pd.DataFrame(
        {
            "upstream_positive": [i % 2 for i in range(12)],
            "cov_calendar_year": [["2020", "2021", "2022"][i % 3] for i in range(12)],
            "cov_calendar_month": [["1", "6", "12"][(i // 3) % 3] for i in range(12)],
        }
    )

    design = regression._build_design_matrix(frame, include_upstream_positive=True)

    assert "cov_calendar_year" not in design.columns
    assert {"cov_calendar_year_2021", "cov_calendar_year_2022"} <= set(design.columns)
    assert not any(column.startswith("cov_calendar_month") for column in design.columns)


def test_both_papers_adjust_for_the_same_calendar_terms() -> None:
    from amr_cascade_platform.surveillance.prevalence_shift_analyzer import PrevalenceShiftAnalyzer

    for covariates in (DownstreamTestingRegression._ADJUSTMENT_COVARIATES, PrevalenceShiftAnalyzer._PRIMARY_MNAR_COVARIATES):
        assert "cov_calendar_year" in covariates
        assert "cov_calendar_month" not in covariates


def test_ridge_penalty_is_configurable_and_never_shrinks_the_upstream_coefficient(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    rng = np.random.default_rng(0)
    n = 400
    upstream = rng.integers(0, 2, n)
    nuisance = rng.normal(size=n)
    target = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.2 * upstream + 0.8 * nuisance)))).astype(int))
    design = pd.DataFrame({"_intercept": 1.0, "upstream_positive": upstream.astype(float), "nuisance": nuisance})
    clusters = pd.Series([f"c{i}" for i in range(n)])

    def fit(penalty: float | None) -> dict:
        return DownstreamTestingRegression(settings, PathManager(tmp_path, settings), ridge_penalty=penalty)._fit_clustered_logit(
            design, target, clusters
        )["coefficients"]

    default, weak, strong = fit(None), fit(0.001), fit(1000.0)

    assert DownstreamTestingRegression(settings, PathManager(tmp_path, settings))._ridge_penalty == 10.0
    assert abs(strong["nuisance"]) < abs(weak["nuisance"])
    assert abs(default["nuisance"]) <= abs(weak["nuisance"])
    assert strong["upstream_positive"] > 0.5


def _simulated_design(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    upstream = rng.integers(0, 2, n).astype(float)
    nuisance = rng.normal(size=(n, 12))
    eta = -1.0 + 0.9 * upstream + nuisance @ np.linspace(-0.6, 0.6, 12)
    target = pd.Series((rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int))
    design = pd.DataFrame(nuisance, columns=[f"x{i}" for i in range(12)])
    design.insert(0, "upstream_positive", upstream)
    design.insert(0, "_intercept", 1.0)
    return design, target


def test_penalized_newton_solves_the_penalized_score_equation() -> None:
    design, target = _simulated_design(60_000, seed=3)
    X, y = design.to_numpy(float), target.to_numpy(float)
    mask = np.array([False, False] + [True] * 12)
    beta, iterations, converged = DownstreamTestingRegression._penalized_newton(X, y, mask, 10.0)

    assert converged and iterations < 20
    probability = 1 / (1 + np.exp(-(X @ beta)))
    score = X.T @ (probability - y) + np.where(mask, 20.0, 0.0) * beta
    assert np.max(np.abs(score)) < 1e-6 * len(y)
    assert abs(beta[1] - 0.9) < 0.05


def test_penalized_newton_matches_an_unpenalized_reference_fit() -> None:
    from scipy.optimize import minimize

    design, target = _simulated_design(3_000, seed=5)
    X, y = design.to_numpy(float), target.to_numpy(float)
    beta, _, converged = DownstreamTestingRegression._penalized_newton(X, y, np.zeros(X.shape[1], bool), 0.0)
    reference = minimize(
        lambda b: float(np.sum(np.logaddexp(0.0, X @ b) - y * (X @ b))),
        np.zeros(X.shape[1]),
        jac=lambda b: X.T @ (1 / (1 + np.exp(-(X @ b))) - y),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000},
    )
    assert converged
    np.testing.assert_allclose(beta, reference.x, atol=1e-5)


def test_fit_converges_and_recovers_the_upstream_effect_on_a_large_pair(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    design, target = _simulated_design(120_000, seed=11)
    clusters = pd.Series(np.arange(len(target)).astype(str))
    fit = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))._fit_clustered_logit(design, target, clusters)

    assert fit is not None
    assert abs(fit["coefficients"]["upstream_positive"] - 0.9) < 0.05
    assert fit["coefficient_ci_lower"]["upstream_positive"] < fit["coefficients"]["upstream_positive"] < fit["coefficient_ci_upper"]["upstream_positive"]


def test_separated_upstream_result_is_flagged_not_reported(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    rng = np.random.default_rng(2)
    n = 4_000
    upstream = rng.integers(0, 2, n)
    tested = np.where(upstream == 1, rng.random(n) < 0.4, False).astype(int)
    group = pd.DataFrame(
        {
            "anon_id": [f"p{i}" for i in range(n)],
            "pat_enc_csn_id_coded": [f"e{i}" for i in range(n)],
            "order_proc_id_coded": [f"o{i}" for i in range(n)],
            "organism": "ESCHERICHIA COLI",
            "source_site": rng.choice(["armd", "armd_ecuh"], n),
            "upstream_positive": upstream,
            "downstream_tested": tested,
            "cov_icu_status": rng.integers(0, 2, n),
            "cov_prior_abx_any_90d": rng.integers(0, 2, n),
        }
    )
    result = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))._fit_pair_model(group, "UP", "DOWN")

    assert result["non_estimable_reason"] == "quasi_separation"
    assert pd.isna(result["adjusted_odds_ratio"]) and pd.isna(result["adjusted_or_ci_e_value"])

