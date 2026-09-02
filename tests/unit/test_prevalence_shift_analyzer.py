from dataclasses import replace
import math
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.surveillance.prevalence_shift_analyzer import PrevalenceShiftAnalyzer


def _settings():
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    return replace(
        settings,
        prevalence=replace(
            settings.prevalence,
            min_eligible_support=1,
            min_tested_support=1,
            min_resistant_support=1,
            delta_curve_points=5,
            mnar_lambda_grid=(-1.0, 0.0, 1.0),
        ),
    )


def _eligible_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "is_eligible": 1},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "is_eligible": 1},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "is_eligible": 1},
            {"anon_id": "p4", "pat_enc_csn_id_coded": "e4", "order_proc_id_coded": "o4", "order_time_jittered": "2020-01-04T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "is_eligible": 1},
        ]
    )


def _tested_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "RESISTANT"},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "SUSCEPTIBLE"},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "RESISTANT"},
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CIPROFLOXACIN", "susceptibility": "RESISTANT"},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CIPROFLOXACIN", "susceptibility": "SUSCEPTIBLE"},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CIPROFLOXACIN", "susceptibility": "RESISTANT"},
        ]
    )


def _covariates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "ordering_mode": "Inpatient", "culture_description": "URINE", "was_positive_numeric": 1, "cov_calendar_year": "2020", "cov_calendar_month": "1", "cov_age_bin": "66_plus", "cov_sex": "female", "cov_icu_status": 1, "cov_icu_available": 1, "cov_prior_abx_any_90d": 1, "cov_prior_abx_available": 1, "cov_prior_same_organism_any_90d": 0, "cov_prior_organism_available": 1, "cov_comorbidity_count": 2, "cov_comorbidity_available": 1},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "ordering_mode": "Inpatient", "culture_description": "URINE", "was_positive_numeric": 1, "cov_calendar_year": "2020", "cov_calendar_month": "1", "cov_age_bin": "41_65", "cov_sex": "male", "cov_icu_status": 0, "cov_icu_available": 1, "cov_prior_abx_any_90d": 0, "cov_prior_abx_available": 1, "cov_prior_same_organism_any_90d": 0, "cov_prior_organism_available": 1, "cov_comorbidity_count": 1, "cov_comorbidity_available": 1},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "ordering_mode": "Outpatient", "culture_description": "URINE", "was_positive_numeric": 1, "cov_calendar_year": "2020", "cov_calendar_month": "1", "cov_age_bin": "18_40", "cov_sex": "female", "cov_icu_status": 0, "cov_icu_available": 1, "cov_prior_abx_any_90d": 1, "cov_prior_abx_available": 1, "cov_prior_same_organism_any_90d": 0, "cov_prior_organism_available": 1, "cov_comorbidity_count": 0, "cov_comorbidity_available": 1},
            {"anon_id": "p4", "pat_enc_csn_id_coded": "e4", "order_proc_id_coded": "o4", "order_time_jittered": "2020-01-04T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "ordering_mode": "Outpatient", "culture_description": "URINE", "was_positive_numeric": 1, "cov_calendar_year": "2020", "cov_calendar_month": "1", "cov_age_bin": "18_40", "cov_sex": "male", "cov_icu_status": 0, "cov_icu_available": 1, "cov_prior_abx_any_90d": 0, "cov_prior_abx_available": 1, "cov_prior_same_organism_any_90d": 0, "cov_prior_organism_available": 1, "cov_comorbidity_count": 0, "cov_comorbidity_available": 1},
        ]
    )


def test_mnar_threshold_crossing_interpolates_and_labels_bounds() -> None:
    curve = pd.DataFrame(
        {
            "mnar_lambda": [-1.0, 0.0, 1.0],
            "mnar_prevalence": [0.30, 0.20, 0.10],
        }
    )

    lambda_star, status = PrevalenceShiftAnalyzer._mnar_threshold_crossing(curve, 0.15)
    assert status == "crosses_threshold"
    assert math.isclose(lambda_star, 0.5, rel_tol=0.0, abs_tol=1e-12)

    low_lambda, low_status = PrevalenceShiftAnalyzer._mnar_threshold_crossing(curve, 0.05)
    assert math.isnan(low_lambda)
    assert low_status == "always_above_threshold"

    high_lambda, high_status = PrevalenceShiftAnalyzer._mnar_threshold_crossing(curve, 0.40)
    assert math.isnan(high_lambda)
    assert high_status == "always_below_threshold"


def test_prevalence_shift_analyzer_computes_bounds_and_mnar_shift() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.5,
                "adjusted_odds_ratio": 1.8,
                "validation_status": "robust",
            }
        ]
    )

    covariates = _covariates()
    covariates["cov_lab_wbc"] = [12.0, 9.0, 8.5, 7.0]
    covariates["cov_vital_temp"] = [38.1, 37.0, 36.8, 36.7]
    bundle = analyzer.analyze(_eligible_pairs(), _tested_rows(), covariates, cascade_edge_report)

    assert len(bundle.pair_results) == 1
    row = bundle.pair_results.iloc[0]
    assert row["organism"] == "ESCHERICHIA COLI"
    assert row["drug"] == "MEROPENEM"
    assert math.isclose(float(row["naive_prevalence"]), 2 / 3, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["prevalence_lower_bound"]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["prevalence_upper_bound"]), 0.75, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["denominator_inflation_effect"]), 1 / 6, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["cascade_trigger_fraction"]), 2 / 3, rel_tol=0.0, abs_tol=1e-12)
    assert pd.notna(row["rho_independent_vs_cascade"])
    assert "legacy_delta_empirical" not in bundle.pair_results.columns
    assert "legacy_shifted_prevalence_empirical" not in bundle.pair_results.columns
    assert pd.notna(row["mnar_lambda0_prevalence"])
    assert row["mnar_status_lambda0"] == "estimated"
    assert "cov_lab_wbc" not in row["mnar_feature_columns"]
    assert "cov_vital_temp" not in row["mnar_feature_columns"]
    assert "cov_prior_abx_any_90d" in row["mnar_feature_columns"]
    assert "cascade_any_validated_positive_trigger" in row["mnar_feature_columns"]
    assert math.isclose(float(row["cascade_eligible_n"]), 2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["independent_eligible_n"]), 2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["w_casc_eligible"]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["w_ind_eligible"]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["standardised_prevalence"]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert row["prevalence_lower_bound"] <= row["mnar_lambda0_prevalence"] <= row["prevalence_upper_bound"]
    assert not bundle.delta_curve_results.empty
    assert not bundle.mnar_curve_results.empty
    assert not bundle.mnar_tipping_point_results.empty
    assert {"crosses_threshold", "always_above_threshold", "always_below_threshold", "not_evaluable"}.issuperset(
        set(bundle.mnar_tipping_point_results["crossing_status"])
    )
    assert set(bundle.mnar_tipping_point_results["decision_threshold"]) == set(
        analyzer._settings.prevalence.decision_thresholds
    )
    assert set(bundle.mnar_curve_results["mnar_lambda"]) == {-1.0, 0.0, 1.0}
    assert not bundle.edge_enrichment_results.empty


def test_prevalence_shift_analyzer_projects_empirical_anchor_to_admissible_range() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    eligible_pairs = _eligible_pairs()
    tested = pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "SUSCEPTIBLE"},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "RESISTANT"},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "MEROPENEM", "susceptibility": "RESISTANT"},
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CIPROFLOXACIN", "susceptibility": "RESISTANT"},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CIPROFLOXACIN", "susceptibility": "RESISTANT"},
        ]
    )
    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.5,
                "adjusted_odds_ratio": 1.8,
                "validation_status": "robust",
            }
        ]
    )

    bundle = analyzer.analyze(eligible_pairs, tested, _covariates(), cascade_edge_report)
    row = bundle.pair_results.iloc[0]
    assert "legacy_delta_empirical_was_projected" not in bundle.pair_results.columns


def test_prevalence_shift_analyzer_handles_no_cascade_anchor() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    bundle = analyzer.analyze(_eligible_pairs(), _tested_rows(), _covariates(), pd.DataFrame())
    assert bundle.pair_results.empty
    assert bundle.episode_level_data.empty
    assert bundle.delta_curve_results.empty
    assert bundle.mnar_curve_results.empty
    assert bundle.edge_enrichment_results.empty


def test_prevalence_shift_ignores_validated_suppression_edges() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 0.4,
                "cascade_direction": "suppression",
                "adjusted_odds_ratio": 0.7,
                "validation_status": "robust",
            }
        ]
    )

    bundle = analyzer.analyze(_eligible_pairs(), _tested_rows(), _covariates(), cascade_edge_report)
    assert bundle.pair_results.empty
    assert bundle.episode_level_data.empty
    assert bundle.edge_enrichment_results.empty


def test_prevalence_shift_filters_old_reports_by_escalation_ratio_when_direction_missing() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 0.4,
                "adjusted_odds_ratio": 0.7,
                "validation_status": "robust",
            }
        ]
    )

    bundle = analyzer.analyze(_eligible_pairs(), _tested_rows(), _covariates(), cascade_edge_report)
    assert bundle.pair_results.empty
    assert bundle.episode_level_data.empty
    assert "legacy_delta_empirical" not in bundle.pair_results.columns


def test_prevalence_shift_analyzer_separates_unobserved_from_non_evaluable_observed() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    eligible_pairs = pd.concat(
        [
            _eligible_pairs(),
            pd.DataFrame(
                [
                    {
                        "anon_id": "p5",
                        "pat_enc_csn_id_coded": "e5",
                        "order_proc_id_coded": "o5",
                        "order_time_jittered": "2020-01-05T00:00:00Z",
                        "organism": "ESCHERICHIA COLI",
                        "source_site": "armd",
                        "antibiotic": "MEROPENEM",
                        "is_eligible": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    tested_rows = pd.concat(
        [
            _tested_rows(),
            pd.DataFrame(
                [
                    {
                        "anon_id": "p4",
                        "pat_enc_csn_id_coded": "e4",
                        "order_proc_id_coded": "o4",
                        "order_time_jittered": "2020-01-04T00:00:00Z",
                        "organism": "ESCHERICHIA COLI",
                        "source_site": "armd",
                        "antibiotic": "MEROPENEM",
                        "susceptibility": "INTERMEDIATE",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    covariates = pd.concat(
        [
            _covariates(),
            pd.DataFrame(
                [
                    {
                        "anon_id": "p5",
                        "pat_enc_csn_id_coded": "e5",
                        "order_proc_id_coded": "o5",
                        "order_time_jittered": "2020-01-05T00:00:00Z",
                        "organism": "ESCHERICHIA COLI",
                        "source_site": "armd",
                        "ordering_mode": "Outpatient",
                        "culture_description": "URINE",
                        "was_positive_numeric": 1,
                        "cov_calendar_year": "2020",
                        "cov_calendar_month": "1",
                        "cov_age_bin": "18_40",
                        "cov_sex": "female",
                        "cov_icu_status": 0,
                        "cov_icu_available": 1,
                        "cov_prior_abx_any_90d": 0,
                        "cov_prior_abx_available": 1,
                        "cov_prior_same_organism_any_90d": 0,
                        "cov_prior_organism_available": 1,
                        "cov_comorbidity_count": 0,
                        "cov_comorbidity_available": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.5,
                "adjusted_odds_ratio": 1.8,
                "validation_status": "robust",
            }
        ]
    )
    bundle = analyzer.analyze(eligible_pairs, tested_rows, covariates, cascade_edge_report)
    row = bundle.pair_results.iloc[0]

    assert int(row["eligible_n"]) == 5
    assert int(row["observed_n"]) == 4
    assert int(row["tested_n"]) == 3
    assert int(row["unobserved_n"]) == 1
    assert int(row["non_evaluable_observed_n"]) == 1
    assert int(row["unknown_binary_outcome_n"]) == 2
    assert math.isclose(float(row["prevalence_lower_bound"]), 2 / 5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(row["prevalence_upper_bound"]), 4 / 5, rel_tol=0.0, abs_tol=1e-12)


def test_prevalence_shift_empty_results_keep_artifact_schema() -> None:
    settings = replace(
        _settings(),
        prevalence=replace(
            _settings().prevalence,
            min_eligible_support=100,
            min_tested_support=100,
            min_resistant_support=100,
        ),
    )
    analyzer = PrevalenceShiftAnalyzer(settings)

    bundle = analyzer.analyze(_eligible_pairs(), _tested_rows(), _covariates(), pd.DataFrame())

    assert bundle.pair_results.empty
    assert "mnar_lambda0_absolute_shift_pct" in bundle.pair_results.columns
    assert "eligible_n" in bundle.pair_results.columns
    assert "tested_n" in bundle.pair_results.columns
    assert bundle.delta_curve_results.empty
    assert "delta" in bundle.delta_curve_results.columns
    assert bundle.mnar_curve_results.empty
    assert "mnar_lambda" in bundle.mnar_curve_results.columns
    assert bundle.edge_enrichment_results.empty
    assert "upstream_resistance_enrichment_pct" in bundle.edge_enrichment_results.columns


def test_prevalence_shift_only_reports_downstream_drugs_from_validated_escalation_edges() -> None:
    analyzer = PrevalenceShiftAnalyzer(_settings())
    eligible_pairs = pd.concat(
        [
            _eligible_pairs(),
            pd.DataFrame(
                [
                    {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CEFEPIME", "is_eligible": 1},
                    {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CEFEPIME", "is_eligible": 1},
                    {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2020-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CEFEPIME", "is_eligible": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    tested_rows = pd.concat(
        [
            _tested_rows(),
            pd.DataFrame(
                [
                    {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2020-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CEFEPIME", "susceptibility": "RESISTANT"},
                    {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2020-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "antibiotic": "CEFEPIME", "susceptibility": "SUSCEPTIBLE"},
                ]
            ),
        ],
        ignore_index=True,
    )
    cascade_edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPROFLOXACIN",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.5,
                "validation_status": "robust",
            }
        ]
    )

    bundle = analyzer.analyze(eligible_pairs, tested_rows, _covariates(), cascade_edge_report)

    assert set(bundle.pair_results["drug"]) == {"MEROPENEM"}
    assert set(bundle.episode_level_data["antibiotic"]) == {"MEROPENEM"}
