import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.statistics.e_values import e_value_from_odds_ratio_interval
from amr_cascade_platform.reporting.builders import supplementary_table_builder as stb
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def settings():
    return ConfigLoader(PROJECT_ROOT).load("mac")


@pytest.fixture(scope="module")
def resolver():
    return AntibioticClassificationResolver(PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")


def _validation_row(up, down, status, er=2.0, **overrides):
    row = {
        "upstream_antibiotic": up,
        "downstream_antibiotic": down,
        "validation_status": status,
        "cascade_direction": "escalation" if er >= 1 else "suppression",
        "observed_escalation_ratio": er,
        "permutation_fdr_q_value_two_sided": 0.01,
        "permutation_p_value_two_sided": 0.001,
        "bootstrap_sign_stability": 0.95,
        "site_replication_n": 3,
        "site_direction_agreement_rate": 1.0,
        "temporal_direction_agreement": True,
        "site_i_squared": 10.0,
        "site_pooled_log_er": math.log(er),
        "site_pooled_se": 0.1,
    }
    row.update(overrides)
    return row


def _validation_frame():
    return pd.DataFrame(
        [
            _validation_row("CEFTRIAXONE", "FOSFOMYCIN", "robust"),
            _validation_row("CEFTRIAXONE", "ERTAPENEM", "robust", er=0.5),
            # Fails the pooled-site check only through I2 = 80%; temporal agreement is false.
            _validation_row(
                "AMPICILLIN", "CEFAZOLIN", "supported", site_i_squared=80.0, temporal_direction_agreement=False
            ),
            _validation_row(
                "AMPICILLIN", "MEROPENEM", "mixed", permutation_fdr_q_value_two_sided=0.5, temporal_direction_agreement=False
            ),
            _validation_row(
                "GENTAMICIN",
                "AMIKACIN",
                "insufficient",
                permutation_fdr_q_value_two_sided=0.5,
                bootstrap_sign_stability=0.5,
                site_replication_n=1,
                site_direction_agreement_rate=1.0,
                site_i_squared=np.nan,
                site_pooled_log_er=np.nan,
                site_pooled_se=np.nan,
                temporal_direction_agreement=False,
            ),
        ]
    )


def test_validation_summary_counts_labels_and_directions():
    table, summary = stb.validation_summary(_validation_frame())
    assert summary["retained_patterns"] == 5
    assert (summary["robust"], summary["supported"], summary["mixed"], summary["insufficient"]) == (2, 1, 1, 1)
    assert summary["validated"] == 3 and summary["validated_share"] == pytest.approx(0.6)
    assert (summary["validated_escalation"], summary["validated_suppression"]) == (2, 1)
    assert list(table.columns)[:2] == ["retained_patterns", "robust"]
    assert summary["escalation_ratio"]["escalation"] == {"min": 2.0, "median": 2.0, "max": 2.0}
    assert summary["escalation_ratio"]["suppression"]["max"] == 0.5
    assert "escalation_ratio" not in table.columns


def test_validation_summary_rejects_unknown_labels():
    frame = _validation_frame()
    frame.loc[0, "validation_status"] = "validated"
    with pytest.raises(ValueError, match="Unexpected validation labels"):
        stb.validation_summary(frame)


def test_heterogeneity_sensitivity_relabels_with_the_primary_classifier(settings):
    table, transitions, summary = stb.heterogeneity_sensitivity(_validation_frame(), settings)
    rows = table.set_index("policy")
    assert rows.iloc[0][["robust", "supported", "mixed", "insufficient"]].tolist() == [2, 1, 1, 1]
    # I2 = 80% passes neither 50% nor 75%, but passes once heterogeneity is ignored.
    assert rows.iloc[1]["robust"] == 2
    assert rows.iloc[2]["robust"] == 3
    assert transitions.to_dict(orient="records") == [
        {
            "policy": "No heterogeneity check (direction agreement only)",
            "from_status": "supported",
            "to_status": "robust",
            "patterns": 1,
        }
    ]
    assert summary["supported_to_robust_without_heterogeneity_check"] == 1
    assert summary["their_i_squared_median"] == 80.0
    assert summary["their_primary_site_failure"] == {"i_squared_above_threshold": 1}


def test_heterogeneity_sensitivity_refuses_labels_it_cannot_reproduce(settings):
    frame = _validation_frame()
    frame.loc[3, "validation_status"] = "robust"
    with pytest.raises(ValueError, match="changes 1 stored labels"):
        stb.heterogeneity_sensitivity(frame, settings)


def _escalation_row(up, down, pos_n, pos_t, neg_n, neg_t):
    positive = (pos_t + 0.5) / (pos_n + 1.0)
    negative = (neg_t + 0.5) / (neg_n + 1.0)
    return {
        "upstream_antibiotic": up,
        "downstream_antibiotic": down,
        "positive_support_n": pos_n,
        "positive_tested_n": pos_t,
        "negative_support_n": neg_n,
        "negative_tested_n": neg_t,
        "total_support_n": pos_n + neg_n,
        "escalation_ratio": positive / negative,
    }


def test_support_threshold_sensitivity_counts_unvalidated_additions_separately(settings):
    escalation = pd.DataFrame(
        [
            _escalation_row("A", "B", 20, 10, 40, 5),  # retained at every threshold here
            _escalation_row("A", "C", 6, 3, 30, 3),  # dropped when min_result_support = 10
            _escalation_row("A", "D", 3, 1, 40, 4),  # only retained when min_result_support = 1
            _escalation_row("A", "E", 8, 4, 12, 2),  # total 20: only retained when min_total_support = 10
        ]
    )
    validation = pd.DataFrame(
        [_validation_row("A", "B", "robust"), _validation_row("A", "C", "supported")]
    )
    table, _ = stb.support_threshold_sensitivity(escalation, validation, settings)
    rows = table.set_index(["min_total_support", "min_result_support"])
    assert bool(rows.loc[(25, 5), "primary"]) and rows.loc[(25, 5), "retained_patterns"] == 2
    assert rows.loc[(10, 5), "added_not_validated"] == 1
    assert rows.loc[(25, 1), "added_not_validated"] == 1 and rows.loc[(25, 1), "robust"] == 1
    assert rows.loc[(25, 10), "dropped_from_primary_set"] == 1 and rows.loc[(25, 10), "validated_dropped"] == 1


def test_support_threshold_sensitivity_detects_results_from_different_runs(settings):
    escalation = pd.DataFrame([_escalation_row("A", "B", 20, 10, 40, 5)])
    validation = pd.DataFrame([_validation_row("A", "B", "robust"), _validation_row("A", "C", "mixed")])
    with pytest.raises(ValueError, match="do not reproduce the validated candidate set"):
        stb.support_threshold_sensitivity(escalation, validation, settings)


def test_patient_cluster_table_sorts_flags_and_abbreviates(resolver):
    validation = _validation_frame()
    patient = pd.DataFrame(
        {
            "upstream_antibiotic": ["CEFTRIAXONE", "CEFTRIAXONE"],
            "downstream_antibiotic": ["FOSFOMYCIN", "ERTAPENEM"],
            "observed_escalation_ratio": [2.0, 0.5],
            "patient_bootstrap_sign_stability": [0.99, 0.70],
            "n_rows": [100, 80],
            "n_patients": [60, 50],
        }
    )
    table, summary = stb.patient_cluster_table(patient, validation, resolver, 0.8)
    assert table["downstream_antibiotic"].tolist() == ["ERTAPENEM", "FOSFOMYCIN"]
    assert table["patient_below_threshold"].tolist() == [True, False]
    assert table["upstream_abbreviation"].tolist() == ["CRO", "CRO"]
    assert summary["patient_below_threshold"] == 1 and summary["episode_below_threshold"] == 0


def test_patient_cluster_table_requires_exactly_the_robust_set(resolver):
    patient = pd.DataFrame(
        {
            "upstream_antibiotic": ["CEFTRIAXONE"],
            "downstream_antibiotic": ["FOSFOMYCIN"],
            "observed_escalation_ratio": [2.0],
            "patient_bootstrap_sign_stability": [0.99],
            "n_rows": [100],
            "n_patients": [60],
        }
    )
    with pytest.raises(ValueError, match="do not cover exactly the robust patterns"):
        stb.patient_cluster_table(patient, _validation_frame(), resolver, 0.8)


def test_episode_multiplicity_keys_patients_by_site():
    episodes = pd.DataFrame({"source_site": ["a", "a", "a", "b"], "anon_id": ["p1", "p1", "p2", "p1"]})
    summary = stb.episode_multiplicity(episodes)
    assert summary["patients"] == 3
    assert summary["patients_with_multiple_episodes"] == 1
    assert summary["share_episodes_from_patients_with_multiple"] == pytest.approx(0.5)
    assert summary["max_episodes_per_patient"] == 2


def _probabilities(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "upstream_antibiotic",
            "downstream_antibiotic",
            "p_downstream_given_upstream",
            "p_upstream_given_downstream",
            "support_n",
            "reverse_support_n",
        ],
    )


def test_pbi_sensitivity_applies_the_screen_rule_including_support(settings):
    validation = _validation_frame()
    probabilities = _probabilities(
        [
            ("CEFTRIAXONE", "FOSFOMYCIN", 0.92, 0.93, 400, 300),  # excluded at 0.90 only
            ("CEFTRIAXONE", "ERTAPENEM", 0.90, 0.97, 400, 300),  # PBI exactly 0.90: excluded at 0.90
            ("AMPICILLIN", "CEFAZOLIN", 0.96, 0.99, 400, 20),  # high PBI, reverse support < 25: exempt
            ("AMPICILLIN", "MEROPENEM", 0.50, 0.40, 400, 300),
            ("GENTAMICIN", "AMIKACIN", 0.89, 0.99, 400, 300),
        ]
    )
    table, summary_table, summary = stb.pbi_sensitivity(probabilities, validation, settings)
    assert table.set_index("downstream_antibiotic").loc["ERTAPENEM", "excluded_at_stricter"]
    assert summary_table["pairs"].tolist() == [2, 3]
    assert summary_table.iloc[0][["robust", "supported", "mixed_or_insufficient"]].tolist() == [2, 0, 0]
    assert summary["pbi_at_or_above_primary_but_support_below_minimum"] == 1


def test_pbi_sensitivity_rejects_pairs_the_primary_screen_should_have_removed(settings):
    probabilities = _probabilities(
        [
            ("CEFTRIAXONE", "FOSFOMYCIN", 0.96, 0.97, 400, 300),
            ("CEFTRIAXONE", "ERTAPENEM", 0.5, 0.5, 400, 300),
            ("AMPICILLIN", "CEFAZOLIN", 0.5, 0.5, 400, 300),
            ("AMPICILLIN", "MEROPENEM", 0.5, 0.5, 400, 300),
            ("GENTAMICIN", "AMIKACIN", 0.5, 0.5, 400, 300),
        ]
    )
    with pytest.raises(ValueError, match="meet the primary co-testing exclusion rule"):
        stb.pbi_sensitivity(probabilities, _validation_frame(), settings)


def _edge_report():
    rows = []
    for up, down, status, odds, low, high, reason in [
        ("CEFTRIAXONE", "FOSFOMYCIN", "robust", 3.0, 2.0, 4.5, None),
        ("CEFTRIAXONE", "ERTAPENEM", "robust", 0.5, 0.3, 0.8, None),
        ("AMPICILLIN", "CEFAZOLIN", "supported", np.nan, np.nan, np.nan, "quasi_separation"),
        ("AMPICILLIN", "MEROPENEM", "mixed", 1.2, 0.9, 1.6, None),
    ]:
        rows.append(
            {
                "upstream_antibiotic": up,
                "downstream_antibiotic": down,
                "validation_status": status,
                "cascade_direction": "escalation",
                "adjusted_odds_ratio": odds,
                "adjusted_odds_ratio_ci_lower": low,
                "adjusted_odds_ratio_ci_upper": high,
                "adjusted_or_ci_e_value": e_value_from_odds_ratio_interval(low, high),
                "escalation_ratio": 2.0,
                "resistant_tested_n": 10,
                "susceptible_tested_n": 0 if down == "ERTAPENEM" else 4,
                "non_estimable_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def test_confounding_strength_table_keeps_validated_patterns_and_explains_gaps():
    report = _edge_report()
    adjusted = report[["upstream_antibiotic", "downstream_antibiotic", "non_estimable_reason"]]
    table, summary = stb.confounding_strength_table(report.drop(columns="non_estimable_reason"), adjusted)
    assert len(table) == 3 and summary["estimable"] == 2 and summary["ci_excludes_null"] == 2
    assert table.iloc[-1]["notes"].startswith("Non-estimable: adjusted OR outside")
    assert table["validation_status"].tolist() == ["robust", "robust", "supported"]


def test_confounding_strength_table_checks_the_formula():
    report = _edge_report()
    report.loc[0, "adjusted_or_ci_e_value"] = 99.0
    adjusted = report[["upstream_antibiotic", "downstream_antibiotic", "non_estimable_reason"]]
    with pytest.raises(ValueError, match="do not match the stated formula"):
        stb.confounding_strength_table(report.drop(columns="non_estimable_reason"), adjusted)


def test_cross_site_replication_uses_one_agreement_definition():
    validation = pd.DataFrame(
        [
            _validation_row("A", "B", "robust", er=2.0, site_replication_n=3, site_direction_agreement_rate=2 / 3),
            _validation_row("A", "C", "supported", er=2.0, site_replication_n=2, site_direction_agreement_rate=1.0),
            _validation_row("A", "D", "robust", er=0.5, site_replication_n=1, site_direction_agreement_rate=1.0),
        ]
    )
    sites = {
        "s1": pd.DataFrame({"upstream_antibiotic": ["A", "A", "A"], "downstream_antibiotic": ["B", "C", "D"], "escalation_ratio": [1.5, 3.0, 0.4]}),
        "s2": pd.DataFrame({"upstream_antibiotic": ["A", "A"], "downstream_antibiotic": ["B", "C"], "escalation_ratio": [0.8, 1.2]}),
        "s3": pd.DataFrame({"upstream_antibiotic": ["A"], "downstream_antibiotic": ["B"], "escalation_ratio": [2.5]}),
    }
    table, summary = stb.cross_site_replication(validation, sites)
    rows = table.set_index("downstream_antibiotic")
    assert rows.loc["B", ["pooled_sites_with_estimate", "pooled_sites_agreeing", "pooled_consistent"]].tolist() == [3, 2, "No"]
    assert rows.loc["B", ["independent_sites_with_estimate", "independent_sites_agreeing", "independent_consistent"]].tolist() == [3, 2, "No"]
    assert rows.loc["C", "independent_consistent"] == "Yes"
    assert rows.loc["D", "independent_consistent"] == "Insufficient"
    assert summary["pooled"]["patterns_with_estimates_at_2_or_more_sites"] == 2
    assert summary["pooled"]["patterns_agreeing_at_2_or_more_sites"] == 2
    assert summary["pooled"]["patterns_all_sites_agree_among_2_or_more"] == 1


def _ridge_long(report, shift=0.0):
    rows = []
    validated = report.loc[report["validation_status"].isin(["robust", "supported"])]
    for penalty, factor in ((1.0, 1.05), (10.0, 1.0), (100.0, 0.9)):
        for row in validated.itertuples(index=False):
            odds = row.adjusted_odds_ratio * factor * (1 + (shift if penalty == 10.0 else 0))
            rows.append(
                {
                    "upstream_antibiotic": row.upstream_antibiotic,
                    "downstream_antibiotic": row.downstream_antibiotic,
                    "ridge_penalty": penalty,
                    "adjusted_odds_ratio": odds,
                    "adjusted_odds_ratio_ci_lower": row.adjusted_odds_ratio_ci_lower * factor,
                    "adjusted_odds_ratio_ci_upper": row.adjusted_odds_ratio_ci_upper * factor,
                    "non_estimable_reason": row.non_estimable_reason,
                    "modeled_n": 100,
                }
            )
    return pd.DataFrame(rows)


def test_ridge_penalty_table_checks_the_primary_refit_and_summarises_directions():
    report = _edge_report()
    wide, summary = stb.ridge_penalty_table(_ridge_long(report), report, 10.0)
    assert list(wide.columns[:3]) == ["upstream_antibiotic", "downstream_antibiotic", "validation_status"]
    assert summary["estimable_at_every_penalty"] == 2 and summary["same_direction_at_every_penalty"] == 2
    assert summary["max_abs_log_or_change_from_primary"] == pytest.approx(math.log(1 / 0.9))
    with pytest.raises(ValueError, match="differs from the reported adjusted OR"):
        stb.ridge_penalty_table(_ridge_long(report, shift=0.01), report, 10.0)


def test_continuity_correction_summary_reports_direction_and_fold_range():
    table_r = pd.DataFrame({"er_lambda_0.1": [10.0, 0.5], "er_lambda_0.5": [5.0, 0.6], "er_lambda_1.0": [2.0, 0.7]})
    summary = stb.continuity_correction_summary(table_r)
    assert summary["same_direction_at_every_lambda"] == 2
    assert summary["fold_range_max"] == pytest.approx(5.0)
    assert summary["fold_range_min"] == pytest.approx(1.4)


def test_downstream_event_profile_counts_sparse_validated_patterns():
    profile = stb.downstream_event_profile(_edge_report())
    assert profile["validated"] == 3
    assert profile["one_branch_without_events"] == 1
    assert profile["events_at_most_10"] == 1


def test_antibiotic_panel_gives_the_first_failing_rule_as_the_reason(settings, resolver):
    culture_drug = pd.DataFrame(
        {
            "antibiotic": ["CEFTRIAXONE", "FOSFOMYCIN", "VANCOMYCIN", "GATIFLOXACIN", "AMIKACIN", "COLISTIN"],
            "susceptibility": ["RESISTANT", "SUSCEPTIBLE", "SUSCEPTIBLE", "SUSCEPTIBLE", "SUSCEPTIBLE", "INCONCLUSIVE"],
        }
    )
    escalation = pd.DataFrame(
        [
            _escalation_row("CEFTRIAXONE", "FOSFOMYCIN", 30, 10, 60, 5),
            _escalation_row("CEFTRIAXONE", "GATIFLOXACIN", 30, 0, 60, 0),
            _escalation_row("CEFTRIAXONE", "AMIKACIN", 3, 1, 60, 5),
        ]
    )
    validation = pd.DataFrame([_validation_row("CEFTRIAXONE", "FOSFOMYCIN", "robust")])
    reference = stb.IntrinsicReference.load(PROJECT_ROOT / "data" / "reference", settings)
    table, summary = stb.antibiotic_panel(
        culture_drug, escalation, pd.DataFrame(columns=stb.PAIR), validation, "ESCHERICHIA COLI", resolver, reference, settings
    )
    reasons = table.set_index("antibiotic")["reason_not_retained"]
    assert "COLISTIN" not in reasons.index  # no R/S/I result
    assert reasons["CEFTRIAXONE"] == "" and reasons["FOSFOMYCIN"] == ""
    assert reasons["VANCOMYCIN"].startswith("Intrinsically resistant")
    assert reasons["GATIFLOXACIN"] == "Fewer than 5 downstream-observed episodes in every support-passing pair"
    assert reasons["AMIKACIN"] == "Below minimum candidate-pair support threshold"
    assert summary["observed_antibiotics"] == 5 and summary["retained_antibiotics"] == 2


def test_prevalence_tables_join_raw_and_display_drug_names(resolver):
    prevalence = pd.DataFrame(
        {
            "drug": ["CEFEPIME", "IMIPENEM/CILASTATIN"],
            "eligible_n": [100, 90],
            "tested_n": [40, 30],
            "unknown_binary_outcome_n": [60, 60],
            "naive_prevalence_pct": [5.0, 1.0],
            "prevalence_lower_bound_pct": [2.0, 0.3],
            "prevalence_upper_bound_pct": [62.0, 67.0],
            "mnar_lambda0_prevalence_pct": [4.0, 1.5],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "Drug": ["CEFEPIM", "IMIPENEM"],
            "Observed n": [41, 31],
            "Binary-evaluable tested n": [40, 30],
            "Mean abs SMD (observed vs unobserved)": [0.11, 0.2],
            "Mean abs SMD (cascade evaluable vs unobserved)": [0.12, 0.3],
            "Mean abs SMD (independent evaluable vs unobserved)": [0.1, 0.25],
            "Flagged covariates (observed vs unobserved)": [3, 7],
        }
    )
    s2 = stb.prevalence_diagnostics_table(prevalence, diagnostics, resolver)
    assert s2["observed_n"].tolist() == [41, 31]
    s3, summary = stb.smd_balance_table(diagnostics, resolver)
    assert s3["drug"].tolist() == ["CEFEPIME", "IMIPENEM/CILASTATIN"]
    assert summary["drug_with_largest_mean_abs_smd"] == "IMIPENEM/CILASTATIN"
    mismatched = diagnostics.assign(**{"Binary-evaluable tested n": [40, 29]})
    with pytest.raises(ValueError, match="Binary-evaluable counts differ"):
        stb.prevalence_diagnostics_table(prevalence, mismatched, resolver)


def test_build_writes_what_it_can_and_lists_missing_inputs(tmp_path, settings):
    paths = PathManager(tmp_path, settings)
    builder = stb.SupplementaryTableBuilder(settings, paths, PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")
    cascade_dir = paths.paths.artifacts / settings.cascade.outputs.result_dir / "combined" / "organisms" / "escherichia_coli"
    cascade_dir.mkdir(parents=True)
    _validation_frame().to_parquet(cascade_dir / "validation_results.parquet", index=False)
    stale = builder.output_dir("ESCHERICHIA COLI") / "table_s7_patient_cluster_bootstrap_sensitivity.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier run")

    result = builder.build("ESCHERICHIA COLI")

    assert not stale.exists()

    assert {"table_s4_validation_summary.csv", "table_s8_heterogeneity_threshold_sensitivity.csv"} <= set(result.written)
    macros = (result.output_dir / "latex" / "generated_numbers.tex").read_text()
    assert "\\csname amrrows:s4\\endcsname{%\nCombined \\textit{E. coli} & 5" in macros
    assert "amrnum:validation.robust\\endcsname{2}" in (result.output_dir / "latex" / "generated_numbers.tex").read_text()
    assert any(path.endswith("edge_report.parquet") for path in result.missing_inputs)
    numbers = json.loads((result.output_dir / "supplementary_numbers.json").read_text())
    assert numbers["validation"]["robust"] == 2
    assert numbers["missing_inputs"] == result.missing_inputs


def test_latex_text_escapes_special_characters_and_refuses_unknown_ones():
    assert stb.latex_text("β-lactam/β-lactamase — 5% & x_y") == r"$\beta$-lactam\slash $\beta$-lactamase --- 5\% \& x\_y"
    assert stb.latex_int(1217) == "1{,}217"
    with pytest.raises(ValueError, match="No LaTeX mapping"):
        stb.latex_text("α")


def test_table_body_fragments_are_complete_rows(settings):
    _, summary = stb.validation_summary(_validation_frame())
    fragment = stb.s4_fragment(summary)
    assert fragment == "Combined \\textit{E. coli} & 5 & 2 & 1 & 3 & 60.0\\% \\\\\n"
    table, transitions, _ = stb.heterogeneity_sensitivity(_validation_frame(), settings)
    rows = stb.s8_fragment(table, 50.0, 75.0)
    assert rows.startswith("$I^2\\le50\\%$ (primary) & 2 & 1 & 1 & 1 \\\\") and rows.endswith("\\\\\n")
    assert stb.transition_sentence(transitions, table["policy"].iloc[2]) == "1 moves from supported to robust"
    assert stb.transition_sentence(transitions, table["policy"].iloc[1]) == "no pattern changes label"


def test_row_flow_fragment_requires_the_binary_upstream_stage():
    flow = pd.DataFrame(
        [(site, stage, 1000 * (i + 1)) for site in ("armd", "armd_ecuh") for i, stage in enumerate(stb._FLOW_STAGES)],
        columns=["site", "stage", "row_count"],
    )
    fragment = stb.s3_fragment(flow, ["armd", "armd_ecuh"])
    assert "ECU Health & 1{,}000 & 2{,}000 & 3{,}000 & 4{,}000 & 5{,}000 \\\\" in fragment
    assert "\\textbf{Total} & \\textbf{2{,}000}" in fragment and "\\midrule" in fragment
    with pytest.raises(ValueError, match="binary_upstream_pair_rows"):
        stb.s3_fragment(flow.loc[flow["stage"].ne("binary_upstream_pair_rows")], ["armd", "armd_ecuh"])


def test_classification_reference_has_no_spelling_shared_by_two_antibiotics(resolver):
    check = stb.classification_reference_check(resolver)
    assert check["antibiotics"] > 150 and check["spellings"] > check["antibiotics"]


def test_numbers_tex_defines_each_number_in_the_formats_the_text_can_ask_for():
    tex = stb.numbers_tex({"validation": {"robust": 1217, "validated_share": 0.4446, "max_er": 2800.4}, "drug": "IMIPENEM/CILASTATIN", "rows": [{"n": 3}], "flag": True, "gap": None}, "unit test")
    assert "\\csname amrnum:validation.max_er|1\\endcsname{2{,}800.4}" in tex
    assert "\\csname amrnum:validation.robust\\endcsname{1{,}217}" in tex
    assert "\\csname amrnum:validation.validated_share|pct1\\endcsname{44.5\\%}" in tex
    assert "\\csname amrnum:validation.validated_share|3\\endcsname{0.445}" in tex
    assert "\\csname amrnum:drug\\endcsname{IMIPENEM\\slash CILASTATIN}" in tex
    assert "\\csname amrnum:rows.0.n\\endcsname{3}" in tex
    assert "amrnum:flag" not in tex and "amrnum:gap" not in tex


def test_estimability_reports_every_reason_even_when_absent():
    validation = _validation_frame()
    adjusted = validation[["upstream_antibiotic", "downstream_antibiotic"]].assign(non_estimable_reason=[None, "quasi_separation", None, None, None])
    table, summary = stb.adjusted_model_estimability(adjusted, validation)
    assert summary["validated"] == 3 and summary["estimable"] == 2 and summary["quasi_separation"] == 1
    assert summary["model_fit_failed"] == 0 and summary["not_estimable"] == 1


def test_prediction_fragments_bold_the_best_model_and_the_selected_threshold():
    metrics = pd.DataFrame(
        [
            (model, split, threshold, 0.524 if split == "test" else 0.5, auc)
            for model, threshold, aucs in (("logistic_regression", 0.05, (0.958, 0.779, 0.825)), ("xgboost", 0.40, (0.963, 0.793, 0.840)))
            for split, auc in zip(("train", "validation", "test"), aucs)
        ],
        columns=["model_name", "split", "decision_threshold", "prevalence", "roc_auc"],
    )
    summary = stb.prediction_summary(metrics)
    assert summary["test_prevalence"] == 0.524 and summary["selected_threshold"]["logistic_regression"] == 0.05
    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost"],
            "ROC-AUC (train)": ["0.958", "0.963"],
            "ROC-AUC (validation)": ["0.779", "0.793"],
            "ROC-AUC (test)": ["0.825", "0.840"],
            "PR-AUC (test)": ["0.8320", "0.8380"],
            "Brier score (test)": ["0.1980", "0.1880"],
            "Balanced accuracy (test)": ["0.692", "0.672"],
        }
    )
    s12 = stb.s12_fragment(comparison, summary["test_prevalence"])
    assert "XGBoost & 0.963 & 0.793 & \\textbf{0.840} & 0.838 & 0.188 & 0.672 \\\\" in s12
    assert "No-skill baseline & 0.500 & 0.500 & 0.500 & 0.524 & --- & 0.500 \\\\" in s12
    thresholds = pd.DataFrame({"Threshold": ["0.05", "0.10"], "Precision": ["0.650", "0.667"]})
    s13 = stb.s13_fragment(thresholds, 0.05)
    assert s13.splitlines() == ["\\textbf{0.05} & \\textbf{0.650} \\\\", "0.10 & 0.667 \\\\"]
    coefficients = pd.DataFrame({"Feature": ["Train pair negative probability", "Ward: Inpatient"], "Coefficient (log-OR)": ["+1.0849", "-0.1010"], "Odds Ratio": ["2.9592", "0.9039"]})
    s14 = stb.s14_fragment(coefficients)
    assert "Train pair negative probability & $+$1.085 & 2.959 \\\\" in s14
    assert "Ward: Inpatient & $-$0.101 & 0.904 \\\\" in s14 and "Top 1 negative" in s14


def test_pair_flow_accounts_for_every_step_to_the_retained_set(settings):
    escalation = pd.DataFrame(
        [
            _escalation_row("A", "B", 20, 10, 40, 5),
            _escalation_row("A", "C", 20, 0, 40, 0),  # meets support, no downstream-observed episode
            _escalation_row("A", "D", 3, 1, 40, 4),  # below the per-branch minimum
        ]
    )
    probabilities = pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "A", "A", "X", "Y", "Z"],
            "downstream_antibiotic": ["B", "C", "D", "Y", "X", "W"],
            "p_downstream_given_upstream": [0.5, 0.2, 0.4, 0.97, 0.96, 0.3],
            "p_upstream_given_downstream": [0.6, 0.3, 0.5, 0.96, 0.97, np.nan],
            "support_n": [60, 60, 43, 400, 400, 30],
            "reverse_support_n": [60, 60, 43, 400, 400, np.nan],
            "flagged": [False, False, False, True, True, False],
        }
    )
    cotesting_pairs = probabilities.loc[probabilities["flagged"]]
    validation = pd.DataFrame([_validation_row("A", "B", "robust")])
    flow = stb.pair_flow(probabilities, cotesting_pairs, escalation, validation, settings)
    assert flow == {
        "pairs_with_eligible_rows": 6,
        "removed_by_cotesting_screen": 2,
        "with_both_upstream_branches": 3,
        "meeting_support": 2,
        "below_downstream_event_minimum": 1,
        "retained": 1,
    }
    distribution = stb.pbi_distribution(probabilities, settings)
    assert (distribution["assessed_pairs"], distribution["one_direction_pairs"], distribution["removed"], distribution["kept"]) == (5, 1, 2, 3)
    assert distribution["max_kept_pbi"] == 0.5 and distribution["min_removed_pbi"] == 0.96
    with pytest.raises(ValueError, match="do not reproduce the retained set"):
        stb.pair_flow(probabilities, cotesting_pairs, escalation, pd.DataFrame([_validation_row("A", "B", "robust"), _validation_row("A", "E", "mixed")]), settings)


def test_build_refuses_inputs_older_than_their_gold_layer(tmp_path, settings):
    import os

    paths = PathManager(tmp_path, settings)
    builder = stb.SupplementaryTableBuilder(settings, paths, PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")
    cascade_dir = paths.paths.artifacts / settings.cascade.outputs.result_dir / "combined" / "organisms" / "escherichia_coli"
    gold_dir = paths.paths.gold / "combined" / "organisms" / "escherichia_coli"
    cascade_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)
    validation_path = cascade_dir / "validation_results.parquet"
    _validation_frame().to_parquet(validation_path, index=False)
    pd.DataFrame({"source_site": ["a"], "anon_id": ["p"]}).to_parquet(gold_dir / "culture_episodes.parquet", index=False)
    earlier = os.path.getmtime(gold_dir / "culture_episodes.parquet") - 3600
    os.utime(validation_path, (earlier, earlier))

    result = builder.build("ESCHERICHIA COLI")

    assert f"{validation_path} (older than this run's gold layer)" in result.missing_inputs
    assert "table_s4_validation_summary.csv" not in result.written
    assert "episode_multiplicity" in result.numbers



def test_organism_label_counts_reports_pooled_annotated_episodes():
    metadata = {"organism_labels": {"ESCHERICHIA COLI": 1000, "ESBL ESCHERICHIA COLI": 40, "ESCHERICHIA COLI BIOTYPE 2": 2}}
    assert stb.organism_label_counts(metadata, "Escherichia coli") == {"annotated_label_episodes": 42, "annotated_labels": 2}
    assert stb.organism_label_counts({"organism_labels": {"ESCHERICHIA COLI": 5}}, "ESCHERICHIA COLI") == {
        "annotated_label_episodes": 0,
        "annotated_labels": 0,
    }
    with pytest.raises(ValueError, match="organism_labels"):
        stb.organism_label_counts({}, "ESCHERICHIA COLI")


def test_latex_text_maps_typographic_characters_and_stays_strict():
    assert stb.latex_text("Comorbidity: long name…") == r"Comorbidity: long name\ldots{}"
    assert stb.latex_text("ER ≥ 2 × baseline, patient’s μg") == r"ER $\geq$ 2 $\times$ baseline, patient's $\mu$g"
    with pytest.raises(ValueError, match="No LaTeX mapping"):
        stb.latex_text("Δ change")


def _edge(up, down, status, er, lower, upper, tested_r=5, tested_s=5, support=100, adjusted=1.5):
    return {
        "upstream_antibiotic": up, "downstream_antibiotic": down, "validation_status": status,
        "cascade_direction": "escalation" if er >= 1 else "suppression", "escalation_ratio": er,
        "er_ci_lower": lower, "er_ci_upper": upper, "resistant_tested_n": tested_r, "susceptible_tested_n": tested_s,
        "total_support_n": support, "resistant_downstream_test_probability": 0.3, "susceptible_downstream_test_probability": 0.1,
        "adjusted_odds_ratio": adjusted,
    }


def test_selected_patterns_follow_the_forest_ranking_and_prefer_robust():
    report = pd.DataFrame([
        _edge("A", "B", "robust", 3.0, 2.0, 4.5),
        _edge("A", "C", "robust", 9.0, 1.1, 60.0),  # larger ratio, weaker conservative limit
        _edge("A", "D", "robust", 30.0, 20.0, 45.0, tested_s=0),  # one branch unobserved: ranked last
        _edge("A", "E", "supported", 5.0, 4.0, 6.0),
        _edge("B", "C", "robust", 0.4, 0.3, 0.5),
        _edge("B", "D", "mixed", 0.1, 0.05, 0.2),
    ])
    chosen = stb.selected_patterns(report, escalation_n=4, suppression_n=3)
    assert chosen["downstream_antibiotic"].tolist() == ["B", "C", "D", "E", "C"]
    assert chosen["validation_status"].tolist()[:3] == ["robust"] * 3


def test_prevalence_summary_signs_follow_naive_minus_mnar():
    prevalence = pd.DataFrame({
        "drug": ["A", "B", "C", "D"],
        "mnar_lambda0_shift_from_naive_pct": [5.0, -2.0, 0.05, -0.2],
        "naive_prevalence_pct": [40.0, 10.0, 2.0, 1.0],
        "prevalence_lower_bound_pct": [10.0, 5.0, 1.0, 0.5],
        "prevalence_upper_bound_pct": [90.0, 60.0, 40.0, 30.0],
        "eligible_n": [10, 10, 10, 10],
    })
    summary = stb.prevalence_summary(prevalence)
    assert (summary["naive_higher"], summary["naive_lower"], summary["negligible"]) == (1, 2, 1)
    assert summary["bound_width_median_pp"] == pytest.approx(47.0)
    assert stb.prevalence_top(prevalence, 2)["drug"].tolist() == ["A", "B"]


def test_aware_summary_writes_arrows_that_latex_can_typeset():
    transitions = pd.DataFrame({
        "aware_transition": ["Access -> Watch", "Watch -> Watch"], "aware_direction": ["upward", "lateral"],
        "validated_edge_n": [5, 7], "support_weighted_mean_escalation_ratio": [1.5, 2.25],
    })
    summary = stb.aware_summary(transitions)
    assert summary["groups_sentence"] == "Watch \u2192 Watch (7), and Access \u2192 Watch (5)"
    assert summary["upward"] == 5 and summary["largest_group_n"] == 7
    assert stb.latex_text(summary["largest_group"]) == "Watch \\,$\\to$\\, Watch"


def test_adjusted_direction_summary_counts_only_estimable_models():
    validation = pd.DataFrame([
        _validation_row("A", "B", "robust", er=2.0), _validation_row("A", "C", "robust", er=0.5),
        _validation_row("A", "D", "supported", er=3.0), _validation_row("A", "E", "mixed", er=3.0),
    ])
    adjusted = pd.DataFrame({
        "upstream_antibiotic": ["A"] * 4, "downstream_antibiotic": ["B", "C", "D", "E"],
        "adjusted_odds_ratio": [1.8, 1.2, 5000.0, 2.0],
        "non_estimable_reason": [None, None, "quasi_separation", None],
    })
    summary = stb.adjusted_direction_summary(adjusted, validation)
    assert (summary["validated"], summary["estimable"], summary["consistent"], summary["discordant"]) == (3, 2, 1, 1)
    assert summary["escalation"]["median_or"] == pytest.approx(1.8)


def test_availability_sensitivity_counts_kept_dropped_and_added(settings):
    import dataclasses

    primary = pd.DataFrame([_validation_row("A", "B", "robust"), _validation_row("A", "C", "supported"), _validation_row("A", "D", "mixed")])
    run = pd.DataFrame([_validation_row("A", "B", "supported"), _validation_row("A", "D", "robust")])
    eligibility = pd.DataFrame({"is_eligible": [1, 1, 1, 0], "is_observed_tested": [1, 0, 0, 0]})
    strict = pd.DataFrame({"is_eligible": [1, 1, 0, 0], "is_observed_tested": [1, 0, 0, 0]})
    run_settings = dataclasses.replace(
        settings, gold=dataclasses.replace(settings.gold, eligibility=dataclasses.replace(settings.gold.eligibility, availability_min_observed=5))
    )
    table, summary = stb.availability_sensitivity(primary, eligibility, settings, {"support_5": (run_settings, run, strict)})
    row = summary["support_5"]
    assert (row["validated_kept"], row["validated_dropped"], row["validated_added"]) == (1, 1, 1)
    assert row["eligible_share_of_primary"] == pytest.approx(2 / 3)
    assert row["retained_in_both_label_changed"] == 2
    fragment = stb.availability_sensitivity_fragment(table)
    assert fragment.startswith("\\textbf{Support $\\ge$1, 5-year era (primary)}") and "Support $\\ge$5, 5-year era" in fragment


def test_exclusion_flow_numbers_flag_rows_without_a_reason():
    stages = {"raw_rows": 110, "not_ingested_rows": 0, "exact_duplicate_rows": 10, "not_harmonized_rows": 0, "other_organism_rows": 40,
              "organism_rows": 60, "no_result_rows": 5, "other_non_interpretive_rows": 3, "excluded_assay_rows": 2, "interpretable_rows": 50,
              "repeated_rows": 3, "conflicting_extra_rows": 2, "observed_episode_drugs": 45, "conflicting_episode_drugs": 2,
              "culture_episodes": 10, "episodes_without_result": 1}
    flow = pd.DataFrame([{"order": i, "block": "b", "stage": k, "label": k, "site": "armd", "count": v} for i, (k, v) in enumerate(stages.items())])
    numbers, problems = stb.exclusion_flow_numbers(flow)
    assert problems == [] and numbers["armd"]["organism_rows"] == 60
    flow.loc[flow["stage"].eq("observed_episode_drugs"), "count"] = 44
    assert stb.exclusion_flow_numbers(flow)[1]
    fragment = stb.exclusion_flow_fragment(flow, ["armd"])
    assert "not_ingested_rows" not in fragment and "\\textbf{raw_rows}".replace("_", "\\_") in fragment


def test_prediction_details_ranges_gaps_threshold_rows_and_top_features():
    metrics = pd.DataFrame({
        "model_name": ["logistic_regression"] * 3 + ["xgboost"] * 3,
        "split": ["train", "validation", "test"] * 2,
        "roc_auc": [0.95, 0.78, 0.82, 0.96, 0.79, 0.84],
        "decision_threshold": [0.05] * 3 + [0.3] * 3,
    })
    thresholds = pd.DataFrame({"Threshold": ["0.05", "0.70"], "Precision": ["0.65", "0.79"], "Recall": ["0.94", "0.65"], "F1": ["0.77", "0.71"]})
    coefficients = pd.DataFrame({"Feature": ["A", "B", "C", "D"], "Coefficient (log-OR)": [1.0, 0.5, -0.1, -2.0], "Odds Ratio": [2.7, 1.6, 0.9, 0.14]})
    details = stb.prediction_details(metrics, thresholds, coefficients)
    assert details["test_best_model"] == "XGBoost"
    assert details["test_gap_to_best"]["logistic_regression"] == pytest.approx(0.02)
    assert details["roc_auc_range"]["validation"] == {"min": 0.78, "max": 0.79}
    assert details["lr_at_selected"]["recall"] == pytest.approx(0.94) and details["lr_at_070"]["precision"] == pytest.approx(0.79)
    assert [row["feature"] for row in details["lr_top_positive"]] == ["A", "B", "C"]
    assert details["lr_top_negative"][0] == {"feature": "D", "odds_ratio": 0.14}


def test_signed_numbers_drop_the_sign_when_they_round_to_zero():
    assert stb._signed(-0.04, 1) == "0.0" and stb._signed(2.46, 1) == "$+$2.5" and stb._signed(-9.71, 1) == "$-$9.7"
    macros = stb.numbers_tex({"shift": -0.04}, "test")
    assert "amrnum:shift|signed1\\endcsname{0.0}" in macros and "amrnum:shift|signed2\\endcsname{$-$0.04}" in macros
