import math
from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.cascade.outputs.cascade_report_builder import CascadeReportBuilder


def test_diagnostics_marks_non_estimable_pairs_unavailable_not_negative(tmp_path: Path) -> None:
    """A non-estimable pair (adjusted_log_odds=NaN) must not read as a "negative" effect.

    NaN > 0 is False in Python, so a naive `"positive" if value > 0 else "negative"`
    mislabels every non-estimable pair as negative. This is the actual shape
    DownstreamTestingRegression._non_estimable_result produces for pairs where the
    model was never fitted (e.g. zero-variance outcome/exposure).
    """
    adjusted_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "adjusted_log_odds": 0.4,
                "adjusted_odds_ratio": 1.5,
                "modeled_n": 50,
            },
            {
                "upstream_antibiotic": "C",
                "downstream_antibiotic": "D",
                "adjusted_log_odds": float("nan"),
                "adjusted_odds_ratio": float("nan"),
                "modeled_n": float("nan"),
                "supports_adjusted_model": False,
                "non_estimable_reason": "zero_variance_outcome_or_exposure",
            },
        ]
    )

    outputs = CascadeReportBuilder().export(pd.DataFrame(), adjusted_results, tmp_path)
    diagnostics = pd.read_parquet(outputs["diagnostics_path"])

    positive_row = diagnostics.loc[diagnostics["upstream_antibiotic"] == "A"].iloc[0]
    non_estimable_row = diagnostics.loc[diagnostics["upstream_antibiotic"] == "C"].iloc[0]
    assert positive_row["adjusted_effect_direction"] == "positive"
    assert non_estimable_row["adjusted_effect_direction"] == "unavailable"


def test_adjustment_concordance_distinguishes_concordant_attenuated_reversed(tmp_path: Path) -> None:
    """Three-way classification, not just same-side-of-1 agreement: a pattern

    whose adjusted CI still straddles 1 is "attenuated" even though its point
    estimate nominally agrees in direction with the raw escalation ratio --
    raw_adjusted_direction_agreement alone would call this "aligned" and lose
    that distinction. adjusted_odds_ratio and its CI live directly on
    retained_edges here (as RetainedEdgeAnalyzer's real output does), not on a
    separately-merged adjusted_results -- _build_edge_report reads them
    straight off retained_edges; adjusted_results only feeds the separate
    per-model diagnostics table.
    """
    retained_edges = pd.DataFrame(
        [
            # Concordant: same direction (OR>1), CI excludes 1.
            {
                "upstream_antibiotic": "A", "downstream_antibiotic": "B",
                "escalation_ratio": 4.0, "total_support_n": 50,
                "adjusted_odds_ratio": 3.5,
                "adjusted_odds_ratio_ci_lower": 1.8, "adjusted_odds_ratio_ci_upper": 6.2,
            },
            # Attenuated: same direction (OR>1), but CI straddles 1.
            {
                "upstream_antibiotic": "C", "downstream_antibiotic": "D",
                "escalation_ratio": 4.0, "total_support_n": 50,
                "adjusted_odds_ratio": 1.05,
                "adjusted_odds_ratio_ci_lower": 0.8, "adjusted_odds_ratio_ci_upper": 1.4,
            },
            # Reversed: raw ER>1 (escalation) but adjusted OR<1.
            {
                "upstream_antibiotic": "E", "downstream_antibiotic": "F",
                "escalation_ratio": 4.0, "total_support_n": 50,
                "adjusted_odds_ratio": 0.6,
                "adjusted_odds_ratio_ci_lower": 0.4, "adjusted_odds_ratio_ci_upper": 0.9,
            },
        ]
    )

    outputs = CascadeReportBuilder().export(retained_edges, pd.DataFrame(), tmp_path)
    edge_report = pd.read_parquet(outputs["edge_report_path"])
    by_pair = edge_report.set_index(["upstream_antibiotic", "downstream_antibiotic"])["adjustment_concordance"]

    assert by_pair[("A", "B")] == "concordant"
    assert by_pair[("C", "D")] == "attenuated"
    assert by_pair[("E", "F")] == "reversed"


def test_adjustment_concordance_unavailable_when_not_estimable(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A", "downstream_antibiotic": "B",
                "escalation_ratio": 4.0, "total_support_n": 50,
                "adjusted_odds_ratio": float("nan"),
            }
        ]
    )

    outputs = CascadeReportBuilder().export(retained_edges, pd.DataFrame(), tmp_path)
    edge_report = pd.read_parquet(outputs["edge_report_path"])

    assert edge_report.iloc[0]["adjustment_concordance"] == "unavailable"


def test_report_builder_creates_two_step_paths(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 50,
            },
            {
                "upstream_antibiotic": "B",
                "downstream_antibiotic": "C",
                "escalation_ratio": 3.0,
                "adjusted_odds_ratio": 1.8,
                "total_support_n": 40,
            },
        ]
    )
    adjusted_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "adjusted_log_odds": 0.4,
                "adjusted_odds_ratio": 1.5,
                "modeled_n": 50,
                "upstream_positive_rate": 0.2,
                "downstream_test_rate": 0.4,
            }
        ]
    )

    outputs = CascadeReportBuilder().export(retained_edges, adjusted_results, tmp_path)
    top_paths = pd.read_parquet(outputs["top_paths_path"])
    assert len(top_paths) == 1
    assert float(top_paths.iloc[0]["path_escalation_ratio"]) == 6.0


def test_report_builder_merges_validation_results(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 50,
                "positive_probability": 0.6,
                "negative_probability": 0.3,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_tested_n": 12,
                "negative_tested_n": 9,
                "passes_support_threshold": True,
                "is_retained_edge": True,
            },
        ]
    )
    adjusted_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "adjusted_log_odds": 0.4,
                "adjusted_odds_ratio": 1.5,
                "modeled_n": 50,
            }
        ]
    )
    validation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "permutation_p_value": 0.01,
                "permutation_fdr_q_value": 0.01,
                "permutation_fdr_supported": True,
                "bootstrap_sign_stability": 0.9,
                "validation_status": "robust",
            }
        ]
    )

    outputs = CascadeReportBuilder().export(
        retained_edges,
        adjusted_results,
        tmp_path,
        validation_results=validation_results,
    )
    edge_report = pd.read_parquet(outputs["edge_report_path"])
    assert float(edge_report.iloc[0]["permutation_p_value"]) == 0.01
    assert float(edge_report.iloc[0]["permutation_fdr_q_value"]) == 0.01
    assert bool(edge_report.iloc[0]["permutation_fdr_supported"]) is True
    assert float(edge_report.iloc[0]["bootstrap_sign_stability"]) == 0.9
    assert edge_report.iloc[0]["validation_status"] == "robust"


def test_report_builder_preserves_cascade_direction_after_validation_merge(tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "cascade_direction": "escalation",
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 50,
                "positive_probability": 0.6,
                "negative_probability": 0.3,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_tested_n": 12,
                "negative_tested_n": 9,
                "passes_support_threshold": True,
                "is_retained_edge": True,
            },
        ]
    )
    validation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "cascade_direction": "escalation",
                "permutation_p_value": 0.01,
                "bootstrap_sign_stability": 0.9,
                "validation_status": "robust",
            }
        ]
    )

    outputs = CascadeReportBuilder().export(
        retained_edges,
        pd.DataFrame(),
        tmp_path,
        validation_results=validation_results,
    )

    edge_report = pd.read_parquet(outputs["edge_report_path"])
    assert "cascade_direction" in edge_report.columns
    assert "cascade_direction_x" not in edge_report.columns
    assert "cascade_direction_y" not in edge_report.columns
    assert edge_report.iloc[0]["cascade_direction"] == "escalation"


def _retained_edge(**overrides: object) -> pd.DataFrame:
    row = {
        "upstream_antibiotic": "A",
        "downstream_antibiotic": "B",
        "escalation_ratio": 2.0,
        "adjusted_odds_ratio": 1.5,
        "total_support_n": 50,
        "positive_probability": 0.6,
        "negative_probability": 0.3,
        "positive_support_n": 20,
        "negative_support_n": 30,
        "positive_tested_n": 12,
        "negative_tested_n": 9,
        "passes_support_threshold": True,
        "is_retained_edge": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_er_confidence_interval_is_reported_when_one_branch_has_no_downstream_observations(tmp_path: Path) -> None:
    escalation_ratio = (4.5 / 21) / (0.5 / 31)
    retained_edges = _retained_edge(
        escalation_ratio=escalation_ratio,
        positive_tested_n=4,
        negative_tested_n=0,
        positive_probability=0.2,
        negative_probability=0.0,
    )

    outputs = CascadeReportBuilder().export(retained_edges, pd.DataFrame(), tmp_path)
    row = pd.read_parquet(outputs["edge_report_path"]).iloc[0]

    se = (1 / 4.5 - 1 / 21 + 1 / 0.5 - 1 / 31) ** 0.5
    assert row["er_ci_lower"] == pytest.approx(escalation_ratio * math.exp(-1.96 * se))
    assert row["er_ci_upper"] == pytest.approx(escalation_ratio * math.exp(1.96 * se))


def test_edge_report_carries_the_two_sided_permutation_evidence_that_governs_labels(tmp_path: Path) -> None:
    validation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "permutation_p_value": 0.01,
                "permutation_fdr_q_value": 0.02,
                "permutation_fdr_supported": True,
                "permutation_p_value_two_sided": 0.03,
                "permutation_fdr_q_value_two_sided": 0.04,
                "permutation_fdr_supported_two_sided": True,
                "validation_status": "robust",
            }
        ]
    )

    outputs = CascadeReportBuilder().export(
        _retained_edge(), pd.DataFrame(), tmp_path, validation_results=validation_results
    )
    row = pd.read_parquet(outputs["edge_report_path"]).iloc[0]

    assert row["permutation_p_value_two_sided"] == 0.03
    assert row["permutation_fdr_q_value_two_sided"] == 0.04
    assert bool(row["permutation_fdr_supported_two_sided"]) is True
