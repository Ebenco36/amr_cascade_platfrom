from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.cascade_report_builder import CascadeReportBuilder


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
