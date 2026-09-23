"""Smoke + contract tests for the direction-pure figure suite.

These don't inspect pixels (that was done by hand against real HPC data while
building this) -- they lock in the two things that must never regress
silently: every figure family stays split into exactly one escalation file and
one suppression file, and nothing raises on the inputs the real pipeline will
actually hand it, including the all-empty case.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.visualization.report.figure_manager import FigureManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _edge_report(n_escalation: int = 3, n_suppression: int = 3) -> pd.DataFrame:
    rows = []
    drugs = ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
    for i in range(n_escalation):
        rows.append({
            "upstream_antibiotic": drugs[i % len(drugs)], "downstream_antibiotic": drugs[(i + 1) % len(drugs)],
            "escalation_ratio": 2.0 + i, "total_support_n": 100 + i, "resistant_support_n": 40, "susceptible_support_n": 60,
            "er_ci_lower": 1.5 + i, "er_ci_upper": 3.0 + i, "adjusted_odds_ratio": 1.8 + i, "validation_status": "robust" if i % 2 == 0 else "supported",
        })
    for i in range(n_suppression):
        rows.append({
            "upstream_antibiotic": drugs[(i + 2) % len(drugs)], "downstream_antibiotic": drugs[(i + 3) % len(drugs)],
            "escalation_ratio": 0.5 / (i + 1), "total_support_n": 200 + i, "resistant_support_n": 20, "susceptible_support_n": 180,
            "er_ci_lower": 0.2 / (i + 1), "er_ci_upper": 0.8 / (i + 1), "adjusted_odds_ratio": 0.4, "validation_status": "robust",
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def figure_manager() -> FigureManager:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    path_manager = PathManager(PROJECT_ROOT, settings)
    return FigureManager(settings, path_manager)


def test_directional_suite_produces_exactly_five_file_types_per_direction(figure_manager, tmp_path):
    outputs = figure_manager.export_directional_suite(_edge_report(), tmp_path, ("png",), tier="validated")
    kinds = {"sankey", "matrix", "network", "forest", "aware"}
    for direction in ("escalation", "suppression"):
        for kind in kinds:
            expected = f"figure_{direction}_{kind}_validated.png"
            assert expected in outputs, f"missing {expected}; got {sorted(outputs)}"
            assert outputs[expected].exists() and outputs[expected].stat().st_size > 0
    assert len(outputs) == 10


def test_directional_suite_is_empty_safe(figure_manager, tmp_path):
    empty = pd.DataFrame(columns=["upstream_antibiotic", "downstream_antibiotic", "escalation_ratio", "total_support_n", "validation_status"])
    assert figure_manager.export_directional_suite(empty, tmp_path, ("png",), tier="validated") == {}


def test_directional_suite_handles_one_direction_only(figure_manager, tmp_path):
    # Matches the codebase's existing convention (see _empty_figure() in the older
    # pooled plotters): a direction with zero validated edges still gets all five
    # files, each a real, non-empty "no data" placeholder -- never silently
    # missing, so a downstream \includegraphics never hits a nonexistent file.
    escalation_only = _edge_report(n_escalation=4, n_suppression=0)
    outputs = figure_manager.export_directional_suite(escalation_only, tmp_path, ("png",), tier="validated")
    assert len(outputs) == 10
    assert sum(name.startswith("figure_escalation_") for name in outputs) == 5
    assert sum(name.startswith("figure_suppression_") for name in outputs) == 5
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def test_directional_suite_robust_tier_only_keeps_robust_edges(figure_manager, tmp_path):
    edges = _edge_report(n_escalation=4, n_suppression=0)  # two robust, two supported
    outputs = figure_manager.export_directional_suite(edges, tmp_path, ("png",), tier="robust")
    assert len(outputs) == 10  # placeholders for suppression and for the supported-only escalation slice
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def _edge_report_with_adjustment_fields(n_escalation: int = 3, n_suppression: int = 3) -> pd.DataFrame:
    """_edge_report() plus the columns the evidence-scatter suite needs.

    A separate helper (not a mutation of _edge_report itself) so the suite
    tests above keep exercising the plain fixture unchanged.
    """
    edges = _edge_report(n_escalation=n_escalation, n_suppression=n_suppression)
    n = len(edges)
    edges["panel_bundling_index"] = [0.1 + 0.05 * i for i in range(n)]
    edges["directional_asymmetry_score"] = [0.5 + 0.1 * i for i in range(n)]
    edges["adjusted_odds_ratio_ci_lower"] = edges["adjusted_odds_ratio"] * 0.6
    edges["adjusted_odds_ratio_ci_upper"] = edges["adjusted_odds_ratio"] * 1.6
    return edges


def test_evidence_scatter_suite_produces_all_three_figures(figure_manager, tmp_path):
    outputs = figure_manager.export_evidence_scatter_suite(
        _edge_report_with_adjustment_fields(), tmp_path, ("png",), tier="validated",
    )
    expected = {
        "figure_das_vs_pbi_validated.png",
        "figure_raw_vs_adjusted_validated.png",
        "figure_adjustment_concordance_summary_validated.png",
    }
    assert expected <= set(outputs), f"missing {expected - set(outputs)}; got {sorted(outputs)}"
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def test_evidence_scatter_suite_is_empty_safe(figure_manager, tmp_path):
    empty = pd.DataFrame(columns=["upstream_antibiotic", "downstream_antibiotic", "escalation_ratio", "total_support_n", "validation_status"])
    assert figure_manager.export_evidence_scatter_suite(empty, tmp_path, ("png",), tier="validated") == {}


def test_adjustment_concordance_summary_counts_match_the_underlying_labels(figure_manager, tmp_path):
    # 4 escalation rows built by _edge_report_with_adjustment_fields all have
    # adjusted_odds_ratio > 1 (1.8, 2.8, 3.8, 4.8) with CI = [0.6x, 1.6x], which
    # straddles 1 for the first (1.08-2.88) and clears it for the rest -- i.e. a
    # known, hand-checkable mix of "attenuated" and "concordant", not all one
    # label, so this test would fail if the category split were ever swapped.
    edges = _edge_report_with_adjustment_fields(n_escalation=4, n_suppression=0)
    outputs = figure_manager.export_evidence_scatter_suite(edges, tmp_path, ("png",), tier="validated")
    assert "figure_adjustment_concordance_summary_validated.png" in outputs
    assert outputs["figure_adjustment_concordance_summary_validated.png"].stat().st_size > 0
