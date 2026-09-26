from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.visualization.report.cohort_rules import CohortRules
from amr_cascade_platform.visualization.report.plotly_consort_plotter import PlotlyConsortPlotter
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_panel_bundling_plotter import PlotlyPanelBundlingPlotter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rules() -> CohortRules:
    return CohortRules.from_settings(ConfigLoader(PROJECT_ROOT).load("mac"))


def _escalation(rows):
    return pd.DataFrame(rows, columns=["upstream_antibiotic", "downstream_antibiotic", "total_support_n", "positive_support_n", "negative_support_n", "positive_tested_n", "negative_tested_n"])


def test_rules_come_from_settings():
    rules = _rules()
    assert (rules.cotesting_threshold, rules.min_total_support, rules.min_result_support) == (0.95, 25, 5)
    assert (rules.min_downstream_events, rules.q_threshold, rules.stability_threshold) == (5, 0.05, 0.8)


def test_consort_counts_each_stage_from_the_step_that_produced_it():
    plotter = PlotlyConsortPlotter(PlotlyFigureExporter(width=800, height=600), "plotly_white", 800, 600, _rules())
    flow = pd.DataFrame(
        [(site, stage, n) for site in ("a", "b") for stage, n in (("raw_ast_rows", 100), ("culture_episodes", 10), ("eligible_directed_pair_rows", 60), ("binary_upstream_pair_rows", 50))],
        columns=["site", "stage", "row_count"],
    )
    escalation = _escalation(
        [
            ("A", "B", 60, 20, 40, 5, 3),  # retained
            ("A", "C", 60, 20, 40, 0, 0),  # passes support, no downstream-observed episode
            ("A", "D", 20, 10, 10, 2, 2),  # total below 25
            ("A", "E", 60, 3, 57, 1, 4),  # resistant branch below 5
        ]
    )
    cotesting = pd.DataFrame({"upstream_antibiotic": ["X", "Y"], "downstream_antibiotic": ["Y", "X"]})
    report = pd.DataFrame({"upstream_antibiotic": ["A"], "downstream_antibiotic": ["B"], "validation_status": ["robust"], "retained_edge": [True]})

    stats = plotter.collect_stats(flow, report, escalation, cotesting)

    assert (stats["raw_ast"], stats["episodes"], stats["pair_rows"]) == (200, 20, 100)
    assert (stats["screened_out"], stats["evaluated"], stats["below_support"], stats["no_event"]) == (2, 4, 2, 1)
    assert (stats["retained"], stats["robust"], stats["supported"], stats["validated"], stats["not_validated"]) == (1, 1, 0, 1, 0)


def test_consort_marks_counts_it_cannot_derive_as_missing(tmp_path):
    plotter = PlotlyConsortPlotter(PlotlyFigureExporter(width=800, height=600), "plotly_white", 800, 600, _rules())
    stats = plotter.collect_stats(pd.DataFrame(), pd.DataFrame())
    assert stats["screened_out"] is None and stats["pair_rows"] is None and stats["validated"] is None
    outputs = plotter.export(pd.DataFrame(), pd.DataFrame(), tmp_path / "consort", ("html", "png"))
    assert all(path.exists() for path in outputs.values())


def test_panel_bundling_classifies_pairs_with_the_screen_rule(tmp_path):
    plotter = PlotlyPanelBundlingPlotter(PlotlyFigureExporter(width=800, height=600), "plotly_white", 800, 600, _rules())
    probabilities = pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "B", "C", "D"],
            "downstream_antibiotic": ["B", "A", "D", "C"],
            "p_downstream_given_upstream": [0.97, 0.96, 0.99, 0.40],
            "p_upstream_given_downstream": [0.96, 0.97, 0.98, 0.99],
            "support_n": [400, 300, 400, 30],
            "reverse_support_n": [300, 400, 10, 400],
        }
    )
    screened = plotter.screened_pairs(probabilities).set_index(["upstream_antibiotic", "downstream_antibiotic"])
    assert screened["removed"].tolist() == [True, True, False, False]
    assert screened["exempt"].tolist() == [False, False, True, False]
    titles = plotter._titles(plotter.screened_pairs(probabilities))
    assert titles["a"] == "Assessed pairs: 2 removed, 2 kept"
    assert titles["note"] == "1 kept pair has PBI ≥ 0.95 but fewer than 25 rows in one direction"

    report = pd.DataFrame({"upstream_antibiotic": ["C"], "downstream_antibiotic": ["D"], "panel_bundling_index": [0.98], "escalation_ratio": [2.5]})
    outputs = plotter.export(probabilities, report, tmp_path / "panel", ("html", "png"))
    assert all(path.exists() for path in outputs.values())
