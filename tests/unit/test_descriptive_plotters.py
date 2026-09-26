from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.reporting.builders import descriptive_summaries as ds
from amr_cascade_platform.reporting.workflows.report_export_workflow import ReportExportWorkflow
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_descriptive_plotters import ABSENT, PlotlyDescriptivePlotter
from amr_cascade_platform.visualization.report.plotly_exporter import PRINT_FONT_SCALE

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def resolver():
    return AntibioticClassificationResolver(PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")


@pytest.fixture()
def plotter():
    return PlotlyDescriptivePlotter(exporter=None, template="plotly_white", min_tested=30)


def _legend_names(fig) -> list[str]:
    return [trace.name for trace in fig.data if trace.showlegend]


def test_opportunity_space_figure_keys_only_categories_present(plotter):
    table = pd.DataFrame({
        "site": ["all_sites", "armd"], "grid_rows": [100, 100], "eligible_observed_rows": [40, 40],
        "eligible_unobserved_rows": [35, 35], "not_available_rows": [25, 25], "intrinsic_rows": [0, 0],
    })
    screen = plotter.opportunity_space_figure(table, "screen")
    assert _legend_names(screen) == ["Eligible, observed", "Eligible, not observed", "Not operationally available"]
    printed = plotter.opportunity_space_figure(table, "print")
    assert printed.layout.width == 2000 and printed.layout.height > screen.layout.height
    keys = [trace for trace in printed.data if trace.showlegend]
    assert keys[0].marker.size == pytest.approx(13 * PRINT_FONT_SCALE)


def test_panel_breadth_figure_has_one_cell_per_specimen_and_scope(plotter):
    distribution = pd.DataFrame({
        "site": ["all_sites", "all_sites", "armd", "armd"], "specimen_group": ["Urine", "Blood", "Urine", "Blood"],
        "observed_antibiotics": [10, 12, 10, 12], "episodes": [30, 5, 30, 5], "share": [1.0, 1.0, 1.0, 1.0],
    })
    summary = pd.DataFrame({"site": ["all_sites", "all_sites", "armd", "armd"], "specimen_group": ["Urine", "Blood", "Urine", "Blood"],
                            "observed_median": [10, 12, 10, 12]})
    fig = plotter.panel_breadth_figure((summary, distribution), "screen")
    bars = [trace for trace in fig.data if trace.type == "bar"]
    assert len(bars) == 4
    assert {bar.marker.color for bar in bars} == {"#2f3b4c", "#a3acb9"}  # the 5-episode blood cells are drawn as sparse
    assert any("median 12" in annotation.text for annotation in fig.layout.annotations)


def test_coverage_by_era_figure_marks_cells_outside_the_eligible_space(plotter, resolver):
    availability = pd.DataFrame({
        "site": ["armd", "armd", "armd_ecuh"], "era": ["2015-2019", "2020-2024", "2020-2024"],
        "antibiotic": ["AMPICILLIN", "AMPICILLIN", "MEROPENEM"], "eligible_n": [10, 10, 5], "observed_n": [5, 9, 1],
    })
    coverage = ds.coverage_by_era(availability, ("armd", "armd_ecuh"), resolver)
    fig = plotter.coverage_by_era_figure(coverage, "print")
    absent_layers = [trace for trace in fig.data if trace.type == "heatmap" and trace.colorscale[0][1] == ABSENT]
    assert len(absent_layers) == 3
    assert "Outside the site's eligible space" in _legend_names(fig)


def test_antibiogram_figure_keys_sparse_and_untested_cells(plotter, resolver):
    episodes = pd.DataFrame({"episode_id": [1, 2], "source_site": ["armd", "armd_ecuh"], "anon_id": ["p1", "p2"],
                             "episode_time": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True)})
    results = pd.DataFrame({"episode_id": [1, 2], "antibiotic": ["AMPICILLIN", "MEROPENEM"], "susceptibility": ["RESISTANT", "SUSCEPTIBLE"]})
    table = ds.antibiogram(results, episodes, ("armd", "armd_ecuh"), resolver, min_tested=30)
    fig = plotter.antibiogram_figure(table, "screen")
    names = _legend_names(fig)
    assert names[:3] == ["Resistant", "Intermediate", "Susceptible"]
    assert "lighter bar: fewer than 30 tested" in names and any("not tested at the site" in name for name in names)


def test_descriptive_figures_are_checked_by_fail_on_missing_figures():
    missing = ReportExportWorkflow._missing_requested_figure_groups(("descriptive_summaries",), {})
    assert missing == {"descriptive_summaries": "figure_opportunity_space., figure_panel_breadth., figure_observation_coverage_by_era., figure_antibiogram."}
