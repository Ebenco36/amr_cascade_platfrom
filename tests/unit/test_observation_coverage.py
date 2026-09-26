from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.reporting.builders import supplementary_table_builder as stb
from amr_cascade_platform.reporting.workflows.report_export_workflow import ReportExportWorkflow
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_observation_coverage_plotter import (
    ALL_SITES,
    MIN_SITE_SUPPORT,
    PlotlyObservationCoveragePlotter,
    observation_coverage_table,
    site_totals,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def resolver():
    return AntibioticClassificationResolver(PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv")


def _opportunities(site: str, antibiotic: str, eligible: int, observed: int, ineligible: int = 0) -> list[dict]:
    rows = [{"antibiotic": antibiotic, "source_site": site, "is_eligible": 1, "is_observed_tested": int(i < observed)} for i in range(eligible)]
    rows += [{"antibiotic": antibiotic, "source_site": site, "is_eligible": 0, "is_observed_tested": 1} for _ in range(ineligible)]
    return rows


@pytest.fixture()
def eligible_pairs():
    rows = []
    rows += _opportunities("armd", "AMPICILLIN", 100, 95, ineligible=7)
    rows += _opportunities("armd_ecuh", "AMPICILLIN", 50, 50)
    rows += _opportunities("armd", "MEROPENEM", 100, 20)
    rows += _opportunities("armd_ecuh", "MEROPENEM", 10, 5)
    rows += _opportunities("armd", "CEFTAZIDIM/AVIBACTAM", 40, 4)
    return pd.DataFrame(rows)


def test_coverage_counts_only_eligible_opportunities_and_pools_sites(eligible_pairs, resolver):
    table = observation_coverage_table(eligible_pairs, resolver, ("armd", "armd_ecuh"))
    row = table.set_index(["antibiotic", "site"])
    # Observed-but-ineligible rows (intrinsic or unavailable) are outside the space.
    assert row.loc[("AMPICILLIN", "armd"), ["eligible_n", "observed_n"]].tolist() == [100, 95]
    assert row.loc[("AMPICILLIN", ALL_SITES), ["eligible_n", "observed_n"]].tolist() == [150, 145]
    assert row.loc[("MEROPENEM", ALL_SITES), "observed_share"] == pytest.approx(25 / 110)
    assert set(table["antibiotic_display"]) >= {"Ampicillin", "Meropenem"}
    # WHO AWaRe order (Access, Watch, Reserve), then all-sites share, all_sites row first per drug.
    order = table.loc[table["site"].eq(ALL_SITES), "aware_category"].tolist()
    assert order == sorted(order, key=["Access", "Watch", "Reserve"].index)
    assert table["site"].iloc[0] == ALL_SITES
    totals = site_totals(table).set_index("site")
    assert totals.loc[ALL_SITES, ["eligible_n", "observed_n"]].tolist() == [300, 174]


def test_coverage_table_rejects_inputs_without_eligibility_flags(resolver):
    with pytest.raises(ValueError, match="is_eligible"):
        observation_coverage_table(pd.DataFrame({"antibiotic": ["X"], "source_site": ["armd"], "is_observed_tested": [1]}), resolver)


def _data_traces(fig):
    # Key entries are marker-only traces with no data; everything else carries customdata or text.
    return [trace for trace in fig.data if not (trace.type == "scatter" and trace.showlegend is not False)]


@pytest.mark.parametrize("medium", ["screen", "print"])
def test_two_panel_layout_groups_by_aware_and_marks_sparse_sites(eligible_pairs, resolver, medium):
    table = observation_coverage_table(eligible_pairs, resolver, ("armd", "armd_ecuh"))
    fig = PlotlyObservationCoveragePlotter(exporter=None, template="plotly_white").figure(table, medium=medium)
    ticktext = list(fig.layout.yaxis2.ticktext)
    assert ticktext[0] == "<b>Access</b>" and "<b>Watch</b>" in ticktext and "<b>Reserve</b>" in ticktext
    sites = {trace.name: trace for trace in _data_traces(fig) if trace.type == "scatter"}
    assert set(sites) == {"Stanford", "ECU Health"}
    ecuh_symbols = dict(zip(sites["ECU Health"].customdata[:, 0], sites["ECU Health"].marker.symbol, strict=True))
    assert ecuh_symbols["Meropenem"].endswith("-open")  # 10 eligible < MIN_SITE_SUPPORT
    assert not ecuh_symbols["Ampicillin"].endswith("-open")
    key = [trace.name for trace in fig.data if trace.showlegend is not False and trace.type == "scatter"]
    assert key[0] == "Observed, all sites" and "Eligible, not observed" in key
    assert MIN_SITE_SUPPORT == 20


def test_print_geometry_scales_canvas_and_marks_but_not_fonts(eligible_pairs, resolver):
    table = observation_coverage_table(eligible_pairs, resolver, ("armd", "armd_ecuh"))
    plotter = PlotlyObservationCoveragePlotter(exporter=None, template="plotly_white")
    screen, paper = plotter.figure(table, medium="screen"), plotter.figure(table, medium="print")
    assert paper.layout.height > 2 * screen.layout.height  # the exporter scales fonts, not the canvas
    assert paper.layout.margin.l == screen.layout.margin.l  # margins are scaled by the exporter itself
    marker = lambda fig: next(t for t in _data_traces(fig) if t.type == "scatter").marker.size
    assert marker(paper) > 2 * marker(screen)


def test_small_multiples_have_one_column_per_scope_and_mark_absent_drugs(eligible_pairs, resolver):
    table = observation_coverage_table(eligible_pairs, resolver, ("armd", "armd_ecuh"))
    fig = PlotlyObservationCoveragePlotter(exporter=None, template="plotly_white").small_multiples_figure(table)
    titles = [annotation.text for annotation in fig.layout.annotations if annotation.text.startswith("<b>")]
    assert titles[:3] == ["<b>All sites</b>", "<b>Stanford</b>", "<b>ECU Health</b>"]
    assert list(fig.layout.yaxis.ticktext)[0] == "<b>All antibiotics</b>"
    values = [trace for trace in fig.data if trace.type == "scatter" and trace.mode == "text"]
    assert len(values) == 3
    # Ceftazidime/avibactam has no eligible opportunities at ECU Health in the fixture: a dash, no track.
    ecuh_text = dict(zip(values[2].y, values[2].text, strict=True))
    avibactam_row = list(fig.layout.yaxis.ticktext).index("Ceftazidime/avibactam")
    assert ecuh_text[avibactam_row] == "\u2013"
    sparse = [trace for trace in fig.data if trace.type == "bar" and trace.name and "fewer than" in trace.name]
    assert sparse and all(len(trace.x) >= 1 for trace in sparse)  # Meropenem at ECU Health (10 eligible)


def test_coverage_numbers_for_the_manuscript(eligible_pairs, resolver):
    numbers = stb.observation_coverage_numbers(observation_coverage_table(eligible_pairs, resolver, ("armd", "armd_ecuh")))
    assert numbers["all_sites"] == {"eligible_n": 300, "observed_n": 174, "observed_share": pytest.approx(0.58)}
    assert numbers["armd_ecuh"]["observed_n"] == 55
    assert numbers["antibiotics"] == 3
    assert numbers["drug_share_median"] == pytest.approx(25 / 110)


def test_observation_coverage_is_checked_by_fail_on_missing_figures():
    missing = ReportExportWorkflow._missing_requested_figure_groups(("observation_coverage",), {})
    assert missing == {"observation_coverage": "figure_observation_coverage., figure_observation_coverage_by_site."}
    # The two-panel file is not mistaken for present when only the small multiples exist.
    only_by_site = {"figure_observation_coverage_by_site.png": Path("figure_observation_coverage_by_site.png")}
    missing = ReportExportWorkflow._missing_requested_figure_groups(("observation_coverage",), only_by_site)
    assert missing == {"observation_coverage": "figure_observation_coverage."}
