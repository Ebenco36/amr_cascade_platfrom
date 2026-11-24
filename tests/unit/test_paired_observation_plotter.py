import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.reporting.builders.directional_views import observation_probability_rows
from amr_cascade_platform.visualization.report.paired_observation_plotter import PairedObservationPlotter
from amr_cascade_platform.visualization.report.figure_presentation import without_plotly_headings


def edges():
    return pd.DataFrame([
        dict(upstream_antibiotic="A", downstream_antibiotic="B", validation_status="robust",
             resistant_tested_n=8, resistant_support_n=10, susceptible_tested_n=10,
             susceptible_support_n=100, escalation_ratio=8.0, er_ci_lower=3.0, er_ci_upper=12.0),
        dict(upstream_antibiotic="C", downstream_antibiotic="D", validation_status="supported",
             resistant_tested_n=2, resistant_support_n=10, susceptible_tested_n=10,
             susceptible_support_n=100, escalation_ratio=2.0, er_ci_lower=1.1, er_ci_upper=4.0),
        dict(upstream_antibiotic="E", downstream_antibiotic="F", validation_status="robust",
             resistant_tested_n=1, resistant_support_n=10, susceptible_tested_n=90,
             susceptible_support_n=100, escalation_ratio=0.12, er_ci_lower=0.03, er_ci_upper=0.4),
    ])


def test_raw_percentages_and_signed_differences():
    out = observation_probability_rows(edges(), "escalation")
    assert out.upstream_antibiotic.tolist() == ["A", "C"]
    assert out.resistant_observed_pct.tolist() == [80, 20]
    assert out.difference_pp.tolist() == [70, 10]
    assert observation_probability_rows(edges(), "suppression").difference_pp.tolist() == [-80]


@pytest.mark.parametrize("column,value", [("resistant_tested_n", 11), ("susceptible_support_n", 0),
                                          ("resistant_tested_n", np.nan), ("resistant_tested_n", 1.5)])
def test_reject_invalid_counts(column, value):
    data = edges()
    data[column] = data[column].astype(float)
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="Invalid"):
        observation_probability_rows(data, "escalation")


def test_missing_count_column_rejected():
    with pytest.raises(ValueError, match="saved branch counts"):
        observation_probability_rows(edges().drop(columns="resistant_tested_n"), "escalation")


def test_stale_probabilities_rejected():
    data = edges()
    data["resistant_downstream_test_probability"] = 0.99
    with pytest.raises(ValueError, match="does not match"):
        observation_probability_rows(data, "escalation")


def test_only_validated_rows_and_stable_ties():
    data = edges()
    data.loc[0, "validation_status"] = "insufficient"
    assert observation_probability_rows(data, "escalation").upstream_antibiotic.tolist() == ["C"]
    data = edges()
    data.loc[1, "resistant_tested_n"] = 8
    assert observation_probability_rows(data.iloc[::-1], "escalation").upstream_antibiotic.tolist() == ["A", "C"]


def test_csv_includes_unshown_and_plot_preserves_semantics(tmp_path):
    class Exporter:
        def write(self, fig, output_stem, formats):
            self.fig = fig
            return {}
    exporter = Exporter()
    plotter = PairedObservationPlotter(exporter)
    outputs = plotter.export(edges(), "escalation", tmp_path / "plot", ("html",), top_n=1)
    saved = pd.read_csv(outputs["plot.csv"])
    assert len(saved) == 2
    assert saved.shown_in_figure.tolist() == [True, False]
    fig = exporter.fig
    dots = [trace for trace in fig.data if trace.mode == "markers"]
    assert list(dots[0].x) == [10]
    assert list(dots[1].x) == [80]
    hidden = without_plotly_headings(fig, width=1600, height=880)
    assert any("Increased observation" in a.text for a in hidden.layout.annotations)
    assert any("no ER magnitude filter" in a.text for a in hidden.layout.annotations)


def test_empty_direction_is_explicit(tmp_path):
    rows = observation_probability_rows(edges().iloc[:2], "suppression")
    fig = PairedObservationPlotter(None).build(rows, "suppression", 0)
    assert any("No validated pairs" in a.text for a in fig.layout.annotations)


def test_no_smoothing_of_zero_or_one():
    data = edges()
    data.loc[0, "resistant_tested_n"] = 10
    data.loc[0, "susceptible_tested_n"] = 0
    row = observation_probability_rows(data, "escalation").iloc[0]
    assert row.resistant_observed_pct == 100
    assert row.susceptible_observed_pct == 0
    assert row.escalation_ratio == 8.0  # saved ER, not recomputed from the raw dots
