from pathlib import Path

import pandas as pd
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.figure_presentation import prepare_matplotlib
from amr_cascade_platform.visualization.report.plotly_mnar_prevalence_shift_distribution_plotter import (
    PlotlyMNARPrevalenceShiftDistributionPlotter,
)

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_prevalence_plotter import PlotlyPrevalencePlotter


def _plotter() -> PlotlyPrevalencePlotter:
    return PlotlyPrevalencePlotter(
        exporter=PlotlyFigureExporter(width=1600, height=900),
        template="plotly_white",
        width=1600,
        height=900,
    )


def _prevalence_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "CEFEPIME",
                "eligible_n": 4700,
                "naive_prevalence_pct": 6.0,
                "mnar_lambda0_prevalence_pct": 4.0,
                "mnar_lambda0_shift_from_naive": 0.02,
                "mnar_lambda0_shift_from_naive_pct": 2.0,
                "prevalence_lower_bound_pct": 3.0,
                "prevalence_upper_bound_pct": 58.0,
                "rho_independent_vs_cascade": 0.45,
                "cascade_trigger_fraction": 0.60,
                "cascade_prevalence_pct": 35.0,
                "independent_prevalence_pct": 10.0,
            },
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "TETRACYCLINE",
                "eligible_n": 4389,
                "naive_prevalence_pct": 33.0,
                "mnar_lambda0_prevalence_pct": 40.0,
                "mnar_lambda0_shift_from_naive": -0.07,
                "mnar_lambda0_shift_from_naive_pct": -7.0,
                "prevalence_lower_bound_pct": 15.0,
                "prevalence_upper_bound_pct": 70.0,
                "rho_independent_vs_cascade": 0.70,
                "cascade_trigger_fraction": 0.25,
                "cascade_prevalence_pct": 20.0,
                "independent_prevalence_pct": 30.0,
            },
        ]
    )


def test_prevalence_forest_uses_compact_layout_and_correct_semantics() -> None:
    figure = _plotter()._build_figure(_prevalence_results())

    assert figure.layout.margin.b == 120
    assert figure.layout.height == 900
    assert "Eligible-denominator resistance summaries" in figure.layout.title.text
    assert all(
        not (annotation.yref == "paper" and isinstance(annotation.y, (int, float)) and annotation.y < 0)
        for annotation in figure.layout.annotations
    )

    names = {trace.name for trace in figure.data if trace.name}
    assert "Eligible-denominator bounds" in names
    assert "Naive tested-row prevalence" in names
    assert "Reference estimate (naive higher)" in names
    assert "Reference estimate (naive lower)" in names
    assert not any("MNAR estimate" in name for name in names)

    bound_trace = next(trace for trace in figure.data if trace.name == "Eligible-denominator bounds")
    assert bound_trace.line.color == PlotlyPrevalencePlotter._BOUNDS_COLOR
    assert PlotlyPrevalencePlotter._shift_direction(1.0) == "naive_overestimates"
    assert PlotlyPrevalencePlotter._shift_direction(-1.0) == "naive_underestimates"


def test_prevalence_static_exports_are_real_and_compact(tmp_path: Path) -> None:
    plotter = _plotter()
    results = _prevalence_results()

    forest = plotter.export_forest(results, tmp_path / "forest", ("png", "pdf", "svg"))
    kappa = plotter.export_kappa_ranked(results, tmp_path / "kappa", ("png",))
    sensitivity = plotter.export_surveillance_sensitivity_map(
        results,
        tmp_path / "sensitivity",
        ("html", "png"),
    )

    for output in [*forest.values(), *kappa.values(), *sensitivity.values()]:
        assert output.exists()
        assert output.stat().st_size > 0
    html = sensitivity["sensitivity.html"].read_text(encoding="utf-8")
    assert "CEFEPIME" in html
    assert "TETRACYCLINE" in html
    assert b"Static export unavailable" not in forest["forest.pdf"].read_bytes()


def test_sensitivity_map_long_edge_labels_stay_inside_axis(monkeypatch, tmp_path):
    data = pd.DataFrame({
        "drug": ["TRIMETHOPRIM/SULFAMETHOXAZOLE", "AMPICILLIN/SULBACTAM"],
        "cascade_trigger_fraction": [0.05, 0.90],
        "_enrichment": [4.0, 12.0], "_bound_width": [10.0, 40.0],
    })
    def inspect(fig, *args, **kwargs):
        prepare_matplotlib(fig, hide_titles=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        labels = [text for text in ax.texts if text.get_text() in set(data["drug"])]
        assert len(labels) == 2
        for label in labels:
            box = label.get_window_extent(renderer)
            assert box.x0 >= ax.bbox.x0 and box.x1 <= ax.bbox.x1
    monkeypatch.setattr(
        "amr_cascade_platform.visualization.report.plotly_prevalence_plotter.save_matplotlib", inspect,
    )
    _plotter()._write_surveillance_sensitivity_map_static(data, "png", tmp_path / "map.png")


def test_reference_shift_labels_do_not_claim_true_prevalence():
    plotter = PlotlyMNARPrevalenceShiftDistributionPlotter(
        PlotlyFigureExporter(1200, 800), "plotly_white", 1200, 800,
    )
    data = pd.DataFrame({
        "pair_label": ["A", "B", "C"], "shift_pct": [10.0, -5.0, 0.0],
        "naive_prevalence_pct": [50.0, 20.0, 30.0],
        "mnar_lambda0_prevalence_pct": [40.0, 25.0, 30.0],
    })
    fig = make_subplots(rows=1, cols=2)
    plotter._panel_a_waterfall(fig, data)
    plotter._panel_b_scatter(fig, data)
    names = {trace.name for trace in fig.data if trace.name}
    assert names == {"Naive above reference", "Naive below reference", "Equal to reference"}
    equal_points = [trace for trace in fig.data if trace.name == "Equal to reference" and trace.mode == "markers"]
    assert len(equal_points) == 1 and list(equal_points[0].x) == [30.0]
