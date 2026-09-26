from pathlib import Path

import pandas as pd

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
