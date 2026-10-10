from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from amr_cascade_platform.visualization.report.figure_presentation import prepare_matplotlib
from amr_cascade_platform.visualization.report.plotly_delta_curves_plotter import PrevalenceDeltaCurvePlotter


def test_every_sensitivity_tier_gets_a_full_size_figure_even_when_empty(tmp_path: Path) -> None:
    # One drug swings 12 percentage points (sensitive); the stable and moderate
    # tiers are empty, yet the manuscript places a figure for each tier.
    curves = pd.DataFrame({
        "drug": ["MEROPENEM"] * 3,
        "mnar_lambda": [-1.0, 0.0, 1.0],
        "mnar_prevalence_pct": [20.0, 14.0, 8.0],
        "mnar_status": ["estimated"] * 3,
    })
    summary = pd.DataFrame({"drug": ["MEROPENEM"], "naive_prevalence_pct": [15.0]})
    outputs = PrevalenceDeltaCurvePlotter().export(
        curves, summary, tmp_path / "figure_prevalence_mnar_sensitivity_curves", ("png", "pdf")
    )
    for tier in ("stable", "moderate", "sensitive"):
        assert f"figure_prevalence_mnar_sensitivity_curves_{tier}.pdf" in outputs
        with Image.open(outputs[f"figure_prevalence_mnar_sensitivity_curves_{tier}.png"]) as image:
            # The readiness audit's minimum PNG size.
            assert image.size[0] >= 1200 and image.size[1] >= 628
    # Only a populated not-evaluable tier is drawn: no manuscript figure depends on it.
    assert not any("not_evaluable" in name for name in outputs)


@pytest.mark.parametrize("drug_count", [1, 2, 9])
@pytest.mark.parametrize("hide_titles", [False, True])
def test_tier_legend_caption_and_drug_labels_do_not_overlap(drug_count, hide_titles):
    drugs = [f"AMPICILLIN/SULBACTAM {number}" for number in range(drug_count)]
    curves = pd.DataFrame([
        {"drug": drug, "mnar_lambda": value, "mnar_prevalence_pct": 20 - value}
        for drug in drugs for value in (-1, 0, 1)
    ])
    plotter = PrevalenceDeltaCurvePlotter(hide_titles=hide_titles)
    fig = plotter._build_tier_figure(
        "moderate", "Moderate", "#B8860B", drugs, curves,
        pd.Series(2.0, index=drugs), dict.fromkeys(drugs, 19.0), "ESCHERICHIA COLI",
    )
    prepare_matplotlib(fig, hide_titles=hide_titles)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    caption = fig.texts[-1].get_window_extent(renderer)
    for ax in fig.axes:
        if ax.get_visible():
            assert not legend.overlaps(ax.title.get_window_extent(renderer))
            assert not caption.overlaps(ax.get_tightbbox(renderer))
    assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width
    assert caption.x0 >= 0 and caption.x1 <= fig.bbox.width
    if not hide_titles:
        assert not legend.overlaps(fig._suptitle.get_window_extent(renderer))
        assert fig._suptitle.get_window_extent(renderer).x0 >= 0
    plt.close(fig)
