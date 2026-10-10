import io
import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image
from plotly.subplots import make_subplots

from amr_cascade_platform.visualization.report.figure_presentation import (
    figure_titles_hidden, prepare_matplotlib, save_matplotlib, title_policy,
    without_plotly_headings,
)
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


def test_plotly_heading_removal_preserves_data_fonts_and_panel_labels():
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Precision", "Recall"))
    figure.add_scatter(x=[1, 2], y=[0.5, 0.8], name="Model", row=1, col=1)
    figure.update_layout(title="Study title<br><sup>Study subtitle</sup>", width=1200, height=800,
                         font_size=18, margin_t=140, margin_b=100)
    figure.update_xaxes(title="Threshold")
    figure.add_annotation(text="Separate subtitle", name="figure_subtitle", x=0.5, y=1.1,
                          xref="paper", yref="paper", showarrow=False)
    figure.add_annotation(text="No evaluable rows", x=0.5, y=0.5, showarrow=False)
    original = json.loads(figure.to_json())
    clean = without_plotly_headings(figure, width=1200, height=800)
    assert json.loads(figure.to_json()) == original
    assert clean.layout.title.text == ""
    assert clean.layout.width == 1200 and clean.layout.height == 800
    assert clean.layout.margin.t < 140
    assert clean.layout.font.size == 18
    assert clean.layout.xaxis.title.text == "Threshold"
    assert [a.text for a in clean.layout.annotations] == ["Precision", "Recall", "No evaluable rows"]
    assert clean.data[0].to_plotly_json() == figure.data[0].to_plotly_json()
    printed = PlotlyFigureExporter._print_scaled(clean, 1200, 800)
    titled = PlotlyFigureExporter._print_scaled(figure, 1200, 800)
    assert printed.layout.font.size == titled.layout.font.size
    assert printed.layout.width >= 1200
    assert printed.layout.height >= 800
    assert printed.layout.height - printed.layout.margin.t - printed.layout.margin.b >= 800 - 140 - 100


def test_matplotlib_removal_keeps_canvas_axis_labels_and_expands_plot():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.70, bottom=0.15)
    ax.plot([0, 1], [0, 1], label="Observed")
    ax.set(title="Figure title", xlabel="Threshold", ylabel="Recall")
    fig.text(0.5, 0.92, "Figure subtitle", gid="figure_subtitle")
    legend = ax.legend(loc="lower right")
    before = ax.get_position().height
    fontsize = ax.xaxis.label.get_fontsize()
    buffer = io.BytesIO()
    save_matplotlib(fig, buffer, hide_titles=True, format="png", dpi=100, bbox_inches="tight")
    assert not ax.title.get_visible()
    assert not fig.texts[0].get_visible()
    assert ax.get_xlabel() == "Threshold" and legend.get_visible()
    assert ax.get_position().height > before
    assert ax.xaxis.label.get_fontsize() == fontsize
    np.testing.assert_array_equal(fig.get_size_inches(), [8, 5])
    assert Image.open(buffer).size[0] >= 800 and Image.open(buffer).size[1] >= 500
    plt.close(fig)


def test_matplotlib_preserves_multiple_panel_titles():
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Global heading")
    axes[0].set_title("Precision")
    axes[1].set_title("Recall")
    prepare_matplotlib(fig, hide_titles=True)
    assert not fig._suptitle.get_visible()
    assert all(ax.title.get_visible() for ax in axes)
    plt.close(fig)


def test_single_drug_panel_label_is_not_a_global_title():
    fig, ax = plt.subplots()
    fig.suptitle("MNAR summary")
    ax.set_title("MEROPENEM")
    ax._amr_panel_label = True
    prepare_matplotlib(fig, hide_titles=True)
    assert ax.title.get_visible()
    plt.close(fig)


def test_exporter_passes_policy_to_fallback_and_resets_after_error(tmp_path):
    exporter = PlotlyFigureExporter(1200, 800, hide_titles=True)
    def fallback(fmt, path):
        assert figure_titles_hidden()
        raise ValueError("test failure")
    with pytest.raises(ValueError, match="test failure"):
        exporter.write(go.Figure(), tmp_path / "figure", ("png",), fallback, True)
    assert not figure_titles_hidden()


def test_titled_default_does_not_change_matplotlib_figure():
    fig, ax = plt.subplots()
    ax.set_title("Retain this title")
    original = ax.get_position().bounds
    prepare_matplotlib(fig, hide_titles=False)
    assert ax.title.get_visible() and ax.get_position().bounds == original
    with title_policy(True):
        assert figure_titles_hidden()
    assert not figure_titles_hidden()
    plt.close(fig)


def test_no_heading_does_not_relayout_existing_figure():
    figure = go.Figure(layout={"margin": {"t": 80}, "width": 900, "height": 600})
    figure.add_annotation(text="No evaluable rows", x=0.5, y=0.5, showarrow=False)
    clean = without_plotly_headings(figure, width=900, height=600)
    assert clean.layout.margin.t == 80
    assert clean.layout.annotations[0].text == "No evaluable rows"


def test_empty_tipping_html_obeys_presentation_option():
    from amr_cascade_platform.visualization.report.plotly_mnar_tipping_point_plotter import PlotlyMNARTippingPointPlotter
    plotter = PlotlyMNARTippingPointPlotter(hide_titles=True)
    output = plotter._empty_html("MNAR summary", "No evaluable drugs")
    assert "<h2>" not in output
    assert "No evaluable drugs" in output
