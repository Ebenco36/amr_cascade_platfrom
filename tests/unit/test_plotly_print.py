import numpy as np
import plotly.graph_objects as go
import pytest

from amr_cascade_platform.visualization.report.plotly_print import PRINT_FONT_SCALE, prepare_for_print, wrap_markup


def _drugs(count: int) -> list[str]:
    return [f"ANTIBIOTIC NUMBER {index:02d}" for index in range(count)]


def test_text_left_at_the_default_size_is_enlarged_too():
    figure = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    assert spec["layout"]["font"]["size"] == pytest.approx(12 * PRINT_FONT_SCALE)
    assert spec["layout"]["margin"]["l"] == pytest.approx(80 * PRINT_FONT_SCALE)


def test_long_title_wraps_and_the_top_margin_holds_it():
    title = "A very long figure title that cannot possibly fit on one printed line " * 3
    figure = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]), layout={"title": {"text": title, "font": {"size": 20}}, "margin": {"t": 40}})
    spec = prepare_for_print(figure.to_plotly_json(), 1200, 800)
    layout = spec["layout"]
    lines = layout["title"]["text"].split("<br>")
    assert len(lines) > 1
    assert layout["title"]["yref"] == "container" and layout["title"]["yanchor"] == "top"
    font = 20 * PRINT_FONT_SCALE
    assert layout["margin"]["t"] >= len(lines) * 1.3 * font
    assert layout["height"] > 800  # the canvas grew by the extra top margin instead of squeezing the plot


def test_every_category_of_a_dense_axis_is_labelled_without_overlap():
    drugs = _drugs(60)
    figure = go.Figure(go.Heatmap(z=np.ones((60, 60)), x=drugs, y=np.array(drugs, dtype=object)),
                       layout={"yaxis": {"scaleanchor": "x", "tickfont": {"size": 12}}, "xaxis": {"tickfont": {"size": 12}}})
    spec = prepare_for_print(figure.to_plotly_json(), 1100, 1100)
    layout = spec["layout"]
    for key in ("xaxis", "yaxis"):
        assert layout[key]["tickmode"] == "linear" and layout[key]["dtick"] == 1
        assert layout[key]["automargin"] is True
    plot_height = layout["height"] - layout["margin"]["t"] - layout["margin"]["b"]
    assert layout["yaxis"]["tickfont"]["size"] * 1.2 * 60 <= plot_height + 1e-6
    assert layout["xaxis"]["tickangle"] == -90  # 60 long labels cannot sit side by side
    assert layout["width"] > 1100 and layout["height"] > 1100  # square plot grown on both sides


def test_numeric_and_date_axes_are_left_alone():
    years = [str(year) for year in range(2008, 2025)]
    dates = [f"2020-01-{day:02d}" for day in range(1, 29)]
    figure = go.Figure([go.Bar(x=years, y=list(range(17))), go.Scatter(x=dates, y=list(range(28)), xaxis="x2", yaxis="y2")],
                       layout={"xaxis2": {"anchor": "y2"}, "yaxis2": {"anchor": "x2"}})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    assert "tickmode" not in spec["layout"]["xaxis"] and "tickmode" not in spec["layout"]["xaxis2"]


def test_bottom_legend_moves_below_the_axis_title():
    figure = go.Figure([go.Bar(x=["a", "b"], y=[1, 2], name=f"Series {index}") for index in range(4)],
                       layout={"legend": {"orientation": "h", "y": -0.15}, "xaxis": {"title": {"text": "Share of episodes"}},
                               "margin": {"b": 60}})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    legend = spec["layout"]["legend"]
    assert legend["yref"] == "container" and legend["yanchor"] == "bottom" and 0 < legend["y"] < 0.05
    assert not any(key.startswith("_print") for key in legend)
    assert spec["layout"]["margin"]["b"] > 60 * PRINT_FONT_SCALE


def test_top_legend_moves_below_the_title():
    figure = go.Figure([go.Bar(x=["a", "b"], y=[1, 2], name=f"Series {index}") for index in range(3)],
                       layout={"title": {"text": "Title"}, "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom"}})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    layout = spec["layout"]
    title_px = 12 * PRINT_FONT_SCALE * 1.4
    assert layout["legend"]["yref"] == "container" and layout["legend"]["yanchor"] == "top"
    assert (1 - layout["legend"]["y"]) * layout["height"] > 1.3 * title_px  # starts under the title
    assert layout["margin"]["t"] > (1 - layout["legend"]["y"]) * layout["height"]


def test_print_geometry_figures_are_only_scaled():
    figure = go.Figure(go.Bar(x=_drugs(30), y=list(range(30))),
                       layout={"meta": {"print_geometry": True}, "font": {"size": 14}, "margin": {"t": 50}, "title": {"text": "x" * 400}})
    spec = prepare_for_print(figure.to_plotly_json(), 2000, 900)
    layout = spec["layout"]
    assert layout["font"]["size"] == pytest.approx(14 * PRINT_FONT_SCALE)
    assert layout["title"]["text"] == "x" * 400 and "tickmode" not in layout.get("xaxis", {})
    assert (layout["width"], layout["height"]) == (2000, 900)


def test_wrapping_keeps_markup_balanced():
    wrapped = wrap_markup("<b>Resistant upstream result</b> with a <i>long italic tail that must wrap</i> here", 20)
    for line in wrapped.split("<br>"):
        assert line.count("<b>") == line.count("</b>") and line.count("<i>") == line.count("</i>")


def test_less_than_sign_in_a_title_is_text_not_a_tag():
    figure = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]), layout={"title": {"text": "<b>Suppression matrix (ER < 1)</b> " + "long words " * 30}})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    text = spec["layout"]["title"]["text"]
    assert text.count("ER < 1") == 1 and "</1>" not in text


def test_multi_line_title_is_pushed_down_by_its_ascent():
    single = prepare_for_print(go.Figure(layout={"title": {"text": "One line", "font": {"size": 20}}}).to_plotly_json(), 1600, 900)
    multi = prepare_for_print(go.Figure(layout={"title": {"text": "One<br><sup>two</sup>", "font": {"size": 20}}}).to_plotly_json(), 1600, 900)
    font = 20 * PRINT_FONT_SCALE
    assert single["layout"]["title"]["pad"]["t"] == pytest.approx(0.35 * font)
    assert multi["layout"]["title"]["pad"]["t"] == pytest.approx(1.25 * font)


def test_plot_area_keeps_its_screen_height_and_margins_hold_only_their_contents():
    figure = go.Figure(go.Bar(x=["a", "b", "c"], y=[1, 2, 3]),
                       layout={"height": 900, "margin": {"t": 200, "b": 300, "l": 80, "r": 80}, "title": {"text": "Short"}})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    layout = spec["layout"]
    assert layout["height"] - layout["margin"]["t"] - layout["margin"]["b"] == pytest.approx(900 - 200 - 300, abs=1)
    assert layout["margin"]["t"] == pytest.approx(200) and layout["margin"]["b"] == pytest.approx(300)  # never below the author's own


def test_few_long_x_labels_wrap_instead_of_standing_up():
    labels = ["All sites in the network", "Stanford Health Care", "ECU Health medical center", "UT Southwestern Medical Center"]
    figure = go.Figure(go.Bar(x=labels, y=[1, 2, 3, 4]), layout={"width": 1600})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 700)
    axis = spec["layout"]["xaxis"]
    assert axis["tickangle"] == 0 and axis["tickvals"] == labels
    assert all("<br>" in text for text in axis["ticktext"])


def test_notes_below_the_plot_get_room_above_a_bottom_legend():
    figure = go.Figure([go.Scatter(x=[1, 2], y=[1, 2], name="Series")],
                       layout={"legend": {"orientation": "h", "y": -0.3}, "margin": {"b": 60},
                               "annotations": [{"text": "Note: one line of explanation", "x": 0, "y": -0.2, "xref": "paper",
                                                "yref": "paper", "showarrow": False, "yanchor": "top"}]})
    spec = prepare_for_print(figure.to_plotly_json(), 1600, 900)
    layout = spec["layout"]
    plot_height = layout["height"] - layout["margin"]["t"] - layout["margin"]["b"]
    note_bottom = 0.2 * plot_height + 1.3 * 12 * PRINT_FONT_SCALE
    legend_top = layout["legend"]["y"] * layout["height"] + 1.45 * 12 * PRINT_FONT_SCALE
    assert layout["margin"]["b"] >= note_bottom + (legend_top - layout["legend"]["y"] * layout["height"])
