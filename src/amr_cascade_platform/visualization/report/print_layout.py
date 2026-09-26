"""Shared geometry and chrome for figures built for both screen (HTML) and print (PNG/SVG/PDF).

The house exporter (PlotlyFigureExporter) enlarges every font and margin by
PRINT_FONT_SCALE for static formats but not the canvas, marker sizes, line
widths or pixel offsets. A figure therefore describes its pixel sizes through
``Geometry.px``, which multiplies them by the same factor for print, while its
margins stay in screen units (the exporter scales those itself). Every figure
built this way reads the same on screen and on paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

from amr_cascade_platform.visualization.report.plotly_exporter import PRINT_FONT_SCALE

INK = "#2f3b4c"
INK_SPARSE = "#a3acb9"
TRACK = "#e4e7ec"
TEXT = "#1a1a2e"
MUTED = "#5b6472"
GRID = "#eef1f5"
RULE = "#c9ced6"
FONT = "Arial"
# Sized so that labels print at about 7 pt once a full-width figure is placed on the page.
FONT_TITLE, FONT_HEADING, FONT_LABEL, FONT_VALUE, FONT_LEGEND = 20, 15, 14, 13, 14
# Screen-pixel gaps: row label to axis, track end (or a marker drawn at 100%) to its value, legend placement.
LABEL_GAP, VALUE_GAP, LEGEND_TOP, LEGEND_ROW = 10, 14, 50, 30


@dataclass(frozen=True)
class Geometry:
    """Pixel sizes the exporter does not scale: canvas, pitches, marks, offsets."""

    width: int
    unit: float  # multiplier for every other pixel size below
    margin_scale: float  # what the exporter multiplies margins by

    @classmethod
    def for_medium(cls, medium: str, screen_width: int, print_width: int) -> Geometry:
        if medium == "screen":
            return cls(width=screen_width, unit=1.0, margin_scale=1.0)
        if medium == "print":
            return cls(width=print_width, unit=PRINT_FONT_SCALE, margin_scale=PRINT_FONT_SCALE)
        raise ValueError(f"medium must be 'screen' or 'print', not {medium!r}")

    def px(self, value: float) -> float:
        return value * self.unit


def export_both_media(exporter, build, data, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
    """HTML from the screen geometry, static formats from the print geometry."""
    outputs: dict[str, Path] = {}
    screen = tuple(fmt for fmt in formats if fmt == "html")
    static = tuple(fmt for fmt in formats if fmt != "html")
    if screen:
        outputs.update(exporter.write(build(data, medium="screen"), output_stem, screen))
    if static:
        outputs.update(exporter.write(build(data, medium="print"), output_stem, static))
    return outputs


def key(fig: go.Figure, g: Geometry, entries: list[tuple[str, str, str, str]], legend: str = "legend", **position) -> None:
    """Legend-only entries (name, symbol, fill, outline) drawn at the medium's marker size.

    Plotly sizes bar swatches and constant-size legend symbols in pixels the
    print exporter does not scale, so the key is built from marker traces
    sized for the medium and every data trace stays out of the legend.
    """
    for rank, (name, symbol, fill, outline) in enumerate(entries, start=1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name, legend=legend, legendrank=rank, hoverinfo="skip", showlegend=True,
            marker={"symbol": symbol, "size": g.px(13), "color": fill, "line": {"color": outline, "width": g.px(1.4)}},
        ), **position)


def frame(
    fig: go.Figure,
    g: Geometry,
    height: int,
    title: str,
    margin: dict,
    legends: dict[str, str | None],
    template: str,
) -> None:
    """Shared layout; ``legends`` maps each legend id to its row title (or ``None``), top row first."""
    rows = {}
    for index, (name, heading) in enumerate(legends.items()):
        rows[name] = {
            "orientation": "h", "x": g.px(16) / g.width, "xref": "container", "xanchor": "left",
            "y": 1 - g.px(LEGEND_TOP + index * LEGEND_ROW) / height, "yref": "container", "yanchor": "top",
            "font": {"size": FONT_LEGEND, "color": TEXT}, "traceorder": "normal", "itemsizing": "trace",
            "itemwidth": max(30, int(g.px(22))),
        }
        if heading:
            rows[name]["title"] = {"text": heading, "side": "left", "font": {"size": FONT_LEGEND, "color": MUTED}}
    fig.update_layout(**rows)
    fig.update_layout(
        template=template, width=g.width, height=height, barmode="overlay",
        plot_bgcolor="white", paper_bgcolor="white", font={"family": FONT, "size": FONT_LABEL, "color": TEXT},
        title={"text": title, "x": g.px(16) / g.width, "xref": "container", "xanchor": "left",
               "y": 1 - g.px(14) / height, "yref": "container", "yanchor": "top",
               "font": {"size": FONT_TITLE, "color": TEXT}},
        margin=dict(margin),
        hoverlabel={"font": {"family": FONT}},
        meta={"print_geometry": True},  # laid out for print already: the exporter only scales its text
    )


def empty_figure(message: str, template: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 14, "color": MUTED})
    fig.update_layout(template=template, width=1400, height=400, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig
