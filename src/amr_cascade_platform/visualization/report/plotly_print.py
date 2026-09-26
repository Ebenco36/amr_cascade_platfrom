"""Make a screen-authored Plotly figure print cleanly (PNG, SVG, PDF).

Static exports are placed at about 6.5 inches wide, so a figure authored for a
1,400-1,600 px screen canvas needs its text enlarged to stay legible. Enlarging
the text on an unchanged canvas is what clipped titles, squeezed the plot until
category axes silently dropped every other label, and drew legends over axis
titles. This module enlarges the text and then lays the page out again around
it, keeping the width (the width is what the page scales to) and the plot area
the author designed:

* text left at Plotly's default size is enlarged too (it has no size in the
  figure's JSON, so scaling explicit sizes alone left ticks and legends tiny);
  markers and lines grow by the square root of the text factor;
* the top and bottom margins are budgeted from what actually sits in them --
  the wrapped title, a horizontal legend, notes and headers placed above or
  below the plot, axis tick labels and titles -- and the canvas grows taller
  to hold them instead of squeezing the plot;
* every label of a categorical axis is shown: x labels wrap at spaces when they
  do not fit side by side and stand up only when a single word is too long;
  a dense axis gets a capped tick font and, if still too dense, a longer plot;
* every cartesian axis reserves room for its own tick labels and title;
* bar value labels are not clipped at the axis edge.

Figures laid out for print geometry themselves (see print_layout) set
``layout.meta.print_geometry`` and only have their fonts and margins scaled.
"""

from __future__ import annotations

import copy
import math
import re

# Every static export (PDF into a ~6.5in LaTeX column, or PNG viewed at anything
# but 1:1) shrinks a figure authored at a screen-oriented canvas (typically
# width=1600) down to a small fraction of its native size. Text is enlarged by
# this factor before static export; the interactive HTML export (which the
# viewer can zoom) is left untouched.
PRINT_FONT_SCALE = 2.6
_DEFAULT_FONT = 12.0  # Plotly's default when a figure sets no font size
_DEFAULT_MARGIN = {"l": 80, "r": 80, "t": 100, "b": 80}  # Plotly's defaults
_CHAR_WIDTH = 0.55  # average glyph width as a share of font size (Arial)
_LINE = 1.3  # Plotly's line spacing, in em of the text element's font
_MIN_TICK_FONT = 14.0  # px before the 2x raster scale; below this the plot grows instead
_MAX_SIDE_MARGINS = 0.6  # left + right margins never take more than this share of the width
_TAG = re.compile(r"(</?[a-zA-Z][^<>]*>)")  # a real tag starts with a letter: "ER < 1" is text
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_NUMERIC_LABEL = re.compile(r"[-+\d.,:/TZ %]+")  # numbers, dates, times and ranges are not category labels
_NON_CARTESIAN = {"sankey", "pie", "table", "parcoords", "parcats", "indicator", "sunburst", "treemap", "icicle", "funnelarea"}


def scale_font_sizes(node: object, parent_key: str | None, factor: float) -> None:
    """Multiply every Plotly font-object ``size`` in place.

    Plotly names every font-bearing key with "font" somewhere in it (font,
    tickfont, textfont, hoverlabel.font, title.font, ...), so gating on the
    immediate parent key containing "font" finds every title/axis/legend/
    annotation/colorbar/trace font without touching unrelated "size" keys such
    as marker.size.
    """
    if isinstance(node, dict):
        if parent_key and "font" in parent_key.lower():
            size = node.get("size")
            if isinstance(size, (int, float)):
                node["size"] = size * factor
        for key, value in node.items():
            scale_font_sizes(value, key, factor)
    elif isinstance(node, list):
        for item in node:
            scale_font_sizes(item, parent_key, factor)


# ---------------------------------------------------------------------------
# Text measurement and wrapping
# ---------------------------------------------------------------------------


def _as_list(values: object) -> list | None:
    if values is None or isinstance(values, (str, bytes, dict)):
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values) if isinstance(values, (list, tuple)) else None


def _visible_length(text: str) -> int:
    return len(re.sub(r"&[a-zA-Z#0-9]+;", "x", _TAG.sub("", str(text))))


def _lines(text: object) -> list[str]:
    return _BREAK.split(str(text))


def _longest_line(text: object) -> int:
    return max((_visible_length(line) for line in _lines(text)), default=0)


def _tag_name(tag: str) -> str:
    match = re.match(r"</?\s*([a-zA-Z][a-zA-Z0-9]*)", tag)
    return match.group(1).lower() if match else ""


def _wrap_line(line: str, max_chars: int) -> list[str]:
    """Break one line at spaces so no piece shows more than ``max_chars`` characters.

    Markup tags are kept intact: a tag open at a break is closed at the end of
    the line and reopened at the start of the next, so <b>, <i>, <sup> and
    styled spans survive wrapping.
    """
    if _visible_length(line) <= max_chars:
        return [line]
    lines: list[str] = []
    stack: list[str] = []
    current, visible = "", 0
    for part in _TAG.split(line):
        if not part:
            continue
        if _TAG.fullmatch(part):
            name = _tag_name(part)
            if part.startswith("</"):
                for index in range(len(stack) - 1, -1, -1):
                    if _tag_name(stack[index]) == name:
                        del stack[index]
                        break
            elif not part.endswith("/>") and name != "br":
                stack.append(part)
            current += part
            continue
        for word in re.split(r"(\s+)", part):
            if not word:
                continue
            if word.isspace():
                if visible:
                    current += " "
                    visible += 1
                continue
            width = _visible_length(word)
            if visible and visible + width > max_chars:
                closing = "".join(f"</{_tag_name(tag)}>" for tag in reversed(stack))
                lines.append(current.rstrip() + closing)
                current, visible = "".join(stack), 0
            current += word
            visible += width
    lines.append(current)
    return [piece for piece in lines if _visible_length(piece)] or [line]


def wrap_markup(text: str, max_chars: float) -> str:
    """Wrap Plotly pseudo-HTML text to at most ``max_chars`` visible characters per line."""
    limit = max(8, int(max_chars))
    return "<br>".join(piece for line in _lines(text) for piece in _wrap_line(line, limit))


def _text_height(text: object, font_px: float) -> float:
    """Plotly spaces every line 1.3 em of the text element's font, <sup>/<sub> lines included."""
    return len(_lines(text)) * _LINE * font_px


def _font_size(node: object, fallback: float) -> float:
    if isinstance(node, dict):
        size = (node.get("font") or {}).get("size")
        if isinstance(size, (int, float)):
            return float(size)
    return fallback


def _title_dict(owner: dict) -> dict | None:
    title = owner.get("title")
    if isinstance(title, str):
        title = {"text": title}
        owner["title"] = title
    return title if isinstance(title, dict) and title.get("text") else None


# ---------------------------------------------------------------------------
# Figure structure
# ---------------------------------------------------------------------------


def _cartesian(trace: dict) -> bool:
    return trace.get("type", "scatter") not in _NON_CARTESIAN


def _trace_axis_ref(trace: dict, letter: str) -> str:
    ref = trace.get(f"{letter}axis") or letter
    return f"{letter}axis" + ref[1:]


def _axis_keys(spec: dict, letter: str) -> list[str]:
    """Every cartesian axis of one letter: those the layout configures and those a trace draws on."""
    keys = {key for key in spec.get("layout", {}) if re.fullmatch(fr"{letter}axis\d*", key)}
    keys.update(_trace_axis_ref(trace, letter) for trace in spec.get("data", []) if _cartesian(trace))
    return sorted(keys)


def _categories(spec: dict, axis_key: str, letter: str) -> list[str]:
    values: list[str] = []
    for trace in spec.get("data", []):
        if not _cartesian(trace) or _trace_axis_ref(trace, letter) != axis_key:
            continue
        if trace.get("orientation") == "h" and trace.get("type") in {"bar", "box", "funnel"} and letter == "x":
            continue
        data = _as_list(trace.get(letter))
        if data and all(isinstance(item, str) for item in data if item is not None):
            values.extend(item for item in data if item is not None)
    return list(dict.fromkeys(values))


def _is_category_labels(labels: list) -> bool:
    return any(not _NUMERIC_LABEL.fullmatch(str(label).strip() or "0") for label in labels)


def _anchor_axis(layout: dict, axis_key: str, other: str) -> dict:
    anchor = (layout.get(axis_key) or {}).get("anchor")
    key = f"{other}axis" + (anchor[1:] if isinstance(anchor, str) and anchor[:1] == other else "")
    return layout.get(key) or {}


def _domain(axis: dict) -> tuple[float, float]:
    domain = axis.get("domain") or [0, 1]
    return float(domain[0]), float(domain[1])


class _Page:
    """The print canvas: fixed width, margins in px, and the plot area's height."""

    def __init__(self, layout: dict, width: float, plot_height: float) -> None:
        self.layout = layout
        self.width = width
        self.margin = layout["margin"]
        self.plot_height = plot_height

    def plot_width(self) -> float:
        return max(1.0, self.width - self.margin["l"] - self.margin["r"])

    def height(self) -> float:
        return self.margin["t"] + self.margin["b"] + self.plot_height


# ---------------------------------------------------------------------------
# Legacy-figure repairs
# ---------------------------------------------------------------------------


def _scale_offsets(layout: dict, factor: float) -> None:
    """Annotation pixel nudges were sized for the screen text; they grow with it."""
    for annotation in layout.get("annotations", []) or []:
        for key in ("xshift", "yshift"):
            if isinstance(annotation.get(key), (int, float)):
                annotation[key] = annotation[key] * factor


def _scale_marks(spec: dict, factor: float) -> None:
    """Markers and lines grow by ``factor`` so they keep their weight next to the larger text."""

    def scale(node: dict, key: str) -> None:
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            node[key] = value * factor
        elif isinstance(value, list) and value and all(isinstance(v, (int, float)) or v is None for v in value):
            node[key] = [v * factor if isinstance(v, (int, float)) else v for v in value]
        elif hasattr(value, "tolist"):
            node[key] = [v * factor if isinstance(v, (int, float)) else v for v in value.tolist()]

    for trace in spec.get("data", []):
        if not _cartesian(trace):
            continue
        marker = trace.get("marker")
        if isinstance(marker, dict):
            if isinstance(marker.get("sizeref"), (int, float)) and marker["sizeref"]:
                power = 2 if marker.get("sizemode") == "area" else 1
                marker["sizeref"] = marker["sizeref"] / factor**power
                scale(marker, "sizemin")
            else:
                scale(marker, "size")
            if isinstance(marker.get("line"), dict):
                scale(marker["line"], "width")
        if isinstance(trace.get("line"), dict):
            scale(trace["line"], "width")
        for key in ("error_x", "error_y"):
            if isinstance(trace.get(key), dict):
                scale(trace[key], "thickness")
                scale(trace[key], "width")
    for shape in spec["layout"].get("shapes", []) or []:
        if isinstance(shape.get("line"), dict):
            scale(shape["line"], "width")


def _fit_title(page: _Page, base_font: float) -> float:
    """Wrap the figure title to the canvas and pin it to the top edge; returns its block height in px."""
    title = _title_dict(page.layout)
    if title is None:
        return 0.0
    font_px = _font_size(title, round(base_font * 1.4))
    max_chars = (page.width * 0.94) / (_CHAR_WIDTH * font_px)
    lines = []
    for line in _lines(title["text"]):
        small = line.strip().lower().startswith(("<sup>", "<sub>"))
        lines.append(wrap_markup(line, max_chars / (0.7 if small else 1.0)))
    title["text"] = "<br>".join(lines)
    # A single-line title is shifted down by its cap height; Plotly drops that
    # shift once the text has line breaks, putting the first baseline on the
    # anchor itself -- so a multi-line title needs the ascent added to its pad.
    pad = 0.35 * font_px + (0.9 * font_px if len(lines) > 1 or "<br>" in title["text"] else 0.0)
    title.update({"yref": "container", "y": 1.0, "yanchor": "top", "pad": {"t": pad, "b": 0, "l": 0, "r": 0}})
    title.setdefault("x", 0.5)
    return 0.35 * font_px + _text_height(title["text"], font_px) + 0.4 * font_px


def _wrap_annotations(page: _Page, base_font: float) -> None:
    for annotation in page.layout.get("annotations", []) or []:
        text = annotation.get("text")
        if not isinstance(text, str) or annotation.get("showarrow"):
            continue
        font_px = _font_size(annotation, base_font)
        if _longest_line(text) * _CHAR_WIDTH * font_px > page.width * 0.94:
            annotation["text"] = wrap_markup(text, (page.width * 0.94) / (_CHAR_WIDTH * font_px))


def _reserve_axis_room(spec: dict) -> None:
    """Let every cartesian axis push the margin out to fit its tick labels and title."""
    if not any(_cartesian(trace) for trace in spec.get("data", [])):
        return
    layout = spec["layout"]
    for letter in ("x", "y"):
        for key in _axis_keys(spec, letter):
            layout.setdefault(key, {}).setdefault("automargin", True)


def _reserve_y_labels(spec: dict, page: _Page, base_font: float) -> None:
    """Widen the side margins to the y axes' tick labels and titles, as Plotly's automargin will.

    Doing it here keeps every later width estimate (x labels, legends, titles)
    on the plot width the figure will really have.
    """
    layout = page.layout
    need = {"l": 0.0, "r": 0.0}
    for key in _axis_keys(spec, "y"):
        axis = layout.get(key) or {}
        if axis.get("visible") is False:
            continue
        low, high = _domain(_anchor_axis(layout, key, "x"))
        side = "r" if axis.get("side") == "right" else "l"
        if (side == "l" and low > 0.02) or (side == "r" and high < 0.98):
            continue  # an inner subplot column: its labels sit between the columns
        font_px = _font_size({"font": axis.get("tickfont")}, base_font) if axis.get("tickfont") else base_font
        labels = _as_list(axis.get("ticktext")) or _categories(spec, key, "y")
        ticks = 0.0
        if axis.get("showticklabels") is not False:
            longest = max((_longest_line(label) for label in labels), default=0) if labels else 5
            ticks = _CHAR_WIDTH * font_px * longest + 0.8 * font_px
        title = _title_dict(axis)
        title_width = _text_height(title["text"], _font_size(title, round(base_font * 1.2))) + 0.4 * base_font if title else 0.0
        need[side] = max(need[side], ticks + title_width)
    for side, value in need.items():
        if value:
            page.margin[side] = max(page.margin[side], min(value + 0.3 * base_font, 0.45 * page.width))


def _square(spec: dict) -> bool:
    layout = spec.get("layout", {})
    return any((layout.get(key) or {}).get("scaleanchor") for letter in ("x", "y") for key in _axis_keys(spec, letter))


def _fit_category_axes(spec: dict, page: _Page, base_font: float) -> None:
    """Show every category label: cap the tick font, wrap or stand up x labels, lengthen the plot if needed."""
    layout = page.layout
    if not any(_cartesian(trace) for trace in spec.get("data", [])):
        return
    for letter in ("y", "x"):
        other = "x" if letter == "y" else "y"
        for key in _axis_keys(spec, letter):
            axis = layout.setdefault(key, {})
            if axis.get("visible") is False or axis.get("showticklabels") is False:
                continue
            if axis.get("type") in {"date", "linear", "log"} or axis.get("tickmode") == "linear":
                continue
            ticks = _as_list(axis.get("tickvals"))
            labels = _as_list(axis.get("ticktext")) or ticks
            categories = [str(label) for label in labels] if labels else _categories(spec, key, letter)
            count = len(categories)
            if count < 2 or not _is_category_labels(categories):
                continue
            low, high = _domain(axis)
            span = max(0.05, high - low)
            square = bool(axis.get("scaleanchor") or any(layout.get(k, {}).get("scaleanchor") for k in _axis_keys(spec, other)))
            font_px = _font_size({"font": axis.get("tickfont")}, base_font) if axis.get("tickfont") else base_font
            if letter == "y":
                lines = max(len(_lines(label)) for label in categories)
                length = (min(page.plot_width(), page.plot_height) if square else page.plot_height) * span
                cap = length / count / (1.2 * lines)
                if cap < _MIN_TICK_FONT:
                    grow = (count * 1.2 * lines * _MIN_TICK_FONT - length) / span
                    page.plot_height += grow
                    if square:
                        page.width += grow
                    cap = _MIN_TICK_FONT
                axis.setdefault("tickfont", {})["size"] = min(font_px, cap)
            else:
                length = (min(page.plot_width(), page.plot_height) if square else page.plot_width()) * span
                per_category = length / count
                fitted = _fit_x_labels(axis, categories, ticks, per_category, font_px)
                if not fitted:  # stand them up: one label per text line's height
                    axis["tickangle"] = -90
                    cap = per_category / 1.2
                    if cap < _MIN_TICK_FONT:
                        grow = (count * 1.2 * _MIN_TICK_FONT - length) / span
                        page.width += grow
                        if square:
                            page.plot_height += grow
                        cap = _MIN_TICK_FONT
                    axis.setdefault("tickfont", {})["size"] = min(font_px, cap)
            if ticks is None and axis.get("type") in (None, "category", "-") and "tickvals" not in axis:
                axis.update({"tickmode": "linear", "dtick": 1, "tick0": 0})


def _fit_x_labels(axis: dict, categories: list[str], ticks: list | None, per_category: float, font_px: float) -> bool:
    """Keep x labels horizontal when they fit, wrapping at spaces if needed; False when they must stand up."""
    angle = axis.get("tickangle")
    if isinstance(angle, (int, float)) and abs(angle) >= 60:
        return False
    available = max(1.0, per_category - 0.6 * font_px)
    if max(_longest_line(label) for label in categories) * _CHAR_WIDTH * font_px <= available:
        axis["tickangle"] = 0
        return True
    words = max(len(word) for label in categories for line in _lines(_TAG.sub("", label)) for word in line.split() or [""])
    max_chars = int(available / (_CHAR_WIDTH * font_px))
    if words > max_chars or max_chars < 4:
        return False
    wrapped = [wrap_markup(label, max_chars) for label in categories]
    axis.update({"tickmode": "array", "tickvals": ticks if ticks is not None else categories, "ticktext": wrapped, "tickangle": 0})
    return True


def _legend(page: _Page) -> dict | None:
    legend = page.layout.get("legend") or {}
    if page.layout.get("showlegend") is False or legend.get("orientation") != "h" or legend.get("yref") == "container":
        return None
    return legend


def _legend_block(spec: dict, page: _Page, legend: dict, base_font: float) -> float:
    """Height of a horizontal legend laid out across the canvas width, in px (0 without entries)."""
    font_px = _font_size(legend, base_font)
    names = [
        str(trace["name"])
        for trace in spec.get("data", [])
        if trace.get("showlegend") is not False and trace.get("name") and trace.get("legend", "legend") == "legend"
        and (trace.get("type") not in {"pie", "sankey"})
    ]
    names = list(dict.fromkeys(names))
    if not names:
        return 0.0
    # Plotly wraps a horizontal legend placed over the plot's x span at the
    # plot area's width, whatever its xref.
    available = page.plot_width()
    rows, used = 1, 0.0
    for name in names:
        width = 40 + _CHAR_WIDTH * font_px * _longest_line(name) + 10
        if used and used + width > available:
            rows, used = rows + 1, 0.0
        used += width
    lines = max(len(_lines(name)) for name in names)
    title = _title_dict(legend)
    title_height = _text_height(title["text"], _font_size(title, font_px)) if title and legend.get("title", {}).get("side", "left") == "top" else 0.0
    return rows * lines * 1.45 * font_px + title_height + 0.5 * font_px


def _place_legend_container_x(page: _Page, legend: dict) -> None:
    x = legend.get("x")
    if legend.get("xref") != "container" and isinstance(x, (int, float)):
        legend["x"] = min(1.0, max(0.0, (page.margin["l"] + x * page.plot_width()) / page.width))
    elif x is None:
        legend.update({"x": 0.5, "xanchor": "center"})
    legend["xref"] = "container"


def _annotation_extents(page: _Page, base_font: float) -> tuple[float, float]:
    """How far paper-anchored notes and headers reach above the plot top and below its bottom, in px."""
    above = below = 0.0
    for annotation in page.layout.get("annotations", []) or []:
        text = annotation.get("text")
        y = annotation.get("y")
        if not isinstance(text, str) or not text or not isinstance(y, (int, float)):
            continue
        if annotation.get("yref", "paper") != "paper" or annotation.get("showarrow"):
            continue
        height = _text_height(text, _font_size(annotation, base_font)) + 0.2 * base_font
        anchor = annotation.get("yanchor", "auto")
        if anchor == "auto":
            anchor = "top" if y >= 2 / 3 else "bottom" if y <= 1 / 3 else "middle"
        shift = float(annotation.get("yshift") or 0.0)
        if y >= 1:
            base = (y - 1) * page.plot_height + shift
            top = base + (height if anchor == "bottom" else height / 2 if anchor == "middle" else 0.0)
            above = max(above, top)
        elif y <= 0:
            base = -y * page.plot_height - shift
            bottom = base + (height if anchor == "top" else height / 2 if anchor == "middle" else 0.0)
            below = max(below, bottom)
    return above, below


def _axis_block(spec: dict, page: _Page, base_font: float, side: str) -> float:
    """Room the outer x axes need on one side (tick labels, then the axis title), in px."""
    layout = page.layout
    need = 0.0
    for key in _axis_keys(spec, "x"):
        axis = layout.get(key) or {}
        if axis.get("visible") is False:
            continue
        axis_side = axis.get("side", "bottom")
        if axis_side != side:
            continue
        low, high = _domain(_anchor_axis(layout, key, "y"))
        if (side == "bottom" and low > 0.02) or (side == "top" and high < 0.98):
            continue  # an inner subplot row: its labels sit between the rows
        font_px = _font_size({"font": axis.get("tickfont")}, base_font) if axis.get("tickfont") else base_font
        labels = _as_list(axis.get("ticktext")) or _categories(spec, key, "x")
        angle = abs(float(axis.get("tickangle") or 0)) if isinstance(axis.get("tickangle"), (int, float)) else 0.0
        if axis.get("showticklabels") is False:
            ticks = 0.0
        elif labels and angle >= 60:
            ticks = _CHAR_WIDTH * font_px * max(_longest_line(label) for label in labels) + 0.8 * font_px
        elif labels and angle > 0:
            ticks = _CHAR_WIDTH * font_px * max(_longest_line(label) for label in labels) * math.sin(math.radians(angle)) + 1.5 * font_px
        else:
            lines = max((len(_lines(label)) for label in labels), default=1) if labels else 1
            ticks = lines * _LINE * font_px + 0.6 * font_px
        title = _title_dict(axis)
        title_height = 0.0
        if title:
            title_font = _font_size(title, round(base_font * 1.2))
            title_height = _text_height(title["text"], title_font) + 0.5 * title_font
        need = max(need, ticks + title_height)
    return need


def _layout_vertical(spec: dict, page: _Page, base_font: float, title_height: float, screen: dict[str, float]) -> None:
    """Budget the top and bottom margins from their contents and pin titles and legends to the canvas edges."""
    layout = page.layout
    legend = _legend(page)
    legend_side = None
    legend_height = 0.0
    if legend is not None:
        y = legend.get("y")
        legend_side = "bottom" if y is None or y < 0 else "top" if y >= 1 else None
        if legend_side:
            legend_height = _legend_block(spec, page, legend, base_font)
            if not legend_height:
                legend_side = None
    gap = 0.5 * base_font
    above, below = _annotation_extents(page, base_font)
    top_axis = _axis_block(spec, page, base_font, "top")
    bottom_axis = _axis_block(spec, page, base_font, "bottom")
    top_legend = legend_height + gap if legend_side == "top" else 0.0
    bottom_legend = legend_height + gap if legend_side == "bottom" else 0.0
    top = title_height + top_legend + max(above, top_axis) + gap
    bottom = max(below, bottom_axis) + bottom_legend + gap
    page.margin["t"] = max(top, screen["t"])
    page.margin["b"] = max(bottom, screen["b"])
    height = page.height()
    if legend_side:
        _place_legend_container_x(page, legend)
        if legend_side == "top":
            legend.update({"yref": "container", "yanchor": "top", "y": 1.0 - (title_height + 0.2 * base_font) / height})
        else:
            legend.update({"yref": "container", "yanchor": "bottom", "y": (0.3 * base_font) / height})
        layout["legend"] = legend


def _unclip_bar_labels(spec: dict) -> None:
    for trace in spec.get("data", []):
        if trace.get("type") == "bar" and trace.get("text") is not None and trace.get("textposition") in (None, "outside", "auto"):
            trace["cliponaxis"] = False


def prepare_for_print(figure_json: dict, width: float, height: float, font_scale: float = PRINT_FONT_SCALE) -> dict:
    """A print-ready copy of a figure's JSON (see module docstring); ``width``/``height`` are the screen canvas.

    The returned layout carries the final canvas size: the width is kept (it is
    what the page scales to) unless a very dense category axis needs more; the
    height is whatever the margins' contents and the plot area need.
    """
    spec = copy.deepcopy(figure_json)
    layout = spec.setdefault("layout", {})
    meta = layout.get("meta") if isinstance(layout.get("meta"), dict) else {}
    if meta.get("print_geometry"):
        scale_font_sizes(layout, None, font_scale)
        scale_font_sizes(spec.get("data", []), None, font_scale)
        margin = layout.get("margin")
        if isinstance(margin, dict):
            for side in ("t", "b", "l", "r"):
                if isinstance(margin.get(side), (int, float)):
                    margin[side] = margin[side] * font_scale
        layout["width"], layout["height"] = int(round(width)), int(round(height))
        return spec

    font = layout.setdefault("font", {})
    if not isinstance(font.get("size"), (int, float)):
        font["size"] = _DEFAULT_FONT
    margin = layout.setdefault("margin", {})
    for side, value in _DEFAULT_MARGIN.items():
        if not isinstance(margin.get(side), (int, float)):
            margin[side] = value
    screen = {side: float(margin[side]) for side in ("t", "b", "l", "r")}
    plot_height = max(1.0, float(height) - screen["t"] - screen["b"])

    scale_font_sizes(layout, None, font_scale)
    scale_font_sizes(spec.get("data", []), None, font_scale)
    _scale_offsets(layout, font_scale)
    _scale_marks(spec, math.sqrt(font_scale))
    sides = screen["l"] * font_scale + screen["r"] * font_scale
    squeeze = min(1.0, _MAX_SIDE_MARGINS * float(width) / sides) if sides else 1.0
    margin["l"], margin["r"] = screen["l"] * font_scale * squeeze, screen["r"] * font_scale * squeeze

    base_font = float(font["size"])
    page = _Page(layout, float(width), plot_height)
    if _square(spec):
        # A matrix drawn with equal x and y scales: keep the plot area as the
        # author proportioned it at the print width, or Plotly pads the range
        # with empty rows to fill the extra height.
        screen_plot_width = max(1.0, float(width) - screen["l"] - screen["r"])
        page.plot_height = page.plot_width() * plot_height / screen_plot_width
    title_height = _fit_title(page, base_font)
    _wrap_annotations(page, base_font)
    _reserve_axis_room(spec)
    _reserve_y_labels(spec, page, base_font)
    _fit_category_axes(spec, page, base_font)
    _layout_vertical(spec, page, base_font, title_height, screen)
    _unclip_bar_labels(spec)
    layout["width"], layout["height"] = int(round(page.width)), int(round(page.height()))
    return spec
