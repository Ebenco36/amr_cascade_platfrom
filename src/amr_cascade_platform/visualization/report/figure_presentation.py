"""Optional figure-level heading removal without shrinking text or the canvas."""

from contextlib import contextmanager
from contextvars import ContextVar
import copy

import plotly.graph_objects as go

_HIDE_TITLES = ContextVar("hide_figure_titles", default=False)
HEADING_ROLES = {"figure_title", "figure_subtitle"}


def figure_titles_hidden() -> bool:
    return _HIDE_TITLES.get()


@contextmanager
def title_policy(hide_titles: bool):
    """Carry the export option into plot-specific Matplotlib fallback callbacks."""
    token = _HIDE_TITLES.set(hide_titles)
    try:
        yield
    finally:
        _HIDE_TITLES.reset(token)


def without_plotly_headings(figure: go.Figure, *, width: int, height: int) -> go.Figure:
    out = go.Figure(copy.deepcopy(figure.to_plotly_json()))
    out.update_layout(title_text="", width=width, height=height)
    kept = [annotation for annotation in out.layout.annotations
            if annotation.name not in HEADING_ROLES]
    out.layout.annotations = kept
    if not figure.layout.title.text and len(kept) == len(figure.layout.annotations):
        return out
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    if meta.get("heading_band"):
        # Print-authored layouts place legends in canvas coordinates. Move those
        # legends with the reclaimed heading band, keeping panel-label clearance.
        band = min(float(meta["heading_band"]), max(0, float(figure.layout.margin.t or 100) - 24))
        out.update_layout(margin_t=float(figure.layout.margin.t or 100) - band)
        shift = band * float(meta.get("geometry_scale", 1)) / height
        for name in out.layout:
            if name.startswith("legend"):
                legend = out.layout[name]
                if legend.yref == "container" and legend.y is not None and legend.y > 0.5:
                    legend.y = min(1.0, legend.y + shift)
        return out
    # Keep semantic panel/column labels above the plot, but reclaim the title band.
    top = 24.0
    old_top = float(figure.layout.margin.t or 100)
    plot_height = max(1.0, height - old_top - float(figure.layout.margin.b or 80))
    for annotation in kept:
        if annotation.yref == "paper" and isinstance(annotation.y, (int, float)) and annotation.y >= 1:
            font = float(annotation.font.size or 12)
            lines = str(annotation.text or "").count("<br>") + 1
            top = max(top, (annotation.y - 1) * plot_height + max(0, annotation.yshift or 0) + font * 1.5 * lines + 12)
    out.update_layout(margin_t=top)
    return out


def prepare_matplotlib(figure, *, hide_titles: bool) -> None:
    """Remove global headings; keep multi-panel titles and all axis/legend labels."""
    if not hide_titles or getattr(figure, "_amr_headings_removed", False):
        return
    figure._amr_headings_removed = True
    removed = False
    heading = getattr(figure, "_suptitle", None)
    if heading is not None and heading.get_text():
        heading.set_visible(False)
        removed = True
    for artist in list(figure.texts) + [text for ax in figure.axes for text in ax.texts]:
        if artist.get_gid() in HEADING_ROLES:
            artist.set_visible(False)
            removed = True
    axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    if len(axes) == 1 and not getattr(axes[0], "_amr_panel_label", False):
        for title in (axes[0].title, axes[0]._left_title, axes[0]._right_title):
            if title.get_text():
                title.set_visible(False)
                removed = True
    if not removed or not figure.axes or getattr(figure, "_amr_preserve_layout", False):
        return
    # Stretch vertically into the vacated heading band. Do not reduce any font,
    # axis width, or canvas dimension; retain room for surviving figure furniture.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    positions = [ax.get_position(original=True).frozen() for ax in figure.axes]
    bottom, top = min(p.y0 for p in positions), max(p.y1 for p in positions)
    target = 0.96 if len(axes) <= 1 else 0.92
    for artist in list(figure.texts) + list(figure.legends):
        if artist.get_visible():
            box = artist.get_window_extent(renderer).transformed(figure.transFigure.inverted())
            if box.y0 >= top:
                target = min(target, box.y0 - 0.02)
    if top <= bottom or target <= top:
        return
    factor = (target - bottom) / (top - bottom)
    for ax, position in zip(figure.axes, positions):
        ax.set_position([position.x0, bottom + (position.y0 - bottom) * factor,
                         position.width, position.height * factor])


def save_matplotlib(figure, destination, *, hide_titles: bool | None = None, **kwargs) -> None:
    hide = _HIDE_TITLES.get() if hide_titles is None else hide_titles
    prepare_matplotlib(figure, hide_titles=hide)
    # The caller's cropping applies in both modes. Forcing the full canvas here left blank
    # bands beside the plot (so it printed smaller at a given page width) and no padding
    # where a label reached past the canvas; the plot has already been stretched into the
    # heading band above, so a tight crop removes margins only.
    figure.savefig(destination, **kwargs)
