"""Shared Plotly export utilities."""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import patches
from matplotlib import pyplot as plt


class PlotlyFigureExporter:
    """Write Plotly figures in multiple formats with publication-safe static fallbacks."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def write(
        self,
        figure: go.Figure,
        output_stem: Path,
        formats: tuple[str, ...],
        static_fallback: Callable[[str, Path], None] | None = None,
        prefer_static_fallback: bool = False,
    ) -> dict[str, Path]:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        width = int(figure.layout.width or self._width)
        height = int(figure.layout.height or self._height)
        for fmt in formats:
            path = output_stem.with_suffix(f".{fmt}")
            if fmt == "html":
                pio.write_html(figure, path, include_plotlyjs="cdn", full_html=True)
            elif static_fallback is not None and prefer_static_fallback:
                static_fallback(fmt, path)
            else:
                try:
                    figure.write_image(path, format=fmt, width=width, height=height, scale=2)
                except Exception as exc:
                    if static_fallback is None:
                        self.write_matplotlib_static_fallback(figure, path, fmt, width=width, height=height)
                    else:
                        try:
                            static_fallback(fmt, path)
                        except Exception as fallback_exc:
                            raise RuntimeError(
                                f"Static export failed for {path}; Plotly/kaleido error={exc!r}; "
                                f"fallback error={fallback_exc!r}"
                            ) from fallback_exc
            outputs[path.name] = path
        return outputs

    @staticmethod
    def write_matplotlib_static_fallback(
        figure: go.Figure,
        path: Path,
        fmt: str,
        *,
        width: int,
        height: int,
    ) -> None:
        """Render a real static approximation when Plotly's image engine is unavailable.

        Kaleido can fail on some local runtimes even when installed. We still need
        non-placeholder files for review, so this fallback draws supported traces,
        layout shapes, and annotations directly with Matplotlib.
        """

        dpi = 300
        figsize = (max(width, 1200) / dpi, max(height, 628) / dpi)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")

        figure_dict = figure.to_dict()
        data = figure_dict.get("data", [])
        layout = figure_dict.get("layout", {})

        plotted_traces = PlotlyFigureExporter._draw_supported_traces(ax, data)
        drew_shapes = PlotlyFigureExporter._draw_layout_shapes(ax, layout)
        drew_annotations = PlotlyFigureExporter._draw_annotations(ax, layout)

        title = PlotlyFigureExporter._clean_text(layout.get("title", {}).get("text", ""))
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

        if plotted_traces:
            x_title = PlotlyFigureExporter._clean_text(layout.get("xaxis", {}).get("title", {}).get("text", ""))
            y_title = PlotlyFigureExporter._clean_text(layout.get("yaxis", {}).get("title", {}).get("text", ""))
            if x_title:
                ax.set_xlabel(x_title)
            if y_title:
                ax.set_ylabel(y_title)
            ax.grid(True, axis="y", alpha=0.20)
            if any(trace.get("name") for trace in data):
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(frameon=False, fontsize=8, loc="best")
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            if not drew_shapes and not drew_annotations:
                ax.text(
                    0.5,
                    0.5,
                    title or path.stem.replace("_", " ").title(),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _draw_supported_traces(ax: plt.Axes, data: list[dict]) -> bool:
        plotted = False
        category_maps: dict[str, dict[str, int]] = {"x": {}, "y": {}}

        for trace in data:
            trace_type = trace.get("type", "scatter")
            name = trace.get("name") or None
            if trace_type in {"scatter", "scattergl"}:
                x_values = PlotlyFigureExporter._axis_values(trace.get("x", []), category_maps["x"])
                y_values = PlotlyFigureExporter._axis_values(trace.get("y", []), category_maps["y"])
                if not x_values or not y_values:
                    continue
                mode = str(trace.get("mode", "lines+markers"))
                line = trace.get("line", {}) or {}
                marker = trace.get("marker", {}) or {}
                color = PlotlyFigureExporter._mpl_color(line.get("color") or marker.get("color"))
                linestyle = "-" if "lines" in mode else "None"
                marker_style = "o" if "markers" in mode or "lines" not in mode else None
                ax.plot(
                    x_values,
                    y_values,
                    label=name,
                    color=color,
                    linestyle=linestyle,
                    marker=marker_style,
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.90,
                )
                plotted = True
            elif trace_type == "bar":
                x_values = PlotlyFigureExporter._axis_values(trace.get("x", []), category_maps["x"])
                y_values = PlotlyFigureExporter._numeric_values(trace.get("y", []))
                if not x_values or not y_values:
                    continue
                marker = trace.get("marker", {}) or {}
                ax.bar(x_values, y_values, label=name, color=PlotlyFigureExporter._mpl_color(marker.get("color")), alpha=0.85)
                plotted = True
            elif trace_type == "heatmap":
                z_values = trace.get("z", [])
                if z_values is None or len(z_values) == 0:
                    continue
                image = ax.imshow(z_values, aspect="auto", cmap="viridis")
                ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                plotted = True

        for axis_name, mapping in category_maps.items():
            if not mapping:
                continue
            ticks = list(mapping.values())
            labels = list(mapping.keys())
            if axis_name == "x":
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            else:
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels, fontsize=8)

        return plotted

    @staticmethod
    def _draw_layout_shapes(ax: plt.Axes, layout: dict) -> bool:
        drew = False
        for shape in layout.get("shapes", []) or []:
            x0 = PlotlyFigureExporter._paper_coordinate(shape.get("x0"))
            x1 = PlotlyFigureExporter._paper_coordinate(shape.get("x1"))
            y0 = PlotlyFigureExporter._paper_coordinate(shape.get("y0"))
            y1 = PlotlyFigureExporter._paper_coordinate(shape.get("y1"))
            if any(value is None for value in (x0, x1, y0, y1)):
                continue
            line = shape.get("line", {}) or {}
            fill = PlotlyFigureExporter._mpl_color(shape.get("fillcolor"), default="none")
            edge = PlotlyFigureExporter._mpl_color(line.get("color"), default="#1f2937")
            width = float(line.get("width", 1.0) or 1.0)
            if shape.get("type") == "line":
                ax.plot([x0, x1], [y0, y1], color=edge, linewidth=width, transform=ax.transAxes, clip_on=False)
            else:
                rect = patches.Rectangle(
                    (min(x0, x1), min(y0, y1)),
                    abs(x1 - x0),
                    abs(y1 - y0),
                    linewidth=width,
                    edgecolor=edge,
                    facecolor=fill,
                    transform=ax.transAxes,
                    clip_on=False,
                )
                ax.add_patch(rect)
            drew = True
        return drew

    @staticmethod
    def _draw_annotations(ax: plt.Axes, layout: dict) -> bool:
        drew = False
        for annotation in layout.get("annotations", []) or []:
            x = PlotlyFigureExporter._paper_coordinate(annotation.get("x", 0.5))
            y = PlotlyFigureExporter._paper_coordinate(annotation.get("y", 0.5))
            if x is None or y is None:
                continue
            text = PlotlyFigureExporter._clean_text(annotation.get("text", ""))
            if not text:
                continue
            font = annotation.get("font", {}) or {}
            arrowprops = None
            if annotation.get("showarrow"):
                ax_ref = PlotlyFigureExporter._paper_coordinate(annotation.get("ax"))
                ay_ref = PlotlyFigureExporter._paper_coordinate(annotation.get("ay"))
                if ax_ref is not None and ay_ref is not None:
                    arrowprops = {"arrowstyle": "->", "color": "#374151", "lw": 1.2}
            ax.annotate(
                text,
                xy=(x, y),
                xycoords=ax.transAxes,
                xytext=(x, y),
                textcoords=ax.transAxes,
                ha=PlotlyFigureExporter._horizontal_alignment(annotation.get("xanchor")),
                va=PlotlyFigureExporter._vertical_alignment(annotation.get("yanchor")),
                fontsize=float(font.get("size", 10) or 10),
                color=PlotlyFigureExporter._mpl_color(font.get("color"), default="#111827"),
                fontweight="bold" if "<b>" in str(annotation.get("text", "")).lower() else "normal",
                arrowprops=arrowprops,
                wrap=True,
            )
            drew = True
        return drew

    @staticmethod
    def _axis_values(values: object, mapping: dict[str, int]) -> list[float]:
        if values is None:
            return []
        result: list[float] = []
        for value in values:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result.append(float(value))
            else:
                label = str(value)
                if label not in mapping:
                    mapping[label] = len(mapping)
                result.append(float(mapping[label]))
        return result

    @staticmethod
    def _numeric_values(values: object) -> list[float]:
        if values is None:
            return []
        result: list[float] = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float("nan")
            result.append(numeric)
        return result

    @staticmethod
    def _paper_coordinate(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _clean_text(value: object) -> str:
        text = str(value or "")
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    @staticmethod
    def _mpl_color(value: object, default: str = "#2563eb") -> object:
        if value is None:
            return default
        if isinstance(value, (list, tuple)) and value:
            if all(isinstance(item, (int, float)) for item in value[:4]):
                numeric = [float(item) for item in value[:4]]
                if len(numeric) >= 3:
                    scale = 255.0 if any(item > 1.0 for item in numeric[:3]) else 1.0
                    rgb = tuple(max(0.0, min(1.0, item / scale)) for item in numeric[:3])
                    alpha = max(0.0, min(1.0, numeric[3])) if len(numeric) >= 4 else 1.0
                    return (*rgb, alpha)
            return PlotlyFigureExporter._mpl_color(value[0], default=default)
        text = str(value)
        if text.startswith("rgba"):
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
            if len(numbers) >= 3:
                rgb = tuple(max(0.0, min(1.0, float(num) / 255.0)) for num in numbers[:3])
                alpha = max(0.0, min(1.0, float(numbers[3]))) if len(numbers) >= 4 else 1.0
                return (*rgb, alpha)
        return text

    @staticmethod
    def _horizontal_alignment(value: object) -> str:
        text = str(value or "center").lower()
        return text if text in {"left", "center", "right"} else "center"

    @staticmethod
    def _vertical_alignment(value: object) -> str:
        text = str(value or "center").lower()
        if text == "middle":
            return "center"
        return text if text in {"top", "center", "bottom", "baseline"} else "center"
