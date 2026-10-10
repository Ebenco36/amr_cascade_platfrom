#!/usr/bin/env python
"""Exercise native static export on the actual compute node before reporting."""
from pathlib import Path
import platform
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


def main() -> None:
    exporter = PlotlyFigureExporter(1200, 800, allow_generic_fallback=False)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Observed", "Eligible"))
    figure.add_trace(go.Bar(x=["A", "B"], y=[2, 5]), row=1, col=1)
    figure.add_trace(go.Scatter(x=[0.1, 0.5], y=[0.8, 0.6], mode="lines+markers"), row=1, col=2)
    with tempfile.TemporaryDirectory(prefix="amr-render-check-") as directory:
        outputs = exporter.write(figure, Path(directory) / "native_renderer", ("pdf", "png", "svg"))
        assert all(path.stat().st_size > 1000 for path in outputs.values())
    print(f"Native PDF/PNG/SVG rendering PASS on {platform.node()} using {sys.executable}")


if __name__ == "__main__":
    main()
