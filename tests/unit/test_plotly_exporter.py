from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


def test_plotly_exporter_writes_real_static_fallback_when_kaleido_fails(tmp_path: Path, monkeypatch) -> None:
    def fail_write_image(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("kaleido unavailable")

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)

    figure = go.Figure(data=[go.Bar(x=["A", "B"], y=[1, 3], name="Observed")])
    figure.update_layout(title={"text": "Test Figure"}, width=1200, height=628)

    outputs = PlotlyFigureExporter(width=1200, height=628).write(
        figure,
        tmp_path / "figure_test",
        ("html", "png", "pdf", "svg"),
    )

    assert set(outputs) == {
        "figure_test.html",
        "figure_test.png",
        "figure_test.pdf",
        "figure_test.svg",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    placeholder = b"Static export unavailable"
    assert placeholder not in (tmp_path / "figure_test.pdf").read_bytes()
    assert placeholder not in (tmp_path / "figure_test.svg").read_bytes()
