"""Threshold charts preserve saved results while using readable common scales."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.visualization.report import plotly_model_plotters as plotting
from amr_cascade_platform.visualization.report.figure_presentation import without_plotly_headings


@pytest.fixture
def supplementary():
    path = Path(__file__).resolve().parents[2] / "scripts/run_supplementary_prediction.py"
    spec = importlib.util.spec_from_file_location("supplementary_threshold_layout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scores():
    return pd.DataFrame({
        "split": ["validation"] * 3,
        "model_name": ["logistic_regression"] * 3,
        "threshold": [0.05, 0.30, 0.95],
        "precision": [0.5, 0.7, 0.9], "recall": [1.0, 0.8, 0.1],
        "f1": [0.67, 0.75, 0.18], "balanced_accuracy": [0.5, 0.8, 0.6],
        "mcc": [-0.1, 0.5, 0.1], "selected_for_decision": [False, True, False],
        "threshold_selection_metric": ["balanced_accuracy"] * 3,
        "threshold_selection_strategy": ["grouped_oof"] * 3,
    })


def test_supplementary_threshold_axes_preserve_data(supplementary, scores, monkeypatch, tmp_path):
    captured = {}
    def capture(fig, stem, **kwargs):
        captured["figure"] = fig
        captured["fallback"] = kwargs["static_fallback"]
        return {}
    monkeypatch.setattr(supplementary, "_write_all_formats", capture)
    supplementary.fig_threshold(scores, tmp_path)
    fig = captured["figure"]
    for index in range(1, 5):
        axis = fig.layout["xaxis" + (str(index) if index > 1 else "")]
        assert tuple(axis.range) == (0, 1)
        np.testing.assert_allclose(axis.tickvals, np.linspace(0, 1, 6))
        assert not axis.title.text
    curves = [trace for trace in fig.data if trace.mode == "lines+markers"]
    assert len(curves) == 4
    for trace, metric in zip(curves, ("precision", "recall", "f1", "balanced_accuracy")):
        np.testing.assert_allclose(trace.x, scores.threshold)
        np.testing.assert_allclose(trace.y, scores[metric])
    markers = [trace for trace in fig.data if trace.mode == "markers" and trace.showlegend is False]
    assert len(markers) == 4
    assert all(tuple(trace.x) == (0.3,) for trace in markers)
    clean = without_plotly_headings(fig, width=fig.layout.width, height=fig.layout.height)
    assert sum(a.name == "shared_axis_label" for a in clean.layout.annotations) == 1
    note = next(a.text for a in clean.layout.annotations if a.name == "selection_note")
    assert "patient-grouped out-of-fold training predictions" in note
    assert "balanced accuracy" in note
    assert all(any(a.text == label for a in clean.layout.annotations)
               for label in ("Precision (PPV)", "Recall (Sensitivity)", "F1 Score", "Balanced Accuracy"))

    static_figures = []
    monkeypatch.setattr(plotting, "save_matplotlib", lambda fig, *a, **k: static_figures.append(fig))
    captured["fallback"]("png", tmp_path / "test.png")
    static = static_figures[0]
    assert len(static.axes) == 4
    for axis in static.axes:
        assert axis.get_xlim() == (0, 1)
        assert not axis.get_xlabel()
        np.testing.assert_allclose(axis.lines[0].get_xdata(), scores.threshold)


def test_generic_threshold_chart_includes_negative_mcc(scores, monkeypatch, tmp_path):
    class CaptureExporter:
        def write(self, fig, *args, **kwargs):
            self.figure = fig
            return {}
    exporter = CaptureExporter()
    plotter = plotting.PlotlyModelEvaluationPlotter(exporter, "plotly_white", 1600, 900)
    plotter.export_threshold_analysis(scores, tmp_path / "threshold", ("html",))
    # Selection metric is first, MCC is second.
    assert exporter.figure.layout.yaxis2.range == (-1.02, 1.02)
    for index in range(1, 6):
        axis = exporter.figure.layout["xaxis" + (str(index) if index > 1 else "")]
        assert tuple(axis.range) == (0, 1)
    scores["threshold_selection_strategy"] = "validation_holdout"
    assert "on the validation split" in plotter.threshold_selection_note(scores)
