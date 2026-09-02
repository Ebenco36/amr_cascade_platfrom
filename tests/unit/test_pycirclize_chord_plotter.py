from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.pycirclize_chord_plotter import PyCirclizeChordPlotter

_CLASSIFICATION_PATH = Path(__file__).resolve().parents[2] / "data" / "antibiotic_classification_complete.csv"


@pytest.fixture
def plotter() -> PyCirclizeChordPlotter:
    resolver = AntibioticClassificationResolver(_CLASSIFICATION_PATH)
    return PyCirclizeChordPlotter(resolver)


def test_chord_export_with_data_includes_tier_label_in_html(plotter: PyCirclizeChordPlotter, tmp_path: Path) -> None:
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "MEROPENEM",
                "total_support_n": 50,
                "escalation_ratio": 3.0,
                "adjusted_odds_ratio": 2.1,
            },
        ]
    )
    outputs = plotter.export(retained_edges, tmp_path / "chord", ("html", "png"), tier_label="Robust")
    assert "chord.html" in outputs
    assert "chord.png" in outputs
    assert outputs["chord.png"].exists()
    html = outputs["chord.html"].read_text(encoding="utf-8")
    assert "Robust Antibiotic Cascade Chord Diagram" in html


def test_chord_export_empty_shows_tier_aware_placeholder(plotter: PyCirclizeChordPlotter, tmp_path: Path) -> None:
    outputs = plotter.export(pd.DataFrame(), tmp_path / "chord", ("png",), tier_label="Supported")
    assert outputs["chord.png"].exists()


def test_chord_export_empty_without_tier_label_uses_generic_placeholder(
    plotter: PyCirclizeChordPlotter, tmp_path: Path
) -> None:
    outputs = plotter.export(pd.DataFrame(), tmp_path / "chord", ("png",))
    assert outputs["chord.png"].exists()
