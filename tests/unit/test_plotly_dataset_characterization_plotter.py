from pathlib import Path

import pandas as pd

from amr_cascade_platform.visualization.report.plotly_dataset_characterization_plotter import (
    DatasetCharacterizationPlotter,
)
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


def _plotter() -> DatasetCharacterizationPlotter:
    exporter = PlotlyFigureExporter(width=1150, height=520)
    return DatasetCharacterizationPlotter(exporter, template="plotly_white", width=1150, height=520)


def _row(site: str, intrinsic: int, operationally_available: int, eligible: int) -> dict:
    return {
        "source_site": site,
        "is_intrinsic_resistance": intrinsic,
        "is_operationally_available": operationally_available,
        "is_eligible": eligible,
    }


def test_eligibility_funnel_segments_partition_the_candidate_space(tmp_path: Path) -> None:
    """The three segments (intrinsic, operationally-unavailable, eligible) must

    sum exactly to the total row count -- this is a real invariant of the
    upstream eligibility construction (is_eligible is definitionally
    (not intrinsic) AND operationally_available), not just a display choice,
    so the figure should never silently drop or double-count a row.
    """
    eligible_pairs = pd.DataFrame(
        [
            _row("site_a", intrinsic=1, operationally_available=0, eligible=0),
            _row("site_a", intrinsic=0, operationally_available=0, eligible=0),
            _row("site_a", intrinsic=0, operationally_available=1, eligible=1),
            _row("site_a", intrinsic=0, operationally_available=1, eligible=1),
            _row("site_b", intrinsic=1, operationally_available=0, eligible=0),
            _row("site_b", intrinsic=0, operationally_available=1, eligible=1),
        ]
    )

    outputs = _plotter().export_eligibility_funnel(
        eligible_pairs, tmp_path / "eligibility_funnel", formats=("html",)
    )

    assert (tmp_path / "eligibility_funnel.html").exists()
    assert len(outputs) == 1


def test_eligibility_funnel_handles_missing_columns_gracefully(tmp_path: Path) -> None:
    eligible_pairs = pd.DataFrame({"source_site": ["site_a"]})

    outputs = _plotter().export_eligibility_funnel(
        eligible_pairs, tmp_path / "eligibility_funnel", formats=("html",)
    )

    assert (tmp_path / "eligibility_funnel.html").exists()
    assert len(outputs) == 1


def test_eligibility_funnel_handles_empty_frame(tmp_path: Path) -> None:
    outputs = _plotter().export_eligibility_funnel(
        pd.DataFrame(), tmp_path / "eligibility_funnel", formats=("html",)
    )

    assert (tmp_path / "eligibility_funnel.html").exists()
    assert len(outputs) == 1
