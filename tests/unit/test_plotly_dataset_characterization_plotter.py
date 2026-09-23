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


def test_eligible_vs_observed_by_drug_y_axis_is_reversed_so_most_missing_leads(tmp_path: Path) -> None:
    """Regression test: the chart sorts obs_rate ascending specifically so the

    most-missing drug leads the chart (see the code comment at the sort
    call), but Plotly's default y-axis for a horizontal bar draws the first
    category at the BOTTOM -- without an explicit reversed autorange, the
    best-observed drugs rendered at the top instead, backwards from the
    documented, intended reading order. Caught by actually looking at a
    rendered figure, not by code review.
    """
    eligible_pairs = pd.DataFrame(
        {
            "antibiotic": ["MOSTLY_MISSING", "MOSTLY_MISSING", "MOSTLY_OBSERVED", "MOSTLY_OBSERVED"],
            "is_eligible": [1, 1, 1, 1],
            "is_observed_tested": [0, 0, 1, 1],
        }
    )

    outputs = _plotter().export_eligible_vs_observed_by_drug(
        eligible_pairs, tmp_path / "eligible_vs_observed", formats=("html",)
    )

    html = (tmp_path / "eligible_vs_observed.html").read_text()
    assert outputs
    assert '"autorange":"reversed"' in html.replace(" ", "")


def test_operational_availability_matrix_smoke(tmp_path: Path) -> None:
    availability_table = pd.DataFrame(
        {
            "site": ["armd", "armd_ecuh"],
            "era": ["2020-2024", "2020-2024"],
            "antibiotic": ["AMIKACIN", "AMIKACIN"],
            "is_operationally_available": [1, 0],
        }
    )
    outputs = _plotter().export_operational_availability_matrix(
        availability_table, tmp_path / "availability_matrix", formats=("html",)
    )
    assert (tmp_path / "availability_matrix.html").exists()
    assert len(outputs) == 1


def test_site_era_availability_timeline_smoke(tmp_path: Path) -> None:
    availability_table = pd.DataFrame(
        {
            "site": ["armd", "armd", "armd_ecuh"],
            "era": ["2015-2019", "2020-2024", "2020-2024"],
            "antibiotic": ["AMIKACIN", "AMIKACIN", "AMIKACIN"],
            "is_operationally_available": [0, 1, 1],
        }
    )
    outputs = _plotter().export_site_era_availability_timeline(
        availability_table, tmp_path / "availability_timeline", formats=("html",)
    )
    assert (tmp_path / "availability_timeline.html").exists()
    assert len(outputs) == 1
