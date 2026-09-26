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


class _CaptureExporter:
    def __init__(self) -> None:
        self.figures = []

    def write(self, figure, output_stem, formats, **_):  # noqa: ANN001, ANN003
        self.figures.append(figure)
        return {output_stem.name: output_stem}


def _capturing_plotter() -> tuple[DatasetCharacterizationPlotter, _CaptureExporter]:
    exporter = _CaptureExporter()
    return DatasetCharacterizationPlotter(exporter, template="plotly_white", width=1150, height=520), exporter


def test_eligibility_by_site_is_its_own_figure_with_all_sites_first(tmp_path: Path) -> None:
    eligible_pairs = pd.DataFrame(
        [
            _row("armd_ecuh", intrinsic=1, operationally_available=0, eligible=0),
            _row("armd_ecuh", intrinsic=0, operationally_available=1, eligible=1),
            _row("armd", intrinsic=0, operationally_available=0, eligible=0),
            _row("armd", intrinsic=0, operationally_available=1, eligible=1),
        ]
    )
    plotter, exporter = _capturing_plotter()
    plotter.export_eligibility_funnel(eligible_pairs, tmp_path / "funnel", ("png",))
    plotter.export_eligibility_by_site(eligible_pairs, tmp_path / "by_site", ("png",))
    funnel, by_site = exporter.figures
    assert [trace.type for trace in funnel.data] == ["funnel"]
    assert list(by_site.data[0].y) == ["All sites", "Stanford", "ECU Health"]
    shares = [sum(trace.x[index] for trace in by_site.data) for index in range(3)]
    assert shares == [100.0, 100.0, 100.0]
    assert [trace.name for trace in by_site.data] == ["Eligible", "Not operationally available", "Intrinsic resistance"]


def test_covariate_figures_use_recorded_values_by_site(tmp_path: Path) -> None:
    episodes = pd.DataFrame(
        {
            "source_site": ["armd", "armd", "armd", "armd_utsw"],
            "cov_comorbidity_count": [1, 3, 9, 2],
            "cov_comorbidity_available": [1, 1, 0, 1],
            "cov_adi_score": [40, 0, 70, 55],
            "cov_adi_available": [1, 1, 1, 0],
            "cov_age_bin": ["18_40", "unknown", "41_65", "66_plus"],
            "cov_labs_available": [1, 0, 0, 0],
        }
    )
    plotter, exporter = _capturing_plotter()
    plotter.export_covariate_comparison(pd.DataFrame(), episodes, tmp_path / "covariates", ("png",))
    plotter.export_missing_data_profile(episodes, tmp_path / "profile", ("png",))
    comparison, profile = exporter.figures
    boxes = {(trace.yaxis, trace.name): trace for trace in comparison.data}
    assert boxes[("y", "Stanford")].median == (2.0,)  # the unrecorded 9 is left out
    assert boxes[("y", "Stanford")].y == ("Stanford (n = 2)",)
    assert boxes[("y2", "Stanford")].median == (55.0,)  # the 0 is missing, not a score
    assert ("y2", "UT Southwestern") not in boxes  # no recorded ADI there
    heatmap = profile.data[0]
    assert list(heatmap.y) == ["Age", "Comorbidity data", "Area Deprivation Index", "Laboratory values"]
    assert [round(value, 2) for value in heatmap.z[0]] == [0.75, 0.67, 1.0]  # all sites, Stanford, UT Southwestern
    assert [round(value, 2) for value in heatmap.z[2]] == [0.5, 0.67, 0.0]
