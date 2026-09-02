from pathlib import Path

from amr_cascade_platform.core.utils.pipeline_preflight import (
    PipelinePreflightChecker,
    PipelinePreflightRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_preflight_flags_invalid_estimator_and_format() -> None:
    checker = PipelinePreflightChecker(PROJECT_ROOT, "mac")
    request = PipelinePreflightRequest(
        environment="mac",
        sites=("armd",),
        organisms=("ESCHERICHIA COLI",),
        estimators=("not_a_model",),
        report_formats=("html", "badfmt"),
        run_sampling=False,
        run_ingestion=False,
        run_preprocessing=False,
        run_harmonization=False,
        run_gold=False,
        run_cascade=False,
        run_site_cascade=False,
        run_sensitivity=False,
        run_features=False,
        run_training=False,
        run_reporting=False,
    )
    result = checker.run(request)
    assert not result.ok
    assert any("Unsupported estimators" in error for error in result.errors)
    assert any("Unsupported report formats" in error for error in result.errors)


def test_preflight_warns_when_sensitivity_requested_without_armd() -> None:
    checker = PipelinePreflightChecker(PROJECT_ROOT, "mac")
    request = PipelinePreflightRequest(
        environment="mac",
        sites=("armd_ecuh",),
        organisms=(),
        report_formats=("html",),
        run_sampling=False,
        run_organism_primary=False,
        run_pooled_sensitivity=True,
        run_ingestion=False,
        run_preprocessing=False,
        run_harmonization=False,
        run_gold=False,
        run_cascade=False,
        run_site_cascade=False,
        run_sensitivity=True,
        run_features=False,
        run_training=False,
        run_reporting=False,
    )
    result = checker.run(request)
    assert any("Sensitivity run requested" in warning for warning in result.warnings)
