from __future__ import annotations

from pathlib import Path

import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.reporting.builders.scientific_audit_builder import ScientificAuditBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _builder() -> ScientificAuditBuilder:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    path_manager = PathManager(PROJECT_ROOT, settings)
    return ScientificAuditBuilder(settings, path_manager)


def _require_output(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Required pipeline output not available: {path}")


def test_combined_pipeline_contracts_pass_for_current_outputs() -> None:
    required = PROJECT_ROOT / "data" / "gold" / "combined" / "culture_episodes.parquet"
    _require_output(required)
    checks = _builder().build_contract_check_table(scope="combined", site=None)
    assert not checks.empty
    failures = checks.loc[checks["status"] != "pass"]
    assert failures.empty, failures.to_string(index=False)


def test_armd_site_pipeline_contracts_pass_for_current_outputs() -> None:
    required = PROJECT_ROOT / "data" / "gold" / "armd" / "culture_episodes.parquet"
    _require_output(required)
    checks = _builder().build_contract_check_table(scope="site", site="armd")
    assert not checks.empty
    failures = checks.loc[checks["status"] != "pass"]
    assert failures.empty, failures.to_string(index=False)


def test_scientific_qa_payload_is_dynamic_and_complete() -> None:
    required = PROJECT_ROOT / "outputs" / "tables" / "combined" / "table_j_downstream_trigger_availability.csv"
    _require_output(required)
    payload = _builder().build_scientific_qa_payload(scope="combined", site=None)
    assert payload["scope"] == "combined"
    assert payload["sampling_context"]["environment"] == "mac_sampled"
    assert payload["downstream_trigger_availability"]["total_downstreams_screened"] >= payload["downstream_trigger_availability"]["estimable_downstreams"]
    assert "best_test_model" in payload["modeling"]
    assert payload["interpretation_boundary"]["primary_analysis"] == "directly observed AST only"
