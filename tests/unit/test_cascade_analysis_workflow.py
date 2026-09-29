from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import CascadeAnalysisWorkflow


def test_validated_edge_results_filters_to_robust_and_supported_edges() -> None:
    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "passes_support_threshold": True,
                "escalation_ratio": 2.0,
            },
            {
                "upstream_antibiotic": "C",
                "downstream_antibiotic": "D",
                "passes_support_threshold": True,
                "escalation_ratio": 3.0,
            },
            {
                "upstream_antibiotic": "E",
                "downstream_antibiotic": "F",
                "passes_support_threshold": True,
                "escalation_ratio": 4.0,
            },
        ]
    )
    validation_results = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "validation_status": "robust"},
            {"upstream_antibiotic": "C", "downstream_antibiotic": "D", "validation_status": "supported"},
            {"upstream_antibiotic": "E", "downstream_antibiotic": "F", "validation_status": "mixed"},
        ]
    )

    filtered = CascadeAnalysisWorkflow._validated_edge_results(
        escalation_results,
        validation_results,
    )

    assert set(zip(filtered["upstream_antibiotic"], filtered["downstream_antibiotic"], strict=False)) == {
        ("A", "B"),
        ("C", "D"),
    }


def _retained_edges(*pairs: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"upstream_antibiotic": u, "downstream_antibiotic": d} for u, d in pairs]
    )


def _current_schema_validation_results(*pairs: tuple[str, str]) -> pd.DataFrame:
    """A validation_results frame carrying every column the current merge_shards() writes."""
    rows = []
    for u, d in pairs:
        row = {"upstream_antibiotic": u, "downstream_antibiotic": d, "validation_status": "robust"}
        for column in CascadeAnalysisWorkflow._EXPECTED_VALIDATION_COLUMNS:
            row.setdefault(column, 0.01)
        rows.append(row)
    return pd.DataFrame(rows)


def test_precomputed_validation_check_passes_for_current_schema_and_matching_keys() -> None:
    retained_edges = _retained_edges(("A", "B"), ("C", "D"))
    validation_results = _current_schema_validation_results(("A", "B"), ("C", "D"))

    CascadeAnalysisWorkflow._validate_precomputed_validation_matches_retained_edges(
        validation_results=validation_results,
        retained_edges=retained_edges,
        validation_path=Path("validation_results.parquet"),
    )


def test_precomputed_validation_check_rejects_file_missing_current_columns() -> None:
    """A validation_results.parquet computed by an older code version has the

    RIGHT set of edges (so the key-correspondence check alone would pass it
    clean) but is missing columns the current pipeline produces -- e.g. the
    two-sided permutation/FDR columns. This is the exact staleness the
    key-only check could not catch: same pairs, stale statistics.
    """
    retained_edges = _retained_edges(("A", "B"))
    stale_validation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "validation_status": "robust",
                # Deliberately missing permutation_p_value_two_sided and every
                # other column in _EXPECTED_VALIDATION_COLUMNS.
            }
        ]
    )

    with pytest.raises(ValueError, match="older code version"):
        CascadeAnalysisWorkflow._validate_precomputed_validation_matches_retained_edges(
            validation_results=stale_validation_results,
            retained_edges=retained_edges,
            validation_path=Path("validation_results.parquet"),
        )


def test_precomputed_validation_check_still_rejects_mismatched_keys() -> None:
    """Existing key-correspondence behavior must be unchanged by the new schema check."""
    retained_edges = _retained_edges(("A", "B"), ("C", "D"))
    validation_results = _current_schema_validation_results(("A", "B"))  # missing C->D

    with pytest.raises(ValueError, match="do not match the current retained-edge set"):
        CascadeAnalysisWorkflow._validate_precomputed_validation_matches_retained_edges(
            validation_results=validation_results,
            retained_edges=retained_edges,
            validation_path=Path("validation_results.parquet"),
        )


def _mandatory_outputs(directory: Path) -> dict[str, Path]:
    outputs = {}
    for name in CascadeAnalysisWorkflow._MANDATORY_OUTPUTS:
        path = directory / f"{name}.parquet"
        path.write_bytes(b"x")
        outputs[name] = path
    return outputs


def test_precomputed_validation_without_provenance_is_refused_unless_allowed(tmp_path: Path) -> None:
    path = tmp_path / "validation_results.parquet"
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="no provenance record"):
        CascadeAnalysisWorkflow._check_validation_provenance(path, "abc", allow_unrecorded=False)
    CascadeAnalysisWorkflow._check_validation_provenance(path, "abc", allow_unrecorded=True)


def test_precomputed_validation_provenance_must_match_the_current_fingerprint(tmp_path: Path) -> None:
    from amr_cascade_platform.core.utils.provenance import atomic_write_json, sidecar_path

    path = tmp_path / "validation_results.parquet"
    path.write_bytes(b"x")
    atomic_write_json({"fingerprint": "abc"}, sidecar_path(path))

    CascadeAnalysisWorkflow._check_validation_provenance(path, "abc", allow_unrecorded=False)
    with pytest.raises(ValueError, match="different gold data"):
        CascadeAnalysisWorkflow._check_validation_provenance(path, "xyz", allow_unrecorded=True)


def test_success_inventory_covers_nested_outputs_and_the_provenance_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "cascade" / "combined"
    (output_dir / "nested").mkdir(parents=True)
    outputs = _mandatory_outputs(output_dir)
    outputs["nested_table"] = output_dir / "nested" / "table.parquet"
    outputs["cross_scope_comparison"] = tmp_path / "cascade" / "comparison.parquet"
    outputs["row_count"] = 3

    inventory = CascadeAnalysisWorkflow._success_inventory(outputs, output_dir)

    assert inventory["nested_table"] == output_dir / "nested" / "table.parquet"
    assert "cross_scope_comparison" not in inventory
    assert "row_count" not in inventory
    assert inventory["validation_results_provenance"] == output_dir / "validation_results.provenance.json"


def test_success_inventory_refuses_a_run_without_its_mandatory_outputs(tmp_path: Path) -> None:
    outputs = _mandatory_outputs(tmp_path)
    del outputs["adjusted_model_shap_domain"]

    with pytest.raises(RuntimeError, match="adjusted_model_shap_domain"):
        CascadeAnalysisWorkflow._success_inventory(outputs, tmp_path)
