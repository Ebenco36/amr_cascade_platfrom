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
