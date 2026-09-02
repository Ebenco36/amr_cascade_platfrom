import pandas as pd

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
