from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_missing_adjusted_or_is_unknown_not_positive() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = RetainedEdgeAnalyzer(settings)

    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.0,
                "total_support_n": 30,
                "positive_support_n": 10,
                "negative_support_n": 20,
                "positive_tested_n": 8,
                "negative_tested_n": 4,
            }
        ]
    )
    adjusted_results = pd.DataFrame(
        columns=["upstream_antibiotic", "downstream_antibiotic", "adjusted_odds_ratio"]
    )

    retained = analyzer.analyze(escalation_results, adjusted_results)

    assert len(retained) == 1
    assert pd.isna(retained.loc[0, "adjusted_or_positive"])


def test_available_adjusted_or_positive_flag_is_boolean() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = RetainedEdgeAnalyzer(settings)

    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.0,
                "total_support_n": 30,
                "positive_support_n": 10,
                "negative_support_n": 20,
                "positive_tested_n": 8,
                "negative_tested_n": 4,
            }
        ]
    )
    adjusted_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "MEROPENEM",
                "adjusted_odds_ratio": 0.8,
            }
        ]
    )

    retained = analyzer.analyze(escalation_results, adjusted_results)

    assert len(retained) == 1
    assert bool(retained.loc[0, "adjusted_or_positive"]) is False
