from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.conditional_probability_analyzer import (
    ConditionalProbabilityAnalyzer,
)
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_conditional_probability_analyzer_groups_positive_and_negative_results() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = ConditionalProbabilityAnalyzer(settings)
    pairs = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_susceptibility": "RESISTANT",
                "downstream_tested": 1,
                "downstream_eligible": 1,
            },
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_susceptibility": "SUSCEPTIBLE",
                "downstream_tested": 0,
                "downstream_eligible": 1,
            },
        ]
    )
    result = analyzer.analyze(pairs)
    assert set(result["upstream_result_group"]) == {"positive", "negative"}
