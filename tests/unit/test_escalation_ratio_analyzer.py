from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.escalation_ratio_analyzer import EscalationRatioAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_escalation_ratio_analyzer_applies_thresholds() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = EscalationRatioAnalyzer(settings)
    conditional = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_result_group": "positive",
                "support_n": 10,
                "downstream_tested_n": 8,
                "conditional_probability": 0.8,
            },
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_result_group": "negative",
                "support_n": 15,
                "downstream_tested_n": 3,
                "conditional_probability": 0.2,
            },
        ]
    )
    result = analyzer.analyze(conditional)
    assert len(result) == 1
    assert abs(float(result.iloc[0]["escalation_ratio"]) - 3.5324675325) < 1e-9
    assert bool(result.iloc[0]["passes_support_threshold"]) is True


def test_escalation_ratio_analyzer_applies_continuity_correction_when_negative_branch_is_zero() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = EscalationRatioAnalyzer(settings)
    conditional = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_result_group": "positive",
                "support_n": 10,
                "downstream_tested_n": 8,
                "conditional_probability": 0.8,
            },
            {
                "upstream_antibiotic": "CIPRO",
                "downstream_antibiotic": "MEROPENEM",
                "upstream_result_group": "negative",
                "support_n": 15,
                "downstream_tested_n": 0,
                "conditional_probability": 0.0,
            },
        ]
    )
    result = analyzer.analyze(conditional)
    assert len(result) == 1
    assert pd.notna(result.iloc[0]["escalation_ratio"])
    assert float(result.iloc[0]["escalation_ratio"]) > 1.0
