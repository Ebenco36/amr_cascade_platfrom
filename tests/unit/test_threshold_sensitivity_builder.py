from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.threshold_sensitivity_builder import ThresholdSensitivityBuilder
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_threshold_sensitivity_builder_exports_grid(tmp_path: Path) -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    escalation = pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
                "positive_support_n": [10],
                "negative_support_n": [20],
                "positive_tested_n": [8],
                "negative_tested_n": [12],
                "total_support_n": [30],
                "escalation_ratio": [2.0],
            "passes_support_threshold": [True],
        }
    )
    adjusted = pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
            "adjusted_log_odds": [0.5],
            "adjusted_odds_ratio": [1.6],
            "modeled_n": [30],
        }
    )
    outputs = ThresholdSensitivityBuilder(settings).export(escalation, adjusted, tmp_path)
    table = pd.read_parquet(outputs["threshold_sensitivity_path"])
    assert len(table) == (
        len(settings.cascade.sensitivity_min_total_supports)
        * len(settings.cascade.sensitivity_min_result_supports)
        * len(settings.cascade.sensitivity_retained_min_escalation_ratios)
    )
