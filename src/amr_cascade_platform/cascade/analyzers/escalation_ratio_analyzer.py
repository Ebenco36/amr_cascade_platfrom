"""Escalation ratio analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class EscalationRatioAnalyzer:
    """Compute escalation ratios and support thresholds."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, conditional_probabilities: pd.DataFrame) -> pd.DataFrame:
        if conditional_probabilities.empty:
            return pd.DataFrame()

        positive = conditional_probabilities[
            conditional_probabilities["upstream_result_group"] == "positive"
        ].rename(
            columns={
                "support_n": "positive_support_n",
                "downstream_tested_n": "positive_tested_n",
                "conditional_probability": "positive_probability",
            }
        )
        negative = conditional_probabilities[
            conditional_probabilities["upstream_result_group"] == "negative"
        ].rename(
            columns={
                "support_n": "negative_support_n",
                "downstream_tested_n": "negative_tested_n",
                "conditional_probability": "negative_probability",
            }
        )
        merged = positive.merge(
            negative,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="inner",
        )
        merged["total_support_n"] = merged["positive_support_n"] + merged["negative_support_n"]
        correction = float(self._settings.cascade.continuity_correction)
        positive_corrected_probability = (
            (merged["positive_tested_n"] + correction)
            / (merged["positive_support_n"] + 2.0 * correction)
        )
        negative_corrected_probability = (
            (merged["negative_tested_n"] + correction)
            / (merged["negative_support_n"] + 2.0 * correction)
        )
        merged["escalation_ratio"] = np.where(
            negative_corrected_probability > 0,
            positive_corrected_probability / negative_corrected_probability,
            np.nan,
        )
        merged["passes_support_threshold"] = (
            (merged["total_support_n"] >= self._settings.cascade.min_total_support)
            & (merged["positive_support_n"] >= self._settings.cascade.min_result_support)
            & (merged["negative_support_n"] >= self._settings.cascade.min_result_support)
        )
        return merged
