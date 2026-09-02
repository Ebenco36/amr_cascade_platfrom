"""Confounding-strength calibration summaries (VanderWeele--Ding approximation).

These quantities are used only as scale calibrations for the adjusted association.
They are not interpreted as evidence for causal robustness, because the estimand
is selection-conditioned (upstream-observed subset), not a population-average
causal effect. The underlying formula (VanderWeele & Ding 2017) was developed in
a causal-inference context; here it is repurposed as a sensitivity scale for the
minimum unmeasured confounding magnitude that would explain the adjusted OR,
without any causal claim.
"""

from __future__ import annotations

import math

import pandas as pd


def e_value_from_ratio(ratio: float | int | None) -> float:
    """Return the E-value for a risk-ratio-like measure or its reciprocal."""
    if pd.isna(ratio) or ratio is None or ratio <= 0:
        return math.nan
    standardized = float(ratio if ratio >= 1 else 1.0 / ratio)
    if standardized <= 1:
        return 1.0
    return float(standardized + math.sqrt(standardized * (standardized - 1.0)))


def e_value_from_interval(lower: float | int | None, upper: float | int | None) -> float:
    """Return the E-value for the confidence interval limit closest to the null."""
    if pd.isna(lower) or pd.isna(upper) or lower is None or upper is None or lower <= 0 or upper <= 0:
        return math.nan
    if lower <= 1 <= upper:
        return 1.0
    if upper < 1:
        return e_value_from_ratio(1.0 / float(upper))
    return e_value_from_ratio(float(lower))


def e_value_from_odds_ratio(odds_ratio: float | int | None) -> float:
    """Return the VanderWeele-Ding E-value approximation for an odds ratio."""
    if pd.isna(odds_ratio) or odds_ratio is None or odds_ratio <= 0:
        return math.nan
    standardized = float(odds_ratio if odds_ratio >= 1 else 1.0 / odds_ratio)
    if standardized <= 1:
        return 1.0
    return e_value_from_ratio(math.sqrt(standardized))


def e_value_from_odds_ratio_interval(lower: float | int | None, upper: float | int | None) -> float:
    """Return the odds-ratio E-value for the confidence interval limit closest to the null."""
    if pd.isna(lower) or pd.isna(upper) or lower is None or upper is None or lower <= 0 or upper <= 0:
        return math.nan
    if lower <= 1 <= upper:
        return 1.0
    if upper < 1:
        return e_value_from_odds_ratio(1.0 / float(upper))
    return e_value_from_odds_ratio(float(lower))
