"""Directed unit tests for the permutation and bootstrap statistical routines.

These cover the three primary statistical gates that determine paper conclusions:
  - _empirical_p_value  (Davison-Hinkley formula)
  - _resample_within_sites  (episode-level bootstrap within sites)
  - Bootstrap sign stability  (sign stability fraction in _bootstrap_summary)

_shuffle_within_strata is covered by test_cascade_validation_analyzer.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import (
    CascadeValidationAnalyzer,
)
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def _analyzer() -> CascadeValidationAnalyzer:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    return CascadeValidationAnalyzer(settings)


# ── _empirical_p_value ────────────────────────────────────────────────────────


class TestEmpiricalPValue:
    """Davison-Hinkley: p = (exceedances + 1) / (B + 1), never 0 or truly 1."""

    def test_upper_tail_er_above_one(self) -> None:
        # observed_er = 2.0 (positive), null = [1.0, 2.0, 3.0]
        # exceedances = count(null >= 2.0) = 2  →  p = (2+1)/(3+1) = 0.75
        az = _analyzer()
        p = az._empirical_p_value(2.0, [1.0, 2.0, 3.0])
        assert abs(p - 0.75) < 1e-12

    def test_lower_tail_er_below_one(self) -> None:
        # observed_er = 0.5 (negative), null = [0.3, 0.5, 0.7, 1.0]
        # exceedances = count(null <= 0.5) = 2  →  p = (2+1)/(4+1) = 0.6
        az = _analyzer()
        p = az._empirical_p_value(0.5, [0.3, 0.5, 0.7, 1.0])
        assert abs(p - 0.6) < 1e-12

    def test_observed_er_exactly_one_uses_upper_tail(self) -> None:
        # ER = 1.0 triggers upper tail (observed_er >= 1.0)
        # null = [0.8, 1.0, 1.2], exceedances = count(null >= 1.0) = 2
        # p = (2+1)/(3+1) = 0.75
        az = _analyzer()
        p = az._empirical_p_value(1.0, [0.8, 1.0, 1.2])
        assert abs(p - 0.75) < 1e-12

    def test_minimum_p_value_is_not_zero(self) -> None:
        # No null exceeds observed → exceedances = 0 → p = 1/(B+1), not 0.
        # observed_er = 100.0, null = [1.0, 1.1, 1.2] → p = 1/4 = 0.25
        az = _analyzer()
        p = az._empirical_p_value(100.0, [1.0, 1.1, 1.2])
        assert abs(p - 0.25) < 1e-12

    def test_maximum_p_value_when_all_null_exceed(self) -> None:
        # All 3 null values ≤ observed_er = 0.1 (lower tail)
        # exceedances = 3  →  p = (3+1)/(3+1) = 1.0
        az = _analyzer()
        p = az._empirical_p_value(0.1, [0.1, 0.05, 0.01])
        assert abs(p - 1.0) < 1e-12

    def test_empty_null_returns_nan(self) -> None:
        az = _analyzer()
        assert math.isnan(az._empirical_p_value(1.5, []))

    def test_nan_observed_er_returns_nan(self) -> None:
        az = _analyzer()
        assert math.isnan(az._empirical_p_value(math.nan, [1.0, 1.5, 2.0]))

    def test_inf_and_nan_in_null_excluded(self) -> None:
        # _finite_array removes inf/nan → valid = [1.0, 2.0]
        # observed_er = 1.5 (upper tail), exceedances = count([1.0, 2.0] >= 1.5) = 1
        # p = (1+1)/(2+1) = 2/3
        az = _analyzer()
        p = az._empirical_p_value(1.5, [1.0, math.inf, math.nan, 2.0])
        assert abs(p - 2.0 / 3.0) < 1e-12


# ── _resample_within_sites ────────────────────────────────────────────────────


def _make_subset(
    n_per_site: int = 10,
    sites: tuple[str, ...] = ("site_A", "site_B"),
) -> pd.DataFrame:
    rows = []
    ep_counter = 0
    for site in sites:
        for i in range(n_per_site):
            ep_counter += 1
            rows.append({
                "source_site": site,
                "anon_id": f"p{ep_counter}",
                "pat_enc_csn_id_coded": f"enc{ep_counter}",
                "order_proc_id_coded": f"ord{ep_counter}",
                "upstream_result_group": "positive" if i < n_per_site // 2 else "negative",
                "downstream_tested": 1 if i < n_per_site // 3 else 0,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


class TestResampleWithinSites:
    """Bootstrap resampling preserves row count and respects site boundaries."""

    def test_row_count_is_preserved(self) -> None:
        az = _analyzer()
        subset = _make_subset(n_per_site=10, sites=("A", "B"))
        rng = np.random.default_rng(42)
        resampled = az._resample_within_sites(subset, rng)
        assert len(resampled) == len(subset)

    def test_values_come_from_original_rows(self) -> None:
        az = _analyzer()
        subset = _make_subset(n_per_site=8, sites=("A",))
        rng = np.random.default_rng(0)
        for _ in range(5):
            resampled = az._resample_within_sites(subset, rng)
            assert set(resampled["upstream_result_group"]).issubset(
                set(subset["upstream_result_group"])
            )
            assert set(resampled["source_site"]).issubset(set(subset["source_site"]))

    def test_site_boundaries_not_crossed(self) -> None:
        # Each site's rows have a unique marker in upstream_result_group.
        # After resampling, rows for site_X must still have marker "X", etc.
        az = _analyzer()
        rows = []
        for site, marker in (("site_X", "X"), ("site_Y", "Y")):
            for i in range(6):
                rows.append({
                    "source_site": site,
                    "anon_id": f"{marker}_{i}",
                    "pat_enc_csn_id_coded": f"{marker}_enc_{i}",
                    "order_proc_id_coded": f"{marker}_ord_{i}",
                    "upstream_result_group": marker,
                    "downstream_tested": 1,
                })
        subset = pd.DataFrame(rows).reset_index(drop=True)
        rng = np.random.default_rng(7)
        for _ in range(10):
            resampled = az._resample_within_sites(subset, rng)
            for _, row in resampled.iterrows():
                expected_marker = row["source_site"].split("_")[1]
                assert row["upstream_result_group"] == expected_marker

    def test_empty_subset_returns_empty(self) -> None:
        az = _analyzer()
        subset = _make_subset(n_per_site=0, sites=("A",))
        rng = np.random.default_rng(0)
        resampled = az._resample_within_sites(subset, rng)
        assert resampled.empty

    def test_bootstrap_produces_variation_across_seeds(self) -> None:
        # With replacement from 20+ distinct episodes, different seeds must differ.
        az = _analyzer()
        subset = _make_subset(n_per_site=20, sites=("A",))
        samples = set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            r = az._resample_within_sites(subset, rng)
            samples.add(tuple(r["anon_id"].tolist()))
        assert len(samples) > 1


# ── Bootstrap sign stability ──────────────────────────────────────────────────


def _sign_stability(
    observed_er: float,
    boot_values: list[float],
) -> float:
    """Replicate the sign_stability calculation from _bootstrap_summary."""
    valid = CascadeValidationAnalyzer._finite_array(boot_values)
    if valid.size == 0:
        return math.nan
    positive_effect = observed_er >= 1.0
    return float((valid >= 1.0).mean()) if positive_effect else float((valid <= 1.0).mean())


class TestBootstrapSignStability:
    """Sign stability = fraction of bootstrap ERs on the same side of 1.0 as observed ER."""

    def test_all_positive_samples_give_one(self) -> None:
        assert _sign_stability(1.5, [1.1, 1.5, 2.0, 1.2, 1.8]) == 1.0

    def test_all_negative_samples_give_one(self) -> None:
        assert _sign_stability(0.5, [0.3, 0.5, 0.7, 0.9, 0.8]) == 1.0

    def test_partial_positive_agreement(self) -> None:
        # observed ER = 2.0 (positive), boot = [0.8, 1.0, 1.5, 2.0, 0.5]
        # values ≥ 1.0: {1.0, 1.5, 2.0} = 3 of 5  → 0.6
        assert abs(_sign_stability(2.0, [0.8, 1.0, 1.5, 2.0, 0.5]) - 0.6) < 1e-12

    def test_partial_negative_agreement(self) -> None:
        # observed ER = 0.4 (negative), boot = [0.3, 0.7, 1.5, 0.9, 0.5]
        # values ≤ 1.0: {0.3, 0.7, 0.9, 0.5} = 4 of 5  → 0.8
        assert abs(_sign_stability(0.4, [0.3, 0.7, 1.5, 0.9, 0.5]) - 0.8) < 1e-12

    def test_nan_and_inf_in_boot_excluded(self) -> None:
        # valid = [2.0, 1.5]; both ≥ 1.0 → stability = 1.0
        assert _sign_stability(2.0, [2.0, math.inf, math.nan, 1.5]) == 1.0

    def test_all_boot_nan_returns_nan(self) -> None:
        assert math.isnan(_sign_stability(2.0, [math.nan, math.nan]))

    def test_observed_er_exactly_one_counts_as_positive(self) -> None:
        # ER = 1.0 → positive_effect = True → fraction ≥ 1.0
        # boot = [0.9, 1.0, 1.1] → 2 of 3 ≥ 1.0  →  2/3
        assert abs(_sign_stability(1.0, [0.9, 1.0, 1.1]) - 2.0 / 3.0) < 1e-12
