"""Cascade pair dependence — quantify bundled-testing structure around retained edges."""

from __future__ import annotations

import math

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class CascadePairDependenceAnalyzer:
    """Summarize episode-level downstream bundle structure for retained cascade pairs."""

    _OUTPUT_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "smoothed_pair_test_rate",
        "smoothed_reverse_test_rate",
        "testing_asymmetry_score",
        "testing_asymmetry_bin",
        "panel_bundling_index",
        "mean_upstream_panel_size",
        "median_upstream_panel_size",
        "panel_size_q90",
        "bundled_panel_fraction",
        "max_other_downstream_test_rate",
        "mean_other_downstream_test_rate",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, drug_pairs: pd.DataFrame, retained_edges: pd.DataFrame) -> pd.DataFrame:
        if drug_pairs.empty or retained_edges.empty:
            return pd.DataFrame(columns=self._OUTPUT_COLUMNS)

        frame = drug_pairs.copy()
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame.loc[frame["downstream_eligible"] == 1].copy()
        if frame.empty:
            return pd.DataFrame(columns=self._OUTPUT_COLUMNS)

        panel_keys = list(self._settings.gold.episode_key_columns) + ["upstream_antibiotic"]
        panel_sizes = (
            frame.groupby(panel_keys, dropna=False, observed=True)["downstream_tested"]
            .sum()
            .rename("upstream_panel_size")
            .reset_index()
        )
        companion_rates = (
            frame.groupby(["upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True)["downstream_tested"]
            .agg(pair_test_rate="mean", pair_tested_n="sum", pair_support_n="size")
            .reset_index()
        )
        reverse_rates = companion_rates.rename(
            columns={
                "upstream_antibiotic": "downstream_antibiotic",
                "downstream_antibiotic": "upstream_antibiotic",
                "pair_test_rate": "reverse_pair_test_rate",
                "pair_tested_n": "reverse_pair_tested_n",
                "pair_support_n": "reverse_pair_support_n",
            }
        )

        merged = frame.merge(panel_sizes, on=panel_keys, how="left", validate="many_to_one")
        retained_keys = retained_edges.loc[:, ["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates()
        retained_frame = (
            merged.merge(retained_keys, on=["upstream_antibiotic", "downstream_antibiotic"], how="inner")
            .merge(
                companion_rates,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
                validate="many_to_one",
            )
            .merge(
                reverse_rates,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
                validate="many_to_one",
            )
        )
        if retained_frame.empty:
            return pd.DataFrame(columns=self._OUTPUT_COLUMNS)

        rows: list[dict[str, object]] = []
        cc = float(self._settings.cascade.continuity_correction)
        for (upstream_antibiotic, downstream_antibiotic), group in retained_frame.groupby(
            ["upstream_antibiotic", "downstream_antibiotic"],
            dropna=False,
            observed=True,
        ):
            companion_subset = companion_rates.loc[
                companion_rates["upstream_antibiotic"].eq(upstream_antibiotic)
                & ~companion_rates["downstream_antibiotic"].eq(downstream_antibiotic)
            ]
            other_rates = pd.to_numeric(companion_subset["pair_test_rate"], errors="coerce")
            panel_size = pd.to_numeric(group["upstream_panel_size"], errors="coerce")
            pair_tested_n = pd.to_numeric(group["pair_tested_n"], errors="coerce").dropna()
            pair_support_n = pd.to_numeric(group["pair_support_n"], errors="coerce").dropna()
            reverse_tested_n = pd.to_numeric(group["reverse_pair_tested_n"], errors="coerce").dropna()
            reverse_support_n = pd.to_numeric(group["reverse_pair_support_n"], errors="coerce").dropna()
            smoothed_pair_rate = self._smoothed_rate(pair_tested_n.iloc[0], pair_support_n.iloc[0], cc) if not pair_tested_n.empty and not pair_support_n.empty else math.nan
            smoothed_reverse_rate = self._smoothed_rate(reverse_tested_n.iloc[0], reverse_support_n.iloc[0], cc) if not reverse_tested_n.empty and not reverse_support_n.empty else math.nan
            rows.append(
                {
                    "upstream_antibiotic": upstream_antibiotic,
                    "downstream_antibiotic": downstream_antibiotic,
                    "smoothed_pair_test_rate": smoothed_pair_rate,
                    "smoothed_reverse_test_rate": smoothed_reverse_rate,
                    "testing_asymmetry_score": self._asymmetry_score(smoothed_pair_rate, smoothed_reverse_rate),
                    "testing_asymmetry_bin": "unassigned",
                    # PBI = min{P(O_k|O_j), P(O_j|O_k)} — quantifies panel-bundling intensity.
                    # Pairs near-deterministically bundled in both directions have PBI → 1.
                    # Retained edges are expected to have moderate PBI (co-testing filter removes PBI > 0.95 pairs).
                    "panel_bundling_index": self._panel_bundling_index(smoothed_pair_rate, smoothed_reverse_rate),
                    "mean_upstream_panel_size": float(panel_size.mean()) if panel_size.notna().any() else math.nan,
                    "median_upstream_panel_size": float(panel_size.median()) if panel_size.notna().any() else math.nan,
                    "panel_size_q90": float(panel_size.quantile(0.9)) if panel_size.notna().any() else math.nan,
                    "bundled_panel_fraction": float(panel_size.ge(2).mean()) if panel_size.notna().any() else math.nan,
                    "max_other_downstream_test_rate": float(other_rates.max()) if not other_rates.empty else math.nan,
                    "mean_other_downstream_test_rate": float(other_rates.mean()) if not other_rates.empty else math.nan,
                }
            )
        result = pd.DataFrame(rows).reindex(columns=self._OUTPUT_COLUMNS)
        result["testing_asymmetry_bin"] = self._assign_asymmetry_bins(result["testing_asymmetry_score"])
        return result

    @staticmethod
    def _smoothed_rate(tested_n: float, support_n: float, continuity_correction: float) -> float:
        return float((tested_n + continuity_correction) / (support_n + 2.0 * continuity_correction))

    @staticmethod
    def _asymmetry_score(pair_rate: float, reverse_rate: float) -> float:
        if pd.isna(pair_rate) or pd.isna(reverse_rate) or pair_rate <= 0 or reverse_rate <= 0:
            return math.nan
        return float(abs(math.log(pair_rate) - math.log(reverse_rate)))

    @staticmethod
    def _panel_bundling_index(pair_rate: float, reverse_rate: float) -> float:
        """PBI = min{P(O_k|O_j), P(O_j|O_k)} — co-testing symmetry measure.

        Values near 1 indicate near-deterministic bilateral bundling.
        Values near 0 indicate highly asymmetric testing (one direction drives the other).
        Pairs with PBI > cotesting_probability_threshold (default 0.95) are removed by the co-testing screen;
        retained edges are therefore bounded PBI < 0.95 by construction.
        """
        if pd.isna(pair_rate) or pd.isna(reverse_rate):
            return math.nan
        return float(min(pair_rate, reverse_rate))

    @staticmethod
    def _assign_asymmetry_bins(scores: pd.Series) -> pd.Series:
        numeric_scores = pd.to_numeric(scores, errors="coerce")
        valid = numeric_scores.notna()
        labels = pd.Series(["unassigned"] * len(scores), index=scores.index, dtype=object)
        if int(valid.sum()) < 3:
            labels.loc[valid] = "undifferentiated"
            return labels
        ranked = numeric_scores.loc[valid].rank(method="first")
        try:
            labels.loc[valid] = pd.qcut(
                ranked,
                q=3,
                labels=["low_asymmetry", "medium_asymmetry", "high_asymmetry"],
            ).astype(str)
        except ValueError:
            labels.loc[valid] = "undifferentiated"
        return labels
