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
        "directional_support_ratio",
        "directional_episode_overlap",
        "mutual_eligible_pair_support_n",
        "mutual_eligible_reverse_support_n",
        "mutual_eligible_pair_test_rate",
        "mutual_eligible_reverse_test_rate",
        "testing_asymmetry_score_mutual_eligible",
        "mutual_eligible_escalation_ratio",
        "mutual_eligible_reverse_escalation_ratio",
        "directional_asymmetry_score_mutual_eligible",
        "mutual_eligible_directional_support_ratio",
        "mutual_eligible_directional_episode_overlap",
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

        episode_key_columns = list(self._settings.gold.episode_key_columns)
        retained_pair_labels = retained_edges.loc[:, ["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates()
        relevant_labels = pd.concat(
            [
                retained_pair_labels,
                retained_pair_labels.rename(
                    columns={"upstream_antibiotic": "downstream_antibiotic", "downstream_antibiotic": "upstream_antibiotic"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()
        overlap_frame = frame.merge(
            relevant_labels, on=["upstream_antibiotic", "downstream_antibiotic"], how="inner"
        )
        if not overlap_frame.empty:
            overlap_frame = overlap_frame.copy()
            overlap_frame["_episode_key"] = (
                overlap_frame[episode_key_columns].astype(str).agg("||".join, axis=1)
            )
            episode_sets = (
                overlap_frame.groupby(["upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True)["_episode_key"]
                .agg(lambda values: frozenset(values))
                .to_dict()
            )
        else:
            episode_sets = {}

        # Mutual-eligibility sensitivity: DAS/PBI above condition each direction only
        # on the DOWNSTREAM drug's eligibility (frame is already filtered to
        # downstream_eligible==1). This restricts BOTH directions further to a
        # common opportunity universe -- upstream_eligible==1 as well -- without
        # requiring both drugs to be OBSERVED (that would coincide with the
        # co-tested population the panel-bundling screen treats as suspect, not a
        # clean comparison population). upstream_eligible is absent from
        # drug_pairs materialised before PairGenerator started carrying it; this
        # diagnostic degrades to NaN rather than raising when that column is missing.
        if "upstream_eligible" in frame.columns:
            mutual_frame = frame.loc[frame["upstream_eligible"] == 1]
        else:
            mutual_frame = frame.iloc[0:0]
        mutual_frame = mutual_frame.merge(
            relevant_labels, on=["upstream_antibiotic", "downstream_antibiotic"], how="inner"
        )
        mutual_rate_lookup = self._mutual_eligible_lookup(mutual_frame, episode_key_columns)

        panel_keys = episode_key_columns + ["upstream_antibiotic"]
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
            pair_rate = self._observed_rate(pair_tested_n, pair_support_n)
            reverse_rate = self._observed_rate(reverse_tested_n, reverse_support_n)
            rows.append(
                {
                    "upstream_antibiotic": upstream_antibiotic,
                    "downstream_antibiotic": downstream_antibiotic,
                    "smoothed_pair_test_rate": smoothed_pair_rate,
                    "smoothed_reverse_test_rate": smoothed_reverse_rate,
                    "testing_asymmetry_score": self._asymmetry_score(smoothed_pair_rate, smoothed_reverse_rate),
                    "testing_asymmetry_bin": "unassigned",
                    # PBI = min{P(O_k|O_j), P(O_j|O_k)} from the unsmoothed rates on the same
                    # downstream-eligible rows the co-testing screen uses, so a reported PBI is
                    # exactly the quantity the screen compared with its threshold.
                    "panel_bundling_index": self._panel_bundling_index(pair_rate, reverse_rate),
                    "mean_upstream_panel_size": float(panel_size.mean()) if panel_size.notna().any() else math.nan,
                    "median_upstream_panel_size": float(panel_size.median()) if panel_size.notna().any() else math.nan,
                    "panel_size_q90": float(panel_size.quantile(0.9)) if panel_size.notna().any() else math.nan,
                    "bundled_panel_fraction": float(panel_size.ge(2).mean()) if panel_size.notna().any() else math.nan,
                    "max_other_downstream_test_rate": float(other_rates.max()) if not other_rates.empty else math.nan,
                    "mean_other_downstream_test_rate": float(other_rates.mean()) if not other_rates.empty else math.nan,
                    # Quantifies the j->k vs k->j population-size mismatch the DAS/PBI
                    # diagnostics do not otherwise control for (min/max of the two
                    # directions' row counts; 1.0 = identically sized, ->0 = highly
                    # mismatched). Does not restrict DAS/PBI themselves to a mutually
                    # eligible population -- reports how large that gap actually is.
                    "directional_support_ratio": self._directional_support_ratio(
                        pair_support_n.iloc[0] if not pair_support_n.empty else math.nan,
                        reverse_support_n.iloc[0] if not reverse_support_n.empty else math.nan,
                    ),
                    # Jaccard overlap of the *actual episodes* contributing to each
                    # direction -- complements directional_support_ratio, which only
                    # compares denominator sizes and cannot detect two equally sized
                    # but compositionally different episode sets.
                    "directional_episode_overlap": self._episode_jaccard_overlap(
                        episode_sets.get((upstream_antibiotic, downstream_antibiotic), frozenset()),
                        episode_sets.get((downstream_antibiotic, upstream_antibiotic), frozenset()),
                    ),
                    **self._mutual_eligible_diagnostics(
                        mutual_rate_lookup, upstream_antibiotic, downstream_antibiotic, cc
                    ),
                }
            )
        result = pd.DataFrame(rows).reindex(columns=self._OUTPUT_COLUMNS)
        result["testing_asymmetry_bin"] = self._assign_asymmetry_bins(result["testing_asymmetry_score"])
        return result

    @staticmethod
    def _smoothed_rate(tested_n: float, support_n: float, continuity_correction: float) -> float:
        return float((tested_n + continuity_correction) / (support_n + 2.0 * continuity_correction))

    @staticmethod
    def _observed_rate(tested_n: pd.Series, support_n: pd.Series) -> float:
        if tested_n.empty or support_n.empty or not support_n.iloc[0] > 0:
            return math.nan
        return float(tested_n.iloc[0] / support_n.iloc[0])

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
        The co-testing screen removes pairs with PBI >= the configured
        ``cotesting_probability_threshold``
        when both directions have at least min_total_support rows, so a retained pair can
        reach that PBI only when one direction has fewer rows than that.
        """
        if pd.isna(pair_rate) or pd.isna(reverse_rate):
            return math.nan
        return float(min(pair_rate, reverse_rate))

    @staticmethod
    def _directional_support_ratio(pair_support_n: float, reverse_support_n: float) -> float:
        """min/max of the two directions' denominator sizes; 1.0 = identically sized."""
        if pd.isna(pair_support_n) or pd.isna(reverse_support_n) or pair_support_n <= 0 or reverse_support_n <= 0:
            return math.nan
        return float(min(pair_support_n, reverse_support_n) / max(pair_support_n, reverse_support_n))

    @staticmethod
    def _episode_jaccard_overlap(forward_episodes: frozenset, reverse_episodes: frozenset) -> float:
        """Jaccard index of the two directions' contributing episode sets.

        1.0 = identical episode composition, 0.0 = fully disjoint episodes, NaN
        when either direction has no episodes at all. Complements
        directional_support_ratio: two directions can have identical support
        counts while sharing few or no episodes, which a size-only ratio cannot
        detect.
        """
        if not forward_episodes or not reverse_episodes:
            return math.nan
        union_size = len(forward_episodes | reverse_episodes)
        if union_size == 0:
            return math.nan
        return float(len(forward_episodes & reverse_episodes) / union_size)

    def _mutual_eligible_lookup(
        self,
        mutual_frame: pd.DataFrame,
        episode_key_columns: list[str],
    ) -> dict[tuple[object, object], dict[str, object]]:
        """Per ordered pair, the counts behind the mutual-eligibility diagnostics.

        All rows (any observed upstream result) give the co-observation rate, the
        support size and the contributing episodes, as for the primary support ratio
        and episode overlap; the resistant and susceptible upstream rows give the
        escalation ratio, as for the primary DAS.
        """
        if mutual_frame.empty:
            return {}
        frame = mutual_frame.copy()
        positive = self._settings.cascade.upstream_result_positive
        negative = self._settings.cascade.upstream_result_negative
        result = (
            frame["upstream_susceptibility"].astype("string")
            if "upstream_susceptibility" in frame.columns
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
        frame["_resistant"] = result.eq(positive).fillna(False).to_numpy()
        frame["_susceptible"] = result.eq(negative).fillna(False).to_numpy()
        frame["_episode_key"] = frame[episode_key_columns].astype(str).agg("||".join, axis=1)
        lookup: dict[tuple[object, object], dict[str, object]] = {}
        for (upstream, downstream), group in frame.groupby(
            ["upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True
        ):
            tested = group["downstream_tested"].astype(float)
            lookup[(upstream, downstream)] = {
                "tested_n": float(tested.sum()),
                "support_n": float(len(group)),
                "resistant_tested_n": float(tested[group["_resistant"]].sum()),
                "resistant_n": float(group["_resistant"].sum()),
                "susceptible_tested_n": float(tested[group["_susceptible"]].sum()),
                "susceptible_n": float(group["_susceptible"].sum()),
                "episodes": frozenset(group["_episode_key"]),
            }
        return lookup

    @staticmethod
    def _escalation_ratio(counts: dict[str, object] | None, continuity_correction: float) -> float:
        """The primary escalation ratio's formula; undefined unless both upstream branches have rows."""
        if not counts or counts["resistant_n"] <= 0 or counts["susceptible_n"] <= 0:
            return math.nan
        cc = continuity_correction
        resistant = (counts["resistant_tested_n"] + cc) / (counts["resistant_n"] + 2.0 * cc)
        susceptible = (counts["susceptible_tested_n"] + cc) / (counts["susceptible_n"] + 2.0 * cc)
        return float(resistant / susceptible) if susceptible > 0 else math.nan

    def _mutual_eligible_diagnostics(
        self,
        mutual_rate_lookup: dict[tuple[object, object], dict[str, object]],
        upstream_antibiotic: object,
        downstream_antibiotic: object,
        continuity_correction: float,
    ) -> dict[str, float]:
        """DAS recomputed with both directions restricted to a common, mutually

        eligible opportunity universe (both upstream_eligible==1 and
        downstream_eligible==1), rather than each direction conditioning only on
        its own downstream drug's eligibility. It is the primary DAS's formula,
        log ER(j->k) - log ER(k->j), with each ER computed from that universe's
        resistant and susceptible upstream rows. Complements, does not replace,
        the primary directional_asymmetry_score in cascade_report_builder.py:
        it shows how much of the primary DAS could reflect the two directions
        drawing from different opportunity populations. The two directions'
        support can still differ afterward (one direction requires j observed,
        the other k), so the support ratio and episode overlap are recomputed
        within the same universe and reported alongside it. The co-observation
        rate asymmetry of that universe is kept as
        testing_asymmetry_score_mutual_eligible.
        """
        pair = mutual_rate_lookup.get((upstream_antibiotic, downstream_antibiotic))
        reverse = mutual_rate_lookup.get((downstream_antibiotic, upstream_antibiotic))
        pair_rate = (
            self._smoothed_rate(pair["tested_n"], pair["support_n"], continuity_correction) if pair else math.nan
        )
        reverse_rate = (
            self._smoothed_rate(reverse["tested_n"], reverse["support_n"], continuity_correction) if reverse else math.nan
        )
        pair_er = self._escalation_ratio(pair, continuity_correction)
        reverse_er = self._escalation_ratio(reverse, continuity_correction)
        das = (
            float(math.log(pair_er) - math.log(reverse_er))
            if pd.notna(pair_er) and pd.notna(reverse_er) and pair_er > 0 and reverse_er > 0
            else math.nan
        )
        return {
            "mutual_eligible_pair_support_n": pair["support_n"] if pair else math.nan,
            "mutual_eligible_reverse_support_n": reverse["support_n"] if reverse else math.nan,
            "mutual_eligible_pair_test_rate": pair_rate,
            "mutual_eligible_reverse_test_rate": reverse_rate,
            "testing_asymmetry_score_mutual_eligible": self._asymmetry_score(pair_rate, reverse_rate),
            "mutual_eligible_escalation_ratio": pair_er,
            "mutual_eligible_reverse_escalation_ratio": reverse_er,
            "directional_asymmetry_score_mutual_eligible": das,
            "mutual_eligible_directional_support_ratio": self._directional_support_ratio(
                pair["support_n"] if pair else math.nan, reverse["support_n"] if reverse else math.nan
            ),
            "mutual_eligible_directional_episode_overlap": self._episode_jaccard_overlap(
                pair["episodes"] if pair else frozenset(), reverse["episodes"] if reverse else frozenset()
            ),
        }

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
