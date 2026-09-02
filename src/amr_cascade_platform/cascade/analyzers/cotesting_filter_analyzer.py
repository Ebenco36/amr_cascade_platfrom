"""Filter symmetric near-deterministic co-testing pairs before cascade estimation."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class CoTestingFilterAnalyzer:
    """Remove symmetric high-probability co-testing pairs that are unlikely to reflect escalation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def filter(self, drug_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if drug_pairs.empty or not self._settings.cascade.filter_symmetric_cotesting:
            return drug_pairs.copy(), pd.DataFrame(
                columns=[
                    "upstream_antibiotic",
                    "downstream_antibiotic",
                    "p_downstream_given_upstream",
                    "p_upstream_given_downstream",
                    "support_n",
                    "reverse_support_n",
                ]
            )

        frame = drug_pairs.copy()
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame.loc[frame["downstream_eligible"] == 1].copy()
        if frame.empty:
            return drug_pairs.copy(), pd.DataFrame()

        pair_probabilities = (
            frame.groupby(["upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True)
            .agg(
                p_downstream_given_upstream=("downstream_tested", "mean"),
                support_n=("downstream_tested", "size"),
            )
            .reset_index()
        )
        reverse = pair_probabilities.rename(
            columns={
                "upstream_antibiotic": "downstream_antibiotic",
                "downstream_antibiotic": "upstream_antibiotic",
                "p_downstream_given_upstream": "p_upstream_given_downstream",
                "support_n": "reverse_support_n",
            }
        )
        flagged = pair_probabilities.merge(
            reverse,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="inner",
        )
        threshold = self._settings.cascade.cotesting_probability_threshold
        min_support = self._settings.cascade.min_total_support
        flagged = flagged.loc[
            flagged["p_downstream_given_upstream"].ge(threshold)
            & flagged["p_upstream_given_downstream"].ge(threshold)
            & flagged["support_n"].ge(min_support)
            & flagged["reverse_support_n"].ge(min_support)
        ].copy()
        if flagged.empty:
            return drug_pairs.copy(), flagged

        filtered = drug_pairs.merge(
            flagged.loc[:, ["upstream_antibiotic", "downstream_antibiotic"]],
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="left",
            indicator=True,
        )
        filtered = filtered.loc[filtered["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)
        # The rename/merge chain above (needed to detect *symmetric* co-testing pairs) merges
        # upstream_antibiotic/downstream_antibiotic against themselves with the column names
        # swapped, so the two sides carry different category sets -- pandas silently falls
        # back to object dtype for a categorical-vs-mismatched-categorical (and later,
        # categorical-vs-object) merge. That downgrade would otherwise persist through every
        # downstream analyzer, since they all groupby/merge on these exact two columns. Restore
        # the original dictionary-encoded dtype here; the values are an exact subset of
        # drug_pairs' own categories, so this is a pure dtype restore, not a data change.
        for column in ("upstream_antibiotic", "downstream_antibiotic"):
            if column in drug_pairs.columns and isinstance(drug_pairs[column].dtype, pd.CategoricalDtype):
                filtered[column] = filtered[column].astype(drug_pairs[column].dtype)
        return filtered, flagged.reset_index(drop=True)
