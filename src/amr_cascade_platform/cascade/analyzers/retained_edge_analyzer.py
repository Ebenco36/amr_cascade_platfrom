"""Retain empirical cascade edges for reporting and network export."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class RetainedEdgeAnalyzer:
    """Filter cascade edges using raw-evidence thresholds only.

    Retention is intentionally based on the raw escalation ratio and support
    counts, not the adjusted odds ratio.  The adjusted OR answers RQ2: whether
    retained raw-cascade edges remain positive after covariate adjustment.
    Using the adjusted OR as a *retention gate* would make RQ2 partly circular —
    edges where adjustment reverses direction would be excluded before validation
    ever runs.  Instead, ``adjusted_or_positive`` is attached as a post-retention
    annotation so callers can inspect direction agreement without confounding the
    retention decision.

    Scope note: when ``validate_suppression_cascades`` is true, both
    escalation-direction edges (ER >= 1) and suppression-direction edges
    (ER < 1) can reach validation if they have sufficient raw support. When it
    is false, retention is limited to edges with
    ``escalation_ratio >= retained_min_escalation_ratio``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(
        self,
        escalation_results: pd.DataFrame,
        adjusted_results: pd.DataFrame,
        min_total_support: int | None = None,
        min_result_support: int | None = None,
        retained_min_escalation_ratio: float | None = None,
        adjusted_or_positive_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Return retained edges.

        Parameters
        ----------
        adjusted_or_positive_threshold:
            Threshold used only for the ``adjusted_or_positive`` annotation
            flag. It does not gate retention.
        """
        if escalation_results.empty:
            return pd.DataFrame()

        merged = escalation_results.copy()
        if not adjusted_results.empty:
            merged = merged.merge(
                adjusted_results,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
            )
        else:
            merged["adjusted_log_odds"] = np.nan
            merged["adjusted_odds_ratio"] = np.nan
            merged["modeled_n"] = np.nan

        total_support_threshold = (
            min_total_support if min_total_support is not None else self._settings.cascade.min_total_support
        )
        result_support_threshold = (
            min_result_support if min_result_support is not None else self._settings.cascade.min_result_support
        )
        escalation_threshold = (
            retained_min_escalation_ratio
            if retained_min_escalation_ratio is not None
            else self._settings.cascade.retained_min_escalation_ratio
        )
        adjusted_threshold = (
            adjusted_or_positive_threshold
            if adjusted_or_positive_threshold is not None
            else self._settings.cascade.adjusted_or_positive_threshold
        )

        merged["passes_support_threshold"] = (
            (merged["total_support_n"] >= total_support_threshold)
            & (merged["positive_support_n"] >= result_support_threshold)
            & (merged["negative_support_n"] >= result_support_threshold)
        )
        merged["has_informative_downstream_testing"] = (
            merged["positive_tested_n"].fillna(0) + merged["negative_tested_n"].fillna(0)
        ) >= self._settings.cascade.min_total_tested_events_for_retention

        # Mark cascade direction before applying the retention gate.
        merged["cascade_direction"] = np.where(
            merged["escalation_ratio"].isna(),
            None,
            np.where(merged["escalation_ratio"] >= 1.0, "escalation", "suppression"),
        )

        # Retention gate: raw evidence only (support + raw ER).
        # Adjusted OR is NOT used here — see class docstring.
        # When validate_suppression_cascades is enabled both escalation (ER ≥ threshold)
        # and suppression (ER < 1.0) edges are retained so the validation layer can
        # assess them symmetrically.
        if self._settings.cascade.validate_suppression_cascades:
            merged["is_retained_edge"] = (
                merged["passes_support_threshold"]
                & merged["has_informative_downstream_testing"]
                & merged["escalation_ratio"].notna()
            )
        else:
            merged["is_retained_edge"] = (
                merged["passes_support_threshold"]
                & merged["has_informative_downstream_testing"]
                & (merged["escalation_ratio"] >= escalation_threshold)
            )

        # Post-retention annotation: flag whether the adjusted OR is positive.
        # Used for RQ2 reporting; does not affect which edges are retained.
        # Missing adjusted models are deliberately left as <NA>, not counted as
        # positive. Treating non-estimable adjusted ORs as positive would overstate
        # RQ2 direction agreement.
        merged["adjusted_or_positive"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
        adjusted_mask = merged["adjusted_odds_ratio"].notna()
        merged.loc[adjusted_mask, "adjusted_or_positive"] = (
            merged.loc[adjusted_mask, "adjusted_odds_ratio"] >= adjusted_threshold
        ).astype("boolean")

        return merged[merged["is_retained_edge"]].reset_index(drop=True)
