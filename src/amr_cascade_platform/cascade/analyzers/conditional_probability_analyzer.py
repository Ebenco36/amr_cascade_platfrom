"""Conditional downstream testing analysis."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class ConditionalProbabilityAnalyzer:
    """Estimate downstream testing probabilities by upstream result state."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, drug_pairs: pd.DataFrame) -> pd.DataFrame:
        # Organism-scoping guard: cascade grouping keys do not include organism, so
        # mixing organisms would silently pool biologically distinct drug-eligibility
        # structures.  Raise early if the caller accidentally passes multi-organism data.
        if "organism" in drug_pairs.columns:
            unique_organisms = drug_pairs["organism"].dropna().unique()
            if len(unique_organisms) > 1:
                raise ValueError(
                    f"ConditionalProbabilityAnalyzer received data for "
                    f"{len(unique_organisms)} distinct organisms "
                    f"({sorted(unique_organisms)}).  Cascade estimation must be run "
                    f"per-organism; pass a single-organism subset to this method."
                )

        analysis_frame = self._prepare(drug_pairs)
        if analysis_frame.empty:
            return pd.DataFrame()

        grouped = (
            analysis_frame.groupby(
                ["upstream_antibiotic", "downstream_antibiotic", "upstream_result_group"],
                dropna=False,
                observed=True,
            )
            .agg(
                support_n=("downstream_tested", "size"),
                downstream_tested_n=("downstream_tested", "sum"),
            )
            .reset_index()
        )
        grouped["conditional_probability"] = grouped["downstream_tested_n"] / grouped["support_n"]
        return grouped

    def _prepare(self, drug_pairs: pd.DataFrame) -> pd.DataFrame:
        frame = drug_pairs.copy()
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame[frame["downstream_eligible"] == 1]
        frame["upstream_result_group"] = frame["upstream_susceptibility"].map(self._map_result_group)
        return frame[frame["upstream_result_group"].isin(["positive", "negative"])]

    def _map_result_group(self, value: object) -> str | None:
        if pd.isna(value):
            return None
        if value == self._settings.cascade.upstream_result_positive:
            return "positive"
        if value == self._settings.cascade.upstream_result_negative:
            return "negative"
        return None
