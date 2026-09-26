"""The screening and validation rules that cohort-flow figures describe, taken from settings."""

from __future__ import annotations

from dataclasses import dataclass

from amr_cascade_platform.core.config.config_models import Settings


@dataclass(frozen=True)
class CohortRules:
    cotesting_threshold: float
    min_total_support: int
    min_result_support: int
    min_downstream_events: int
    q_threshold: float
    stability_threshold: float

    @classmethod
    def from_settings(cls, settings: Settings) -> CohortRules:
        cascade = settings.cascade
        return cls(
            cotesting_threshold=float(cascade.cotesting_probability_threshold),
            min_total_support=int(cascade.min_total_support),
            min_result_support=int(cascade.min_result_support),
            min_downstream_events=int(cascade.min_total_tested_events_for_retention),
            q_threshold=float(cascade.validation.permutation_p_value_threshold),
            stability_threshold=float(cascade.validation.bootstrap_sign_stability_threshold),
        )
