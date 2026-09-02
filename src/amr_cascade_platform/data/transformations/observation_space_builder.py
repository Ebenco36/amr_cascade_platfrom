"""Build observed culture-drug episode tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class ObservationSpaceBuilder:
    """Build one row per observed culture-drug episode."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None) -> pd.DataFrame:
        frame = cohort.copy()
        if organism_list:
            frame = frame[frame["organism"].isin(organism_list)].copy()
        observed = (
            frame.loc[:, list(self._settings.gold.culture_drug_columns)]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        observed["was_tested"] = 1
        observed["observation_layer"] = "observed_ast"
        return observed
