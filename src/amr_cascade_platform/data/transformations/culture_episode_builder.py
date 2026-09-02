"""Build culture episode gold tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class CultureEpisodeBuilder:
    """Build one row per organism-level culture episode."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, cohort: pd.DataFrame, organism_list: tuple[str, ...] | None = None) -> pd.DataFrame:
        frame = cohort.copy()
        if organism_list:
            frame = frame[frame["organism"].isin(organism_list)].copy()
        return (
            frame.loc[:, list(self._settings.gold.culture_episode_columns)]
            .drop_duplicates()
            .reset_index(drop=True)
        )
