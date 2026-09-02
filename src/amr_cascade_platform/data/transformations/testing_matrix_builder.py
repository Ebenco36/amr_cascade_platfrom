"""Build testing-matrix gold tables."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.text import safe_feature_name


class TestingMatrixBuilder:
    """Build a wide testing matrix from observed culture-drug episodes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, culture_drug_episodes: pd.DataFrame) -> pd.DataFrame:
        keys = list(self._settings.gold.episode_key_columns)
        if culture_drug_episodes.empty:
            return pd.DataFrame(columns=keys)

        pivot_source = culture_drug_episodes.loc[:, keys + ["antibiotic", "was_tested"]].copy()
        matrix = (
            pivot_source.pivot_table(
                index=keys,
                columns="antibiotic",
                values="was_tested",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        matrix.columns = [
            column
            if column in keys
            else f"{self._settings.gold.testing_matrix_prefix}{safe_feature_name(str(column))}"
            for column in matrix.columns
        ]
        return matrix
