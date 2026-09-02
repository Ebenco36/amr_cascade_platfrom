"""History feature builder for prior antibiotics and prior organisms."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.text import normalize_label


class HistoryFeatureBuilder:
    """Build episode-level history features with explicit recency windows."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        culture_episodes: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        for site in self._settings.platform.sites:
            site_base = self._site_episode_index(culture_episodes, site)
            site_episode_context = self._site_episode_context(culture_episodes, site)
            frame = site_base.copy()
            frame = frame.merge(
                self._aggregate_antibiotic_history(site, site_base, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_prior_organisms(site, site_base, site_episode_context, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_prior_procedures(site, site_base, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_antibiotic_subtype_history(site, site_base, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_nursing_home_visits(site, site_base, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            feature_columns = [column for column in frame.columns if column not in join_keys]
            for column in feature_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        for column in [column for column in combined.columns if column not in join_keys]:
            if column.endswith("_available") or column.endswith("_count") or column.endswith("_unique_count") or column.endswith("d") or column.endswith("365d"):
                combined[column] = combined[column].astype(int)
        return combined

    def _aggregate_antibiotic_history(
        self,
        site: str,
        site_base: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        exposures = load_site_table(site, "antibiotic_class_exposure")
        if exposures.empty:
            return self._empty_abx(site_base)
        id_keys = list(self._settings.platform.id_columns)
        exposures = exposures.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if exposures.empty:
            return self._empty_abx(site_base)

        days_column = "time_to_culturetime"
        table = exposures.loc[:, join_keys + ["antibiotic_class", days_column]].copy()
        table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
        # The exposure tables encode days before culture. Restricting to non-negative values
        # prevents any post-culture medication rows from leaking into history features.
        table = table.loc[table[days_column].ge(0).fillna(False)].copy()
        if table.empty:
            return self._empty_abx(site_base)
        grouped = table.groupby(join_keys, dropna=False)
        result = grouped.agg(
            history_abx_available=(days_column, lambda series: 1),
            history_abx_exposure_count=("antibiotic_class", "size"),
            history_abx_unique_class_count=("antibiotic_class", "nunique"),
            history_abx_min_days=(days_column, "min"),
        ).reset_index()
        for window in self._settings.feature_build.history_windows_days:
            window_frame = grouped[days_column].apply(
                lambda series, window=window: int(series.between(0, window, inclusive="both").fillna(False).any())
            )
            result[f"history_abx_any_{window}d"] = window_frame.to_numpy()
        return site_base.merge(result, on=join_keys, how="left", validate="one_to_one").loc[:, result.columns]

    def _aggregate_prior_organisms(
        self,
        site: str,
        site_base: pd.DataFrame,
        site_episode_context: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        priors = load_site_table(site, "prior_infecting_organism")
        if priors.empty:
            return self._empty_prior(site_base)
        id_keys = list(self._settings.platform.id_columns)
        priors = priors.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if priors.empty:
            return self._empty_prior(site_base)

        days_column = (
            "prior_infecting_organism_days_to_culture"
            if "prior_infecting_organism_days_to_culture" in priors.columns
            else "prior_infecting_organism_days_to_culutre"
        )
        table = priors.loc[:, join_keys + ["prior_organism", days_column]].copy()
        table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
        # Prior organism history must be strictly pre-culture. Negative lags would mean the
        # "history" event actually happened after the index culture and would leak the future.
        table = table.loc[table[days_column].ge(0).fillna(False)].copy()
        if table.empty:
            return self._empty_prior(site_base)
        table = table.merge(
            site_episode_context.loc[:, join_keys + ["organism"]],
            on=join_keys,
            how="left",
            validate="many_to_one",
        )
        table["prior_organism_normalized"] = table["prior_organism"].map(normalize_label)
        table["episode_organism_normalized"] = table["organism"].map(normalize_label)
        table["same_organism"] = (
            table["prior_organism_normalized"].notna()
            & table["episode_organism_normalized"].notna()
            & (table["prior_organism_normalized"] == table["episode_organism_normalized"])
        ).astype(int)
        grouped = table.groupby(join_keys, dropna=False)
        result = grouped.agg(
            history_prior_organism_available=(days_column, lambda series: 1),
            history_prior_organism_count=("prior_organism", "size"),
            history_prior_organism_unique_count=("prior_organism", "nunique"),
            history_prior_organism_min_days=(days_column, "min"),
        ).reset_index()
        for window in self._settings.feature_build.history_windows_days:
            window_frame = grouped[days_column].apply(
                lambda series, window=window: int(series.between(0, window, inclusive="both").fillna(False).any())
            )
            result[f"history_prior_organism_any_{window}d"] = window_frame.to_numpy()
            same_window = (
                table.loc[
                    table[days_column].between(0, window, inclusive="both").fillna(False) & table["same_organism"].eq(1),
                    join_keys,
                ]
                .drop_duplicates()
                .assign(**{f"history_prior_same_organism_any_{window}d": 1})
            )
            result = result.merge(same_window, on=join_keys, how="left", validate="one_to_one")
            result[f"history_prior_same_organism_any_{window}d"] = (
                result[f"history_prior_same_organism_any_{window}d"].fillna(0).astype(int)
            )
        return site_base.merge(result, on=join_keys, how="left", validate="one_to_one").loc[:, result.columns]

    def _aggregate_prior_procedures(
        self,
        site: str,
        site_base: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        procedures = load_site_table(site, "prior_procedures")
        if procedures.empty:
            return self._empty_procedures(site_base)
        id_keys = list(self._settings.platform.id_columns)
        procedures = procedures.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if procedures.empty:
            return self._empty_procedures(site_base)

        days_column = "procedure_time_to_culturetime"
        table = procedures.loc[:, join_keys + ["procedure_description", days_column]].copy()
        table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
        # Mirrors the antibiotic-history guard: only strictly pre-culture procedures count as history.
        table = table.loc[table[days_column].ge(0).fillna(False)].copy()
        if table.empty:
            return self._empty_procedures(site_base)
        grouped = table.groupby(join_keys, dropna=False)
        result = grouped.agg(
            history_procedure_available=(days_column, lambda series: 1),
            history_procedure_count=("procedure_description", "size"),
            history_procedure_unique_count=("procedure_description", "nunique"),
            history_procedure_min_days=(days_column, "min"),
        ).reset_index()
        for window in self._settings.feature_build.history_windows_days:
            window_frame = grouped[days_column].apply(
                lambda series, window=window: int(series.between(0, window, inclusive="both").fillna(False).any())
            )
            result[f"history_procedure_any_{window}d"] = window_frame.to_numpy()
        return site_base.merge(result, on=join_keys, how="left", validate="one_to_one").loc[:, result.columns]

    def _aggregate_antibiotic_subtype_history(
        self,
        site: str,
        site_base: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        exposures = load_site_table(site, "antibiotic_subtype_exposure")
        if exposures.empty:
            return self._empty_abx_subtype(site_base)
        id_keys = list(self._settings.platform.id_columns)
        exposures = exposures.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if exposures.empty:
            return self._empty_abx_subtype(site_base)

        days_column = "medication_time_to_culturetime"
        table = exposures.loc[:, join_keys + ["antibiotic_subtype", days_column]].copy()
        table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
        table = table.loc[table[days_column].ge(0).fillna(False)].copy()
        if table.empty:
            return self._empty_abx_subtype(site_base)
        grouped = table.groupby(join_keys, dropna=False)
        result = grouped.agg(
            history_abx_subtype_available=(days_column, lambda series: 1),
            history_abx_subtype_exposure_count=("antibiotic_subtype", "size"),
            history_abx_subtype_unique_count=("antibiotic_subtype", "nunique"),
            history_abx_subtype_min_days=(days_column, "min"),
        ).reset_index()
        for window in self._settings.feature_build.history_windows_days:
            window_frame = grouped[days_column].apply(
                lambda series, window=window: int(series.between(0, window, inclusive="both").fillna(False).any())
            )
            result[f"history_abx_subtype_any_{window}d"] = window_frame.to_numpy()
        return site_base.merge(result, on=join_keys, how="left", validate="one_to_one").loc[:, result.columns]

    def _aggregate_nursing_home_visits(
        self,
        site: str,
        site_base: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        visits = load_site_table(site, "nursing_home_visits")
        if visits.empty:
            return self._empty_nursing_home(site_base)
        id_keys = list(self._settings.platform.id_columns)
        visits = visits.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if visits.empty:
            return self._empty_nursing_home(site_base)

        # `nursing_home_visit_culture` is not a boolean flag: it carries the day-offset from
        # the nursing-home visit to the culture (0 = same day), the same convention as the
        # other tables' `*_time_to_culturetime` columns. `order_time_jittered` in this raw
        # table is constant per episode (it mirrors the culture's own order time, not an
        # independent visit timestamp), so it carries no extra recency signal.
        days_column = "nursing_home_visit_culture"
        table = visits.loc[:, join_keys + [days_column]].copy()
        table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
        table = table.loc[table[days_column].notna() & table[days_column].ge(0)].copy()
        if table.empty:
            return self._empty_nursing_home(site_base)
        grouped = table.groupby(join_keys, dropna=False)
        result = grouped.agg(
            history_nursing_home_available=(days_column, lambda series: 1),
            history_nursing_home_visit_count=(days_column, "size"),
            history_nursing_home_min_days=(days_column, "min"),
        ).reset_index()
        for window in self._settings.feature_build.history_windows_days:
            window_frame = grouped[days_column].apply(
                lambda series, window=window: int(series.between(0, window, inclusive="both").fillna(False).any())
            )
            result[f"history_nursing_home_any_{window}d"] = window_frame.to_numpy()
        return site_base.merge(result, on=join_keys, how="left", validate="one_to_one").loc[:, result.columns]

    @staticmethod
    def _empty_abx(site_base: pd.DataFrame) -> pd.DataFrame:
        frame = site_base.copy()
        frame["history_abx_available"] = 0
        frame["history_abx_exposure_count"] = 0
        frame["history_abx_unique_class_count"] = 0
        frame["history_abx_min_days"] = 0.0
        return frame

    def _empty_prior(self, site_base: pd.DataFrame) -> pd.DataFrame:
        frame = site_base.copy()
        frame["history_prior_organism_available"] = 0
        frame["history_prior_organism_count"] = 0
        frame["history_prior_organism_unique_count"] = 0
        frame["history_prior_organism_min_days"] = 0.0
        for window in self._settings.feature_build.history_windows_days:
            frame[f"history_prior_same_organism_any_{window}d"] = 0
        return frame

    def _empty_procedures(self, site_base: pd.DataFrame) -> pd.DataFrame:
        frame = site_base.copy()
        frame["history_procedure_available"] = 0
        frame["history_procedure_count"] = 0
        frame["history_procedure_unique_count"] = 0
        frame["history_procedure_min_days"] = 0.0
        for window in self._settings.feature_build.history_windows_days:
            frame[f"history_procedure_any_{window}d"] = 0
        return frame

    def _empty_abx_subtype(self, site_base: pd.DataFrame) -> pd.DataFrame:
        frame = site_base.copy()
        frame["history_abx_subtype_available"] = 0
        frame["history_abx_subtype_exposure_count"] = 0
        frame["history_abx_subtype_unique_count"] = 0
        frame["history_abx_subtype_min_days"] = 0.0
        for window in self._settings.feature_build.history_windows_days:
            frame[f"history_abx_subtype_any_{window}d"] = 0
        return frame

    def _empty_nursing_home(self, site_base: pd.DataFrame) -> pd.DataFrame:
        frame = site_base.copy()
        frame["history_nursing_home_available"] = 0
        frame["history_nursing_home_visit_count"] = 0
        frame["history_nursing_home_min_days"] = 0.0
        for window in self._settings.feature_build.history_windows_days:
            frame[f"history_nursing_home_any_{window}d"] = 0
        return frame

    def _site_episode_index(self, culture_episodes: pd.DataFrame, site: str) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        return (
            culture_episodes[culture_episodes["source_site"] == site]
            .loc[:, join_keys]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _site_episode_context(self, culture_episodes: pd.DataFrame, site: str) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        return (
            culture_episodes[culture_episodes["source_site"] == site]
            .loc[:, join_keys + ["organism"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

