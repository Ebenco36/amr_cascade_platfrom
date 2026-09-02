"""Acute labs, vitals, and ward feature builder."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class AcuteFeatureBuilder:
    """Build acute episode-level features from labs, vitals, and ward context."""

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
                self._aggregate_labs(site, site_base, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_vitals(site, site_base, site_episode_context, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            frame = frame.merge(
                self._aggregate_ward(site, site_base, site_episode_context, load_site_table),
                on=join_keys,
                how="left",
                validate="one_to_one",
            )
            feature_columns = [column for column in frame.columns if column not in join_keys]
            for column in feature_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        numeric_columns = [column for column in combined.columns if column not in join_keys]
        for column in numeric_columns:
            if column.endswith("_available") or column.startswith("ward_"):
                combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0).astype(int)
        return combined

    def _aggregate_labs(
        self,
        site: str,
        site_base: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        # The raw labs extract's `Period_Day` column is not a per-row signed offset from the
        # culture order -- per the source dataset's own documentation it is the *window size*
        # used to build the whole extract ("days from the culture order during which lab
        # measurements were recorded"), and it is constant across every row at every site
        # (confirmed empirically: a single value, "14", for the entire armd labs table). There
        # is no other column in this table that indicates whether an individual row's window
        # falls before or after the culture. We therefore cannot verify pre-culture timing for
        # labs at all, and -- exactly like `_aggregate_vitals` below when no timestamp is
        # available -- we intentionally suppress lab features rather than risk leaking
        # possibly-post-culture values into a "prior" feature.
        return self._empty_frame(site_base, "acute_labs_available")

    def _aggregate_vitals(
        self,
        site: str,
        site_base: pd.DataFrame,
        site_episode_context: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        vitals = load_site_table(site, "vitals")
        if vitals.empty:
            return self._empty_frame(site_base, "acute_vitals_available")
        id_keys = list(self._settings.platform.id_columns)
        vitals = vitals.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if vitals.empty:
            return self._empty_frame(site_base, "acute_vitals_available")

        # If harmonization has already removed the vitals timestamp, we cannot prove the rows are
        # pre-culture. In that case we intentionally suppress vitals features instead of leaking
        # future information into training or evaluation.
        time_column = self._find_time_column(vitals)
        if time_column is None:
            return self._empty_frame(site_base, "acute_vitals_available")

        numeric_columns = [
            column
            for column in vitals.columns
            if column not in set(join_keys + [time_column])
        ]
        table = vitals.loc[:, join_keys + [time_column] + numeric_columns].copy()
        table = self._filter_to_pre_culture_rows(table, site_episode_context, time_column)
        if table.empty:
            return self._empty_frame(site_base, "acute_vitals_available")
        for column in numeric_columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        grouped = table.groupby(join_keys, dropna=False).median(numeric_only=True).reset_index()
        rename_map = {column: f"vital_{column.lower()}" for column in grouped.columns if column not in join_keys}
        grouped = grouped.rename(columns=rename_map)
        grouped["acute_vitals_available"] = 1
        return site_base.merge(grouped, on=join_keys, how="left", validate="one_to_one").loc[:, grouped.columns]

    def _aggregate_ward(
        self,
        site: str,
        site_base: pd.DataFrame,
        site_episode_context: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        ward = load_site_table(site, "ward_info")
        if ward.empty:
            return self._empty_frame(site_base, "acute_ward_available")
        id_keys = list(self._settings.platform.id_columns)
        ward = ward.merge(site_base[id_keys].drop_duplicates(), on=id_keys, how="inner")
        if ward.empty:
            return self._empty_frame(site_base, "acute_ward_available")

        time_column = self._find_time_column(ward)
        if time_column is None:
            return self._empty_frame(site_base, "acute_ward_available")

        feature_columns = [
            column
            for column in ward.columns
            if column not in set(join_keys + [time_column])
        ]
        table = ward.loc[:, join_keys + [time_column] + feature_columns].copy()
        table = self._filter_to_pre_culture_rows(table, site_episode_context, time_column)
        if table.empty:
            return self._empty_frame(site_base, "acute_ward_available")
        for column in feature_columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        grouped = table.groupby(join_keys, dropna=False).max(numeric_only=True).reset_index()
        rename_map = {column: f"ward_{column.lower()}" for column in grouped.columns if column not in join_keys}
        grouped = grouped.rename(columns=rename_map)
        grouped["acute_ward_available"] = 1
        return site_base.merge(grouped, on=join_keys, how="left", validate="one_to_one").loc[:, grouped.columns]

    @staticmethod
    def _empty_frame(site_base: pd.DataFrame, availability_column: str) -> pd.DataFrame:
        frame = site_base.copy()
        frame[availability_column] = 0
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
            .loc[:, join_keys + ["order_time_jittered"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    @staticmethod
    def _find_time_column(frame: pd.DataFrame) -> str | None:
        for candidate in ("order_time_jittered", "order_time_jittered_utc"):
            if candidate in frame.columns:
                return candidate
        return None

    def _filter_to_pre_culture_rows(
        self,
        frame: pd.DataFrame,
        site_episode_context: pd.DataFrame,
        time_column: str,
    ) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        table = frame.merge(
            site_episode_context.rename(columns={"order_time_jittered": "culture_order_time_jittered"}),
            on=join_keys,
            how="inner",
            validate="many_to_one",
        )
        # format="mixed": table is combined across sites, and raw source timestamp
        # strings differ in format per site (space-separated with offset, "T"/"Z"
        # ISO, and naive). Without format="mixed", pandas infers one format from
        # the array and silently coerces every row that doesn't match it to NaT,
        # which the row_time.notna() & culture_time.notna() mask below would then
        # silently drop -- losing whichever sites don't match the inferred format,
        # with no error.
        row_time = pd.to_datetime(table[time_column], errors="coerce", format="mixed", utc=True)
        culture_time = pd.to_datetime(table["culture_order_time_jittered"], errors="coerce", format="mixed", utc=True)
        mask = row_time.notna() & culture_time.notna() & row_time.le(culture_time)
        return table.loc[mask, frame.columns].copy()
