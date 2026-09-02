"""Baseline demographics feature builder."""

from __future__ import annotations

import math
import re
from typing import Callable

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class DemographicsFeatureBuilder:
    """Build episode-level baseline features from demographics tables."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        culture_episodes: pd.DataFrame,
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        output_columns = [
            "baseline_demographics_available",
            "demo_age",
            "demo_gender_female",
            "demo_gender_male",
            "demo_gender_unknown",
            "demo_gender_category",
            "baseline_adi_available",
            "demo_adi_score",
            "demo_adi_state_rank",
        ]
        for site in self._settings.platform.sites:
            site_base = self._site_episode_index(culture_episodes, site)
            demographics = load_site_table(site, "demographics")
            if demographics.empty:
                frame = site_base.copy()
                frame["baseline_demographics_available"] = 0
                frame["demo_age"] = 0.0
                frame["demo_gender_female"] = 0
                frame["demo_gender_male"] = 0
                frame["demo_gender_unknown"] = 0
                frame["demo_gender_category"] = "unknown"
            else:
                table = demographics.loc[:, join_keys + ["age", "gender"]].copy()
                table = (
                    table.groupby(join_keys, dropna=False)
                    .agg({"age": "first", "gender": "first"})
                    .reset_index()
                )
                frame = site_base.merge(table, on=join_keys, how="left", validate="one_to_one")
                gender = frame["gender"].fillna("").astype(str).str.strip().str.upper()
                frame["baseline_demographics_available"] = (
                    frame["age"].notna() | frame["gender"].notna()
                ).astype(int)
                frame["demo_age"] = frame["age"].map(self._parse_age_value).fillna(0.0)
                frame["demo_gender_female"] = gender.isin(["F", "FEMALE"]).astype(int)
                frame["demo_gender_male"] = gender.isin(["M", "MALE"]).astype(int)
                frame["demo_gender_unknown"] = (
                    (frame["baseline_demographics_available"] == 1)
                    & (frame["demo_gender_female"] == 0)
                    & (frame["demo_gender_male"] == 0)
                ).astype(int)
                frame["demo_gender_category"] = gender.map(self._normalize_gender_category)

            frame = self._merge_adi_scores(site, frame, join_keys, load_site_table)
            frames.append(frame.loc[:, join_keys + output_columns])
        # Sites with no matching culture episodes in this scope contribute a
        # zero-row (but correctly-columned) frame. Concatenating those alongside
        # non-empty frames triggers pandas' empty/all-NA dtype-inference
        # FutureWarning; excluding them first cannot change the row count (an
        # empty frame contributes none either way) and is the fix pandas itself
        # recommends for this pattern.
        non_empty_frames = [frame for frame in frames if not frame.empty]
        if not non_empty_frames:
            return pd.DataFrame(columns=join_keys + output_columns)
        return pd.concat(non_empty_frames, ignore_index=True)

    def _merge_adi_scores(
        self,
        site: str,
        frame: pd.DataFrame,
        join_keys: list[str],
        load_site_table: Callable[[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        adi = load_site_table(site, "adi_scores")
        if adi.empty:
            frame = frame.copy()
            frame["baseline_adi_available"] = 0
            frame["demo_adi_score"] = 0.0
            frame["demo_adi_state_rank"] = 0.0
            return frame

        table = adi.loc[:, join_keys + ["adi_score", "adi_state_rank"]].copy()
        table = (
            table.groupby(join_keys, dropna=False)
            .agg({"adi_score": "first", "adi_state_rank": "first"})
            .reset_index()
        )
        frame = frame.merge(table, on=join_keys, how="left", validate="one_to_one")
        frame["baseline_adi_available"] = (
            frame["adi_score"].notna() | frame["adi_state_rank"].notna()
        ).astype(int)
        frame["demo_adi_score"] = pd.to_numeric(frame["adi_score"], errors="coerce").fillna(0.0)
        frame["demo_adi_state_rank"] = pd.to_numeric(frame["adi_state_rank"], errors="coerce").fillna(0.0)
        return frame

    def _site_episode_index(self, culture_episodes: pd.DataFrame, site: str) -> pd.DataFrame:
        join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        return (
            culture_episodes[culture_episodes["source_site"] == site]
            .loc[:, join_keys]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    @staticmethod
    def _parse_age_value(value: object) -> float:
        """Parse either numeric ages or source age-bin labels to numeric midpoints.

        Always returns a float (math.nan for unparseable input), never None: a
        Series.map() over a function that sometimes returns Python None produces
        an object-dtype result, which makes the caller's .fillna(0.0) trigger
        pandas' object-dtype-downcast FutureWarning. Returning math.nan instead
        keeps the mapped Series a native float64 dtype from the start.
        """
        if pd.isna(value):
            return math.nan
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return float(numeric)

        text = str(value).strip().lower()
        if not text:
            return math.nan
        if "above" in text or "over" in text:
            match = re.search(r"\d+", text)
            return float(match.group(0)) if match else math.nan
        if "under" in text or "below" in text:
            match = re.search(r"\d+", text)
            return max(float(match.group(0)) - 1.0, 0.0) if match else math.nan
        bounds = [float(match) for match in re.findall(r"\d+", text)]
        if len(bounds) >= 2:
            return (bounds[0] + bounds[1]) / 2.0
        if len(bounds) == 1:
            return bounds[0]
        return math.nan

    @staticmethod
    def _normalize_gender_category(value: object) -> str:
        """Keep undocumented numeric gender encodings as source codes, not sex labels."""
        if pd.isna(value):
            return "unknown"
        text = str(value).strip().upper()
        if not text:
            return "unknown"
        if text in {"F", "FEMALE"}:
            return "female"
        if text in {"M", "MALE"}:
            return "male"
        safe = re.sub(r"[^A-Z0-9]+", "_", text).strip("_").lower()
        return f"code_{safe}" if safe else "unknown"
