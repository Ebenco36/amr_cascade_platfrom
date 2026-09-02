"""Eligibility and structural-omission logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.core.utils.text import normalize_label


class EligibilityService:
    """Label biological and operational organism-drug opportunities."""

    def __init__(self, settings: Settings, reference_dir: Path) -> None:
        self._settings = settings
        self._reference_dir = reference_dir
        self._intrinsic_reference: pd.DataFrame | None = None

    def build_episode_eligibility(
        self,
        culture_episodes: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_denominator()
        antibiotic_universe = sorted(
            culture_drug_episodes["antibiotic"].dropna().astype("string").unique().tolist()
        )
        if not antibiotic_universe:
            return pd.DataFrame(
                columns=[
                    *self._settings.gold.episode_key_columns,
                    "availability_era",
                    "antibiotic",
                    "is_observed_tested",
                    "is_intrinsic_resistance",
                    "is_biologically_eligible",
                    "availability_support_n",
                    "is_operationally_available",
                    "eligibility_denominator",
                    "is_eligible",
                    "observation_status",
                ]
            )

        base = culture_episodes.loc[:, list(self._settings.gold.episode_key_columns)].drop_duplicates().copy()
        base["availability_era"] = self._availability_era(base)
        base["_merge_key"] = 1
        antibiotics = pd.DataFrame({"antibiotic": antibiotic_universe, "_merge_key": 1})
        eligible_space = base.merge(antibiotics, on="_merge_key", how="inner").drop(columns="_merge_key")

        observed = culture_drug_episodes.loc[
            :, list(self._settings.gold.episode_key_columns) + ["antibiotic"]
        ].drop_duplicates()
        observed["is_observed_tested"] = 1
        eligible_space = eligible_space.merge(
            observed,
            on=[*self._settings.gold.episode_key_columns, "antibiotic"],
            how="left",
        )
        eligible_space["is_observed_tested"] = eligible_space["is_observed_tested"].fillna(0).astype("int64")

        intrinsic_reference = self._load_intrinsic_reference()
        eligible_space["organism_normalized"] = eligible_space["organism"].map(normalize_label)
        eligible_space["antibiotic_normalized"] = eligible_space["antibiotic"].map(normalize_antibiotic_label)
        eligible_space = eligible_space.merge(
            intrinsic_reference,
            on=["organism_normalized", "antibiotic_normalized"],
            how="left",
        )
        eligible_space["is_intrinsic_resistance"] = (
            eligible_space["is_intrinsic_resistance"].fillna(0).astype("int64")
        )
        eligible_space["is_biologically_eligible"] = (1 - eligible_space["is_intrinsic_resistance"]).astype("int64")

        availability = self._build_operational_availability(culture_drug_episodes)
        eligible_space = eligible_space.merge(
            availability,
            on=["source_site", "organism_normalized", "availability_era", "antibiotic_normalized"],
            how="left",
        )
        eligible_space["availability_support_n"] = (
            eligible_space["availability_support_n"].fillna(0).astype("int64")
        )
        eligible_space["is_operationally_available"] = (
            eligible_space["availability_support_n"] >= self._settings.gold.eligibility.availability_min_observed
        ).astype("int64")
        eligible_space["eligibility_denominator"] = self._settings.gold.eligibility.denominator
        if self._settings.gold.eligibility.denominator == "biological":
            eligible_space["is_eligible"] = eligible_space["is_biologically_eligible"]
        else:
            eligible_space["is_eligible"] = (
                (eligible_space["is_biologically_eligible"] == 1)
                & (eligible_space["is_operationally_available"] == 1)
            ).astype("int64")

        # Vectorised equivalent of _observation_status — same priority order,
        # no Python-level row iteration over the ~4 M-row eligibility space.
        _obs_conditions = [
            eligible_space["is_observed_tested"].eq(1),
            eligible_space["is_intrinsic_resistance"].eq(1),
            eligible_space["is_operationally_available"].eq(0),
        ]
        _obs_choices = [
            "observed_tested",
            "structural_intrinsic_resistance",
            "operationally_unavailable",
        ]
        eligible_space["observation_status"] = np.select(
            _obs_conditions, _obs_choices, default="not_observed"
        )
        return eligible_space.drop(columns=["organism_normalized", "antibiotic_normalized"])

    def _validate_denominator(self) -> None:
        allowed = {"biological", "operational_site_organism_era"}
        denominator = self._settings.gold.eligibility.denominator
        if denominator not in allowed:
            raise ValueError(f"Unsupported eligibility denominator '{denominator}'. Expected one of {sorted(allowed)}.")
        if self._settings.gold.eligibility.availability_era_years < 1:
            raise ValueError("gold.eligibility.availability_era_years must be >= 1.")
        if self._settings.gold.eligibility.availability_min_observed < 1:
            raise ValueError("gold.eligibility.availability_min_observed must be >= 1.")

    def _availability_era(self, frame: pd.DataFrame) -> pd.Series:
        time_column = self._settings.gold.eligibility.availability_time_column
        if time_column not in frame.columns:
            if self._settings.gold.eligibility.denominator == "operational_site_organism_era":
                raise ValueError(
                    f"Operational eligibility requires time column '{time_column}' to build availability eras."
                )
            return pd.Series(["unknown"] * len(frame), index=frame.index, dtype="string")
        # format="mixed": frame is combined across sites, and raw source
        # timestamp strings differ in format per site (e.g. "2017-12-11
        # 01:16:00+00:00" vs "2022-02-11T09:37:00.000Z" vs a naive
        # "2009-09-03 08:51:00"). Without format="mixed", pandas infers one
        # format from the array and silently coerces every row that doesn't
        # match it to NaT -- which previously collapsed 2 of 3 sites' rows to
        # era="unknown" (a single site-wide-all-time bucket instead of proper
        # per-era buckets), while the guard below only catches *total*
        # parse failure and missed this partial one. utc=True is required
        # alongside it since the mix includes both tz-aware and tz-naive
        # strings.
        years = pd.to_datetime(frame[time_column], errors="coerce", format="mixed", utc=True).dt.year
        if (
            self._settings.gold.eligibility.denominator == "operational_site_organism_era"
            and len(frame) > 0
            and years.notna().sum() == 0
        ):
            raise ValueError(
                f"Operational eligibility could not parse any years from '{time_column}'."
            )
        width = self._settings.gold.eligibility.availability_era_years
        starts = (years // width) * width
        ends = starts + width - 1
        era = starts.astype("Int64").astype("string") + "-" + ends.astype("Int64").astype("string")
        return era.fillna("unknown")

    def _build_operational_availability(self, culture_drug_episodes: pd.DataFrame) -> pd.DataFrame:
        if culture_drug_episodes.empty:
            return pd.DataFrame(
                columns=[
                    "source_site",
                    "organism_normalized",
                    "availability_era",
                    "antibiotic_normalized",
                    "availability_support_n",
                ]
            )
        frame = culture_drug_episodes.copy()
        frame["availability_era"] = self._availability_era(frame)
        frame["organism_normalized"] = frame["organism"].map(normalize_label)
        frame["antibiotic_normalized"] = frame["antibiotic"].map(normalize_antibiotic_label)
        group_columns = ["source_site", "organism_normalized", "availability_era", "antibiotic_normalized"]
        return (
            frame.dropna(subset=["source_site", "organism_normalized", "antibiotic_normalized"])
            .groupby(group_columns, dropna=False)
            .size()
            .rename("availability_support_n")
            .reset_index()
        )

    def _load_intrinsic_reference(self) -> pd.DataFrame:
        if self._intrinsic_reference is not None:
            return self._intrinsic_reference

        path = self._reference_dir / self._settings.platform.reference_files["intrinsic_resistance"]
        reference = pd.read_csv(path, sep=";")
        frames = [
            pd.DataFrame(
                {
                    "organism_normalized": reference["microorganism"].map(normalize_label),
                    "antibiotic_normalized": reference["antibiotic"].map(normalize_antibiotic_label),
                }
            )
        ]

        # Curated additions not covered by the primary reference list (e.g. drugs
        # with no CLSI/EUCAST breakpoint for a given organism class). Optional:
        # some environments may not ship this file.
        supplemental_name = self._settings.platform.reference_files.get("supplemental_intrinsic_resistance")
        supplemental_path = self._reference_dir / supplemental_name if supplemental_name else None
        if supplemental_path and supplemental_path.exists():
            supplemental = pd.read_csv(supplemental_path)
            frames.append(
                pd.DataFrame(
                    {
                        "organism_normalized": supplemental["microorganism"].map(normalize_label),
                        "antibiotic_normalized": supplemental["antibiotic"].map(normalize_antibiotic_label),
                    }
                )
            )

        normalized = pd.concat(frames, ignore_index=True)
        normalized["is_intrinsic_resistance"] = 1
        self._intrinsic_reference = normalized.drop_duplicates()
        return self._intrinsic_reference
