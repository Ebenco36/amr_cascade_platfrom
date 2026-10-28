"""Eligibility and structural-omission logic."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.core.utils.organism_names import normalize_organism_label

logger = logging.getLogger(__name__)


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
                    "eligibility_state",
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
        # normalize_organism_label (not the bare normalize_label) resolves known
        # phenotype/resistance-annotated, abbreviated, and legacy-coded spelling
        # variants (e.g. "STAPH AUREUS {MRSA}", "MUCOID PSEUDOMONAS AERUGINOSA")
        # to the same canonical organism the plain-spelled rows already match --
        # see organism_names.py's module docstring for how this alias table was
        # built and how to extend it. Using the bare normalizer here silently
        # defaulted every unmatched variant's is_intrinsic_resistance to 0.
        eligible_space["organism_normalized"] = eligible_space["organism"].map(normalize_organism_label)
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

        # Eligibility and observation are separate facts about a row: an intrinsically
        # resistant pair can still carry a recorded result, and it stays ineligible.
        # eligibility_state says why a row is (not) in the opportunity space;
        # is_observed_tested says whether a result was recorded; observation_status
        # combines them with eligibility first, so an observed ineligible row is
        # reported by its eligibility, never as "observed_tested". Vectorised: no
        # Python-level iteration over the multi-million-row eligibility space.
        intrinsic = eligible_space["is_intrinsic_resistance"].eq(1)
        unavailable = eligible_space["is_operationally_available"].eq(0) & (
            self._settings.gold.eligibility.denominator != "biological"
        )
        eligible_space["eligibility_state"] = pd.Categorical(
            np.select([intrinsic, unavailable], ["intrinsic_resistance", "operationally_unavailable"], default="eligible"),
            categories=["eligible", "intrinsic_resistance", "operationally_unavailable"],
        )
        eligible_space["observation_status"] = pd.Categorical(
            np.select(
                [intrinsic, unavailable, eligible_space["is_observed_tested"].eq(1)],
                ["structural_intrinsic_resistance", "operationally_unavailable", "observed_tested"],
                default="not_observed",
            ),
            categories=["observed_tested", "not_observed", "structural_intrinsic_resistance", "operationally_unavailable"],
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
        unparsed = years.isna()
        if (
            self._settings.gold.eligibility.denominator == "operational_site_organism_era"
            and unparsed.any()
        ):
            by_site = (
                frame.loc[unparsed, "source_site"]
                .astype("string")
                .fillna("unknown")
                .value_counts()
                .to_dict()
                if "source_site" in frame.columns
                else {"unknown": int(unparsed.sum())}
            )
            logger.warning(
                "Operational eligibility could not parse %d of %d timestamps from '%s'; "
                "those rows will use availability_era='unknown'. Unparsed rows by site: %s",
                int(unparsed.sum()),
                len(frame),
                time_column,
                by_site,
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
        # Must use the same normalize_organism_label as build_episode_eligibility's
        # eligible_space (not bare normalize_label) -- this table is joined back
        # onto eligible_space by organism_normalized, so a mismatched normalizer
        # here would silently re-fragment the very spelling-variant pooling the
        # alias table is meant to fix.
        frame["organism_normalized"] = frame["organism"].map(normalize_organism_label)
        frame["antibiotic_normalized"] = frame["antibiotic"].map(normalize_antibiotic_label)
        group_columns = ["source_site", "organism_normalized", "availability_era", "antibiotic_normalized"]
        # Support is the number of distinct episode-antibiotic observations, not rows:
        # an episode whose drug carries two results (discordant R and S, or two
        # labels that normalise to one drug) is one observation of availability.
        episode_keys = [column for column in self._settings.gold.episode_key_columns if column in frame.columns]
        return (
            frame.dropna(subset=["source_site", "organism_normalized", "antibiotic_normalized"])
            .drop_duplicates(subset=[*episode_keys, *[c for c in group_columns if c not in episode_keys]])
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
                    "organism_normalized": reference["microorganism"].map(normalize_organism_label),
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
                        "organism_normalized": supplemental["microorganism"].map(normalize_organism_label),
                        "antibiotic_normalized": supplemental["antibiotic"].map(normalize_antibiotic_label),
                    }
                )
            )

        normalized = pd.concat(frames, ignore_index=True)
        normalized["is_intrinsic_resistance"] = 1
        self._intrinsic_reference = normalized.drop_duplicates()
        return self._intrinsic_reference
