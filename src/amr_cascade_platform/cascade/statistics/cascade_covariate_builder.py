"""Assemble episode-level covariates for adjusted cascade models."""

from __future__ import annotations

import gc

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.text import normalize_label, safe_feature_name
from amr_cascade_platform.features.builders.acute_feature_builder import AcuteFeatureBuilder
from amr_cascade_platform.features.builders.demographics_feature_builder import DemographicsFeatureBuilder
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


class CascadeCovariateBuilder:
    """Build episode covariates for adjusted downstream-testing models.

    Primary adjustment covariates are selected downstream in
    DownstreamTestingRegression. This builder also materializes timing-limited
    acuity summaries (labs/vitals) for diagnostics and sensitivity analyses; those
    columns are not primary leakage-safe covariates unless measurement timing is
    proven upstream.
    """

    _LAB_SEVERITY_COLS: tuple[str, ...] = (
        "median_wbc",
        "median_neutrophils",
        "median_hgb",
        "median_cr",
        "median_lactate",
        "median_procalcitonin",
    )
    _VITAL_SEVERITY_COLS: tuple[str, ...] = (
        "median_heartrate",
        "median_sysbp",
        "median_temp",
        "median_resprate",
    )

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)
        self._baseline_builder = DemographicsFeatureBuilder(settings)
        self._acute_builder = AcuteFeatureBuilder(settings)

    def build(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        if culture_episodes.empty:
            return pd.DataFrame(columns=episode_keys)

        base = culture_episodes.loc[:, episode_keys].drop_duplicates().copy()
        episode_context = self._build_episode_context(culture_episodes)
        baseline = self._baseline_builder.build(culture_episodes, self._load_site_table)
        acute = self._acute_builder.build(culture_episodes, self._load_site_table)
        comorbidity = self._build_comorbidity_counts(culture_episodes)
        labs_severity = self._build_labs_severity(culture_episodes)
        vitals_severity = self._build_vitals_severity(culture_episodes)
        nursing_home = self._build_nursing_home(culture_episodes)
        prior_procedures_cov = self._build_prior_procedures(culture_episodes)
        calendar = self._build_calendar_features(culture_episodes)
        prior_antibiotic = self._build_prior_antibiotic(culture_episodes)
        prior_same_organism = self._build_prior_same_organism(culture_episodes)

        frame = base.merge(calendar, on=episode_keys, how="left", validate="one_to_one")
        frame = frame.merge(episode_context, on=episode_keys, how="left", validate="one_to_one")
        for covariates in (baseline, acute, comorbidity, labs_severity, vitals_severity, nursing_home, prior_procedures_cov):
            frame = frame.merge(covariates, on=feature_join_keys, how="left", validate="many_to_one")
        del episode_context, baseline, acute, comorbidity, labs_severity, vitals_severity, nursing_home, prior_procedures_cov
        gc.collect()
        frame = frame.merge(prior_antibiotic, on=episode_keys, how="left", validate="one_to_one")
        frame = frame.merge(prior_same_organism, on=episode_keys, how="left", validate="one_to_one")

        frame["cov_calendar_year"] = frame["cov_calendar_year"].fillna("unknown").astype(str)
        frame["cov_calendar_month"] = frame["cov_calendar_month"].fillna("unknown").astype(str)
        frame["cov_ordering_mode"] = frame["cov_ordering_mode"].fillna("unknown").astype(str)
        frame["cov_specimen_type"] = frame["cov_specimen_type"].fillna("unknown").astype(str)
        frame["cov_age_bin"] = self._age_bin(frame.get("demo_age", pd.Series(index=frame.index, dtype="float64")))
        demographics_available = pd.to_numeric(
            self._series_or_default(frame, "baseline_demographics_available"), errors="coerce"
        ).fillna(0).astype(int)
        frame.loc[demographics_available.eq(0), "cov_age_bin"] = "unknown"
        frame["cov_sex"] = self._sex_category(frame)
        frame["cov_icu_status"] = pd.to_numeric(
            self._series_or_default(frame, "ward_hosp_ward_icu"),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_icu_available"] = pd.to_numeric(
            self._series_or_default(frame, "acute_ward_available"),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_er_status"] = pd.to_numeric(
            self._series_or_default(frame, "ward_hosp_ward_er"),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_er_available"] = pd.to_numeric(
            self._series_or_default(frame, "acute_ward_available"),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_prior_abx_any_90d"] = pd.to_numeric(
            self._series_or_default(frame, "cov_prior_abx_any_90d"), errors="coerce"
        ).fillna(0).astype(int)
        frame["cov_prior_abx_available"] = pd.to_numeric(
            self._series_or_default(frame, "cov_prior_abx_available"), errors="coerce"
        ).fillna(0).astype(int)
        frame["cov_prior_same_organism_any_90d"] = pd.to_numeric(
            self._series_or_default(
                frame,
                "cov_prior_same_organism_any_90d",
                fallback_column="history_prior_same_organism_any_90d",
            ),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_prior_organism_available"] = pd.to_numeric(
            self._series_or_default(
                frame,
                "cov_prior_organism_available",
                fallback_column="history_prior_organism_available",
            ),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_comorbidity_count"] = pd.to_numeric(
            self._series_or_default(frame, "comorbidity_count"),
            errors="coerce",
        ).fillna(0).astype(int)
        frame["cov_comorbidity_available"] = pd.to_numeric(
            self._series_or_default(frame, "cov_comorbidity_available"), errors="coerce"
        ).fillna(0).astype(int)
        frame["cov_adi_score"] = pd.to_numeric(
            self._series_or_default(frame, "demo_adi_score"), errors="coerce"
        ).fillna(0.0)
        frame["cov_adi_state_rank"] = pd.to_numeric(
            self._series_or_default(frame, "demo_adi_state_rank"), errors="coerce"
        ).fillna(0.0)
        frame["cov_adi_available"] = pd.to_numeric(
            self._series_or_default(frame, "baseline_adi_available"), errors="coerce"
        ).fillna(0).astype(int)

        comorbidity_component_columns = sorted(column for column in frame.columns if column.startswith("comorb_"))
        for column in comorbidity_component_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        procedure_component_columns = sorted(column for column in frame.columns if column.startswith("proc_"))
        for column in procedure_component_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)

        covariate_columns = [
            "cov_calendar_year",
            "cov_calendar_month",
            "cov_ordering_mode",
            "cov_specimen_type",
            "cov_age_bin",
            "cov_sex",
            "cov_icu_status",
            "cov_icu_available",
            "cov_er_status",
            "cov_er_available",
            "cov_prior_abx_any_90d",
            "cov_prior_abx_available",
            "cov_prior_same_organism_any_90d",
            "cov_prior_organism_available",
            "cov_comorbidity_count",
            "cov_comorbidity_available",
            "cov_adi_score",
            "cov_adi_state_rank",
            "cov_adi_available",
            "cov_labs_available",
            "cov_lab_wbc",
            "cov_lab_neutrophils",
            "cov_lab_hgb",
            "cov_lab_cr",
            "cov_lab_lactate",
            "cov_lab_procalcitonin",
            "cov_vitals_available",
            "cov_vital_heartrate",
            "cov_vital_sysbp",
            "cov_vital_temp",
            "cov_vital_resprate",
            "cov_nursing_home_90d",
            "cov_nursing_home_available",
            "cov_prior_procedure_90d",
            "cov_prior_procedure_available",
        ] + comorbidity_component_columns + procedure_component_columns
        return frame.loc[:, episode_keys + covariate_columns]

    def _build_episode_context(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        available = [column for column in ("ordering_mode", "culture_description") if column in culture_episodes.columns]
        if not available:
            frame = culture_episodes.loc[:, episode_keys].drop_duplicates().copy()
            frame["cov_ordering_mode"] = "unknown"
            frame["cov_specimen_type"] = "unknown"
            return frame

        context = culture_episodes.loc[:, episode_keys + available].copy()
        grouped = (
            context.groupby(episode_keys, dropna=False, observed=True)
            .agg({column: self._first_nonempty_string for column in available})
            .reset_index()
        )
        if "ordering_mode" in grouped.columns:
            grouped["cov_ordering_mode"] = grouped["ordering_mode"].map(self._ordering_mode_category)
        else:
            grouped["cov_ordering_mode"] = "unknown"
        if "culture_description" in grouped.columns:
            grouped["cov_specimen_type"] = grouped["culture_description"].map(self._specimen_type_category)
        else:
            grouped["cov_specimen_type"] = "unknown"
        return grouped.loc[:, episode_keys + ["cov_ordering_mode", "cov_specimen_type"]]

    def _build_calendar_features(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        join_keys = list(self._settings.gold.episode_key_columns)
        frame = culture_episodes.loc[:, join_keys].drop_duplicates().copy()
        # format="mixed": frame is combined across sites, and raw source timestamp
        # strings differ in format per site (space-separated with offset, "T"/"Z"
        # ISO, and naive). Without format="mixed", pandas infers one format from
        # the array and silently coerces every row that doesn't match it to NaT --
        # collapsing cov_calendar_year/month to "unknown" for whichever sites don't
        # match the inferred format, with no error.
        timestamps = pd.to_datetime(frame["order_time_jittered"], errors="coerce", format="mixed", utc=True)
        frame["cov_calendar_year"] = timestamps.dt.year.fillna(-1).astype(int).astype(str).replace({"-1": "unknown"})
        frame["cov_calendar_month"] = timestamps.dt.month.fillna(-1).astype(int).astype(str).replace({"-1": "unknown"})
        return frame

    def _build_comorbidity_counts(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        results: list[pd.DataFrame] = []
        active_component_results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            table = self._load_site_table(
                site,
                "comorbidity",
                columns=feature_join_keys
                + [
                    "comorbidity_component_start_days_culture",
                    "comorbidity_component_end_days_culture",
                    "comorbidity_component",
                ],
            )
            if table.empty:
                empty = site_base.loc[:, feature_join_keys].drop_duplicates().copy()
                empty["comorbidity_count"] = 0
                empty["cov_comorbidity_available"] = 0
                results.append(empty)
                continue
            active = table.copy()
            active["comorbidity_component_start_days_culture"] = pd.to_numeric(
                active["comorbidity_component_start_days_culture"], errors="coerce"
            )
            active["comorbidity_component_end_days_culture"] = pd.to_numeric(
                active["comorbidity_component_end_days_culture"], errors="coerce"
            )
            # The ARMD day-offset convention is culture_time - event_time:
            # positive values occurred before culture; negative values occurred after culture.
            # Active-at-culture comorbidities therefore need a known start on/before
            # culture and no recorded end before culture. Missing starts are not used
            # in the primary leakage-safe burden because their timing is unverifiable.
            active = active[
                active["comorbidity_component_start_days_culture"].ge(0).fillna(False)
                & (
                    active["comorbidity_component_end_days_culture"].isna()
                    | (active["comorbidity_component_end_days_culture"] <= 0)
                )
            ].copy()
            component = active["comorbidity_component"].astype("string").str.strip()
            active = active.loc[component.notna() & component.ne("")]
            active = active.loc[:, feature_join_keys + ["comorbidity_component"]].drop_duplicates()
            if not active.empty:
                active_component_results.append(active)
            grouped = (
                active.groupby(feature_join_keys, dropna=False, observed=True)
                .size()
                .rename("comorbidity_count")
                .reset_index()
            )
            site_frame = (
                site_base.loc[:, feature_join_keys]
                .drop_duplicates()
                .merge(grouped, on=feature_join_keys, how="left", validate="one_to_one")
            )
            site_frame["comorbidity_count"] = site_frame["comorbidity_count"].fillna(0).astype(int)
            site_frame["cov_comorbidity_available"] = 1
            results.append(site_frame)
            del table, active, grouped
            gc.collect()
        if not results:
            return pd.DataFrame(columns=feature_join_keys + ["comorbidity_count", "cov_comorbidity_available"])
        combined = pd.concat(results, ignore_index=True)
        if not active_component_results:
            return combined

        active_components = pd.concat(active_component_results, ignore_index=True)
        selected_components = (
            active_components["comorbidity_component"]
            .astype(str)
            .value_counts()
            .head(self._settings.comorbidity.default_top_k)
            .index
            .tolist()
        )
        if not selected_components:
            return combined
        selected_rows = active_components[
            active_components["comorbidity_component"].isin(selected_components)
        ].copy()
        selected_rows["component_feature"] = selected_rows["comorbidity_component"].map(
            lambda value: f"comorb_{safe_feature_name(str(value))}"
        )
        selected_rows["value"] = 1
        selected_rows = selected_rows.loc[:, feature_join_keys + ["component_feature", "value"]].drop_duplicates(
            subset=feature_join_keys + ["component_feature"]
        )
        component_flags = (
            selected_rows.pivot_table(
                index=feature_join_keys,
                columns="component_feature",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        combined = combined.merge(component_flags, on=feature_join_keys, how="left", validate="one_to_one")
        component_columns = [column for column in component_flags.columns if column not in feature_join_keys]
        if component_columns:
            combined[component_columns] = combined[component_columns].fillna(0).astype(int)
        return combined

    def _build_prior_antibiotic(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            exposures = self._load_site_table(
                site,
                "antibiotic_class_exposure",
                columns=feature_join_keys + ["time_to_culturetime"],
            )
            if exposures.empty:
                empty = site_base.copy()
                empty["cov_prior_abx_any_90d"] = 0
                empty["cov_prior_abx_available"] = 0
                results.append(empty)
                continue
            table = exposures.loc[:, feature_join_keys + ["time_to_culturetime"]].copy()
            table["time_to_culturetime"] = pd.to_numeric(table["time_to_culturetime"], errors="coerce")
            # Adjusted cascade models must only see pre-culture history, so we drop any
            # exposure rows with negative lags before constructing lookback indicators.
            table = table.loc[table["time_to_culturetime"].ge(0).fillna(False)].copy()
            if table.empty:
                empty = site_base.copy()
                empty["cov_prior_abx_any_90d"] = 0
                empty["cov_prior_abx_available"] = 0
                results.append(empty)
                continue
            table = table.merge(
                site_base.loc[:, episode_keys],
                on=feature_join_keys,
                how="inner",
                validate="many_to_many",
            )
            grouped = (
                table.loc[:, episode_keys]
                .drop_duplicates()
                .assign(cov_prior_abx_available=1)
            )
            recent = (
                table.loc[table["time_to_culturetime"].between(0, 90, inclusive="both").fillna(False), episode_keys]
                .drop_duplicates()
                .assign(cov_prior_abx_any_90d=1)
            )
            grouped = grouped.merge(recent, on=episode_keys, how="left", validate="one_to_one")
            grouped["cov_prior_abx_any_90d"] = grouped["cov_prior_abx_any_90d"].fillna(0).astype(int)
            site_frame = site_base.merge(grouped, on=episode_keys, how="left", validate="one_to_one")
            site_frame["cov_prior_abx_any_90d"] = site_frame["cov_prior_abx_any_90d"].fillna(0).astype(int)
            site_frame["cov_prior_abx_available"] = site_frame["cov_prior_abx_available"].fillna(0).astype(int)
            results.append(site_frame)
            del exposures, table, grouped, recent
            gc.collect()
        if not results:
            return pd.DataFrame(columns=episode_keys + ["cov_prior_abx_any_90d", "cov_prior_abx_available"])
        return pd.concat(results, ignore_index=True)

    def _build_prior_same_organism(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            priors = self._load_site_table(site, "prior_infecting_organism")
            if priors.empty:
                empty = site_base.copy()
                empty["cov_prior_same_organism_any_90d"] = 0
                empty["cov_prior_organism_available"] = 0
                results.append(empty)
                continue
            days_candidates = (
                "prior_infecting_organism_days_to_culture",
                "prior_infecting_organism_days_to_culutre",
            )
            days_column = next((column for column in days_candidates if column in priors.columns), None)
            if days_column is None or "prior_organism" not in priors.columns:
                empty = site_base.copy()
                empty["cov_prior_same_organism_any_90d"] = 0
                empty["cov_prior_organism_available"] = 0
                results.append(empty)
                continue
            table = priors.loc[:, feature_join_keys + ["prior_organism", days_column]].copy()
            table[days_column] = pd.to_numeric(table[days_column], errors="coerce")
            # Prior-organism flags are only meaningful if the prior isolate predates the index
            # culture. Negative lags would inject post-culture knowledge into the adjustment set.
            table = table.loc[table[days_column].ge(0).fillna(False)].copy()
            if table.empty:
                empty = site_base.copy()
                empty["cov_prior_same_organism_any_90d"] = 0
                empty["cov_prior_organism_available"] = 0
                results.append(empty)
                continue
            table = table.merge(
                site_base.loc[:, episode_keys],
                on=feature_join_keys,
                how="inner",
                validate="many_to_many",
            )
            table["prior_organism_normalized"] = table["prior_organism"].map(normalize_label)
            table["episode_organism_normalized"] = table["organism"].map(normalize_label)
            table["same_organism"] = (
                table["prior_organism_normalized"].notna()
                & table["episode_organism_normalized"].notna()
                & table["prior_organism_normalized"].eq(table["episode_organism_normalized"])
            ).astype(int)
            grouped = (
                table.loc[:, episode_keys]
                .drop_duplicates()
                .assign(cov_prior_organism_available=1)
            )
            same_keys = (
                table.loc[
                    table["same_organism"].eq(1)
                    & table[days_column].between(0, 90, inclusive="both").fillna(False),
                    episode_keys,
                ]
                .drop_duplicates()
                .assign(cov_prior_same_organism_any_90d=1)
            )
            grouped = grouped.merge(same_keys, on=episode_keys, how="left", validate="one_to_one")
            grouped["cov_prior_same_organism_any_90d"] = grouped["cov_prior_same_organism_any_90d"].fillna(0).astype(int)
            site_frame = site_base.merge(grouped, on=episode_keys, how="left", validate="one_to_one")
            site_frame["cov_prior_same_organism_any_90d"] = site_frame["cov_prior_same_organism_any_90d"].fillna(0).astype(int)
            site_frame["cov_prior_organism_available"] = site_frame["cov_prior_organism_available"].fillna(0).astype(int)
            results.append(site_frame)
            del priors, table, grouped, same_keys
            gc.collect()
        if not results:
            return pd.DataFrame(columns=episode_keys + ["cov_prior_same_organism_any_90d", "cov_prior_organism_available"])
        return pd.concat(results, ignore_index=True)

    def _build_labs_severity(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        # labs uses its own order_proc_id_coded (lab order IDs, not culture IDs);
        # join at encounter level so every culture episode inherits the same labs.
        encounter_join = ["anon_id", "pat_enc_csn_id_coded", "source_site"]
        cov_names = [f"cov_lab_{c.replace('median_', '')}" for c in self._LAB_SEVERITY_COLS]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            base_frame = site_base.loc[:, feature_join_keys].drop_duplicates().copy()
            labs = self._load_site_table(site, "labs")
            if labs.empty:
                base_frame["cov_labs_available"] = 0
                for col in cov_names:
                    base_frame[col] = 0.0
                results.append(base_frame)
                continue
            avail = [c for c in self._LAB_SEVERITY_COLS if c in labs.columns]
            table = labs.loc[:, encounter_join + avail].copy()
            for c in avail:
                table[c] = pd.to_numeric(table[c], errors="coerce")
            grouped = (
                table.groupby(encounter_join, dropna=False)
                .median(numeric_only=True)
                .reset_index()
            )
            grouped = grouped.rename(columns={c: f"cov_lab_{c.replace('median_', '')}" for c in avail})
            grouped["cov_labs_available"] = 1
            site_frame = base_frame.merge(grouped, on=encounter_join, how="left", validate="many_to_one")
            site_frame["cov_labs_available"] = site_frame["cov_labs_available"].fillna(0).astype(int)
            for col in cov_names:
                if col in site_frame.columns:
                    site_frame[col] = pd.to_numeric(site_frame[col], errors="coerce").fillna(0.0)
                else:
                    site_frame[col] = 0.0
            results.append(site_frame)
            del labs, table, grouped
            gc.collect()
        if not results:
            return pd.DataFrame(columns=feature_join_keys + ["cov_labs_available"] + cov_names)
        return pd.concat(results, ignore_index=True)

    def _build_vitals_severity(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        # vitals also uses its own order_proc_id_coded; join at encounter level.
        encounter_join = ["anon_id", "pat_enc_csn_id_coded", "source_site"]
        cov_names = [f"cov_vital_{c.replace('median_', '')}" for c in self._VITAL_SEVERITY_COLS]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            base_frame = site_base.loc[:, feature_join_keys].drop_duplicates().copy()
            vitals = self._load_site_table(site, "vitals")
            if vitals.empty:
                base_frame["cov_vitals_available"] = 0
                for col in cov_names:
                    base_frame[col] = 0.0
                results.append(base_frame)
                continue
            avail = [c for c in self._VITAL_SEVERITY_COLS if c in vitals.columns]
            table = vitals.loc[:, encounter_join + avail].copy()
            for c in avail:
                table[c] = pd.to_numeric(table[c], errors="coerce")
            grouped = (
                table.groupby(encounter_join, dropna=False)
                .median(numeric_only=True)
                .reset_index()
            )
            grouped = grouped.rename(columns={c: f"cov_vital_{c.replace('median_', '')}" for c in avail})
            grouped["cov_vitals_available"] = 1
            site_frame = base_frame.merge(grouped, on=encounter_join, how="left", validate="many_to_one")
            site_frame["cov_vitals_available"] = site_frame["cov_vitals_available"].fillna(0).astype(int)
            for col in cov_names:
                if col in site_frame.columns:
                    site_frame[col] = pd.to_numeric(site_frame[col], errors="coerce").fillna(0.0)
                else:
                    site_frame[col] = 0.0
            results.append(site_frame)
            del vitals, table, grouped
            gc.collect()
        if not results:
            return pd.DataFrame(columns=feature_join_keys + ["cov_vitals_available"] + cov_names)
        return pd.concat(results, ignore_index=True)

    def _build_nursing_home(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            table = self._load_site_table(site, "nursing_home_visits")
            base_frame = site_base.loc[:, feature_join_keys].drop_duplicates().copy()
            if table.empty:
                base_frame["cov_nursing_home_90d"] = 0
                base_frame["cov_nursing_home_available"] = 0
                results.append(base_frame)
                continue
            table = table.loc[:, feature_join_keys + ["nursing_home_visit_culture"]].copy()
            table["nursing_home_visit_culture"] = pd.to_numeric(
                table["nursing_home_visit_culture"], errors="coerce"
            )
            table = table.merge(
                site_base.loc[:, feature_join_keys], on=feature_join_keys, how="inner"
            )
            available = table.loc[:, feature_join_keys].drop_duplicates().assign(cov_nursing_home_available=1)
            recent = (
                table.loc[
                    table["nursing_home_visit_culture"].between(0, 90, inclusive="both").fillna(False),
                    feature_join_keys,
                ]
                .drop_duplicates()
                .assign(cov_nursing_home_90d=1)
            )
            grouped = available.merge(recent, on=feature_join_keys, how="left", validate="one_to_one")
            grouped["cov_nursing_home_90d"] = grouped["cov_nursing_home_90d"].fillna(0).astype(int)
            site_frame = base_frame.merge(grouped, on=feature_join_keys, how="left", validate="one_to_one")
            site_frame["cov_nursing_home_90d"] = site_frame["cov_nursing_home_90d"].fillna(0).astype(int)
            site_frame["cov_nursing_home_available"] = site_frame["cov_nursing_home_available"].fillna(0).astype(int)
            results.append(site_frame)
            del table, available, recent, grouped
            gc.collect()
        if not results:
            return pd.DataFrame(columns=feature_join_keys + ["cov_nursing_home_90d", "cov_nursing_home_available"])
        return pd.concat(results, ignore_index=True)

    def _build_prior_procedures(self, culture_episodes: pd.DataFrame) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        feature_join_keys = list(self._settings.platform.id_columns) + ["source_site"]
        results: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            site_base = (
                culture_episodes[culture_episodes["source_site"] == site]
                .loc[:, episode_keys]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if site_base.empty:
                continue
            table = self._load_site_table(site, "prior_procedures")
            base_frame = site_base.loc[:, feature_join_keys].drop_duplicates().copy()
            if table.empty:
                base_frame["cov_prior_procedure_90d"] = 0
                base_frame["cov_prior_procedure_available"] = 0
                results.append(base_frame)
                continue
            columns = feature_join_keys + ["procedure_time_to_culturetime"]
            if "procedure_description" in table.columns:
                columns.append("procedure_description")
            table = table.loc[:, columns].copy()
            table["procedure_time_to_culturetime"] = pd.to_numeric(
                table["procedure_time_to_culturetime"], errors="coerce"
            )
            # positive values = days before the culture order
            table = table.loc[table["procedure_time_to_culturetime"].ge(0).fillna(False)].copy()
            if table.empty:
                base_frame["cov_prior_procedure_90d"] = 0
                base_frame["cov_prior_procedure_available"] = 0
                results.append(base_frame)
                continue
            table = table.merge(
                site_base.loc[:, feature_join_keys], on=feature_join_keys, how="inner"
            )
            available = table.loc[:, feature_join_keys].drop_duplicates().assign(cov_prior_procedure_available=1)
            recent = (
                table.loc[
                    table["procedure_time_to_culturetime"].between(0, 90, inclusive="both").fillna(False),
                    feature_join_keys + (["procedure_description"] if "procedure_description" in table.columns else []),
                ]
                .drop_duplicates()
            )
            recent_any = recent.loc[:, feature_join_keys].drop_duplicates().assign(cov_prior_procedure_90d=1)
            grouped = available.merge(recent_any, on=feature_join_keys, how="left", validate="one_to_one")
            grouped["cov_prior_procedure_90d"] = grouped["cov_prior_procedure_90d"].fillna(0).astype(int)
            site_frame = base_frame.merge(grouped, on=feature_join_keys, how="left", validate="one_to_one")
            site_frame["cov_prior_procedure_90d"] = site_frame["cov_prior_procedure_90d"].fillna(0).astype(int)
            site_frame["cov_prior_procedure_available"] = site_frame["cov_prior_procedure_available"].fillna(0).astype(int)
            if "procedure_description" in recent.columns and not recent.empty:
                procedure_names = recent["procedure_description"].astype("string").str.strip()
                recent = recent.loc[procedure_names.notna() & procedure_names.ne("")].copy()
                if not recent.empty:
                    selected_procedures = (
                        recent["procedure_description"]
                        .astype(str)
                        .value_counts()
                        .head(self._settings.comorbidity.default_top_k)
                        .index
                        .tolist()
                    )
                    selected_rows = recent[recent["procedure_description"].isin(selected_procedures)].copy()
                    selected_rows["procedure_feature"] = selected_rows["procedure_description"].map(
                        lambda value: f"proc_{safe_feature_name(str(value))}"
                    )
                    selected_rows["value"] = 1
                    selected_rows = selected_rows.loc[
                        :, feature_join_keys + ["procedure_feature", "value"]
                    ].drop_duplicates(subset=feature_join_keys + ["procedure_feature"])
                    procedure_flags = (
                        selected_rows.pivot_table(
                            index=feature_join_keys,
                            columns="procedure_feature",
                            values="value",
                            aggfunc="max",
                            fill_value=0,
                        )
                        .reset_index()
                        .rename_axis(None, axis=1)
                    )
                    site_frame = site_frame.merge(
                        procedure_flags, on=feature_join_keys, how="left", validate="one_to_one"
                    )
                    procedure_columns = [column for column in procedure_flags.columns if column not in feature_join_keys]
                    if procedure_columns:
                        site_frame[procedure_columns] = site_frame[procedure_columns].fillna(0).astype(int)
            results.append(site_frame)
            del table, available, recent, recent_any, grouped
            gc.collect()
        if not results:
            return pd.DataFrame(columns=feature_join_keys + ["cov_prior_procedure_90d", "cov_prior_procedure_available"])
        return pd.concat(results, ignore_index=True)

    def _load_site_table(
        self,
        site: str,
        canonical_table: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        path = self._paths.paths.harmonized / "site_aligned" / site / f"{canonical_table}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return self._dataset_store.read_pandas(path, columns=columns)

    @staticmethod
    def _first_nonempty_string(values: pd.Series) -> str:
        cleaned = values.astype("string").str.strip()
        cleaned = cleaned.loc[cleaned.notna() & cleaned.ne("")]
        if cleaned.empty:
            return "unknown"
        return str(cleaned.iloc[0])

    @staticmethod
    def _ordering_mode_category(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        text = str(value).strip().lower()
        if not text:
            return "unknown"
        if text in {"ip", "inpatient", "in patient"}:
            return "inpatient"
        if text in {"op", "outpatient", "out patient"}:
            return "outpatient"
        if text in {"er", "ed", "emergency", "emergency_department", "emergency department"}:
            return "emergency"
        return safe_feature_name(text) or "unknown"

    @staticmethod
    def _specimen_type_category(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        text = str(value).strip().lower()
        if not text:
            return "unknown"
        if "urine" in text:
            return "urine"
        if "blood" in text:
            return "blood"
        if any(token in text for token in ("respir", "sputum", "bronch", "trach", "lung")):
            return "respiratory"
        if any(token in text for token in ("wound", "skin", "tissue", "abscess")):
            return "wound_skin_soft_tissue"
        if any(token in text for token in ("stool", "fec", "rectal", "gi ", "gastro")):
            return "gastrointestinal"
        if any(token in text for token in ("csf", "fluid", "pleural", "peritoneal", "synovial")):
            return "sterile_fluid"
        return "other"

    @staticmethod
    def _series_or_default(
        frame: pd.DataFrame,
        column: str,
        *,
        fallback_column: str | None = None,
    ) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        if fallback_column and fallback_column in frame.columns:
            return frame[fallback_column]
        return pd.Series(0, index=frame.index, dtype="float64")

    @staticmethod
    def _age_bin(age: pd.Series) -> pd.Series:
        values = pd.to_numeric(age, errors="coerce")
        bins = pd.cut(
            values,
            bins=[-float("inf"), 17, 40, 65, float("inf")],
            labels=["lt18", "18_40", "41_65", "66_plus"],
            right=True,
        )
        return bins.astype("string").fillna("unknown")

    @staticmethod
    def _sex_category(frame: pd.DataFrame) -> pd.Series:
        if "demo_gender_category" in frame.columns:
            return frame["demo_gender_category"].astype("string").fillna("unknown")
        female = pd.to_numeric(frame.get("demo_gender_female", 0), errors="coerce").fillna(0).astype(int)
        male = pd.to_numeric(frame.get("demo_gender_male", 0), errors="coerce").fillna(0).astype(int)
        unknown = pd.to_numeric(frame.get("demo_gender_unknown", 0), errors="coerce").fillna(0).astype(int)
        sex = pd.Series("unknown", index=frame.index, dtype="string")
        sex.loc[female.eq(1)] = "female"
        sex.loc[male.eq(1)] = "male"
        sex.loc[unknown.eq(1)] = "unknown"
        return sex
