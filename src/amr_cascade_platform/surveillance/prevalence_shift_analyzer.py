"""Compute denominator-aware prevalence-shift summaries under selective AST testing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.modeling.estimators.preprocessing import build_tabular_preprocessor


@dataclass(frozen=True)
class PrevalenceShiftBundle:
    pair_results: pd.DataFrame
    episode_level_data: pd.DataFrame
    delta_curve_results: pd.DataFrame
    mnar_curve_results: pd.DataFrame
    mnar_tipping_point_results: pd.DataFrame
    edge_enrichment_results: pd.DataFrame


class PrevalenceShiftAnalyzer:
    """Estimate selective-testing prevalence shift over the eligible denominator."""

    _EVALUABLE_NEGATIVE = "SUSCEPTIBLE"
    _POSITIVE_RESULT = "RESISTANT"
    _VALIDATED_STATUSES = CascadeValidationAnalyzer.VALIDATED_STATUSES
    _CATEGORY_MISSING = "__MISSING__"
    _TIMING_LIMITED_ACUITY_COVARIATES = frozenset(
        {
            "cov_lab_wbc",
            "cov_lab_neutrophils",
            "cov_lab_hgb",
            "cov_lab_cr",
            "cov_lab_lactate",
            "cov_lab_procalcitonin",
            "cov_labs_available",
            "cov_vital_heartrate",
            "cov_vital_sysbp",
            "cov_vital_temp",
            "cov_vital_resprate",
            "cov_vitals_available",
        }
    )
    _PRIMARY_MNAR_COVARIATES = (
        "organism",
        "source_site",
        "cov_ordering_mode",
        "cov_specimen_type",
        "cov_calendar_year",
        "cov_calendar_month",
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
        "cov_nursing_home_90d",
        "cov_nursing_home_available",
        "cov_prior_procedure_90d",
        "cov_prior_procedure_available",
    )
    _MNAR_CASCADE_FEATURES = (
        "cascade_any_validated_positive_trigger",
        "cascade_validated_positive_trigger_count",
    )
    _PAIR_RESULT_COLUMNS = (
        "organism",
        "drug",
        "eligible_n",
        "observed_n",
        "tested_n",
        "evaluable_tested_n",
        "unobserved_n",
        "untested_n",
        "non_evaluable_observed_n",
        "unknown_binary_outcome_n",
        "resistant_n",
        "naive_prevalence",
        "naive_prevalence_pct",
        "prevalence_lower_bound",
        "prevalence_lower_bound_pct",
        "prevalence_upper_bound",
        "prevalence_upper_bound_pct",
        "prevalence_bound_width",
        "prevalence_bound_width_pct",
        "denominator_inflation_effect",
        "denominator_inflation_effect_pct",
        "cascade_triggered_observed_n",
        "independent_observed_n",
        "cascade_triggered_tested_n",
        "independent_tested_n",
        "cascade_resistant_n",
        "independent_resistant_n",
        "cascade_prevalence",
        "cascade_prevalence_pct",
        "independent_prevalence",
        "independent_prevalence_pct",
        "cascade_eligible_n",
        "independent_eligible_n",
        "w_casc_eligible",
        "w_ind_eligible",
        "standardised_prevalence",
        "standardised_prevalence_pct",
        "rho_independent_vs_cascade",
        "delta_max",
        "cascade_trigger_fraction",
        "mean_abs_smd_tested_vs_untested",
        "max_abs_smd_tested_vs_untested",
        "flagged_covariate_n_tested_vs_untested",
        "mean_abs_smd_independent_vs_untested",
        "mean_abs_smd_cascade_vs_untested",
        "mnar_lambda0_prevalence",
        "mnar_lambda0_prevalence_pct",
        "mnar_lambda0_shift_from_naive",
        "mnar_lambda0_shift_from_naive_pct",
        "mnar_lambda0_absolute_shift_pct",
        "mnar_status_lambda0",
        "mnar_feature_count",
        "mnar_feature_columns",
    )
    _DELTA_CURVE_COLUMNS = (
        "organism",
        "drug",
        "delta",
        "shifted_prevalence",
        "shifted_prevalence_pct",
        "delta_prevalence_pct",
        "is_empirical_anchor",
        "is_null_selection_reference",
        "is_upper_bound",
    )
    _MNAR_CURVE_COLUMNS = (
        "organism",
        "drug",
        "eligible_n",
        "tested_n",
        "resistant_n",
        "unknown_binary_outcome_n",
        "naive_prevalence",
        "naive_prevalence_pct",
        "mnar_feature_count",
        "mnar_feature_columns",
        "mnar_lambda",
        "mnar_prevalence",
        "mnar_prevalence_pct",
        "mnar_shift_from_naive",
        "mnar_shift_from_naive_pct",
        "mnar_unknown_expected_resistant_n",
        "mnar_status",
    )
    _MNAR_TIPPING_POINT_COLUMNS = (
        "organism",
        "drug",
        "decision_threshold",
        "decision_threshold_pct",
        "naive_prevalence",
        "naive_prevalence_pct",
        "mnar_lambda_star",
        "crossing_status",
        "min_curve_prevalence",
        "min_curve_prevalence_pct",
        "max_curve_prevalence",
        "max_curve_prevalence_pct",
        "prevalence_at_lambda0",
        "prevalence_at_lambda0_pct",
        "shift_at_lambda0",
        "shift_at_lambda0_pct",
        "eligible_n",
        "tested_n",
        "unknown_binary_outcome_n",
        "cascade_trigger_fraction",
        "rho_independent_vs_cascade",
    )
    _EDGE_ENRICHMENT_COLUMNS = (
        "organism",
        "upstream_antibiotic",
        "downstream_antibiotic",
        "upstream_observed_eligible_n",
        "downstream_tested_n",
        "upstream_positive_rate_when_downstream_tested",
        "upstream_positive_rate_when_downstream_tested_pct",
        "upstream_positive_rate_all_eligible",
        "upstream_positive_rate_all_eligible_pct",
        "upstream_resistance_enrichment",
        "upstream_resistance_enrichment_pct",
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._covariate_columns_cache: tuple[str, ...] | None = None

    def analyze(
        self,
        eligible_pairs: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
        covariates: pd.DataFrame,
        cascade_edge_report: pd.DataFrame | None = None,
    ) -> PrevalenceShiftBundle:
        analysis_frame = self._build_analysis_frame(
            eligible_pairs,
            culture_drug_episodes,
            covariates,
            cascade_edge_report,
        )
        if analysis_frame.empty:
            return PrevalenceShiftBundle(
                pair_results=self._empty_pair_results(),
                episode_level_data=analysis_frame,
                delta_curve_results=self._empty_delta_curve_results(),
                mnar_curve_results=self._empty_mnar_curve_results(),
                mnar_tipping_point_results=self._empty_mnar_tipping_point_results(),
                edge_enrichment_results=self._empty_edge_enrichment_results(),
            )

        pair_rows: list[dict[str, object]] = []
        curve_rows: list[dict[str, object]] = []
        mnar_rows: list[dict[str, object]] = []
        covariate_columns = self._candidate_covariate_columns(analysis_frame)
        for (organism, drug), subset in analysis_frame.groupby(["organism", "antibiotic"], dropna=False, observed=True):
            summary = self._summarize_pair(
                organism=str(organism),
                drug=str(drug),
                subset=subset.copy(),
                covariate_columns=covariate_columns,
            )
            if summary is None:
                continue
            drug_mnar_rows = self._build_mnar_curve_rows(subset.copy(), summary, covariate_columns)
            self._attach_mnar_reference(summary, drug_mnar_rows)
            pair_rows.append(summary)
            curve_rows.extend(self._build_delta_curve_rows(summary))
            mnar_rows.extend(drug_mnar_rows)

        pair_results = pd.DataFrame(pair_rows, columns=self._PAIR_RESULT_COLUMNS)
        if not pair_results.empty:
            pair_results = pair_results.sort_values(
                by=["mnar_lambda0_absolute_shift_pct", "eligible_n", "tested_n"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        delta_curve_results = pd.DataFrame(curve_rows, columns=self._DELTA_CURVE_COLUMNS)
        mnar_curve_results = pd.DataFrame(mnar_rows, columns=self._MNAR_CURVE_COLUMNS)
        mnar_tipping_point_results = self._build_mnar_tipping_point_results(pair_results, mnar_curve_results)
        edge_enrichment_results = self._build_edge_enrichment_results(
            analysis_frame=analysis_frame,
            culture_drug_episodes=culture_drug_episodes,
            cascade_edge_report=cascade_edge_report,
        )
        return PrevalenceShiftBundle(
            pair_results=pair_results,
            episode_level_data=analysis_frame,
            delta_curve_results=delta_curve_results,
            mnar_curve_results=mnar_curve_results,
            mnar_tipping_point_results=mnar_tipping_point_results,
            edge_enrichment_results=edge_enrichment_results,
        )

    @classmethod
    def _empty_pair_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._PAIR_RESULT_COLUMNS)

    @classmethod
    def _empty_delta_curve_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._DELTA_CURVE_COLUMNS)

    @classmethod
    def _empty_mnar_curve_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._MNAR_CURVE_COLUMNS)

    @classmethod
    def _empty_mnar_tipping_point_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._MNAR_TIPPING_POINT_COLUMNS)

    @classmethod
    def _empty_edge_enrichment_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._EDGE_ENRICHMENT_COLUMNS)

    def _build_mnar_tipping_point_results(
        self,
        pair_results: pd.DataFrame,
        mnar_curve_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarise when MNAR curves cross decision thresholds.

        This is a decision-threshold sensitivity diagnostic. It does not
        estimate a true clinical safety point; it reports the lambda value at
        which the assumption-indexed eligible-denominator prevalence would cross
        each configured threshold.
        """

        thresholds = tuple(float(value) for value in self._settings.prevalence.decision_thresholds)
        if not thresholds:
            return self._empty_mnar_tipping_point_results()

        summary_lookup: dict[tuple[str, str], dict[str, object]] = {}
        if not pair_results.empty:
            for _, row in pair_results.iterrows():
                key = (str(row.get("organism", "")), str(row.get("drug", "")))
                summary_lookup[key] = row.to_dict()

        curve_groups: dict[tuple[str, str], pd.DataFrame] = {}
        if not mnar_curve_results.empty:
            for key, group in mnar_curve_results.groupby(["organism", "drug"], dropna=False, observed=True):
                curve_groups[(str(key[0]), str(key[1]))] = group.copy()

        keys = sorted(set(summary_lookup) | set(curve_groups))
        rows: list[dict[str, object]] = []
        for organism, drug in keys:
            summary = summary_lookup.get((organism, drug), {})
            curves = curve_groups.get((organism, drug), pd.DataFrame())
            estimated = (
                curves.loc[curves["mnar_status"].eq("estimated")].copy()
                if "mnar_status" in curves.columns
                else curves.head(0).copy()
            )
            if not estimated.empty:
                estimated["mnar_lambda"] = pd.to_numeric(estimated["mnar_lambda"], errors="coerce")
                estimated["mnar_prevalence"] = pd.to_numeric(estimated["mnar_prevalence"], errors="coerce")
                estimated = (
                    estimated.dropna(subset=["mnar_lambda", "mnar_prevalence"])
                    .sort_values("mnar_lambda")
                    .drop_duplicates(subset=["mnar_lambda"], keep="first")
                )

            lambda0_row = (
                estimated.loc[np.isclose(estimated["mnar_lambda"], 0.0)].head(1)
                if not estimated.empty
                else pd.DataFrame()
            )
            prevalence_at_lambda0 = (
                float(lambda0_row["mnar_prevalence"].iloc[0])
                if not lambda0_row.empty
                else self._safe_float(summary.get("mnar_lambda0_prevalence"))
            )
            naive_prevalence = self._safe_float(summary.get("naive_prevalence"))
            shift_at_lambda0 = (
                float(naive_prevalence - prevalence_at_lambda0)
                if pd.notna(naive_prevalence) and pd.notna(prevalence_at_lambda0)
                else math.nan
            )

            min_curve = float(estimated["mnar_prevalence"].min()) if not estimated.empty else math.nan
            max_curve = float(estimated["mnar_prevalence"].max()) if not estimated.empty else math.nan
            for threshold in thresholds:
                lambda_star, crossing_status = self._mnar_threshold_crossing(estimated, threshold)
                rows.append(
                    {
                        "organism": organism,
                        "drug": drug,
                        "decision_threshold": threshold,
                        "decision_threshold_pct": threshold * 100.0,
                        "naive_prevalence": naive_prevalence,
                        "naive_prevalence_pct": naive_prevalence * 100.0 if pd.notna(naive_prevalence) else math.nan,
                        "mnar_lambda_star": lambda_star,
                        "crossing_status": crossing_status,
                        "min_curve_prevalence": min_curve,
                        "min_curve_prevalence_pct": min_curve * 100.0 if pd.notna(min_curve) else math.nan,
                        "max_curve_prevalence": max_curve,
                        "max_curve_prevalence_pct": max_curve * 100.0 if pd.notna(max_curve) else math.nan,
                        "prevalence_at_lambda0": prevalence_at_lambda0,
                        "prevalence_at_lambda0_pct": (
                            prevalence_at_lambda0 * 100.0 if pd.notna(prevalence_at_lambda0) else math.nan
                        ),
                        "shift_at_lambda0": shift_at_lambda0,
                        "shift_at_lambda0_pct": shift_at_lambda0 * 100.0 if pd.notna(shift_at_lambda0) else math.nan,
                        "eligible_n": int(summary.get("eligible_n", 0) or 0),
                        "tested_n": int(summary.get("tested_n", 0) or 0),
                        "unknown_binary_outcome_n": int(summary.get("unknown_binary_outcome_n", 0) or 0),
                        "cascade_trigger_fraction": self._safe_float(summary.get("cascade_trigger_fraction")),
                        "rho_independent_vs_cascade": self._safe_float(summary.get("rho_independent_vs_cascade")),
                    }
                )

        return pd.DataFrame(rows, columns=self._MNAR_TIPPING_POINT_COLUMNS)

    @staticmethod
    def _mnar_threshold_crossing(estimated: pd.DataFrame, threshold: float) -> tuple[float, str]:
        if estimated.empty or len(estimated) < 2:
            return math.nan, "not_evaluable"

        x = estimated["mnar_lambda"].astype(float).to_numpy()
        y = estimated["mnar_prevalence"].astype(float).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
            return math.nan, "not_evaluable"

        min_y = float(np.min(y))
        max_y = float(np.max(y))
        if threshold < min_y:
            return math.nan, "always_above_threshold"
        if threshold > max_y:
            return math.nan, "always_below_threshold"

        hits = np.where(np.isclose(y, threshold, rtol=0.0, atol=1e-12))[0]
        candidates = [float(x[idx]) for idx in hits]
        for idx in range(len(x) - 1):
            y0 = float(y[idx])
            y1 = float(y[idx + 1])
            if (y0 - threshold) * (y1 - threshold) > 0:
                continue
            if math.isclose(y0, y1, rel_tol=0.0, abs_tol=1e-12):
                continue
            x0 = float(x[idx])
            x1 = float(x[idx + 1])
            frac = (threshold - y0) / (y1 - y0)
            candidates.append(x0 + frac * (x1 - x0))

        if not candidates:
            return math.nan, "not_evaluable"
        return min(candidates, key=lambda value: abs(value)), "crosses_threshold"

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return math.nan
        return result if math.isfinite(result) else math.nan

    def _build_analysis_frame(
        self,
        eligible_pairs: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
        covariates: pd.DataFrame,
        cascade_edge_report: pd.DataFrame | None,
    ) -> pd.DataFrame:
        join_keys = list(self._settings.gold.episode_key_columns)
        pair_keys = join_keys + ["antibiotic"]
        # Raw site data records the same antibiotic under different spellings
        # (e.g. "AMP/SULBACTAM" vs "AMPICILLIN/SULBACTAM"). Without this, the
        # groupby below silently splits one drug into multiple summary rows
        # with different partial-site denominators.
        eligible_pairs = eligible_pairs.copy()
        eligible_pairs["antibiotic"] = eligible_pairs["antibiotic"].map(normalize_antibiotic_label)
        culture_drug_episodes = culture_drug_episodes.copy()
        culture_drug_episodes["antibiotic"] = culture_drug_episodes["antibiotic"].map(normalize_antibiotic_label)

        downstream_antibiotics = self._validated_downstream_antibiotics(cascade_edge_report)
        if not downstream_antibiotics:
            return pd.DataFrame()
        eligible_pairs = eligible_pairs.loc[eligible_pairs["antibiotic"].isin(downstream_antibiotics)].copy()

        eligible = eligible_pairs.loc[eligible_pairs["is_eligible"] == 1, pair_keys].drop_duplicates().copy()
        if eligible.empty:
            return pd.DataFrame()

        observed = self._collapse_observed_results(
            culture_drug_episodes.loc[:, pair_keys + ["susceptibility"]].copy(),
            pair_keys,
        )
        frame = eligible.merge(observed, on=pair_keys, how="left", validate="one_to_one")
        frame["is_tested"] = frame["susceptibility"].notna().astype(int)
        frame["is_resistant"] = (
            frame["susceptibility"].eq(self._settings.cascade.upstream_result_positive).fillna(False).astype(int)
        )
        frame["is_evaluable_test"] = frame["susceptibility"].isin(
            [self._settings.cascade.upstream_result_positive, self._EVALUABLE_NEGATIVE]
        ).fillna(False).astype(int)
        frame = frame.merge(covariates.copy(), on=join_keys, how="left", validate="many_to_one")

        cascade_features = self._build_cascade_features(
            eligible_pairs=eligible,
            culture_drug_episodes=culture_drug_episodes,
            cascade_edge_report=cascade_edge_report,
        )
        if not cascade_features.empty:
            frame = frame.merge(cascade_features, on=pair_keys, how="left", validate="one_to_one")
        for column, default in {
            "cascade_any_validated_trigger_observed": 0,
            "cascade_any_validated_positive_trigger": 0,
            "cascade_validated_trigger_count": 0,
            "cascade_validated_positive_trigger_count": 0,
            "cascade_max_validated_trigger_er": math.nan,
            "cascade_max_validated_trigger_adjusted_or": math.nan,
        }.items():
            if column not in frame.columns:
                frame[column] = default
        return frame

    def _build_cascade_features(
        self,
        eligible_pairs: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
        cascade_edge_report: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if cascade_edge_report is None or cascade_edge_report.empty or culture_drug_episodes.empty or eligible_pairs.empty:
            return pd.DataFrame()
        # eligible_pairs/culture_drug_episodes antibiotic names are already
        # normalized by the caller (_build_analysis_frame); normalize
        # cascade_edge_report's the same way so the merge below on
        # (upstream_antibiotic, downstream_antibiotic) is on canonical names
        # from both sides.
        cascade_edge_report = cascade_edge_report.copy()
        cascade_edge_report["upstream_antibiotic"] = cascade_edge_report["upstream_antibiotic"].map(
            normalize_antibiotic_label
        )
        cascade_edge_report["downstream_antibiotic"] = cascade_edge_report["downstream_antibiotic"].map(
            normalize_antibiotic_label
        )
        edge_frame = self._validated_escalation_edges(
            cascade_edge_report,
            columns=[
                "upstream_antibiotic",
                "downstream_antibiotic",
                "escalation_ratio",
                "adjusted_odds_ratio",
            ],
        )
        if edge_frame.empty:
            return pd.DataFrame()

        episode_keys = list(self._settings.gold.episode_key_columns)
        pair_keys = episode_keys + ["antibiotic"]
        target = eligible_pairs.loc[:, pair_keys].drop_duplicates().rename(columns={"antibiotic": "downstream_antibiotic"})
        observed = self._collapse_observed_results(
            culture_drug_episodes.loc[:, episode_keys + ["antibiotic", "susceptibility"]].copy(),
            episode_keys + ["antibiotic"],
        )
        observed = observed.loc[
            observed["susceptibility"].isin([self._settings.cascade.upstream_result_positive, self._EVALUABLE_NEGATIVE])
        ].rename(columns={"antibiotic": "upstream_antibiotic", "susceptibility": "upstream_susceptibility"})
        candidate = target.merge(observed, on=episode_keys, how="left")
        candidate = candidate[candidate["upstream_antibiotic"] != candidate["downstream_antibiotic"]].copy()
        candidate = candidate.merge(edge_frame, on=["upstream_antibiotic", "downstream_antibiotic"], how="left")
        candidate = candidate[candidate["upstream_antibiotic"].notna() & candidate["escalation_ratio"].notna()].copy()
        if candidate.empty:
            return pd.DataFrame()
        if "adjusted_odds_ratio" not in candidate.columns:
            candidate["adjusted_odds_ratio"] = math.nan

        candidate["validated_positive_trigger"] = candidate["upstream_susceptibility"].eq(
            self._settings.cascade.upstream_result_positive
        ).astype(int)
        grouped = (
            candidate.groupby(episode_keys + ["downstream_antibiotic"], dropna=False, observed=True)
            .agg(
                cascade_any_validated_trigger_observed=("upstream_antibiotic", lambda values: int(len(values) > 0)),
                cascade_any_validated_positive_trigger=("validated_positive_trigger", "max"),
                cascade_validated_trigger_count=("upstream_antibiotic", "nunique"),
                cascade_validated_positive_trigger_count=("validated_positive_trigger", "sum"),
                cascade_max_validated_trigger_er=("escalation_ratio", "max"),
                cascade_max_validated_trigger_adjusted_or=("adjusted_odds_ratio", "max"),
            )
            .reset_index()
            .rename(columns={"downstream_antibiotic": "antibiotic"})
        )
        return grouped

    def _summarize_pair(
        self,
        organism: str,
        drug: str,
        subset: pd.DataFrame,
        covariate_columns: tuple[str, ...],
    ) -> dict[str, object] | None:
        eligible_n = int(len(subset))
        observed_mask = subset["is_tested"].eq(1)
        evaluable_mask = subset["is_evaluable_test"].eq(1)
        observed_n = int(observed_mask.sum())
        tested_n = int(evaluable_mask.sum())
        resistant_n = int(subset.loc[evaluable_mask, "is_resistant"].sum())
        if eligible_n < self._settings.prevalence.min_eligible_support:
            return None
        if tested_n < self._settings.prevalence.min_tested_support:
            return None
        if resistant_n < self._settings.prevalence.min_resistant_support:
            return None

        unobserved_n = max(eligible_n - observed_n, 0)
        non_evaluable_observed_n = max(observed_n - tested_n, 0)
        unknown_binary_outcome_n = max(eligible_n - tested_n, 0)
        naive_prevalence = float(resistant_n / tested_n) if tested_n > 0 else math.nan
        lower_bound = float(resistant_n / eligible_n) if eligible_n > 0 else math.nan
        upper_bound = float((resistant_n + unknown_binary_outcome_n) / eligible_n) if eligible_n > 0 else math.nan
        denominator_inflation = naive_prevalence - lower_bound if pd.notna(naive_prevalence) else math.nan

        cascade_trigger_mask = subset["cascade_any_validated_positive_trigger"].eq(1)
        cascade_observed_mask = observed_mask & cascade_trigger_mask
        independent_observed_mask = observed_mask & ~cascade_trigger_mask
        cascade_mask = evaluable_mask & cascade_trigger_mask
        independent_mask = evaluable_mask & ~cascade_trigger_mask
        cascade_observed_n = int(cascade_observed_mask.sum())
        independent_observed_n = int(independent_observed_mask.sum())
        cascade_tested_n = int(cascade_mask.sum())
        independent_tested_n = int(independent_mask.sum())
        cascade_resistant_n = int(subset.loc[cascade_mask, "is_resistant"].sum())
        independent_resistant_n = int(subset.loc[independent_mask, "is_resistant"].sum())
        cascade_prevalence = (
            float(cascade_resistant_n / cascade_tested_n) if cascade_tested_n > 0 else math.nan
        )
        independent_prevalence = (
            float(independent_resistant_n / independent_tested_n) if independent_tested_n > 0 else math.nan
        )

        # Eligible-population weights for subgroup standardisation estimator.
        # cascade_trigger_mask applies to ALL eligible rows (not just tested), so these
        # weights reflect the composition of the opportunity space rather than the
        # conditioned-on-tested subpopulation.
        cascade_eligible_n = int(cascade_trigger_mask.sum())
        independent_eligible_n = eligible_n - cascade_eligible_n
        w_casc_eligible = float(cascade_eligible_n / eligible_n) if eligible_n > 0 else math.nan
        w_ind_eligible = float(independent_eligible_n / eligible_n) if eligible_n > 0 else math.nan
        standardised_prevalence = self._standardised_prevalence(
            cascade_prevalence=cascade_prevalence,
            independent_prevalence=independent_prevalence,
            w_casc_eligible=w_casc_eligible,
            w_ind_eligible=w_ind_eligible,
        )

        rho_independent_vs_cascade = math.nan
        if cascade_tested_n > 0 and independent_tested_n > 0 and pd.notna(cascade_prevalence) and cascade_prevalence > 0:
            rho_independent_vs_cascade = float(independent_prevalence / cascade_prevalence)

        delta_max = float(1.0 / naive_prevalence) if naive_prevalence > 0 else math.nan
        legacy_delta_empirical = self._project_delta_to_admissible(rho_independent_vs_cascade, delta_max)
        legacy_shifted_prevalence_empirical = self._shifted_prevalence(
            naive_prevalence=naive_prevalence,
            unknown_binary_outcome_n=unknown_binary_outcome_n,
            eligible_n=eligible_n,
            delta=legacy_delta_empirical,
        )
        legacy_delta_prevalence = (
            float(naive_prevalence - legacy_shifted_prevalence_empirical)
            if pd.notna(naive_prevalence) and pd.notna(legacy_shifted_prevalence_empirical)
            else math.nan
        )

        selection_all = self._selection_summary(
            group_a=subset.loc[observed_mask],
            group_b=subset.loc[~observed_mask],
            covariate_columns=covariate_columns,
        )
        selection_independent = self._selection_summary(
            group_a=subset.loc[independent_mask],
            group_b=subset.loc[~observed_mask],
            covariate_columns=covariate_columns,
        )
        selection_cascade = self._selection_summary(
            group_a=subset.loc[cascade_mask],
            group_b=subset.loc[~observed_mask],
            covariate_columns=covariate_columns,
        )

        cascade_trigger_fraction = float(cascade_observed_n / observed_n) if observed_n > 0 else math.nan
        legacy_shift_direction = self._shift_direction(legacy_delta_prevalence)
        return {
            "organism": organism,
            "drug": drug,
            "eligible_n": eligible_n,
            "observed_n": observed_n,
            "tested_n": tested_n,
            "evaluable_tested_n": tested_n,
            "unobserved_n": unobserved_n,
            "untested_n": unobserved_n,
            "non_evaluable_observed_n": non_evaluable_observed_n,
            "unknown_binary_outcome_n": unknown_binary_outcome_n,
            "resistant_n": resistant_n,
            "naive_prevalence": naive_prevalence,
            "naive_prevalence_pct": naive_prevalence * 100.0,
            "prevalence_lower_bound": lower_bound,
            "prevalence_lower_bound_pct": lower_bound * 100.0,
            "prevalence_upper_bound": upper_bound,
            "prevalence_upper_bound_pct": upper_bound * 100.0,
            "prevalence_bound_width": upper_bound - lower_bound if pd.notna(lower_bound) and pd.notna(upper_bound) else math.nan,
            "prevalence_bound_width_pct": (upper_bound - lower_bound) * 100.0 if pd.notna(lower_bound) and pd.notna(upper_bound) else math.nan,
            "denominator_inflation_effect": denominator_inflation,
            "denominator_inflation_effect_pct": denominator_inflation * 100.0 if pd.notna(denominator_inflation) else math.nan,
            "cascade_triggered_observed_n": cascade_observed_n,
            "independent_observed_n": independent_observed_n,
            "cascade_triggered_tested_n": cascade_tested_n,
            "independent_tested_n": independent_tested_n,
            "cascade_resistant_n": cascade_resistant_n,
            "independent_resistant_n": independent_resistant_n,
            "cascade_prevalence": cascade_prevalence,
            "cascade_prevalence_pct": cascade_prevalence * 100.0 if pd.notna(cascade_prevalence) else math.nan,
            "independent_prevalence": independent_prevalence,
            "independent_prevalence_pct": independent_prevalence * 100.0 if pd.notna(independent_prevalence) else math.nan,
            "cascade_eligible_n": cascade_eligible_n,
            "independent_eligible_n": independent_eligible_n,
            "w_casc_eligible": w_casc_eligible,
            "w_ind_eligible": w_ind_eligible,
            "standardised_prevalence": standardised_prevalence,
            "standardised_prevalence_pct": standardised_prevalence * 100.0 if pd.notna(standardised_prevalence) else math.nan,
            "rho_independent_vs_cascade": rho_independent_vs_cascade,
            "legacy_delta_empirical_raw": rho_independent_vs_cascade,
            "legacy_delta_empirical": legacy_delta_empirical,
            "legacy_delta_empirical_was_projected": bool(
                pd.notna(rho_independent_vs_cascade) and pd.notna(legacy_delta_empirical) and not math.isclose(rho_independent_vs_cascade, legacy_delta_empirical, rel_tol=0.0, abs_tol=1e-12)
            ),
            "delta_max": delta_max,
            "legacy_shifted_prevalence_empirical": legacy_shifted_prevalence_empirical,
            "legacy_shifted_prevalence_empirical_pct": legacy_shifted_prevalence_empirical * 100.0 if pd.notna(legacy_shifted_prevalence_empirical) else math.nan,
            "legacy_delta_prevalence": legacy_delta_prevalence,
            "legacy_delta_prevalence_pct": legacy_delta_prevalence * 100.0 if pd.notna(legacy_delta_prevalence) else math.nan,
            "legacy_absolute_shift_pct": abs(legacy_delta_prevalence * 100.0) if pd.notna(legacy_delta_prevalence) else math.nan,
            "legacy_shift_ratio_empirical": (
                float(legacy_shifted_prevalence_empirical / naive_prevalence)
                if pd.notna(legacy_shifted_prevalence_empirical) and pd.notna(naive_prevalence) and naive_prevalence > 0
                else math.nan
            ),
            "cascade_trigger_fraction": cascade_trigger_fraction,
            "mean_abs_smd_tested_vs_untested": selection_all["mean_abs_smd"],
            "max_abs_smd_tested_vs_untested": selection_all["max_abs_smd"],
            "flagged_covariate_n_tested_vs_untested": selection_all["flagged_covariate_n"],
            "mean_abs_smd_independent_vs_untested": selection_independent["mean_abs_smd"],
            "mean_abs_smd_cascade_vs_untested": selection_cascade["mean_abs_smd"],
            "legacy_shift_direction": legacy_shift_direction,
        }

    def _build_delta_curve_rows(self, summary: dict[str, object]) -> list[dict[str, object]]:
        naive_prevalence = float(summary["naive_prevalence"])
        eligible_n = int(summary["eligible_n"])
        unknown_binary_outcome_n = int(summary["unknown_binary_outcome_n"])
        delta_max = summary.get("delta_max")
        if pd.isna(delta_max):
            return []
        delta_max = float(delta_max)
        point_count = max(int(self._settings.prevalence.delta_curve_points), 2)
        sampled = [delta_max * idx / (point_count - 1) for idx in range(point_count)]
        exact = [0.0, 0.25, 0.5, 0.75, 1.0]
        if pd.notna(summary.get("legacy_delta_empirical")):
            exact.append(float(summary["legacy_delta_empirical"]))
        exact.append(delta_max)
        deltas = sorted({round(value, 10) for value in sampled + [value for value in exact if 0.0 <= value <= delta_max]})

        rows: list[dict[str, object]] = []
        for delta in deltas:
            shifted = self._shifted_prevalence(
                naive_prevalence=naive_prevalence,
                unknown_binary_outcome_n=unknown_binary_outcome_n,
                eligible_n=eligible_n,
                delta=delta,
            )
            rows.append(
                {
                    "organism": summary["organism"],
                    "drug": summary["drug"],
                    "delta": delta,
                    "shifted_prevalence": shifted,
                    "shifted_prevalence_pct": shifted * 100.0 if pd.notna(shifted) else math.nan,
                    "delta_prevalence_pct": (naive_prevalence - shifted) * 100.0 if pd.notna(shifted) else math.nan,
                    "is_empirical_anchor": bool(
                        pd.notna(summary.get("legacy_delta_empirical")) and math.isclose(delta, float(summary["legacy_delta_empirical"]), rel_tol=0.0, abs_tol=1e-10)
                    ),
                    "is_null_selection_reference": math.isclose(delta, 1.0, rel_tol=0.0, abs_tol=1e-10),
                    "is_upper_bound": math.isclose(delta, delta_max, rel_tol=0.0, abs_tol=1e-10),
                }
            )
        return rows

    def _build_mnar_curve_rows(
        self,
        subset: pd.DataFrame,
        summary: dict[str, object],
        covariate_columns: tuple[str, ...],
    ) -> list[dict[str, object]]:
        """Build cascade-aware MNAR odds-tilt sensitivity rows for one drug.

        The resistance model estimates P(Y=1 | X, observed binary AST). For
        opportunities without a binary-evaluable result, lambda tilts the
        observed-data resistance odds:

            odds(Y=1 | X, unknown; lambda) = odds(Y=1 | X, observed) * exp(-lambda)

        Positive lambda means resistant outcomes were preferentially observed
        after measured covariates/cascade triggers, so unobserved outcomes are
        tilted toward lower resistance. Lambda is not estimated from the data;
        it is the explicit MNAR sensitivity parameter.
        """
        lambda_grid = tuple(float(value) for value in self._settings.prevalence.mnar_lambda_grid)
        if not lambda_grid:
            return []

        organism = str(summary["organism"])
        drug = str(summary["drug"])
        eligible_n = int(summary["eligible_n"])
        tested_n = int(summary["tested_n"])
        resistant_n = int(summary["resistant_n"])
        naive_prevalence = float(summary["naive_prevalence"])
        unknown_mask = ~subset["is_evaluable_test"].eq(1)
        unknown_n = int(unknown_mask.sum())
        base = {
            "organism": organism,
            "drug": drug,
            "eligible_n": eligible_n,
            "tested_n": tested_n,
            "resistant_n": resistant_n,
            "unknown_binary_outcome_n": unknown_n,
            "naive_prevalence": naive_prevalence,
            "naive_prevalence_pct": naive_prevalence * 100.0,
        }

        feature_columns = self._mnar_feature_columns(subset, covariate_columns)
        base["mnar_feature_count"] = len(feature_columns)
        base["mnar_feature_columns"] = ", ".join(feature_columns)

        training = subset.loc[subset["is_evaluable_test"].eq(1)].copy()
        if training.empty or training["is_resistant"].nunique(dropna=True) < 2:
            return [
                {
                    **base,
                    "mnar_lambda": lambda_value,
                    "mnar_prevalence": math.nan,
                    "mnar_prevalence_pct": math.nan,
                    "mnar_shift_from_naive": math.nan,
                    "mnar_shift_from_naive_pct": math.nan,
                    "mnar_unknown_expected_resistant_n": math.nan,
                    "mnar_status": "not_evaluable_resistance_model",
                }
                for lambda_value in lambda_grid
            ]

        if not feature_columns:
            return [
                {
                    **base,
                    "mnar_lambda": lambda_value,
                    "mnar_prevalence": math.nan,
                    "mnar_prevalence_pct": math.nan,
                    "mnar_shift_from_naive": math.nan,
                    "mnar_shift_from_naive_pct": math.nan,
                    "mnar_unknown_expected_resistant_n": math.nan,
                    "mnar_status": "not_evaluable_no_features",
                }
                for lambda_value in lambda_grid
            ]

        try:
            model = self._fit_mnar_resistance_model(training, feature_columns)
            observed_scale_prob = pd.Series(
                model.predict_proba(subset.loc[:, feature_columns])[:, 1],
                index=subset.index,
            ).clip(1e-6, 1.0 - 1e-6)
        except Exception:
            return [
                {
                    **base,
                    "mnar_lambda": lambda_value,
                    "mnar_prevalence": math.nan,
                    "mnar_prevalence_pct": math.nan,
                    "mnar_shift_from_naive": math.nan,
                    "mnar_shift_from_naive_pct": math.nan,
                    "mnar_unknown_expected_resistant_n": math.nan,
                    "mnar_status": "model_failed",
                }
                for lambda_value in lambda_grid
            ]

        logit_observed_scale = self._logit(observed_scale_prob)
        rows: list[dict[str, object]] = []
        for lambda_value in lambda_grid:
            unknown_prob = self._expit(logit_observed_scale.loc[unknown_mask] - lambda_value)
            unknown_expected = float(unknown_prob.sum()) if unknown_n > 0 else 0.0
            mnar_prevalence = float((resistant_n + unknown_expected) / eligible_n) if eligible_n > 0 else math.nan
            mnar_shift = (
                float(naive_prevalence - mnar_prevalence)
                if pd.notna(naive_prevalence) and pd.notna(mnar_prevalence)
                else math.nan
            )
            rows.append(
                {
                    **base,
                    "mnar_lambda": lambda_value,
                    "mnar_prevalence": mnar_prevalence,
                    "mnar_prevalence_pct": mnar_prevalence * 100.0 if pd.notna(mnar_prevalence) else math.nan,
                    "mnar_shift_from_naive": mnar_shift,
                    "mnar_shift_from_naive_pct": mnar_shift * 100.0 if pd.notna(mnar_shift) else math.nan,
                    "mnar_unknown_expected_resistant_n": unknown_expected,
                    "mnar_status": "estimated",
                }
            )
        return rows

    @staticmethod
    def _attach_mnar_reference(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
        reference = next(
            (row for row in rows if math.isclose(float(row["mnar_lambda"]), 0.0, rel_tol=0.0, abs_tol=1e-12)),
            None,
        )
        if reference is None:
            summary["mnar_lambda0_prevalence"] = math.nan
            summary["mnar_lambda0_prevalence_pct"] = math.nan
            summary["mnar_lambda0_shift_from_naive"] = math.nan
            summary["mnar_lambda0_shift_from_naive_pct"] = math.nan
            summary["mnar_lambda0_absolute_shift_pct"] = math.nan
            summary["mnar_status_lambda0"] = "not_evaluable"
            summary["mnar_feature_count"] = 0
            summary["mnar_feature_columns"] = ""
            return
        shift_pct = reference.get("mnar_shift_from_naive_pct", math.nan)
        summary["mnar_lambda0_prevalence"] = reference.get("mnar_prevalence", math.nan)
        summary["mnar_lambda0_prevalence_pct"] = reference.get("mnar_prevalence_pct", math.nan)
        summary["mnar_lambda0_shift_from_naive"] = reference.get("mnar_shift_from_naive", math.nan)
        summary["mnar_lambda0_shift_from_naive_pct"] = shift_pct
        summary["mnar_lambda0_absolute_shift_pct"] = abs(float(shift_pct)) if pd.notna(shift_pct) else math.nan
        summary["mnar_status_lambda0"] = reference.get("mnar_status", "unavailable")
        summary["mnar_feature_count"] = reference.get("mnar_feature_count", 0)
        summary["mnar_feature_columns"] = reference.get("mnar_feature_columns", "")

    def _build_edge_enrichment_results(
        self,
        analysis_frame: pd.DataFrame,
        culture_drug_episodes: pd.DataFrame,
        cascade_edge_report: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if cascade_edge_report is None or cascade_edge_report.empty or analysis_frame.empty or culture_drug_episodes.empty:
            return self._empty_edge_enrichment_results()
        validated_edges = self._validated_escalation_edges(
            cascade_edge_report,
            columns=["upstream_antibiotic", "downstream_antibiotic", "validation_status"],
        )
        if validated_edges.empty:
            return self._empty_edge_enrichment_results()

        episode_keys = list(self._settings.gold.episode_key_columns)
        organism_merge_keys = episode_keys if "organism" in episode_keys else episode_keys + ["organism"]
        base = analysis_frame.loc[
            :,
            episode_keys + ["organism", "antibiotic", "is_tested"],
        ].rename(columns={"antibiotic": "downstream_antibiotic"})
        upstream_keys = organism_merge_keys + ["antibiotic"]
        upstream_columns = upstream_keys + ["susceptibility"]
        observed_upstream = self._collapse_observed_results(
            culture_drug_episodes.loc[:, upstream_columns].copy(),
            upstream_keys,
        )
        observed_upstream = observed_upstream.loc[
            observed_upstream["susceptibility"].isin([self._settings.cascade.upstream_result_positive, self._EVALUABLE_NEGATIVE])
        ].rename(columns={"antibiotic": "upstream_antibiotic"})
        observed_upstream["upstream_positive"] = observed_upstream["susceptibility"].eq(
            self._settings.cascade.upstream_result_positive
        ).astype(int)
        observed_upstream = observed_upstream.drop(columns=["susceptibility"])
        base = base.loc[:, ~base.columns.duplicated()].copy()
        observed_upstream = observed_upstream.loc[:, ~observed_upstream.columns.duplicated()].copy()
        merged = base.merge(observed_upstream, on=organism_merge_keys, how="left")
        merged = merged.merge(validated_edges, on=["upstream_antibiotic", "downstream_antibiotic"], how="inner")
        merged = merged[merged["upstream_antibiotic"] != merged["downstream_antibiotic"]].copy()
        if merged.empty:
            return self._empty_edge_enrichment_results()

        rows: list[dict[str, object]] = []
        for (organism, upstream, downstream), subset in merged.groupby(
            ["organism", "upstream_antibiotic", "downstream_antibiotic"],
            dropna=False,
            observed=True,
        ):
            upstream_observed = subset["upstream_positive"].notna()
            if not upstream_observed.any():
                continue
            denominator = subset.loc[upstream_observed]
            tested_subset = denominator.loc[denominator["is_tested"].eq(1)]
            if denominator.empty or tested_subset.empty:
                continue
            p_tested = float(tested_subset["upstream_positive"].mean())
            p_all = float(denominator["upstream_positive"].mean())
            rows.append(
                {
                    "organism": organism,
                    "upstream_antibiotic": upstream,
                    "downstream_antibiotic": downstream,
                    "upstream_observed_eligible_n": int(len(denominator)),
                    "downstream_tested_n": int(len(tested_subset)),
                    "upstream_positive_rate_when_downstream_tested": p_tested,
                    "upstream_positive_rate_when_downstream_tested_pct": p_tested * 100.0,
                    "upstream_positive_rate_all_eligible": p_all,
                    "upstream_positive_rate_all_eligible_pct": p_all * 100.0,
                    "upstream_resistance_enrichment": p_tested - p_all,
                    "upstream_resistance_enrichment_pct": (p_tested - p_all) * 100.0,
                }
            )
        return pd.DataFrame(rows, columns=self._EDGE_ENRICHMENT_COLUMNS)

    def _validated_escalation_edges(
        self,
        cascade_edge_report: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """Return robust/supported escalation edges for prevalence-shift analyses.

        The prevalence bridge is resistance-triggered: it asks how downstream
        testing after upstream resistance changes eligible-denominator
        prevalence summaries. Suppression-direction edges are valid cascade
        observations, but they are not used as resistance-triggered prevalence
        anchors in the primary analysis.
        """
        required = {"upstream_antibiotic", "downstream_antibiotic", "validation_status"}
        if not required.issubset(cascade_edge_report.columns):
            return pd.DataFrame()

        validation_status = cascade_edge_report["validation_status"].isin(self._VALIDATED_STATUSES)
        if "cascade_direction" in cascade_edge_report.columns:
            direction = cascade_edge_report["cascade_direction"].eq("escalation")
        elif "escalation_ratio" in cascade_edge_report.columns:
            direction = pd.to_numeric(cascade_edge_report["escalation_ratio"], errors="coerce").ge(1.0)
        else:
            return pd.DataFrame()

        selected_columns = [
            column
            for column in dict.fromkeys(columns)
            if column in cascade_edge_report.columns
        ]
        return cascade_edge_report.loc[validation_status & direction, selected_columns].drop_duplicates()

    def _validated_downstream_antibiotics(
        self,
        cascade_edge_report: pd.DataFrame | None,
    ) -> frozenset[str]:
        """Return downstream drugs from robust/supported escalation edges.

        Primary prevalence-shift analysis is a downstream-consequence module:
        it asks how validated resistance-triggered cascades affect the drugs
        that become visible downstream. Drugs with no validated escalation edge
        are outside this primary prevalence universe.
        """
        if cascade_edge_report is None or cascade_edge_report.empty:
            return frozenset()
        edge_report = cascade_edge_report.copy()
        if "upstream_antibiotic" in edge_report.columns:
            edge_report["upstream_antibiotic"] = edge_report["upstream_antibiotic"].map(normalize_antibiotic_label)
        if "downstream_antibiotic" in edge_report.columns:
            edge_report["downstream_antibiotic"] = edge_report["downstream_antibiotic"].map(normalize_antibiotic_label)
        validated_edges = self._validated_escalation_edges(edge_report, columns=["downstream_antibiotic"])
        if validated_edges.empty or "downstream_antibiotic" not in validated_edges.columns:
            return frozenset()
        return frozenset(validated_edges["downstream_antibiotic"].dropna().astype(str))

    def _selection_summary(
        self,
        group_a: pd.DataFrame,
        group_b: pd.DataFrame,
        covariate_columns: tuple[str, ...],
    ) -> dict[str, float]:
        if group_a.empty or group_b.empty:
            return {"mean_abs_smd": math.nan, "max_abs_smd": math.nan, "flagged_covariate_n": 0}
        smds: list[float] = []
        for column in covariate_columns:
            smd = self._column_abs_smd(group_a[column], group_b[column])
            if pd.notna(smd):
                smds.append(float(smd))
        if not smds:
            return {"mean_abs_smd": math.nan, "max_abs_smd": math.nan, "flagged_covariate_n": 0}
        return {
            "mean_abs_smd": float(sum(smds) / len(smds)),
            "max_abs_smd": float(max(smds)),
            "flagged_covariate_n": int(sum(value >= 0.1 for value in smds)),
        }

    def _column_abs_smd(self, series_a: pd.Series, series_b: pd.Series) -> float:
        if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b):
            return self._numeric_abs_smd(pd.to_numeric(series_a, errors="coerce"), pd.to_numeric(series_b, errors="coerce"))

        categorical_a = series_a.fillna(self._CATEGORY_MISSING).astype(str)
        categorical_b = series_b.fillna(self._CATEGORY_MISSING).astype(str)
        levels = sorted(set(categorical_a.unique()).union(categorical_b.unique()))
        if not levels:
            return math.nan
        smds: list[float] = []
        for level in levels:
            indicator_a = categorical_a.eq(level).astype(float)
            indicator_b = categorical_b.eq(level).astype(float)
            smd = self._numeric_abs_smd(indicator_a, indicator_b)
            if pd.notna(smd):
                smds.append(float(smd))
        return float(max(smds)) if smds else math.nan

    @staticmethod
    def _numeric_abs_smd(series_a: pd.Series, series_b: pd.Series) -> float:
        a = series_a.dropna().astype(float)
        b = series_b.dropna().astype(float)
        if a.empty or b.empty:
            return math.nan
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
        var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
        pooled_sd = math.sqrt(max((var_a + var_b) / 2.0, 0.0))
        if pooled_sd > 0:
            return abs(mean_a - mean_b) / pooled_sd
        return abs(mean_a - mean_b)

    def _candidate_covariate_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        if self._covariate_columns_cache is not None:
            return self._covariate_columns_cache
        columns = tuple(
            column
            for column in self._PRIMARY_MNAR_COVARIATES
            if column in frame.columns and column not in self._TIMING_LIMITED_ACUITY_COVARIATES
        )
        component_columns = tuple(
            sorted(
                column
                for column in frame.columns
                if (column.startswith("comorb_") or column.startswith("proc_"))
                and column not in self._TIMING_LIMITED_ACUITY_COVARIATES
            )
        )
        columns = tuple(dict.fromkeys(columns + component_columns))
        self._covariate_columns_cache = columns
        return columns

    def _mnar_feature_columns(
        self,
        frame: pd.DataFrame,
        covariate_columns: tuple[str, ...],
    ) -> tuple[str, ...]:
        candidate_columns = tuple(
            dict.fromkeys(
                tuple(covariate_columns)
                + self._MNAR_CASCADE_FEATURES
            )
        )
        return tuple(
            column
            for column in candidate_columns
            if column in frame.columns and column not in self._TIMING_LIMITED_ACUITY_COVARIATES
        )

    def _fit_mnar_resistance_model(
        self,
        training: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> Pipeline:
        categorical_columns = tuple(
            column for column in feature_columns if not pd.api.types.is_numeric_dtype(training[column])
        )
        numeric_columns = tuple(column for column in feature_columns if column not in categorical_columns)
        preprocess = build_tabular_preprocessor(categorical_columns, numeric_columns)
        # No class_weight here: this model's output IS the prevalence estimate, not a
        # classification decision. An unweighted MLE logistic fit is calibrated to the
        # training base rate by construction (mean predicted probability == observed
        # positive rate); class_weight="balanced" deliberately breaks that property to
        # trade calibration for classification recall on the minority class, which is
        # the wrong trade for a probability we're about to report as "prevalence."
        model = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=self._settings.modeling.random_state,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("scale", MaxAbsScaler()),
                ("model", model),
            ]
        )
        pipeline.fit(training.loc[:, feature_columns], training["is_resistant"].astype(int))
        return pipeline

    @staticmethod
    def _collapse_observed_results(observed: pd.DataFrame, pair_keys: list[str]) -> pd.DataFrame:
        if observed.empty:
            return observed
        grouped = (
            observed.groupby(pair_keys, dropna=False, observed=True)["susceptibility"]
            .agg(lambda values: PrevalenceShiftAnalyzer._collapse_susceptibility_values(values.dropna().astype(str).tolist()))
            .reset_index()
        )
        return grouped

    @staticmethod
    def _collapse_susceptibility_values(values: list[str]) -> str | None:
        unique = sorted(set(values))
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        evaluable = [value for value in unique if value in {PrevalenceShiftAnalyzer._POSITIVE_RESULT, PrevalenceShiftAnalyzer._EVALUABLE_NEGATIVE}]
        if len(set(evaluable)) == 1 and evaluable:
            return evaluable[0]
        return "CONFLICTING_RESULT"

    @staticmethod
    def _project_delta_to_admissible(delta: float | int | None, delta_max: float | int | None) -> float:
        if pd.isna(delta) or pd.isna(delta_max):
            return math.nan
        return float(min(max(float(delta), 0.0), float(delta_max)))

    @staticmethod
    def _standardised_prevalence(
        *,
        cascade_prevalence: float,
        independent_prevalence: float,
        w_casc_eligible: float,
        w_ind_eligible: float,
    ) -> float:
        if pd.isna(w_casc_eligible) or pd.isna(w_ind_eligible):
            return math.nan
        total = 0.0
        if w_casc_eligible > 0:
            if pd.isna(cascade_prevalence):
                return math.nan
            total += w_casc_eligible * cascade_prevalence
        if w_ind_eligible > 0:
            if pd.isna(independent_prevalence):
                return math.nan
            total += w_ind_eligible * independent_prevalence
        return float(total)

    @staticmethod
    def _shifted_prevalence(
        naive_prevalence: float,
        unknown_binary_outcome_n: int,
        eligible_n: int,
        delta: float | int | None,
    ) -> float:
        if eligible_n <= 0 or pd.isna(naive_prevalence) or pd.isna(delta):
            return math.nan
        observed_resistant = naive_prevalence * (eligible_n - unknown_binary_outcome_n)
        unknown_resistant = float(delta) * naive_prevalence * unknown_binary_outcome_n
        shifted = (observed_resistant + unknown_resistant) / eligible_n
        if shifted < 0:
            return math.nan
        return float(min(shifted, 1.0))

    @staticmethod
    def _logit(probability: pd.Series) -> pd.Series:
        clipped = probability.clip(1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _expit(value: pd.Series) -> pd.Series:
        return pd.Series(1.0 / (1.0 + np.exp(-value)), index=value.index)

    @staticmethod
    def _shift_direction(prevalence_shift: float | int | None) -> str:
        if pd.isna(prevalence_shift):
            return "unavailable"
        if prevalence_shift > 0:
            return "naive_overestimates"
        if prevalence_shift < 0:
            return "naive_underestimates"
        return "no_difference"
