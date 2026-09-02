"""Adjusted downstream-testing regression."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from amr_cascade_platform.core.statistics.e_values import (
    e_value_from_odds_ratio,
    e_value_from_odds_ratio_interval,
)
from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager


logger = logging.getLogger(__name__)


class DownstreamTestingRegression:
    """Fit per-edge adjusted logistic regression models for downstream testing."""

    # These are materialized by CascadeCovariateBuilder but excluded from the
    # primary adjusted model because the ARMD lab/vital extracts do not provide
    # measurement-level timestamps proving that every summarized value predates
    # the culture order. They are suitable for timing-limited sensitivity or
    # descriptive acuity diagnostics, not primary leakage-safe adjustment.
    _TIMING_LIMITED_ACUITY_COVARIATES = (
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
    )
    _ADJUSTMENT_COVARIATES = (
        "organism",
        "source_site",
        "cov_ordering_mode",
        "cov_specimen_type",
        "cov_icu_status",
        "cov_er_status",
        "cov_prior_abx_any_90d",
        "cov_calendar_year",
        "cov_calendar_month",
        "cov_age_bin",
        "cov_sex",
        "cov_prior_same_organism_any_90d",
        "cov_comorbidity_count",
        "cov_icu_available",
        "cov_er_available",
        "cov_prior_abx_available",
        "cov_prior_organism_available",
        "cov_comorbidity_available",
        "cov_adi_score",
        "cov_adi_state_rank",
        "cov_adi_available",
        "cov_nursing_home_90d",
        "cov_nursing_home_available",
        "cov_prior_procedure_90d",
        "cov_prior_procedure_available",
    )
    _CLUSTER_CANDIDATES = (
        "anon_id",
        "pat_enc_csn_id_coded",
        "order_proc_id_coded",
    )
    _RESULT_COLUMNS = (
        "upstream_antibiotic",
        "downstream_antibiotic",
        "adjusted_log_odds",
        "adjusted_odds_ratio",
        "adjusted_standard_error",
        "adjusted_log_odds_ci_lower",
        "adjusted_log_odds_ci_upper",
        "adjusted_odds_ratio_ci_lower",
        "adjusted_odds_ratio_ci_upper",
        "adjusted_p_value",
        "quasi_separation_flagged",
        "adjusted_or_e_value",
        "adjusted_or_ci_e_value",
        "modeled_n",
        "clustered_n",
        "clustering_variable",
        "upstream_positive_rate",
        "downstream_test_rate",
        "supports_adjusted_model",
        "adjustment_model_type",
        "adjustment_covariates",
        "non_estimable_reason",
    )
    # Ridge (L2) penalty on nuisance coefficients; the upstream_positive
    # coefficient is exempt because it is the parameter of interest.
    # A value of 10.0 matches sklearn C ≈ 0.5 for the typical group sizes seen
    # in cascade pair regressions (n ≈ 50–10 000). This prevents nuisance
    # covariate separation from destabilizing the adjusted fit while remaining
    # negligible (< 0.1 % of the data gradient) for large groups.
    # The previous value of 1e-6 was effectively no regularisation and produced
    # coefficients > 100 (OR > 10⁴³) for near-separated pairs.
    _RIDGE_PENALTY: float = 10.0
    _QUASI_SEP_OR_THRESHOLD: float = 1000.0
    _MIN_COMORBIDITY_COMPONENT_BRANCH_SUPPORT: int = 5
    # Symmetric on the log-odds scale (ln(1000) ~= -ln(0.001)): quasi-complete
    # separation drives the upstream_positive coefficient to +-infinity, and an
    # OR collapsing toward 0 is the same divergence in the other direction, not
    # a genuine near-zero effect. Checking only the upper tail let these through
    # as if they were legitimate small-OR estimates -- including cases directly
    # contradicting the row's own escalation ratio (e.g. OR ~ 2e-6 alongside an
    # ER of 14), instead of being flagged non-estimable like their >1000 mirror.
    _QUASI_SEP_OR_LOWER_THRESHOLD: float = 1.0 / _QUASI_SEP_OR_THRESHOLD
    _MIN_PROCEDURE_COMPONENT_BRANCH_SUPPORT: int = 5

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._covariate_builder = CascadeCovariateBuilder(settings, path_manager)

    def analyze(
        self,
        drug_pairs: pd.DataFrame,
        escalation_results: pd.DataFrame,
        culture_episodes: pd.DataFrame,
    ) -> pd.DataFrame:
        if drug_pairs.empty or escalation_results.empty:
            return self._empty_results()

        covariates = self._covariate_builder.build(culture_episodes)
        candidate_edges = escalation_results[escalation_results["passes_support_threshold"]].loc[
            :, ["upstream_antibiotic", "downstream_antibiotic"]
        ]
        # merge() already returns a new, non-view frame -- a trailing .copy() here would
        # just duplicate the (still nearly-full-size) pooled table for no reason while
        # the caller's `drug_pairs` reference stays alive for later pipeline stages.
        frame = drug_pairs.merge(
            candidate_edges,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="inner",
        )
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame[frame["downstream_eligible"] == 1]

        frame = frame.copy()
        frame["upstream_positive"] = frame["upstream_susceptibility"].map(self._map_positive)
        frame = frame[frame["upstream_positive"].isin([0, 1])]
        if frame.empty:
            return self._empty_results()

        # Merging covariates onto the full pooled frame once (rather than once per
        # antibiotic-pair group) measured lower peak memory in practice: doing hundreds of
        # separate small merges of categorical-vs-object join keys costs more in per-call
        # merge/index-construction overhead and allocator fragmentation than one merge does,
        # even though each individual per-group merge touches less data. Cast covariates'
        # join columns to match frame's categorical dtype first so this is a clean
        # categorical-vs-categorical join rather than categorical-vs-object (which would
        # silently decay frame's dictionary-encoded columns back to object for the rest of
        # this function, undoing the memory win from DatasetStore.read_pandas).
        join_keys = list(self._settings.gold.episode_key_columns)
        covariates = covariates.copy()
        categorical_join_keys = [
            column
            for column in join_keys
            if column in covariates.columns and isinstance(frame[column].dtype, pd.CategoricalDtype)
        ]
        if categorical_join_keys:
            # covariates comes from the full culture_episodes table, which can include
            # episodes with zero eligible antibiotic pairs (excluded during gold pair
            # generation, e.g. discordant-result exclusion) -- those episode keys never
            # appear in drug_pairs' category dictionary at all. Such covariate rows could
            # never match anything in the left-join below regardless, so dropping them here
            # is a pure no-op on the merge result, not a data-loss shortcut. This must run
            # before the astype below, since astype to a fixed CategoricalDtype would
            # otherwise silently turn any out-of-category value into NaN.
            keep = pd.Series(True, index=covariates.index)
            for column in categorical_join_keys:
                keep &= covariates[column].isin(frame[column].dtype.categories)
            covariates = covariates.loc[keep]
            for column in categorical_join_keys:
                covariates[column] = covariates[column].astype(frame[column].dtype)
        frame = frame.merge(covariates, on=join_keys, how="left", validate="many_to_one")

        results: list[dict[str, object]] = []
        for (upstream_antibiotic, downstream_antibiotic), group in frame.groupby(
            ["upstream_antibiotic", "downstream_antibiotic"],
            observed=True,
        ):
            results.append(self._fit_pair_model(group, upstream_antibiotic, downstream_antibiotic))
        return pd.DataFrame(results, columns=self._RESULT_COLUMNS)

    @classmethod
    def _empty_results(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._RESULT_COLUMNS)

    def _non_estimable_result(
        self,
        upstream_antibiotic: object,
        downstream_antibiotic: object,
        reason: str,
    ) -> dict[str, object]:
        """All-NaN result row carrying *why* the adjusted OR is missing.

        Previously this case was a bare `return None`, silently dropped by the
        caller, so a pair's absence from the regression output looked
        identical whether the model diverged (quasi-separation) or was never
        attempted at all (e.g. zero variance in the outcome or exposure
        within this pair's subgroup -- empirically the dominant reason, not
        an edge case). The downstream-facing tables already assume a
        per-row reason exists to report (see paper1 Supplementary Table S6);
        this makes that reason a real, auditable field instead of something
        inferred after the fact from which other columns happen to be null.
        """
        return {
            "upstream_antibiotic": upstream_antibiotic,
            "downstream_antibiotic": downstream_antibiotic,
            "adjusted_log_odds": math.nan,
            "adjusted_odds_ratio": math.nan,
            "adjusted_standard_error": math.nan,
            "adjusted_log_odds_ci_lower": math.nan,
            "adjusted_log_odds_ci_upper": math.nan,
            "adjusted_odds_ratio_ci_lower": math.nan,
            "adjusted_odds_ratio_ci_upper": math.nan,
            "adjusted_p_value": math.nan,
            "quasi_separation_flagged": reason == "quasi_separation",
            "adjusted_or_e_value": math.nan,
            "adjusted_or_ci_e_value": math.nan,
            "modeled_n": math.nan,
            "clustered_n": math.nan,
            "clustering_variable": None,
            "upstream_positive_rate": math.nan,
            "downstream_test_rate": math.nan,
            "supports_adjusted_model": False,
            "adjustment_model_type": None,
            "adjustment_covariates": None,
            "non_estimable_reason": reason,
        }

    def _fit_pair_model(
        self,
        group: pd.DataFrame,
        upstream_antibiotic: object,
        downstream_antibiotic: object,
    ) -> dict[str, object]:
        if group["downstream_tested"].nunique() < 2 or group["upstream_positive"].nunique() < 2:
            # The regression cannot be specified at all here, which is a
            # stricter condition than classic quasi-separation: there is no
            # variation in the outcome (downstream_tested) or the exposure
            # (upstream_positive) for this pair to fit a coefficient to in
            # the first place. In local verification this accounted for
            # ~99% of non-estimable pairs -- far more common than a model
            # that fits and then diverges.
            return self._non_estimable_result(
                upstream_antibiotic, downstream_antibiotic, "zero_variance_outcome_or_exposure"
            )

        cluster_ids = self._resolve_cluster_ids(group)
        design_matrix = self._build_design_matrix(group, include_upstream_positive=True)
        if design_matrix is None or "upstream_positive" not in design_matrix.columns:
            return self._non_estimable_result(
                upstream_antibiotic, downstream_antibiotic, "design_matrix_unavailable"
            )
        target = group["downstream_tested"].astype(int)
        event_n = int(target.sum())
        nonevent_n = int((1 - target).sum())
        parameter_n = int(design_matrix.shape[1])
        if event_n < parameter_n or nonevent_n < parameter_n:
            return self._non_estimable_result(
                upstream_antibiotic,
                downstream_antibiotic,
                "outcome_events_below_model_dimension",
            )

        fit = self._fit_clustered_logit(design_matrix, target, cluster_ids)
        if fit is None:
            return self._non_estimable_result(
                upstream_antibiotic, downstream_antibiotic, "model_fit_failed"
            )

        coefficient = fit["coefficients"].get("upstream_positive", math.nan)
        standard_error = fit["standard_errors"].get("upstream_positive", math.nan)
        ci_lower = fit["coefficient_ci_lower"].get("upstream_positive", math.nan)
        ci_upper = fit["coefficient_ci_upper"].get("upstream_positive", math.nan)
        p_value = fit["p_values"].get("upstream_positive", math.nan)
        adjusted_or = self._safe_exp(coefficient) if pd.notna(coefficient) else math.nan
        adjusted_or_ci_lower = self._safe_exp(ci_lower) if pd.notna(ci_lower) else math.nan
        adjusted_or_ci_upper = self._safe_exp(ci_upper) if pd.notna(ci_upper) else math.nan
        quasi_separation = pd.notna(adjusted_or) and (
            adjusted_or > self._QUASI_SEP_OR_THRESHOLD
            or adjusted_or < self._QUASI_SEP_OR_LOWER_THRESHOLD
        )
        if quasi_separation:
            adjusted_or = math.nan
            adjusted_or_ci_lower = math.nan
            adjusted_or_ci_upper = math.nan
        # Even when the point-estimate OR is fine, the CI bounds can be inf
        # if the sandwich standard error is enormous (e.g. < 5 distinct clusters).
        # Keep CI bounds as a matched pair: null both if either is non-finite.
        elif math.isinf(adjusted_or_ci_lower) or math.isinf(adjusted_or_ci_upper):
            adjusted_or_ci_lower = math.nan
            adjusted_or_ci_upper = math.nan
        return {
            "upstream_antibiotic": upstream_antibiotic,
            "downstream_antibiotic": downstream_antibiotic,
            "adjusted_log_odds": coefficient,
            "adjusted_odds_ratio": adjusted_or,
            "adjusted_standard_error": standard_error,
            "adjusted_log_odds_ci_lower": ci_lower,
            "adjusted_log_odds_ci_upper": ci_upper,
            "adjusted_odds_ratio_ci_lower": adjusted_or_ci_lower,
            "adjusted_odds_ratio_ci_upper": adjusted_or_ci_upper,
            "adjusted_p_value": p_value,
            "quasi_separation_flagged": quasi_separation,
            "adjusted_or_e_value": e_value_from_odds_ratio(adjusted_or),
            "adjusted_or_ci_e_value": e_value_from_odds_ratio_interval(adjusted_or_ci_lower, adjusted_or_ci_upper),
            "modeled_n": len(group),
            "clustered_n": int(cluster_ids.nunique()),
            "clustering_variable": fit["clustering_variable"],
            "upstream_positive_rate": float(group["upstream_positive"].mean()),
            "downstream_test_rate": float(group["downstream_tested"].mean()),
            "supports_adjusted_model": True,
            "adjustment_model_type": fit["adjustment_model_type"],
            "adjustment_covariates": ", ".join(self._model_covariates_for_frame(group)),
            "non_estimable_reason": "quasi_separation" if quasi_separation else None,
        }

    def _fit_clustered_logit(
        self,
        design_matrix: pd.DataFrame,
        target: pd.Series,
        cluster_ids: pd.Series,
        model_type: str = "clustered_logistic_regression",
    ) -> dict[str, object] | None:
        X = design_matrix.to_numpy(dtype=float)
        y = target.to_numpy(dtype=float)
        weights_array = np.ones(len(target), dtype=float)
        feature_names = list(design_matrix.columns)
        ridge = self._RIDGE_PENALTY

        # Penalised regression with parameter-of-interest exemption
        # (motivation: Belloni, Chernozhukov & Hansen 2014, ReStud).
        # upstream_positive is the parameter of interest; exempting it from the ridge
        # penalty ensures the adjusted OR reflects the true association rather than
        # shrinkage toward the null.  All other non-intercept covariates remain penalised
        # at _RIDGE_PENALTY to guard against quasi-complete separation on sparse edges.
        _ridge_mask = np.ones(X.shape[1], dtype=bool)
        _ridge_mask[0] = False  # intercept (_intercept column)
        _poi_idx = feature_names.index("upstream_positive") if "upstream_positive" in feature_names else None
        if _poi_idx is not None:
            _ridge_mask[_poi_idx] = False  # parameter of interest: exempt from penalty

        def objective(beta: np.ndarray) -> float:
            eta = X @ beta
            probability = np.clip(expit(eta), 1e-8, 1.0 - 1e-8)
            penalty = ridge * float(np.square(beta[_ridge_mask]).sum())
            log_likelihood = weights_array * (y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
            return float(-log_likelihood.sum() + penalty)

        def gradient(beta: np.ndarray) -> np.ndarray:
            probability = expit(X @ beta)
            grad = X.T @ (weights_array * (probability - y))
            grad[_ridge_mask] += 2.0 * ridge * beta[_ridge_mask]
            return grad

        optimization = minimize(
            objective,
            x0=np.zeros(X.shape[1], dtype=float),
            jac=gradient,
            method="BFGS",
        )
        if not optimization.success:
            logger.warning(
                "BFGS did not converge (message=%r, nit=%d, nfev=%d): pair dropped from adjusted-OR output.",
                optimization.message, optimization.nit, optimization.nfev,
            )
            return None

        beta = optimization.x
        probability = np.clip(expit(X @ beta), 1e-8, 1.0 - 1e-8)
        model_weights = weights_array * probability * (1.0 - probability)
        hessian = X.T @ (X * model_weights[:, None])
        # Penalty matrix for the bread must match _ridge_mask so that the
        # sandwich SE for upstream_positive uses the same regularisation level
        # (zero) as the BFGS objective.  Using a flat non-intercept penalty here
        # would over-regularise the upstream_positive standard error, producing
        # CIs that are slightly too conservative for the parameter of interest.
        penalty = np.diag(np.where(_ridge_mask, 2.0 * ridge, 0.0))
        bread_inv = hessian + penalty
        bread = np.linalg.pinv(bread_inv)

        residual = weights_array * (y - probability)
        meat = np.zeros_like(bread)
        cluster_frame = pd.DataFrame({"cluster_id": cluster_ids.astype(str), "row": np.arange(len(cluster_ids))})
        for _, cluster_rows in cluster_frame.groupby("cluster_id", sort=False, observed=True):
            row_index = cluster_rows["row"].to_numpy(dtype=int)
            score = X[row_index].T @ residual[row_index]
            meat += np.outer(score, score)

        cluster_n = int(cluster_ids.nunique())
        sample_n = len(target)
        parameter_n = X.shape[1]
        if cluster_n > 1 and sample_n > parameter_n:
            small_sample = (cluster_n / (cluster_n - 1.0)) * ((sample_n - 1.0) / (sample_n - parameter_n))
            meat *= small_sample
        covariance = bread @ meat @ bread
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        z_scores = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
        p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))
        ci_lower = beta - 1.96 * standard_errors
        ci_upper = beta + 1.96 * standard_errors

        coefficient_map = dict(zip(feature_names, beta, strict=False))
        se_map = dict(zip(feature_names, standard_errors, strict=False))
        p_value_map = dict(zip(feature_names, p_values, strict=False))
        ci_lower_map = dict(zip(feature_names, ci_lower, strict=False))
        ci_upper_map = dict(zip(feature_names, ci_upper, strict=False))
        return {
            "coefficients": coefficient_map,
            "standard_errors": se_map,
            "p_values": p_value_map,
            "coefficient_ci_lower": ci_lower_map,
            "coefficient_ci_upper": ci_upper_map,
            "clustering_variable": self._select_cluster_variable(cluster_ids),
            "adjustment_model_type": model_type,
        }

    @staticmethod
    def _safe_exp(x: float) -> float:
        """Return exp(x) without raising OverflowError.

        For very large coefficients (|x| > 709) Python's math.exp raises
        OverflowError and scipy BFGS can occasionally produce such values for
        near-separated groups even with a ridge penalty.  This wrapper returns
        math.inf for any x that would overflow so the downstream quasi-separation
        check (OR > _QUASI_SEP_OR_THRESHOLD) fires correctly and the result is
        nulled, rather than crashing the whole cascade analysis.
        """
        try:
            if not math.isfinite(x):
                return math.nan if math.isnan(x) else math.inf
            return math.exp(x)
        except OverflowError:
            return math.inf

    def _build_design_matrix(self, frame: pd.DataFrame, *, include_upstream_positive: bool) -> pd.DataFrame | None:
        columns = list(self._model_covariates_for_frame(frame))
        if not columns and not include_upstream_positive:
            return None
        if not columns:
            columns = []
        if include_upstream_positive:
            columns = ["upstream_positive"] + columns
        design = frame.loc[:, columns].copy()
        categorical_columns = [
            "organism",
            "source_site",
            "cov_ordering_mode",
            "cov_specimen_type",
            "cov_age_bin",
            "cov_sex",
        ]
        categorical_columns = [column for column in categorical_columns if column in design.columns]
        for column in categorical_columns:
            # astype(object) first: if the source column is pandas Categorical dtype
            # (organism/source_site are, post dictionary-encoding), fillna() with a value
            # outside the existing category set raises TypeError. Casting to object drops
            # the Categorical constraint while preserving actual NaN/None as null (unlike
            # astype(str), which would stringify NaN into the literal text "nan").
            design[column] = design[column].astype(object).fillna("unknown").astype(str)
        numeric_columns = [column for column in design.columns if column not in categorical_columns]
        for column in numeric_columns:
            design[column] = pd.to_numeric(design[column], errors="coerce").fillna(0.0)
        design = pd.get_dummies(design, columns=categorical_columns, drop_first=True)
        constant_columns = [column for column in design.columns if design[column].nunique(dropna=False) <= 1]
        protected = {"upstream_positive"} if include_upstream_positive else set()
        constant_columns = [column for column in constant_columns if column not in protected]
        if constant_columns:
            design = design.drop(columns=constant_columns)
        if design.shape[1] == 0:
            return None
        design_matrix = design.astype(float)
        if include_upstream_positive and "upstream_positive" not in design_matrix.columns:
            return None
        design_matrix.insert(0, "_intercept", 1.0)
        design_matrix = self._drop_linearly_dependent_columns(
            design_matrix,
            protected=("_intercept", "upstream_positive") if include_upstream_positive else ("_intercept",),
        )
        return self._standardize_nuisance_columns(
            design_matrix,
            protected=("_intercept", "upstream_positive") if include_upstream_positive else ("_intercept",),
        )

    @staticmethod
    def _drop_linearly_dependent_columns(
        design_matrix: pd.DataFrame,
        *,
        protected: tuple[str, ...],
    ) -> pd.DataFrame:
        """Remove redundant nuisance columns before logistic fitting.

        Availability flags are often nested within site in the ARMD extracts. Leaving
        those exact linear dependencies in the design does not add adjustment
        information, but it does destabilize optimization and sandwich variance
        calculations. Protected columns are evaluated first so the intercept and
        upstream result are retained whenever they contribute rank.
        """
        selected: list[str] = []
        current_rank = 0
        ordered_columns = [
            column for column in protected if column in design_matrix.columns
        ] + [
            column for column in design_matrix.columns if column not in set(protected)
        ]
        for column in ordered_columns:
            candidate_columns = selected + [column]
            candidate = design_matrix.loc[:, candidate_columns].to_numpy(dtype=float)
            candidate_rank = int(np.linalg.matrix_rank(candidate))
            if candidate_rank > current_rank:
                selected.append(column)
                current_rank = candidate_rank
        return design_matrix.loc[:, selected]

    @staticmethod
    def _standardize_nuisance_columns(
        design_matrix: pd.DataFrame,
        *,
        protected: tuple[str, ...],
    ) -> pd.DataFrame:
        """Z-scale nuisance columns so ridge penalty and BFGS are well conditioned."""
        scaled = design_matrix.copy()
        protected_set = set(protected)
        for column in scaled.columns:
            if column in protected_set:
                continue
            values = scaled[column].astype(float)
            std = float(values.std(ddof=0))
            if std > 0 and math.isfinite(std):
                scaled[column] = (values - float(values.mean())) / std
        return scaled

    def _adjustment_covariates_for_frame(self, frame: pd.DataFrame) -> tuple[str, ...]:
        columns = list(self._ADJUSTMENT_COVARIATES)
        # Individual comorbidity components are allowed into the primary adjusted
        # model only when they have enough within-pair support. This preserves the
        # clinical detail of active diagnosis domains without forcing rare binary
        # flags into every sparse pair-specific regression.
        component_columns = sorted(column for column in frame.columns if column.startswith("comorb_"))
        for column in component_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
            positive_n = int(values.eq(1).sum())
            negative_n = int(values.eq(0).sum())
            if (
                positive_n >= self._MIN_COMORBIDITY_COMPONENT_BRANCH_SUPPORT
                and negative_n >= self._MIN_COMORBIDITY_COMPONENT_BRANCH_SUPPORT
                and column not in columns
            ):
                columns.append(column)
        procedure_columns = sorted(column for column in frame.columns if column.startswith("proc_"))
        for column in procedure_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
            positive_n = int(values.eq(1).sum())
            negative_n = int(values.eq(0).sum())
            if (
                positive_n >= self._MIN_PROCEDURE_COMPONENT_BRANCH_SUPPORT
                and negative_n >= self._MIN_PROCEDURE_COMPONENT_BRANCH_SUPPORT
                and column not in columns
            ):
                columns.append(column)
        if self._settings.cascade.include_acuity_covariates:
            columns.extend(column for column in self._TIMING_LIMITED_ACUITY_COVARIATES if column not in columns)
        return tuple(column for column in columns if column in frame.columns)

    def _model_covariates_for_frame(self, frame: pd.DataFrame) -> tuple[str, ...]:
        columns = list(self._adjustment_covariates_for_frame(frame))
        if "organism" in columns and frame["organism"].nunique(dropna=False) <= 1:
            columns.remove("organism")
        return tuple(columns)

    def _resolve_cluster_ids(self, frame: pd.DataFrame) -> pd.Series:
        for column in self._CLUSTER_CANDIDATES:
            if column in frame.columns:
                values = frame[column].astype(str)
                if values.nunique(dropna=False) >= 2:
                    values.name = column
                    return values
        fallback = frame.loc[:, list(self._settings.gold.episode_key_columns)].astype(str).agg("||".join, axis=1)
        fallback.name = "episode_key"
        return fallback

    @staticmethod
    def _select_cluster_variable(cluster_ids: pd.Series) -> str:
        return str(cluster_ids.name) if cluster_ids.name is not None else "episode_key"

    def _map_positive(self, value: object) -> int | None:
        if pd.isna(value):
            return None
        if value == self._settings.cascade.upstream_result_positive:
            return 1
        if value == self._settings.cascade.upstream_result_negative:
            return 0
        return None
