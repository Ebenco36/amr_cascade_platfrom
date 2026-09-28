"""SHAP attribution for adjusted downstream-testing models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from amr_cascade_platform.core.config.config_models import CascadeExplainabilityConfig


@dataclass(frozen=True)
class ShapExplanationArtifacts:
    """Auditable SHAP outputs for one fitted antibiotic-pair model."""

    global_importance: pd.DataFrame
    sampled_values: pd.DataFrame
    metadata: pd.DataFrame
    # Attribution by feature domain: per explained row the domain's SHAP values
    # are summed (additivity), then averaged -- see _domain_table.
    domain_importance: pd.DataFrame


class DownstreamTestingShapExplainer:
    """Explain a fitted linear-logistic model on its modeled log-odds scale.

    The adjusted model is fitted to an encoded and standardized design matrix.
    SHAP therefore explains that exact fitted model rather than a surrogate fit.
    An independent masker gives interventional linear SHAP values relative to a
    deterministic background sample. These are model attributions, not causal
    effects and not tests of confounding.
    """

    _GLOBAL_COLUMNS = (
        "upstream_antibiotic",
        "downstream_antibiotic",
        "feature",
        "feature_domain",
        "mean_absolute_shap",
        "mean_signed_shap",
        "positive_attribution_fraction",
        "importance_rank",
        "explained_n",
        "output_scale",
    )
    _DOMAIN_COLUMNS = (
        "upstream_antibiotic",
        "downstream_antibiotic",
        "feature_domain",
        "feature_n",
        "mean_absolute_shap",
        "mean_signed_shap",
        "explained_n",
        "output_scale",
    )
    _LOCAL_COLUMNS = (
        "upstream_antibiotic",
        "downstream_antibiotic",
        "sample_row",
        "model_row_position",
        "feature",
        "feature_domain",
        "feature_value",
        "shap_value",
        "absolute_shap_value",
        "output_scale",
    )
    _METADATA_COLUMNS = (
        "upstream_antibiotic",
        "downstream_antibiotic",
        "explainer",
        "dependence_assumption",
        "output_scale",
        "model_rows",
        "background_n",
        "explained_n",
        "feature_n",
        "expected_value",
        "maximum_additivity_error",
        "additivity_tolerance",
        "additivity_passed",
        "random_seed",
        "design_transform",
        "interpretation_boundary",
    )

    def __init__(self, config: CascadeExplainabilityConfig) -> None:
        self._config = config

    @classmethod
    def empty_global_importance(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._GLOBAL_COLUMNS)

    @classmethod
    def empty_sampled_values(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._LOCAL_COLUMNS)

    @classmethod
    def empty_metadata(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._METADATA_COLUMNS)

    @classmethod
    def empty_domain_importance(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._DOMAIN_COLUMNS)

    def explain(
        self,
        *,
        design_matrix: pd.DataFrame,
        coefficients: dict[str, float],
        upstream_antibiotic: object,
        downstream_antibiotic: object,
    ) -> ShapExplanationArtifacts:
        if not self._config.enabled:
            return ShapExplanationArtifacts(
                self.empty_global_importance(),
                self.empty_sampled_values(),
                self.empty_metadata(),
                self.empty_domain_importance(),
            )
        if "_intercept" not in design_matrix.columns:
            raise ValueError("SHAP requires the fitted design matrix to contain _intercept.")

        feature_frame = design_matrix.drop(columns="_intercept").astype(float)
        if feature_frame.empty:
            raise ValueError("SHAP requires at least one non-intercept model feature.")
        missing_coefficients = [column for column in feature_frame.columns if column not in coefficients]
        if missing_coefficients:
            raise ValueError(f"SHAP coefficient map is missing design columns: {missing_coefficients}")

        coefficient_vector = np.asarray(
            [coefficients[column] for column in feature_frame.columns], dtype=float
        )
        intercept = float(coefficients["_intercept"])
        if not np.isfinite(coefficient_vector).all() or not np.isfinite(intercept):
            raise ValueError("SHAP cannot explain non-finite fitted coefficients.")

        seed = self._pair_seed(upstream_antibiotic, downstream_antibiotic)
        rng = np.random.default_rng(seed)
        background_positions = self._sample_positions(
            len(feature_frame), self._config.background_size, rng
        )
        explanation_positions = self._sample_positions(
            len(feature_frame), self._config.explanation_sample_size, rng
        )
        background = feature_frame.iloc[background_positions]
        explained = feature_frame.iloc[explanation_positions]

        # Imported lazily so configuration loading and non-cascade commands do
        # not pay SHAP's import cost. SHAP is nevertheless a declared runtime
        # dependency because an enabled production cascade must fail closed if
        # its requested explainability artifact cannot be generated.
        import shap

        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(
            (coefficient_vector, intercept),
            masker=masker,
            link=shap.links.identity,
        )
        explanation = explainer(explained)
        shap_values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        if base_values.ndim == 0:
            base_values = np.repeat(float(base_values), len(explained))
        else:
            base_values = base_values.reshape(-1)

        model_log_odds = intercept + explained.to_numpy(dtype=float) @ coefficient_vector
        reconstructed = base_values + shap_values.sum(axis=1)
        maximum_additivity_error = float(np.max(np.abs(model_log_odds - reconstructed)))
        additivity_passed = maximum_additivity_error <= self._config.additivity_tolerance
        if not additivity_passed:
            raise RuntimeError(
                "SHAP additivity check failed for "
                f"{upstream_antibiotic} -> {downstream_antibiotic}: "
                f"maximum error {maximum_additivity_error:.3g} exceeds "
                f"{self._config.additivity_tolerance:.3g}."
            )

        global_importance = self._global_table(
            shap_values,
            list(feature_frame.columns),
            upstream_antibiotic,
            downstream_antibiotic,
        )
        sampled_values = self._local_table(
            explained,
            explanation_positions,
            shap_values,
            upstream_antibiotic,
            downstream_antibiotic,
        )
        metadata = pd.DataFrame(
            [
                {
                    "upstream_antibiotic": upstream_antibiotic,
                    "downstream_antibiotic": downstream_antibiotic,
                    "explainer": "shap.LinearExplainer",
                    "dependence_assumption": "interventional_independent_background",
                    "output_scale": "log_odds",
                    "model_rows": len(feature_frame),
                    "background_n": len(background),
                    "explained_n": len(explained),
                    "feature_n": feature_frame.shape[1],
                    "expected_value": float(base_values[0]),
                    "maximum_additivity_error": maximum_additivity_error,
                    "additivity_tolerance": self._config.additivity_tolerance,
                    "additivity_passed": additivity_passed,
                    "random_seed": seed,
                    "design_transform": "one_hot_encoded_then_nuisance_z_standardized",
                    "interpretation_boundary": "model_attribution_not_causal_effect",
                }
            ],
            columns=self._METADATA_COLUMNS,
        )
        domain_importance = self._domain_table(
            shap_values,
            list(feature_frame.columns),
            upstream_antibiotic,
            downstream_antibiotic,
        )
        return ShapExplanationArtifacts(global_importance, sampled_values, metadata, domain_importance)

    def _pair_seed(self, upstream: object, downstream: object) -> int:
        digest = hashlib.sha256(f"{upstream}|{downstream}".encode("utf-8")).digest()
        pair_offset = int.from_bytes(digest[:4], "little")
        return int((self._config.random_seed + pair_offset) % (2**32 - 1))

    @staticmethod
    def _sample_positions(n_rows: int, requested: int, rng: np.random.Generator) -> np.ndarray:
        sample_n = min(n_rows, max(1, int(requested)))
        if sample_n == n_rows:
            return np.arange(n_rows, dtype=int)
        return np.sort(rng.choice(n_rows, size=sample_n, replace=False))

    def _global_table(
        self,
        values: np.ndarray,
        features: list[str],
        upstream: object,
        downstream: object,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "upstream_antibiotic": upstream,
                "downstream_antibiotic": downstream,
                "feature": features,
                "feature_domain": [self.feature_domain(feature) for feature in features],
                "mean_absolute_shap": np.mean(np.abs(values), axis=0),
                "mean_signed_shap": np.mean(values, axis=0),
                "positive_attribution_fraction": np.mean(values > 0, axis=0),
                "explained_n": values.shape[0],
                "output_scale": "log_odds",
            }
        ).sort_values("mean_absolute_shap", ascending=False, kind="stable")
        frame["importance_rank"] = np.arange(1, len(frame) + 1)
        return frame.loc[:, self._GLOBAL_COLUMNS]

    def _domain_table(
        self,
        values: np.ndarray,
        features: list[str],
        upstream: object,
        downstream: object,
    ) -> pd.DataFrame:
        """Each feature domain's attribution for one pair.

        SHAP values add, so a domain's contribution to an explained row is the sum
        of its features' values -- all dummies of a one-hot category together.
        Summing per-feature mean |SHAP| instead can only overstate a domain spread
        over many columns (calendar year, site): the mean of absolute values is at
        least the absolute value of their sum, and within a row the dummies of one
        category often have opposite signs.
        """
        domains = np.asarray([self.feature_domain(feature) for feature in features], dtype=object)
        rows = []
        for domain in sorted(set(domains.tolist())):
            grouped = values[:, domains == domain].sum(axis=1)
            rows.append(
                {
                    "upstream_antibiotic": upstream,
                    "downstream_antibiotic": downstream,
                    "feature_domain": domain,
                    "feature_n": int((domains == domain).sum()),
                    "mean_absolute_shap": float(np.mean(np.abs(grouped))),
                    "mean_signed_shap": float(np.mean(grouped)),
                    "explained_n": int(values.shape[0]),
                    "output_scale": "log_odds",
                }
            )
        return pd.DataFrame(rows, columns=self._DOMAIN_COLUMNS)

    def _local_table(
        self,
        explained: pd.DataFrame,
        positions: np.ndarray,
        values: np.ndarray,
        upstream: object,
        downstream: object,
    ) -> pd.DataFrame:
        row_n, feature_n = values.shape
        features = np.asarray(explained.columns, dtype=object)
        frame = pd.DataFrame(
            {
                "upstream_antibiotic": np.repeat(upstream, row_n * feature_n),
                "downstream_antibiotic": np.repeat(downstream, row_n * feature_n),
                "sample_row": np.repeat(np.arange(row_n, dtype=int), feature_n),
                "model_row_position": np.repeat(positions, feature_n),
                "feature": np.tile(features, row_n),
                "feature_domain": np.tile(
                    [self.feature_domain(feature) for feature in features], row_n
                ),
                "feature_value": explained.to_numpy(dtype=float).reshape(-1),
                "shap_value": values.reshape(-1),
                "absolute_shap_value": np.abs(values).reshape(-1),
                "output_scale": "log_odds",
            }
        )
        return frame.loc[:, self._LOCAL_COLUMNS]

    @staticmethod
    def feature_domain(feature: str) -> str:
        """Map encoded design columns into interpretable clinical domains."""
        exact = {
            "upstream_positive": "upstream_result",
            "source_site": "site",
            "cov_ordering_mode": "ordering_context",
            "cov_specimen_type": "specimen_type",
            "cov_icu_status": "care_setting",
            "cov_er_status": "care_setting",
            "cov_prior_abx_any_90d": "prior_antibiotic_exposure",
            "cov_calendar_year": "calendar_time",
            "cov_age_bin": "demographics",
            "cov_sex": "demographics",
            "cov_prior_same_organism_any_90d": "prior_organism_history",
            "cov_comorbidity_count": "comorbidity",
            "cov_adi_score": "area_deprivation",
            "cov_adi_state_rank": "area_deprivation",
            "cov_nursing_home_90d": "nursing_home",
            "cov_prior_procedure_90d": "prior_procedure",
        }
        for base, domain in exact.items():
            if feature == base or feature.startswith(f"{base}_"):
                return domain
        if feature.startswith("comorb_"):
            return "comorbidity"
        if feature.startswith("proc_"):
            return "prior_procedure"
        if feature.endswith("_available"):
            return "source_availability"
        return "other"
