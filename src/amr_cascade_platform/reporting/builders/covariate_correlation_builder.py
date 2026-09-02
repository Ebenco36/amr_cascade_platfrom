"""Covariate dependence diagnostics for episode-level adjustment features."""

from __future__ import annotations

import math

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class CovariateCorrelationBuilder:
    """Build pairwise correlation diagnostics for model-ready episode covariates."""

    _DIRECT_CONTEXT_CATEGORICAL = (
        "source_site",
        "organism",
        "cov_ordering_mode",
        "cov_specimen_type",
        "cov_calendar_year",
        "cov_calendar_month",
        "cov_age_bin",
        "cov_sex",
    )
    _DIRECT_CONTEXT_NUMERIC = (
        "cov_icu_status",
        "cov_icu_available",
        "cov_prior_abx_any_90d",
        "cov_prior_abx_available",
        "cov_prior_same_organism_any_90d",
        "cov_prior_organism_available",
        "cov_comorbidity_count",
        "cov_comorbidity_available",
        "cov_adi_score",
        "cov_adi_state_rank",
        "cov_adi_available",
        "cov_er_status",
        "cov_er_available",
        "cov_nursing_home_90d",
        "cov_nursing_home_available",
        "cov_prior_procedure_90d",
        "cov_prior_procedure_available",
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()

        design_matrix, feature_groups = self._build_design_matrix(frame)
        if design_matrix.empty or design_matrix.shape[1] < 2:
            return pd.DataFrame()

        correlations = design_matrix.corr(method="pearson")
        rows: list[dict[str, object]] = []
        feature_names = list(correlations.columns)
        for left_index, feature_a in enumerate(feature_names):
            for feature_b in feature_names[left_index + 1 :]:
                correlation = correlations.at[feature_a, feature_b]
                if pd.isna(correlation):
                    continue
                if feature_groups.get(feature_a) == feature_groups.get(feature_b):
                    continue
                absolute = abs(float(correlation))
                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "feature_a_group": feature_groups.get(feature_a, feature_a),
                        "feature_b_group": feature_groups.get(feature_b, feature_b),
                        "correlation": float(correlation),
                        "absolute_correlation": absolute,
                        "correlation_flag": self._correlation_flag(absolute),
                        "episode_n": int(len(design_matrix)),
                        "design_feature_n": int(design_matrix.shape[1]),
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame.from_records(rows).sort_values(
            by=["absolute_correlation", "feature_a", "feature_b"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    def _build_design_matrix(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        prepared = frame.copy()
        feature_groups: dict[str, str] = {}

        categorical_columns = [column for column in self._DIRECT_CONTEXT_CATEGORICAL if column in prepared.columns]
        numeric_columns = [column for column in self._DIRECT_CONTEXT_NUMERIC if column in prepared.columns]
        component_columns = [
            column
            for column in sorted(prepared.columns)
            if (column.startswith("comorb_") or column.startswith("proc_")) and column not in numeric_columns
        ]
        numeric_columns.extend(component_columns)

        for column in categorical_columns:
            prepared[column] = prepared[column].fillna("unknown").astype(str)
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).astype(float)

        matrices: list[pd.DataFrame] = []
        if categorical_columns:
            categorical = pd.get_dummies(
                prepared.loc[:, categorical_columns],
                prefix=categorical_columns,
                prefix_sep="=",
                dtype=float,
            )
            matrices.append(categorical)
            for column in categorical.columns:
                feature_groups[column] = column.split("=", 1)[0]
        if numeric_columns:
            numeric = prepared.loc[:, numeric_columns].astype(float)
            matrices.append(numeric)
            for column in numeric.columns:
                feature_groups[column] = column

        if not matrices:
            return pd.DataFrame(), {}

        design = pd.concat(matrices, axis=1)
        design = design.loc[:, design.nunique(dropna=False) > 1].copy()
        feature_groups = {column: feature_groups[column] for column in design.columns}
        return design, feature_groups

    @staticmethod
    def _correlation_flag(value: float) -> str:
        if pd.isna(value):
            return "unknown"
        if value >= 0.9:
            return "very_high"
        if value >= 0.7:
            return "high"
        if value >= 0.5:
            return "moderate"
        if value >= 0.3:
            return "low_to_moderate"
        if value > 0:
            return "low"
        return "none"
