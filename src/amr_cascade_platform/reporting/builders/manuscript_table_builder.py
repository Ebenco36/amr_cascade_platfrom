"""Build manuscript-facing tables from pipeline metadata and cascade artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.cascade.outputs.cascade_comparison_builder import CascadeComparisonBuilder
from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.reporting.builders.covariate_correlation_builder import CovariateCorrelationBuilder
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver


class ManuscriptTableBuilder:
    """Construct publication-facing tables A-D from current pipeline outputs."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._dataset_store = DatasetStore(settings)
        self._classification = AntibioticClassificationResolver(
            path_manager.project_root / "data" / "antibiotic_classification_complete.csv"
        )
        self._cascade_covariates = CascadeCovariateBuilder(settings, path_manager)
        self._covariate_correlation_builder = CovariateCorrelationBuilder(settings)

    def build_provenance_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        sites = self._resolve_sites(scope, site)
        rows: list[dict[str, object]] = []
        combined_gold_dir = scoped_output_dir(self._paths.paths.gold, "combined", organism=organism)
        # These three are only ever fed into _site_row_count, which reads a single
        # column (source_site) -- projecting it here avoids materializing the full
        # table (drug_pair_episodes.parquet can be 100M+ rows) just to count rows.
        combined_culture = self._safe_read(combined_gold_dir / "culture_episodes.parquet", columns=["source_site"])
        combined_eligible = self._safe_read(combined_gold_dir / "eligible_pairs.parquet", columns=["source_site"])
        combined_pairs = self._safe_read(combined_gold_dir / "drug_pair_episodes.parquet", columns=["source_site"])

        for current_site in sites:
            microbial_resistance_metadata = self._read_silver_metadata(current_site, "microbial_resistance")
            row = {
                "site": current_site,
                "raw_file_count": len(list((self._paths.paths.raw / current_site).glob("*.csv"))),
                "bronze_table_count": len(list((self._paths.paths.metadata / "datasets" / "bronze" / current_site).glob("*.json"))),
                "silver_table_count": len(list((self._paths.paths.metadata / "datasets" / "silver" / current_site).glob("*.json"))),
                "culture_episode_n": self._site_row_count(
                    combined_culture, "source_site", current_site, fallback_path=scoped_output_dir(self._paths.paths.gold, "site", site=current_site, organism=organism) / "culture_episodes.parquet"
                ),
                "eligible_episode_drug_n": self._site_row_count(
                    combined_eligible, "source_site", current_site, fallback_path=scoped_output_dir(self._paths.paths.gold, "site", site=current_site, organism=organism) / "eligible_pairs.parquet"
                ),
                "eligible_directed_pair_n": self._site_row_count(
                    combined_pairs, "source_site", current_site, fallback_path=scoped_output_dir(self._paths.paths.gold, "site", site=current_site, organism=organism) / "drug_pair_episodes.parquet"
                ),
                "microbial_resistance_duplicates_removed": int(microbial_resistance_metadata.get("duplicates_removed", 0) or 0),
                "discordant_susceptibility_group_n": int(microbial_resistance_metadata.get("discordant_susceptibility_group_n", 0) or 0),
                "discordant_susceptibility_row_n": int(microbial_resistance_metadata.get("discordant_susceptibility_row_n", 0) or 0),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def build_data_quality_flow_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        sites = self._resolve_sites(scope, site)
        rows: list[dict[str, object]] = []
        combined_gold_dir = scoped_output_dir(self._paths.paths.gold, "combined", organism=organism)
        # Same _site_row_count-only usage as build_provenance_table above -- project
        # just source_site to avoid materializing the full tables.
        combined_culture = self._safe_read(combined_gold_dir / "culture_episodes.parquet", columns=["source_site"])
        combined_observed = self._safe_read(combined_gold_dir / "culture_drug_episodes.parquet", columns=["source_site"])
        combined_eligible = self._safe_read(combined_gold_dir / "eligible_pairs.parquet", columns=["source_site"])
        combined_pairs = self._safe_read(combined_gold_dir / "drug_pair_episodes.parquet", columns=["source_site"])

        for current_site in sites:
            raw_dir = self._paths.paths.raw / current_site
            cohort_candidates = sorted(raw_dir.glob("*cohort*.csv"))
            raw_ast_rows = self._count_csv_rows(cohort_candidates[0]) if cohort_candidates else 0
            site_gold_dir = scoped_output_dir(self._paths.paths.gold, "site", site=current_site, organism=organism)
            culture_n = self._site_row_count(combined_culture, "source_site", current_site, site_gold_dir / "culture_episodes.parquet")
            observed_n = self._site_row_count(combined_observed, "source_site", current_site, site_gold_dir / "culture_drug_episodes.parquet")
            eligible_n = self._site_row_count(combined_eligible, "source_site", current_site, site_gold_dir / "eligible_pairs.parquet")
            pair_n = self._site_row_count(combined_pairs, "source_site", current_site, site_gold_dir / "drug_pair_episodes.parquet")
            rows.extend(
                [
                    {"site": current_site, "stage": "raw_ast_rows", "row_count": raw_ast_rows},
                    {"site": current_site, "stage": "culture_episodes", "row_count": culture_n},
                    {"site": current_site, "stage": "observed_episode_drug_rows", "row_count": observed_n},
                    {"site": current_site, "stage": "eligible_episode_drug_rows", "row_count": eligible_n},
                    {"site": current_site, "stage": "eligible_directed_pair_rows", "row_count": pair_n},
                ]
            )
        return pd.DataFrame(rows)

    def build_eligibility_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        eligible_pairs = self._load_scope_gold(scope, site, filename="eligible_pairs.parquet", organism=organism)
        if eligible_pairs.empty:
            return pd.DataFrame()
        grouped = (
            eligible_pairs.groupby("source_site", observed=True)
            .agg(
                total_episode_drug_rows=("antibiotic", "size"),
                eligible_rows=("is_eligible", "sum"),
                structural_omission_rows=("is_intrinsic_resistance", "sum"),
                observed_tested_rows=("is_observed_tested", "sum"),
            )
            .reset_index()
            .rename(columns={"source_site": "site"})
        )
        grouped["unobserved_but_eligible_rows"] = (
            grouped["eligible_rows"] - grouped["observed_tested_rows"]
        )
        grouped["intrinsic_reference_file"] = self._settings.platform.reference_files["intrinsic_resistance"]
        grouped["canonical_map_file"] = self._settings.platform.reference_files["canonical_table_map"]
        return grouped

    def build_upstream_selection_balance_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
    ) -> pd.DataFrame:
        culture_episodes = self._load_scope_gold(scope, site, filename="culture_episodes.parquet", organism=organism)
        eligible_pairs = self._load_scope_gold(scope, site, filename="eligible_pairs.parquet", organism=organism)
        if culture_episodes.empty or eligible_pairs.empty:
            return pd.DataFrame()

        covariates = self._cascade_covariates.build(culture_episodes)
        if covariates.empty:
            return pd.DataFrame()

        episode_keys = list(self._settings.gold.episode_key_columns)
        frame = (
            eligible_pairs.loc[eligible_pairs["is_eligible"] == 1, episode_keys + ["antibiotic", "is_observed_tested"]]
            .drop_duplicates()
            .rename(columns={"antibiotic": "upstream_antibiotic", "is_observed_tested": "upstream_tested"})
        )
        frame = frame.merge(covariates, on=episode_keys, how="left", validate="many_to_one")
        if frame.empty:
            return pd.DataFrame()

        frame["upstream_antibiotic"] = frame["upstream_antibiotic"].map(self._display_antibiotic_label)
        rows: list[dict[str, object]] = []
        group_columns = ["source_site", "organism", "upstream_antibiotic"]

        binary_covariates = [
            ("cov_icu_status", "ICU status"),
            ("cov_icu_available", "ICU data available"),
            ("cov_er_status", "Emergency-department presentation"),
            ("cov_er_available", "Emergency-department data available"),
            ("cov_prior_abx_any_90d", "Any prior antibiotic exposure (90d)"),
            ("cov_prior_abx_available", "Prior antibiotic data available"),
            ("cov_prior_same_organism_any_90d", "Prior same-organism history (90d)"),
            ("cov_prior_organism_available", "Prior organism-history data available"),
            ("cov_comorbidity_available", "Comorbidity data available"),
            ("cov_adi_available", "ADI data available"),
            ("cov_nursing_home_90d", "Nursing-home residence (90d)"),
            ("cov_nursing_home_available", "Nursing-home data available"),
            ("cov_prior_procedure_90d", "Any prior procedure (90d)"),
            ("cov_prior_procedure_available", "Prior procedure data available"),
        ]
        continuous_covariates = [
            ("cov_comorbidity_count", "Comorbidity count"),
            ("cov_adi_score", "ADI score"),
            ("cov_adi_state_rank", "ADI state rank"),
        ]
        categorical_covariates = [
            ("cov_age_bin", "Age bin"),
            ("cov_sex", "Sex"),
            ("cov_ordering_mode", "Ordering context"),
            ("cov_specimen_type", "Specimen type"),
        ]

        for (source_site, group_organism, upstream_antibiotic), group in frame.groupby(group_columns, dropna=False, observed=True):
            tested = group.loc[group["upstream_tested"] == 1].copy()
            untested = group.loc[group["upstream_tested"] == 0].copy()
            tested_n = int(len(tested))
            untested_n = int(len(untested))
            for column, label in binary_covariates:
                rows.append(
                    self._selection_balance_row(
                        source_site=source_site,
                        organism=group_organism,
                        upstream_antibiotic=upstream_antibiotic,
                        covariate=label,
                        covariate_level=None,
                        metric_type="binary",
                        tested_n=tested_n,
                        untested_n=untested_n,
                        tested_value=self._safe_mean(tested.get(column)),
                        untested_value=self._safe_mean(untested.get(column)),
                    )
                )
            for column, label in continuous_covariates:
                rows.append(
                    self._selection_balance_row(
                        source_site=source_site,
                        organism=group_organism,
                        upstream_antibiotic=upstream_antibiotic,
                        covariate=label,
                        covariate_level=None,
                        metric_type="continuous",
                        tested_n=tested_n,
                        untested_n=untested_n,
                        tested_value=self._safe_mean(tested.get(column)),
                        untested_value=self._safe_mean(untested.get(column)),
                        tested_sd=self._safe_std(tested.get(column)),
                        untested_sd=self._safe_std(untested.get(column)),
                    )
                )
            for column, label in categorical_covariates:
                tested_levels = tested.get(column, pd.Series(dtype="object")).fillna("unknown").astype(str)
                untested_levels = untested.get(column, pd.Series(dtype="object")).fillna("unknown").astype(str)
                levels = sorted(set(tested_levels.tolist()) | set(untested_levels.tolist()))
                for level in levels:
                    rows.append(
                        self._selection_balance_row(
                            source_site=source_site,
                            organism=group_organism,
                            upstream_antibiotic=upstream_antibiotic,
                            covariate=label,
                            covariate_level=level,
                            metric_type="categorical_level",
                            tested_n=tested_n,
                            untested_n=untested_n,
                            tested_value=self._safe_mean((tested_levels == level).astype(int)),
                            untested_value=self._safe_mean((untested_levels == level).astype(int)),
                        )
                    )
            component_columns = sorted(
                column for column in group.columns if column.startswith("comorb_") or column.startswith("proc_")
            )
            for column in component_columns:
                rows.append(
                    self._selection_balance_row(
                        source_site=source_site,
                        organism=group_organism,
                        upstream_antibiotic=upstream_antibiotic,
                        covariate=self._component_covariate_label(column),
                        covariate_level=None,
                        metric_type="binary",
                        tested_n=tested_n,
                        untested_n=untested_n,
                        tested_value=self._safe_mean(tested.get(column)),
                        untested_value=self._safe_mean(untested.get(column)),
                    )
                )

        if not rows:
            return pd.DataFrame()
        table = pd.DataFrame.from_records(rows)
        table["absolute_standardised_mean_difference"] = table["standardised_mean_difference"].abs()
        table["balance_flag"] = table["absolute_standardised_mean_difference"].map(
            lambda value: "high_imbalance" if pd.notna(value) and value >= 0.1 else "acceptable_or_unknown"
        )
        return table.sort_values(
            by=["site", "organism", "upstream_antibiotic", "absolute_standardised_mean_difference", "covariate", "covariate_level"],
            ascending=[True, True, True, False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    def build_covariate_correlation_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
    ) -> pd.DataFrame:
        culture_episodes = self._load_scope_gold(scope, site, filename="culture_episodes.parquet", organism=organism)
        if culture_episodes.empty:
            return pd.DataFrame()

        episode_keys = list(self._settings.gold.episode_key_columns)
        model_ready_context = [
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
        ]
        component_context = [
            column for column in culture_episodes.columns if column.startswith("comorb_") or column.startswith("proc_")
        ]
        direct_context_columns = [
            column
            for column in [
                "source_site",
                "organism",
                "ordering_mode",
                "culture_description",
                *model_ready_context,
                *component_context,
            ]
            if column in culture_episodes.columns and column not in episode_keys
        ]
        base = culture_episodes.loc[:, episode_keys + direct_context_columns].drop_duplicates().copy()
        covariates = self._cascade_covariates.build(culture_episodes)
        if not covariates.empty:
            overlapping_columns = [column for column in covariates.columns if column in base.columns and column not in episode_keys]
            if overlapping_columns:
                covariates = covariates.drop(columns=overlapping_columns)
            base = base.merge(covariates, on=episode_keys, how="left", validate="one_to_one")
        if base.empty:
            return pd.DataFrame()

        return self._covariate_correlation_builder.build(base)

    def build_primary_cascade_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty:
            return pd.DataFrame()
        if "validation_status" in edge_report.columns:
            validation_rank = {"robust": 0, "supported": 1, "mixed": 2, "insufficient": 3}
            edge_report["_validation_rank"] = edge_report["validation_status"].map(validation_rank).fillna(9)
        else:
            edge_report["_validation_rank"] = 9
        filtered = edge_report.loc[
            pd.to_numeric(edge_report["escalation_ratio"], errors="coerce").notna()
            & pd.to_numeric(edge_report["escalation_ratio"], errors="coerce").gt(1.0)
            & pd.to_numeric(edge_report["escalation_ratio"], errors="coerce").le(50.0)
        ].copy()
        if not filtered.empty:
            edge_report = filtered
        columns = [
            "upstream_antibiotic",
            "downstream_antibiotic",
            "validation_status",
            "resistant_support_n",
            "susceptible_support_n",
            "resistant_downstream_test_probability",
            "susceptible_downstream_test_probability",
            "escalation_ratio",
            "adjusted_odds_ratio",
            "adjusted_or_e_value",
            "adjusted_or_ci_e_value",
            "total_support_n",
            "rank_by_escalation_ratio",
            "_validation_rank",
        ]
        table = edge_report.loc[:, columns].rename(
            columns={
                "upstream_antibiotic": "upstream_drug",
                "downstream_antibiotic": "downstream_drug",
                "resistant_support_n": "n_R",
                "susceptible_support_n": "n_S",
                "resistant_downstream_test_probability": "p_R",
                "susceptible_downstream_test_probability": "p_S",
                "adjusted_odds_ratio": "adjusted_OR",
                "adjusted_or_e_value": "conf_strength",
                "adjusted_or_ci_e_value": "conf_strength_ci",
                "total_support_n": "total_support",
            }
        )
        return table.sort_values(
            by=["_validation_rank", "total_support", "escalation_ratio", "adjusted_OR"],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop(columns=["_validation_rank"]).reset_index(drop=True)

    def build_validated_primary_cascade_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> pd.DataFrame:
        output_columns = [
            "upstream_drug",
            "downstream_drug",
            "validation_status",
            "upstream_aware_category",
            "downstream_aware_category",
            "aware_transition",
            "aware_direction",
            "aware_step",
            "n_R",
            "n_S",
            "p_R",
            "p_S",
            "escalation_ratio",
            "adjusted_OR",
            "conf_strength",
            "conf_strength_ci",
            "total_support",
            "permutation_p_value",
            "permutation_fdr_q_value",
            "bootstrap_sign_stability",
            "site_replication_n",
            "site_direction_agreement_rate",
        ]
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return pd.DataFrame(columns=output_columns)
        filtered = edge_report.loc[edge_report["validation_status"].isin(allowed_statuses)].copy()
        if filtered.empty:
            return pd.DataFrame(columns=output_columns)
        filtered = self._annotate_aware_transitions(filtered)
        columns = [
            "upstream_antibiotic",
            "downstream_antibiotic",
            "validation_status",
            "upstream_aware_category",
            "downstream_aware_category",
            "aware_transition",
            "aware_direction",
            "aware_step",
            "resistant_support_n",
            "susceptible_support_n",
            "resistant_downstream_test_probability",
            "susceptible_downstream_test_probability",
            "escalation_ratio",
            "adjusted_odds_ratio",
            "adjusted_or_e_value",
            "adjusted_or_ci_e_value",
            "total_support_n",
            "permutation_p_value",
            "permutation_fdr_q_value",
            "bootstrap_sign_stability",
            "site_replication_n",
            "site_direction_agreement_rate",
        ]
        available = [column for column in columns if column in filtered.columns]
        table = filtered.loc[:, available].rename(
            columns={
                "upstream_antibiotic": "upstream_drug",
                "downstream_antibiotic": "downstream_drug",
                "resistant_support_n": "n_R",
                "susceptible_support_n": "n_S",
                "resistant_downstream_test_probability": "p_R",
                "susceptible_downstream_test_probability": "p_S",
                "adjusted_odds_ratio": "adjusted_OR",
                "adjusted_or_e_value": "conf_strength",
                "adjusted_or_ci_e_value": "conf_strength_ci",
                "total_support_n": "total_support",
            }
        )
        validation_rank = {"robust": 0, "supported": 1, "mixed": 2, "insufficient": 3}
        table["_validation_rank"] = table["validation_status"].map(validation_rank).fillna(9)
        return table.sort_values(
            by=["_validation_rank", "escalation_ratio", "adjusted_OR", "total_support"],
            ascending=[True, False, False, False],
        ).drop(columns=["_validation_rank"]).reset_index(drop=True)

    def build_aware_transition_summary_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> pd.DataFrame:
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return pd.DataFrame()
        filtered = edge_report.loc[edge_report["validation_status"].isin(allowed_statuses)].copy()
        if filtered.empty:
            return pd.DataFrame()
        filtered = self._annotate_aware_transitions(filtered)
        filtered["support_weighted_er_component"] = (
            pd.to_numeric(filtered["total_support_n"], errors="coerce").fillna(0.0)
            * pd.to_numeric(filtered["escalation_ratio"], errors="coerce").fillna(0.0)
        )
        grouped = (
            filtered.groupby(
                [
                    "upstream_aware_category",
                    "downstream_aware_category",
                    "aware_transition",
                    "aware_direction",
                    "aware_step",
                ],
                dropna=False,
                observed=True,
            )
            .agg(
                validated_edge_n=("aware_transition", "size"),
                robust_edge_n=("validation_status", lambda values: int((values == "robust").sum())),
                supported_edge_n=("validation_status", lambda values: int((values == "supported").sum())),
                total_support_sum=("total_support_n", "sum"),
                median_escalation_ratio=("escalation_ratio", "median"),
                mean_escalation_ratio=("escalation_ratio", "mean"),
                median_adjusted_or=("adjusted_odds_ratio", "median"),
                mean_aware_step=("aware_step", "mean"),
                support_weighted_er_component=("support_weighted_er_component", "sum"),
            )
            .reset_index()
        )
        support_sum = pd.to_numeric(grouped["total_support_sum"], errors="coerce").fillna(0.0)
        grouped["support_weighted_mean_escalation_ratio"] = grouped["support_weighted_er_component"] / support_sum.where(
            support_sum > 0,
            pd.NA,
        )
        grouped = grouped.drop(columns=["support_weighted_er_component"])
        total_edge_n = max(int(grouped["validated_edge_n"].sum()), 1)
        total_support_all = max(float(grouped["total_support_sum"].sum()), 1.0)
        grouped["validated_edge_share"] = grouped["validated_edge_n"] / total_edge_n
        grouped["support_share"] = grouped["total_support_sum"] / total_support_all
        return grouped.sort_values(
            by=["aware_step", "validated_edge_n", "total_support_sum", "aware_transition"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def build_aware_direction_hypothesis_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> pd.DataFrame:
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return pd.DataFrame()
        filtered = edge_report.loc[edge_report["validation_status"].isin(allowed_statuses)].copy()
        if filtered.empty:
            return pd.DataFrame()
        filtered = self._annotate_aware_transitions(filtered)
        filtered["support_weighted_er_component"] = (
            pd.to_numeric(filtered["total_support_n"], errors="coerce").fillna(0.0)
            * pd.to_numeric(filtered["escalation_ratio"], errors="coerce").fillna(0.0)
        )
        grouped = (
            filtered.groupby("aware_direction", dropna=False, observed=True)
            .agg(
                validated_edge_n=("aware_direction", "size"),
                robust_edge_n=("validation_status", lambda values: int((values == "robust").sum())),
                supported_edge_n=("validation_status", lambda values: int((values == "supported").sum())),
                total_support_sum=("total_support_n", "sum"),
                support_weighted_er_component=("support_weighted_er_component", "sum"),
            )
            .reset_index()
        )
        support_sum = pd.to_numeric(grouped["total_support_sum"], errors="coerce").fillna(0.0)
        grouped["support_weighted_mean_escalation_ratio"] = grouped["support_weighted_er_component"] / support_sum.where(
            support_sum > 0,
            pd.NA,
        )
        grouped = grouped.drop(columns=["support_weighted_er_component"])
        total_edges = max(int(grouped["validated_edge_n"].sum()), 1)
        total_support = max(float(grouped["total_support_sum"].sum()), 1.0)
        grouped["validated_edge_share"] = grouped["validated_edge_n"] / total_edges
        grouped["support_share"] = grouped["total_support_sum"] / total_support
        grouped["supports_upward_hypothesis"] = grouped["aware_direction"].eq("upward")
        direction_order = {"upward": 0, "lateral": 1, "downward": 2, "unclassified": 3}
        grouped["_sort"] = grouped["aware_direction"].map(direction_order).fillna(99)
        return grouped.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    def build_asymmetry_sensitivity_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> pd.DataFrame:
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty or "testing_asymmetry_bin" not in edge_report.columns:
            return pd.DataFrame()
        filtered = edge_report.copy()
        if filtered["validation_status"].notna().any():
            filtered["is_validated"] = filtered["validation_status"].isin(allowed_statuses)
        else:
            filtered["is_validated"] = False
        grouped = (
            filtered.groupby("testing_asymmetry_bin", dropna=False, observed=True)
            .agg(
                retained_edge_n=("testing_asymmetry_bin", "size"),
                validated_edge_n=("is_validated", "sum"),
                robust_edge_n=("validation_status", lambda values: int((values == "robust").sum())),
                supported_edge_n=("validation_status", lambda values: int((values == "supported").sum())),
                median_escalation_ratio=("escalation_ratio", "median"),
                mean_escalation_ratio=("escalation_ratio", "mean"),
                median_adjusted_or=("adjusted_odds_ratio", "median"),
                total_support_sum=("total_support_n", "sum"),
                mean_bundled_panel_fraction=("bundled_panel_fraction", "mean"),
                mean_testing_asymmetry_score=("testing_asymmetry_score", "mean"),
            )
            .reset_index()
        )
        total_edges = max(int(grouped["retained_edge_n"].sum()), 1)
        grouped["retained_edge_share"] = grouped["retained_edge_n"] / total_edges
        grouped["validated_edge_rate"] = grouped["validated_edge_n"] / grouped["retained_edge_n"].clip(lower=1)
        # Add PBI and DAS summaries if available (new anti-artefact diagnostic columns)
        for col, summary_col in [
            ("panel_bundling_index", "mean_panel_bundling_index"),
            ("directional_asymmetry_score", "mean_directional_asymmetry_score"),
        ]:
            if col in filtered.columns:
                col_summary = (
                    filtered.groupby("testing_asymmetry_bin", dropna=False, observed=True)[col]
                    .mean()
                    .rename(summary_col)
                    .reset_index()
                )
                grouped = grouped.merge(col_summary, on="testing_asymmetry_bin", how="left")
        order = {
            "low_asymmetry": 0,
            "medium_asymmetry": 1,
            "high_asymmetry": 2,
            "undifferentiated": 3,
            "unassigned": 4,
        }
        grouped["_sort"] = grouped["testing_asymmetry_bin"].map(order).fillna(99)
        return grouped.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    def build_concordance_summary_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        concordance = self._load_scope_cascade(scope, site, filename="rule_concordance.parquet", organism=organism)
        retained = self._load_scope_cascade(scope, site, filename="retained_edges.parquet", organism=organism)
        if concordance.empty:
            return pd.DataFrame()

        summary = pd.DataFrame(
            [
                {
                    "scope": site if scope == "site" and site else "combined",
                    "empirical_retained_edge_count": len(retained),
                    "structurally_aligned_edge_count": int((concordance["structural_concordance"] == "aligned_with_eligibility_logic").sum()),
                    "structural_conflict_edge_count": int((concordance["structural_concordance"] == "contains_structural_conflict").sum()),
                    "mean_structural_alignment_rate": float(concordance["structural_alignment_rate"].mean()),
                    "documented_rule_match_count": int(pd.to_numeric(concordance.get("documented_rule_match", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int).sum()),
                }
            ]
        )
        return summary

    def build_concordance_examples_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        concordance = self._load_scope_cascade(scope, site, filename="rule_concordance.parquet", organism=organism)
        if concordance.empty:
            return pd.DataFrame()
        return concordance.sort_values(
            by=["structural_alignment_rate", "escalation_ratio"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def build_threshold_sensitivity_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        table = self._load_scope_cascade(scope, site, filename="threshold_sensitivity.parquet", organism=organism)
        if table.empty:
            return pd.DataFrame()
        sort_columns = [
            column for column in (
                "min_total_support",
                "min_result_support",
                "retained_min_escalation_ratio",
                "adjusted_or_positive_threshold",
                "retained_min_adjusted_odds_ratio",
            )
            if column in table.columns
        ]
        return table.sort_values(by=sort_columns).reset_index(drop=True)

    def build_correction_sensitivity_table(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        top_n: int = 30,
    ) -> pd.DataFrame:
        """Show escalation ratios under λ ∈ sensitivity_continuity_corrections for top-N pairs.

        Uses raw support counts from the edge report to recompute ER at each λ so
        the table is self-contained and reproducible without rerunning the pipeline.
        """
        edge_report = self._normalize_edge_report(
            self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        )
        if edge_report.empty:
            return pd.DataFrame()
        lambdas = list(self._settings.cascade.sensitivity_continuity_corrections)
        required_cols = ["resistant_support_n", "susceptible_support_n"]
        for col in ["resistant_tested_n", "susceptible_tested_n"]:
            if col not in edge_report.columns:
                # Attempt legacy column names
                edge_report = edge_report.rename(columns={
                    "positive_support_n": "resistant_support_n",
                    "negative_support_n": "susceptible_support_n",
                }, errors="ignore")
        # Restrict to validated pairs ranked by escalation_ratio
        if "validation_status" in edge_report.columns:
            validated = edge_report.loc[
                edge_report["validation_status"].isin({"robust", "supported"})
            ].copy()
            if validated.empty:
                validated = edge_report.copy()
        else:
            validated = edge_report.copy()
        er_numeric = pd.to_numeric(validated.get("escalation_ratio"), errors="coerce")
        validated = validated.loc[er_numeric.notna()].copy()
        validated["_er"] = er_numeric.loc[validated.index]
        validated = validated.sort_values("_er", ascending=False).head(top_n)
        rows: list[dict[str, object]] = []
        for _, row in validated.iterrows():
            n_R = int(pd.to_numeric(row.get("resistant_support_n", 0), errors="coerce") or 0)
            n_S = int(pd.to_numeric(row.get("susceptible_support_n", 0), errors="coerce") or 0)
            k_R = int(pd.to_numeric(row.get("resistant_tested_n", 0), errors="coerce") or 0)
            k_S = int(pd.to_numeric(row.get("susceptible_tested_n", 0), errors="coerce") or 0)
            record: dict[str, object] = {
                "upstream_drug": row.get("upstream_antibiotic"),
                "downstream_drug": row.get("downstream_antibiotic"),
                "validation_status": row.get("validation_status"),
                "n_R": n_R,
                "n_S": n_S,
                "k_R": k_R,
                "k_S": k_S,
            }
            for λ in lambdas:
                if n_R > 0 and n_S > 0:
                    p_R = (k_R + λ) / (n_R + 2.0 * λ)
                    p_S = (k_S + λ) / (n_S + 2.0 * λ)
                    er = p_R / p_S if p_S > 0 else math.nan
                else:
                    er = math.nan
                record[f"er_lambda_{λ:.1f}"] = round(er, 3) if pd.notna(er) else math.nan
            rows.append(record)
        if not rows:
            return pd.DataFrame()
        table = pd.DataFrame(rows)
        return table.drop(columns=["_er"], errors="ignore").reset_index(drop=True)

    def build_pathway_flow_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        table = self._load_scope_cascade(scope, site, filename="pathway_flows.parquet", organism=organism)
        if table.empty:
            return pd.DataFrame()
        return table.sort_values(by=["path_rank", "stage"]).reset_index(drop=True)

    def build_site_vs_combined_comparison(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        scope_dir = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope,
            site=site,
            organism=organism,
        )
        table = self._safe_read(scope_dir / "site_vs_combined_edge_comparison.parquet")
        if table.empty:
            self._build_site_comparison_if_possible(scope_dir)
            table = self._safe_read(scope_dir / "site_vs_combined_edge_comparison.parquet")
        if table.empty:
            return pd.DataFrame()
        return table.sort_values(
            by=["site", "edge_presence", "combined_escalation_ratio", "site_escalation_ratio"],
            ascending=[True, True, False, False],
            na_position="last",
        ).reset_index(drop=True)

    def build_site_vs_combined_summary(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        scope_dir = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope,
            site=site,
            organism=organism,
        )
        table = self._safe_read(scope_dir / "site_vs_combined_summary.parquet")
        if table.empty:
            self._build_site_comparison_if_possible(scope_dir)
            table = self._safe_read(scope_dir / "site_vs_combined_summary.parquet")
        if table.empty:
            return pd.DataFrame()
        return table.sort_values(by=["site", "edge_presence"]).reset_index(drop=True)

    def _build_site_comparison_if_possible(self, scope_dir: Path) -> None:
        cascade_root = self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir
        try:
            scope_name = scope_dir.relative_to(cascade_root).as_posix()
        except ValueError:
            return
        if not (scope_dir / "edge_report.parquet").exists():
            return
        CascadeComparisonBuilder().export(
            artifact_root=cascade_root,
            scope_name=scope_name,
        )

    def build_implied_delta_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        if organism is not None or not (scope == "site" and site == "armd"):
            return pd.DataFrame()
        table = self._safe_read(
            self._paths.paths.artifacts
            / self._settings.cascade.outputs.result_dir
            / "armd_sensitivity_implied"
            / "implied_delta_edges.parquet"
        )
        if table.empty:
            return pd.DataFrame()
        return table.sort_values(
            by=["presence_change", "sensitivity_rank", "primary_rank"],
            ascending=[True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    def build_downstream_trigger_availability_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        edge_report = self._load_scope_cascade(scope, site, filename="edge_report.parquet", organism=organism)
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return pd.DataFrame()
        summary = edge_report.loc[
            edge_report["validation_status"].astype("string").str.lower().isin({"robust", "supported"})
        ].copy()
        if summary.empty:
            return pd.DataFrame()
        summary["downstream_key"] = summary["downstream_antibiotic"].map(self._classification.normalize_label)
        summary["upstream_key"] = summary["upstream_antibiotic"].map(self._classification.normalize_label)
        summary["downstream_antibiotic"] = summary["downstream_antibiotic"].map(self._classification.canonical_display_label)
        summary["escalation_ratio"] = pd.to_numeric(summary["escalation_ratio"], errors="coerce")
        summary["has_validated_trigger"] = summary["escalation_ratio"].notna()
        grouped_rows: list[dict[str, object]] = []
        for (downstream_key, downstream_antibiotic), subset in summary.groupby(
            ["downstream_key", "downstream_antibiotic"], dropna=False, observed=True
        ):
            valid = subset.loc[subset["has_validated_trigger"]].copy()
            grouped_rows.append(
                {
                    "downstream_key": downstream_key,
                    "downstream_antibiotic": downstream_antibiotic,
                    "validated_upstream_n": int(valid["upstream_key"].nunique()),
                    "validated_escalation_upstream_n": int(
                        valid.loc[valid["escalation_ratio"].gt(1.0), "upstream_key"].nunique()
                    ),
                    "validated_suppression_upstream_n": int(
                        valid.loc[valid["escalation_ratio"].lt(1.0), "upstream_key"].nunique()
                    ),
                    "max_escalation_ratio": float(valid["escalation_ratio"].max()) if not valid.empty else pd.NA,
                    "max_total_support_n": int(valid["total_support_n"].max()) if not valid.empty else 0,
                }
            )
        grouped = pd.DataFrame.from_records(grouped_rows)
        grouped["has_validated_trigger_forest"] = grouped["validated_upstream_n"] > 0
        return grouped.drop(columns=["downstream_key"]).sort_values(
            by=["has_validated_trigger_forest", "validated_upstream_n", "max_escalation_ratio", "downstream_antibiotic"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    def build_cascade_existence_summary(
        self,
        scope: str,
        site: str | None,
        organism: str | None = None,
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> pd.DataFrame:
        if organism is None:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        summary_specs: list[tuple[str, str | None, str]] = [("combined", None, "combined")]
        summary_specs.extend(("site", current_site, current_site) for current_site in self._settings.platform.sites)

        for current_scope, current_site, label in summary_specs:
            edge_report = self._normalize_edge_report(
                self._load_scope_cascade(current_scope, current_site, filename="edge_report.parquet", organism=organism)
            )
            if edge_report.empty:
                rows.append(
                    {
                        "scope_label": label,
                        "retained_edge_n": 0,
                        "robust_edge_n": 0,
                        "supported_edge_n": 0,
                        "validated_edge_n": 0,
                        "validated_edge_share": 0.0,
                        "max_validated_er": pd.NA,
                        "median_validated_er": pd.NA,
                        "cascade_exists": False,
                        "decision_rule": "No retained edges",
                    }
                )
                continue

            validation_status = edge_report.get("validation_status", pd.Series(index=edge_report.index, dtype=object))
            robust_n = int((validation_status == "robust").sum())
            supported_n = int((validation_status == "supported").sum())
            validated_mask = validation_status.isin(allowed_statuses)
            validated = edge_report.loc[validated_mask].copy()
            validated_edge_n = int(validated_mask.sum())
            retained_edge_n = int(len(edge_report))
            escalation = pd.to_numeric(validated.get("escalation_ratio"), errors="coerce")
            rows.append(
                {
                    "scope_label": label,
                    "retained_edge_n": retained_edge_n,
                    "robust_edge_n": robust_n,
                    "supported_edge_n": supported_n,
                    "validated_edge_n": validated_edge_n,
                    "validated_edge_share": float(validated_edge_n / retained_edge_n) if retained_edge_n > 0 else 0.0,
                    "max_validated_er": float(escalation.max()) if not escalation.dropna().empty else pd.NA,
                    "median_validated_er": float(escalation.median()) if not escalation.dropna().empty else pd.NA,
                    "cascade_exists": validated_edge_n > 0,
                    "decision_rule": "Validated robust/supported edges present" if validated_edge_n > 0 else "No validated robust/supported edges",
                }
            )
        return pd.DataFrame(rows)

    def build_prevalence_shift_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        prevalence = self._load_scope_prevalence(scope, site, filename="prevalence_shift.parquet", organism=organism)
        if prevalence.empty:
            return pd.DataFrame()
        columns = [
            "organism",
            "drug",
            "eligible_n",
            "tested_n",
            "unknown_binary_outcome_n",
            "naive_prevalence_pct",
            "mnar_lambda0_prevalence_pct",
            "mnar_lambda0_shift_from_naive_pct",
            "mnar_status_lambda0",
            "prevalence_lower_bound_pct",
            "prevalence_upper_bound_pct",
            "standardised_prevalence_pct",
            "rho_independent_vs_cascade",
            "cascade_trigger_fraction",
        ]
        available = [column for column in columns if column in prevalence.columns]
        table = prevalence.loc[:, available].copy()
        if "mnar_lambda0_absolute_shift_pct" in prevalence.columns:
            table["_absolute_shift_pct"] = pd.to_numeric(prevalence["mnar_lambda0_absolute_shift_pct"], errors="coerce")
        if "drug" in table.columns:
            table["drug"] = table["drug"].map(self._display_antibiotic_label)
        percent_columns = [
            "naive_prevalence_pct",
            "mnar_lambda0_prevalence_pct",
            "mnar_lambda0_shift_from_naive_pct",
            "prevalence_lower_bound_pct",
            "prevalence_upper_bound_pct",
            "standardised_prevalence_pct",
        ]
        for column in percent_columns:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
        for column in ("rho_independent_vs_cascade", "cascade_trigger_fraction"):
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce").round(4)
        sort_columns = [column for column in ["_absolute_shift_pct", "eligible_n", "tested_n"] if column in table.columns]
        if sort_columns:
            table = table.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
        return table.drop(columns=["_absolute_shift_pct"], errors="ignore").reset_index(drop=True)

    def build_prevalence_shift_diagnostics_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        prevalence = self._load_scope_prevalence(scope, site, filename="prevalence_shift.parquet", organism=organism)
        if prevalence.empty:
            return pd.DataFrame()
        columns = [
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
            "cascade_triggered_observed_n",
            "independent_observed_n",
            "cascade_triggered_tested_n",
            "independent_tested_n",
            "cascade_eligible_n",
            "independent_eligible_n",
            "w_casc_eligible",
            "w_ind_eligible",
            "standardised_prevalence_pct",
            "cascade_trigger_fraction",
            "mean_abs_smd_tested_vs_untested",
            "mean_abs_smd_independent_vs_untested",
            "mean_abs_smd_cascade_vs_untested",
            "flagged_covariate_n_tested_vs_untested",
            "denominator_inflation_effect_pct",
            "mnar_lambda0_prevalence_pct",
            "mnar_lambda0_shift_from_naive_pct",
            "mnar_status_lambda0",
            "mnar_feature_count",
            "mnar_feature_columns",
            "rho_independent_vs_cascade",
        ]
        available = [column for column in columns if column in prevalence.columns]
        table = prevalence.loc[:, available].copy()
        rename_map = {
            "organism": "Organism",
            "drug": "Drug",
            "eligible_n": "Eligible n",
            "observed_n": "Observed n",
            "tested_n": "Binary-evaluable tested n",
            "evaluable_tested_n": "Binary-evaluable tested n",
            "unobserved_n": "Unobserved n",
            "untested_n": "Unobserved n",
            "non_evaluable_observed_n": "Observed non-evaluable n",
            "unknown_binary_outcome_n": "Unknown binary outcome n",
            "cascade_triggered_observed_n": "Cascade-triggered observed n",
            "independent_observed_n": "Independent observed n",
            "cascade_triggered_tested_n": "Cascade-triggered tested n",
            "independent_tested_n": "Independent tested n",
            "cascade_eligible_n": "Cascade eligible n",
            "independent_eligible_n": "Independent eligible n",
            "w_casc_eligible": "Cascade eligible weight",
            "w_ind_eligible": "Independent eligible weight",
            "standardised_prevalence_pct": "Subgroup-standardised prevalence",
            "cascade_trigger_fraction": "Cascade trigger fraction",
            "mean_abs_smd_tested_vs_untested": "Mean abs SMD (observed vs unobserved)",
            "mean_abs_smd_independent_vs_untested": "Mean abs SMD (independent evaluable vs unobserved)",
            "mean_abs_smd_cascade_vs_untested": "Mean abs SMD (cascade evaluable vs unobserved)",
            "flagged_covariate_n_tested_vs_untested": "Flagged covariates (observed vs unobserved)",
            "denominator_inflation_effect_pct": "Denominator inflation effect",
            "mnar_lambda0_prevalence_pct": "MNAR prevalence (lambda=0)",
            "mnar_lambda0_shift_from_naive_pct": "MNAR shift (percentage points)",
            "mnar_status_lambda0": "MNAR status (lambda=0)",
            "mnar_feature_count": "MNAR feature count",
            "mnar_feature_columns": "MNAR feature columns",
            "rho_independent_vs_cascade": "rho independent/cascade",
        }
        table = table.rename(columns=rename_map)
        table = table.loc[:, ~table.columns.duplicated()].copy()
        if "Cascade trigger fraction" in table.columns:
            table["Cascade trigger fraction"] = table["Cascade trigger fraction"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "NA"
            )
        for column in ("Cascade eligible weight", "Independent eligible weight"):
            if column in table.columns:
                table[column] = table[column].map(
                    lambda value: f"{value:.3f}" if pd.notna(value) else "NA"
                )
        if "Subgroup-standardised prevalence" in table.columns:
            table["Subgroup-standardised prevalence"] = table["Subgroup-standardised prevalence"].map(
                lambda value: f"{value:.1f}%" if pd.notna(value) else "NA"
            )
        for column in (
            "Mean abs SMD (observed vs unobserved)",
            "Mean abs SMD (independent evaluable vs unobserved)",
            "Mean abs SMD (cascade evaluable vs unobserved)",
            "MNAR shift (percentage points)",
            "rho independent/cascade",
            "Legacy empirical delta",
            "Legacy delta shift (percentage points)",
        ):
            if column in table.columns:
                table[column] = table[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "NA")
        if "MNAR prevalence (lambda=0)" in table.columns:
            table["MNAR prevalence (lambda=0)"] = table["MNAR prevalence (lambda=0)"].map(
                lambda value: f"{value:.1f}%" if pd.notna(value) else "NA"
            )
        if "Denominator inflation effect" in table.columns:
            table["Denominator inflation effect"] = table["Denominator inflation effect"].map(
                lambda value: f"{value:.1f}%" if pd.notna(value) else "NA"
            )
        if "Legacy delta projected" in table.columns:
            table["Legacy delta projected"] = table["Legacy delta projected"].map(
                lambda value: "Yes" if bool(value) else "No"
            )
        sort_by = ["Eligible n"] if "Eligible n" in table.columns else [table.columns[0]]
        ascending = [False] if "Eligible n" in table.columns else [True]
        return table.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
        ).reset_index(drop=True)

    def _resolve_sites(self, scope: str, site: str | None) -> list[str]:
        if scope == "site" and site:
            return [site]
        return list(self._settings.platform.sites)

    def _load_scope_gold(self, scope: str, site: str | None, filename: str, organism: str | None = None) -> pd.DataFrame:
        return self._safe_read(
            scoped_output_dir(self._paths.paths.gold, scope, site=site, organism=organism) / filename
        )

    def _load_scope_cascade(self, scope: str, site: str | None, filename: str, organism: str | None = None) -> pd.DataFrame:
        return self._safe_read(
            scoped_output_dir(
                self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
                scope,
                site=site,
                organism=organism,
            ) / filename
        )

    def _load_scope_prevalence(self, scope: str, site: str | None, filename: str, organism: str | None = None) -> pd.DataFrame:
        return self._safe_read(
            scoped_output_dir(
                self._paths.paths.artifacts / self._settings.prevalence.output_dir,
                scope,
                site=site,
                organism=organism,
            ) / filename
        )

    def _safe_read(self, path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._dataset_store.read_pandas(path, columns=columns)

    def _read_silver_metadata(self, site: str, canonical_table: str) -> dict[str, object]:
        return self._read_json(
            self._paths.paths.metadata / "datasets" / "silver" / site / f"{canonical_table}.json"
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _normalize_edge_report(self, edge_report: pd.DataFrame) -> pd.DataFrame:
        if edge_report.empty:
            return edge_report
        rename_map = {
            "positive_support_n": "resistant_support_n",
            "negative_support_n": "susceptible_support_n",
            "positive_probability": "resistant_downstream_test_probability",
            "negative_probability": "susceptible_downstream_test_probability",
            "is_retained_edge": "retained_edge",
        }
        normalized = edge_report.rename(columns=rename_map).copy()
        for column in ("upstream_antibiotic", "downstream_antibiotic"):
            if column in normalized.columns:
                key_column = f"__{column}_key"
                raw_column = f"__{column}_raw"
                normalized[raw_column] = normalized[column].astype("string").fillna("Unknown").str.upper()
                normalized[key_column] = normalized[column].map(self._classification.normalize_label)
                display_map = (
                    normalized.groupby(key_column, observed=True)[raw_column]
                    .apply(
                        lambda values: self._classification.canonical_display_label(values.name)
                        or self._choose_preferred_display_label(values, values.name)
                    )
                    .to_dict()
                )
                normalized[column] = normalized[key_column].map(display_map)
        sort_columns = []
        ascending = []
        if "escalation_ratio" in normalized.columns:
            normalized["__has_estimable_ratio"] = normalized["escalation_ratio"].notna() & normalized["escalation_ratio"].gt(1.0)
            sort_columns.append("__has_estimable_ratio")
            ascending.append(False)
        if "total_support_n" in normalized.columns:
            sort_columns.append("total_support_n")
            ascending.append(False)
        sort_columns.extend(["__upstream_antibiotic_key", "__downstream_antibiotic_key"])
        ascending.extend([True, True])
        normalized = normalized.sort_values(sort_columns, ascending=ascending)
        dedup_keys = ["__upstream_antibiotic_key", "__downstream_antibiotic_key"]
        collision_mask = normalized.duplicated(subset=dedup_keys, keep=False)
        if collision_mask.any():
            # Two rows resolved to the same canonical (upstream, downstream) key but
            # were computed as separate statistics upstream — keeping only the first
            # silently discards the other's support/escalation-ratio/validation
            # entirely. This should be rare after normalize_antibiotic_label is
            # applied at ingest; if it fires, the source data or classification
            # reference likely has an unresolved naming collision worth checking
            # before trusting the counts below.
            colliding_pairs = (
                normalized.loc[collision_mask, dedup_keys]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )
            self._logger.warning(
                "_normalize_edge_report: dropping %d row(s) at %d colliding "
                "(upstream, downstream) key(s) after keep='first' dedup: %s",
                int(collision_mask.sum()) - len(colliding_pairs),
                len(colliding_pairs),
                colliding_pairs,
            )
        normalized = normalized.drop_duplicates(dedup_keys, keep="first")
        return normalized.drop(
            columns=[
                "__has_estimable_ratio",
                "__upstream_antibiotic_key",
                "__downstream_antibiotic_key",
                "__upstream_antibiotic_raw",
                "__downstream_antibiotic_raw",
            ],
            errors="ignore",
        )

    def _annotate_aware_transitions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        annotated = self._classification.add_metadata(dataframe.copy(), "upstream_antibiotic", "upstream")
        annotated = self._classification.add_metadata(annotated, "downstream_antibiotic", "downstream")
        annotated["aware_transition"] = annotated.apply(
            lambda row: self._classification.transition_label(
                row.get("upstream_aware_category"),
                row.get("downstream_aware_category"),
            ),
            axis=1,
        )
        annotated["aware_direction"] = annotated.apply(
            lambda row: self._classification.transition_direction(
                row.get("upstream_aware_category"),
                row.get("downstream_aware_category"),
            ),
            axis=1,
        )
        annotated["aware_step"] = annotated.apply(
            lambda row: self._classification.transition_step(
                row.get("upstream_aware_category"),
                row.get("downstream_aware_category"),
            ),
            axis=1,
        )
        return annotated

    def _site_row_count(
        self,
        dataframe: pd.DataFrame,
        site_column: str,
        site: str,
        fallback_path: Path,
    ) -> int:
        if not dataframe.empty and site_column in dataframe.columns:
            return int((dataframe[site_column] == site).sum())
        fallback = self._safe_read(fallback_path)
        return len(fallback)

    @staticmethod
    def _edge_presence_label(row: pd.Series) -> str:
        site_present = pd.notna(row.get("site_escalation_ratio"))
        combined_present = pd.notna(row.get("combined_escalation_ratio"))
        if site_present and combined_present:
            return "shared"
        if site_present:
            return "site_only"
        return "combined_only"

    @staticmethod
    def _choose_preferred_display_label(values: pd.Series, canonical_key: str | None = None) -> str:
        candidates = sorted({str(value).strip().upper() for value in values if str(value).strip()})
        if not candidates:
            return "UNKNOWN"
        if canonical_key:
            for candidate in candidates:
                if candidate == canonical_key:
                    return candidate

        def score(label: str) -> tuple[int, int, int, int]:
            alpha_count = sum(character.isalpha() for character in label)
            separator_bonus = label.count("/") + label.count(" ")
            rm_penalty = 0 if " RM" not in label else -1
            abbreviation_penalty = 0 if len(label) > 4 else -1
            return (abbreviation_penalty, rm_penalty, alpha_count + separator_bonus, len(label))

        return max(candidates, key=score)

    def _display_antibiotic_label(self, value: object) -> str:
        if pd.isna(value):
            return "UNKNOWN"
        text = str(value).strip()
        canonical = self._classification.normalize_label(text)
        return self._classification.canonical_display_label(canonical) or text.upper()

    @staticmethod
    def _component_covariate_label(column: str) -> str:
        if column.startswith("comorb_"):
            name = column[len("comorb_"):].replace("_", " ").strip()
            return f"{name.capitalize()} (comorbidity)" if name else column
        if column.startswith("proc_"):
            name = column[len("proc_"):].replace("_", " ").strip()
            return f"{name.capitalize()} (procedure, 90d)" if name else column
        return column

    def _selection_balance_row(
        self,
        *,
        source_site: object,
        organism: object,
        upstream_antibiotic: object,
        covariate: str,
        covariate_level: str | None,
        metric_type: str,
        tested_n: int,
        untested_n: int,
        tested_value: float,
        untested_value: float,
        tested_sd: float | None = None,
        untested_sd: float | None = None,
    ) -> dict[str, object]:
        if metric_type == "continuous":
            smd = self._continuous_smd(tested_value, untested_value, tested_sd, untested_sd)
        else:
            smd = self._binary_smd(tested_value, untested_value)
        return {
            "site": source_site,
            "organism": organism,
            "upstream_antibiotic": upstream_antibiotic,
            "covariate": covariate,
            "covariate_level": covariate_level or "",
            "metric_type": metric_type,
            "upstream_tested_n": tested_n,
            "upstream_untested_n": untested_n,
            "upstream_tested_value": tested_value,
            "upstream_untested_value": untested_value,
            "standardised_mean_difference": smd,
        }

    @staticmethod
    def _safe_mean(values: pd.Series | None) -> float:
        if values is None or len(values) == 0:
            return math.nan
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == 0:
            return math.nan
        return float(numeric.mean())

    @staticmethod
    def _safe_std(values: pd.Series | None) -> float:
        if values is None or len(values) == 0:
            return math.nan
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() < 2:
            return math.nan
        return float(numeric.std(ddof=1))

    @staticmethod
    def _binary_smd(tested_value: float, untested_value: float) -> float:
        if pd.isna(tested_value) or pd.isna(untested_value):
            return math.nan
        pooled_variance = ((tested_value * (1 - tested_value)) + (untested_value * (1 - untested_value))) / 2
        if pooled_variance <= 0:
            if tested_value == untested_value:
                return 0.0
            return math.copysign(math.inf, tested_value - untested_value)
        return float((tested_value - untested_value) / math.sqrt(pooled_variance))

    @staticmethod
    def _continuous_smd(
        tested_mean: float,
        untested_mean: float,
        tested_sd: float | None,
        untested_sd: float | None,
    ) -> float:
        if pd.isna(tested_mean) or pd.isna(untested_mean) or tested_sd is None or untested_sd is None:
            return math.nan
        if pd.isna(tested_sd) or pd.isna(untested_sd):
            return math.nan
        pooled_sd = math.sqrt(((tested_sd ** 2) + (untested_sd ** 2)) / 2)
        if pooled_sd <= 0:
            return math.nan
        return float((tested_mean - untested_mean) / pooled_sd)

    @staticmethod
    def _count_csv_rows(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
