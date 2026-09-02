"""Manuscript-ready cascade report exports."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer


class CascadeReportBuilder:
    """Build ranked edge tables, path summaries, and adjusted-model diagnostics."""

    def __init__(self, continuity_correction: float = 0.5) -> None:
        self._continuity_correction = continuity_correction

    _EDGE_REPORT_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "resistant_tested_n",
        "susceptible_tested_n",
        "resistant_downstream_test_probability",
        "susceptible_downstream_test_probability",
        "escalation_ratio",
        "er_ci_lower",
        "er_ci_upper",
        "reverse_escalation_ratio",
        "directional_asymmetry_score",
        "resistant_support_n",
        "susceptible_support_n",
        "total_support_n",
        "adjusted_log_odds",
        "adjusted_odds_ratio",
        "adjusted_standard_error",
        "adjusted_log_odds_ci_lower",
        "adjusted_log_odds_ci_upper",
        "adjusted_odds_ratio_ci_lower",
        "adjusted_odds_ratio_ci_upper",
        "adjusted_p_value",
        "adjusted_or_e_value",
        "adjusted_or_ci_e_value",
        "raw_effect_direction",
        "adjusted_effect_direction",
        "raw_adjusted_direction_agreement",
        "adjusted_or_positive",
        "modeled_n",
        "clustered_n",
        "clustering_variable",
        "supports_adjusted_model",
        "smoothed_pair_test_rate",
        "smoothed_reverse_test_rate",
        "testing_asymmetry_score",
        "testing_asymmetry_bin",
        "panel_bundling_index",
        "mean_upstream_panel_size",
        "median_upstream_panel_size",
        "panel_size_q90",
        "bundled_panel_fraction",
        "max_other_downstream_test_rate",
        "mean_other_downstream_test_rate",
        "passes_support_threshold",
        "retained_edge",
        "permutation_p_value",
        "permutation_fdr_q_value",
        "permutation_fdr_supported",
        "permutation_null_median_er",
        "permutation_null_q05_er",
        "permutation_null_q95_er",
        "bootstrap_median_er",
        "bootstrap_er_ci_lower",
        "bootstrap_er_ci_upper",
        "bootstrap_sign_stability",
        "bootstrap_retention_rate",
        "site_replication_n",
        "site_direction_agreement_rate",
        "site_median_er",
        "cascade_direction",
        "temporal_early_er",
        "temporal_late_er",
        "temporal_n_early",
        "temporal_n_late",
        "temporal_direction_agreement",
        "temporal_log_ratio_delta",
        "site_pooled_log_er",
        "site_pooled_se",
        "site_i_squared",
        "site_cochran_q",
        "site_heterogeneity_p",
        "between_site_permutation_p_value",
        "between_site_permutation_fdr_q_value",
        "between_site_permutation_supported",
        "validation_status",
        "rank_by_escalation_ratio",
        "rank_by_adjusted_odds_ratio",
    ]
    _TOP_PATH_COLUMNS = [
        "path_start",
        "path_mid",
        "edge_1_escalation_ratio",
        "edge_1_adjusted_odds_ratio",
        "edge_1_support_n",
        "path_end",
        "edge_2_escalation_ratio",
        "edge_2_adjusted_odds_ratio",
        "edge_2_support_n",
        "path_escalation_ratio",
        "path_log_weight",
        "path_min_support_n",
    ]
    _DIAGNOSTIC_COLUMNS = [
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
        "adjusted_or_e_value",
        "adjusted_or_ci_e_value",
        "modeled_n",
        "clustered_n",
        "clustering_variable",
        "supports_adjusted_model",
        "adjusted_effect_direction",
        "model_rank",
    ]

    def export(
        self,
        retained_edges: pd.DataFrame,
        adjusted_results: pd.DataFrame,
        output_dir: Path,
        validation_results: pd.DataFrame | None = None,
        dependence_results: pd.DataFrame | None = None,
        escalation_results: pd.DataFrame | None = None,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        edge_report_path = output_dir / "edge_report.parquet"
        top_paths_path = output_dir / "top_paths.parquet"
        diagnostics_path = output_dir / "adjusted_model_diagnostics.parquet"
        summary_path = output_dir / "report_summary.json"

        edge_report = self._build_edge_report(retained_edges, validation_results, dependence_results, escalation_results)
        top_path_edges = (
            self._validated_edges_for_primary_outputs(edge_report)
            if validation_results is not None
            else retained_edges
        )
        top_paths = self.build_top_paths(top_path_edges)
        diagnostics = self._build_diagnostics(adjusted_results)

        edge_report.to_parquet(edge_report_path, index=False)
        top_paths.to_parquet(top_paths_path, index=False)
        diagnostics.to_parquet(diagnostics_path, index=False)

        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "row_counts": {
                "edge_report": len(edge_report),
                "top_paths": len(top_paths),
                "adjusted_model_diagnostics": len(diagnostics),
            },
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        return {
            "edge_report_path": edge_report_path,
            "top_paths_path": top_paths_path,
            "diagnostics_path": diagnostics_path,
            "report_summary_path": summary_path,
        }

    def _build_edge_report(
        self,
        retained_edges: pd.DataFrame,
        validation_results: pd.DataFrame | None,
        dependence_results: pd.DataFrame | None,
        escalation_results: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if retained_edges.empty:
            return pd.DataFrame(columns=self._EDGE_REPORT_COLUMNS)
        report = retained_edges.rename(
            columns={
                "positive_probability": "resistant_downstream_test_probability",
                "negative_probability": "susceptible_downstream_test_probability",
                "positive_support_n": "resistant_support_n",
                "negative_support_n": "susceptible_support_n",
                "positive_tested_n": "resistant_tested_n",
                "negative_tested_n": "susceptible_tested_n",
                "is_retained_edge": "retained_edge",
            }
        ).copy()
        defaults = {
            "resistant_tested_n": 0,
            "susceptible_tested_n": 0,
            "resistant_support_n": 0,
            "susceptible_support_n": 0,
            "resistant_downstream_test_probability": 0.0,
            "susceptible_downstream_test_probability": 0.0,
        }
        for column, default in defaults.items():
            if column not in report.columns:
                report[column] = default
        if "supports_adjusted_model" not in report.columns:
            report["supports_adjusted_model"] = report["modeled_n"].notna() if "modeled_n" in report.columns else False
        if "adjusted_or_positive" not in report.columns:
            report["adjusted_or_positive"] = pd.Series(pd.NA, index=report.index, dtype="boolean")
        else:
            report["adjusted_or_positive"] = report["adjusted_or_positive"].astype("boolean")
        for column in ("adjusted_or_e_value", "adjusted_or_ci_e_value"):
            if column not in report.columns:
                report[column] = np.nan
        report = report.sort_values(
            by=["escalation_ratio", "adjusted_odds_ratio", "total_support_n"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        report["rank_by_escalation_ratio"] = report["escalation_ratio"].rank(
            method="dense", ascending=False
        ).astype(int)
        report["rank_by_adjusted_odds_ratio"] = (
            report["adjusted_odds_ratio"]
            .fillna(-math.inf)
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        report["raw_effect_direction"] = report["escalation_ratio"].map(self._effect_direction_from_ratio)
        report["adjusted_effect_direction"] = report["adjusted_odds_ratio"].map(self._effect_direction_from_ratio)
        report["raw_adjusted_direction_agreement"] = report.apply(
            lambda row: self._compare_effect_directions(
                row.get("raw_effect_direction"),
                row.get("adjusted_effect_direction"),
            ),
            axis=1,
        )
        if validation_results is not None and not validation_results.empty:
            report = report.merge(
                validation_results,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
                suffixes=("", "_validation"),
            )
            report = self._resolve_duplicate_cascade_direction(report)
        if dependence_results is not None and not dependence_results.empty:
            report = report.merge(
                dependence_results,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
            )
        validation_defaults = {
            "permutation_p_value": np.nan,
            "permutation_fdr_q_value": np.nan,
            "permutation_fdr_supported": False,
            "permutation_null_median_er": np.nan,
            "permutation_null_q05_er": np.nan,
            "permutation_null_q95_er": np.nan,
            "bootstrap_median_er": np.nan,
            "bootstrap_er_ci_lower": np.nan,
            "bootstrap_er_ci_upper": np.nan,
            "bootstrap_sign_stability": np.nan,
            "bootstrap_retention_rate": np.nan,
            "site_replication_n": 0,
            "site_direction_agreement_rate": np.nan,
            "site_median_er": np.nan,
            "temporal_early_er": np.nan,
            "temporal_late_er": np.nan,
            "temporal_direction_agreement": False,
            "temporal_log_ratio_delta": np.nan,
            "validation_status": "unavailable",
        }
        for column, default in validation_defaults.items():
            if column not in report.columns:
                report[column] = default
        dependence_defaults = {
            "smoothed_pair_test_rate": np.nan,
            "smoothed_reverse_test_rate": np.nan,
            "testing_asymmetry_score": np.nan,
            "testing_asymmetry_bin": "unavailable",
            "panel_bundling_index": np.nan,
            "mean_upstream_panel_size": np.nan,
            "median_upstream_panel_size": np.nan,
            "panel_size_q90": np.nan,
            "bundled_panel_fraction": np.nan,
            "max_other_downstream_test_rate": np.nan,
            "mean_other_downstream_test_rate": np.nan,
        }
        for column, default in dependence_defaults.items():
            if column not in report.columns:
                report[column] = default
        report["er_ci_lower"] = np.nan
        report["er_ci_upper"] = np.nan
        valid = (
            report["escalation_ratio"].notna()
            & report["escalation_ratio"].gt(0)
            & report["resistant_tested_n"].gt(0)
            & report["susceptible_tested_n"].gt(0)
            & report["resistant_support_n"].gt(0)
            & report["susceptible_support_n"].gt(0)
        )
        if valid.any():
            cc = float(self._continuity_correction)
            a = report.loc[valid, "resistant_tested_n"].astype(float) + cc
            n1 = report.loc[valid, "resistant_support_n"].astype(float) + 2.0 * cc
            c = report.loc[valid, "susceptible_tested_n"].astype(float) + cc
            n0 = report.loc[valid, "susceptible_support_n"].astype(float) + 2.0 * cc
            se = ((1 / a) - (1 / n1) + (1 / c) - (1 / n0)).map(math.sqrt)
            log_rr = report.loc[valid, "escalation_ratio"].map(math.log)
            report.loc[valid, "er_ci_lower"] = (log_rr - 1.96 * se).map(math.exp)
            report.loc[valid, "er_ci_upper"] = (log_rr + 1.96 * se).map(math.exp)

        # Directional Asymmetry Score: DAS_{j,k} = log(ER_{j→k}) - log(ER_{k→j}).
        # Bundled panels produce symmetric co-observation (DAS ≈ 0); result-conditioned
        # cascade patterns produce systematic asymmetry.
        report["reverse_escalation_ratio"] = np.nan
        report["directional_asymmetry_score"] = np.nan
        if escalation_results is not None and not escalation_results.empty and "escalation_ratio" in escalation_results.columns:
            er_lookup = (
                escalation_results.loc[:, ["upstream_antibiotic", "downstream_antibiotic", "escalation_ratio"]]
                .dropna(subset=["escalation_ratio"])
                .rename(columns={
                    "upstream_antibiotic": "downstream_antibiotic",
                    "downstream_antibiotic": "upstream_antibiotic",
                    "escalation_ratio": "reverse_escalation_ratio",
                })
            )
            report = report.merge(
                er_lookup,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
                suffixes=("", "_lookup"),
            )
            if "reverse_escalation_ratio_lookup" in report.columns:
                # Merge created a duplicate; resolve by taking the non-NaN value
                report["reverse_escalation_ratio"] = report["reverse_escalation_ratio_lookup"].combine_first(
                    report["reverse_escalation_ratio"]
                )
                report = report.drop(columns=["reverse_escalation_ratio_lookup"])
            has_both = (
                report["escalation_ratio"].gt(0).fillna(False)
                & report["reverse_escalation_ratio"].gt(0).fillna(False)
            )
            report.loc[has_both, "directional_asymmetry_score"] = (
                report.loc[has_both, "escalation_ratio"].map(math.log)
                - report.loc[has_both, "reverse_escalation_ratio"].map(math.log)
            )

        return report.reindex(columns=self._EDGE_REPORT_COLUMNS)

    @staticmethod
    def _resolve_duplicate_cascade_direction(report: pd.DataFrame) -> pd.DataFrame:
        """Keep one canonical cascade_direction column after validation merge."""
        validation_column = "cascade_direction_validation"
        if validation_column not in report.columns:
            return report
        if "cascade_direction" not in report.columns:
            report["cascade_direction"] = report[validation_column]
            return report.drop(columns=[validation_column])

        left = report["cascade_direction"]
        right = report[validation_column]
        disagree = left.notna() & right.notna() & left.ne(right)
        if disagree.any():
            examples = report.loc[
                disagree,
                ["upstream_antibiotic", "downstream_antibiotic", "cascade_direction", validation_column],
            ].head(5)
            raise ValueError(
                "cascade_direction mismatch between retained_edges and validation_results: "
                f"{examples.to_dict(orient='records')}"
            )
        report["cascade_direction"] = left.combine_first(right)
        return report.drop(columns=[validation_column])

    @staticmethod
    def build_top_paths(retained_edges: pd.DataFrame) -> pd.DataFrame:
        """Discover 2-hop chains (A->B->C) by inner-joining edges on the shared midpoint.

        Pure function of `retained_edges` (needs only upstream/downstream_antibiotic,
        escalation_ratio, adjusted_odds_ratio, total_support_n) so any edge subset can be
        seeded through it -- e.g. report_export_workflow's adjusted-OR-confirmed pathway
        tier, which chains a differently-filtered edge pool than the primary export() call
        below does.
        """
        if retained_edges.empty:
            return pd.DataFrame(columns=CascadeReportBuilder._TOP_PATH_COLUMNS)

        edges = retained_edges.loc[
            :,
            [
                "upstream_antibiotic",
                "downstream_antibiotic",
                "escalation_ratio",
                "adjusted_odds_ratio",
                "total_support_n",
            ],
        ].copy()
        first = edges.rename(
            columns={
                "upstream_antibiotic": "path_start",
                "downstream_antibiotic": "path_mid",
                "escalation_ratio": "edge_1_escalation_ratio",
                "adjusted_odds_ratio": "edge_1_adjusted_odds_ratio",
                "total_support_n": "edge_1_support_n",
            }
        )
        second = edges.rename(
            columns={
                "upstream_antibiotic": "path_mid",
                "downstream_antibiotic": "path_end",
                "escalation_ratio": "edge_2_escalation_ratio",
                "adjusted_odds_ratio": "edge_2_adjusted_odds_ratio",
                "total_support_n": "edge_2_support_n",
            }
        )
        paths = first.merge(second, on="path_mid", how="inner")
        paths = paths[
            (paths["path_start"] != paths["path_end"])
            & (paths["path_start"] != paths["path_mid"])
            & (paths["path_mid"] != paths["path_end"])
        ].copy()
        if paths.empty:
            return pd.DataFrame(columns=CascadeReportBuilder._TOP_PATH_COLUMNS)

        paths["path_escalation_ratio"] = (
            paths["edge_1_escalation_ratio"] * paths["edge_2_escalation_ratio"]
        )
        paths["path_log_weight"] = paths["path_escalation_ratio"].map(
            lambda value: math.log(value) if pd.notna(value) and value > 0 else None
        )
        paths["path_min_support_n"] = paths[
            ["edge_1_support_n", "edge_2_support_n"]
        ].min(axis=1)
        return paths.sort_values(
            by=["path_escalation_ratio", "path_min_support_n"],
            ascending=[False, False],
        ).reset_index(drop=True).reindex(columns=CascadeReportBuilder._TOP_PATH_COLUMNS)

    @staticmethod
    def _validated_edges_for_primary_outputs(edge_report: pd.DataFrame) -> pd.DataFrame:
        """Use only robust/supported edges for primary pathway outputs."""
        if edge_report.empty or "validation_status" not in edge_report.columns:
            return edge_report.head(0).copy()
        return edge_report.loc[
            edge_report["validation_status"].isin(CascadeValidationAnalyzer.VALIDATED_STATUSES)
        ].reset_index(drop=True)

    def _build_diagnostics(self, adjusted_results: pd.DataFrame) -> pd.DataFrame:
        if adjusted_results.empty:
            return pd.DataFrame(columns=self._DIAGNOSTIC_COLUMNS)
        diagnostics = adjusted_results.copy()
        diagnostics["adjusted_effect_direction"] = diagnostics["adjusted_log_odds"].map(
            lambda value: "positive" if value > 0 else "negative"
        )
        diagnostics["model_rank"] = (
            diagnostics["adjusted_odds_ratio"]
            .fillna(-math.inf)
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        return diagnostics.sort_values(
            by=["adjusted_odds_ratio", "modeled_n"],
            ascending=[False, False],
        ).reset_index(drop=True).reindex(columns=self._DIAGNOSTIC_COLUMNS)

    @staticmethod
    def _effect_direction_from_ratio(value: float | int | None) -> str:
        if pd.isna(value):
            return "unavailable"
        if value > 1:
            return "positive"
        if value < 1:
            return "negative"
        return "neutral"

    @classmethod
    def _compare_effect_directions(cls, raw_direction: str, adjusted_direction: str) -> str:
        if adjusted_direction == "unavailable":
            return "adjusted_unavailable"
        if raw_direction == adjusted_direction:
            return "directionally_aligned"
        return "directionally_misaligned"
