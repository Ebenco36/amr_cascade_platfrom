"""Workflow for denominator-aware prevalence-shift analysis."""

from __future__ import annotations

import json
from dataclasses import replace
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.statistics.cascade_covariate_builder import CascadeCovariateBuilder
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.surveillance.prevalence_shift_analyzer import PrevalenceShiftAnalyzer
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_prevalence_plotter import PlotlyPrevalencePlotter


@dataclass(frozen=True)
class PrevalenceShiftRequest:
    scope: str = "combined"
    site: str | None = None
    organism: str | None = None
    drugs: tuple[str, ...] | None = None
    top_n: int | None = None
    figure_formats: tuple[str, ...] | None = None
    min_eligible_support: int | None = None
    min_tested_support: int | None = None
    min_resistant_support: int | None = None


class PrevalenceShiftWorkflow:
    """Run prevalence-shift analysis from gold datasets."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)
        self._covariate_builder = CascadeCovariateBuilder(settings, path_manager)
        self._plotter = PlotlyPrevalencePlotter(
            exporter=PlotlyFigureExporter(
                width=settings.reporting.image_width,
                height=settings.reporting.image_height,
            ),
            template=settings.reporting.plotly_template,
            width=settings.reporting.image_width,
            height=settings.reporting.image_height,
        )

    def run(self, request: PrevalenceShiftRequest) -> dict[str, Path]:
        gold_dir = scoped_output_dir(
            root=self._paths.paths.gold,
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        eligible_path = gold_dir / "eligible_pairs.parquet"
        culture_drug_path = gold_dir / "culture_drug_episodes.parquet"
        culture_episode_path = gold_dir / "culture_episodes.parquet"
        cascade_edge_report_path = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        ) / "edge_report.parquet"
        for path in (eligible_path, culture_drug_path, culture_episode_path):
            if not path.exists():
                raise DataDiscoveryError(f"Required prevalence analysis input not found: {path}")

        eligible_pairs = self._dataset_store.read_pandas(eligible_path)
        culture_drug = self._dataset_store.read_pandas(culture_drug_path)
        culture_episodes = self._dataset_store.read_pandas(culture_episode_path)
        if request.drugs:
            allowed = {drug.upper() for drug in request.drugs}
            eligible_pairs = eligible_pairs.loc[
                eligible_pairs["antibiotic"].astype(str).str.upper().isin(allowed)
            ].copy()
            culture_drug = culture_drug.loc[
                culture_drug["antibiotic"].astype(str).str.upper().isin(allowed)
            ].copy()
        covariates = self._covariate_builder.build(culture_episodes)
        cascade_edge_report = (
            self._dataset_store.read_pandas(cascade_edge_report_path) if cascade_edge_report_path.exists() else pd.DataFrame()
        )
        analyzer = self._build_analyzer(request)
        bundle = analyzer.analyze(eligible_pairs, culture_drug, covariates, cascade_edge_report)

        results = bundle.pair_results.copy()
        if request.drugs:
            allowed = {drug.upper() for drug in request.drugs}
            results = results.loc[results["drug"].astype(str).str.upper().isin(allowed)].copy()
        top_n = request.top_n or self._settings.prevalence.default_top_n
        sort_columns = ["mnar_lambda0_absolute_shift_pct", "eligible_n", "tested_n"]
        if results.empty or not set(sort_columns).issubset(results.columns):
            plot_results = results.head(0).copy()
        else:
            plot_results = results.sort_values(
                by=sort_columns,
                ascending=[False, False, False],
            ).head(top_n).copy()

        output_dir = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.prevalence.output_dir,
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = self._paths.project_root / self._settings.reporting.tables_dir / scoped_output_dir(
            root=Path("."),
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        figures_dir = self._paths.project_root / self._settings.reporting.figures_dir / scoped_output_dir(
            root=Path("."),
            scope=request.scope,
            site=request.site,
            organism=request.organism,
        )
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, Path] = {}
        pair_path = output_dir / "prevalence_shift.parquet"
        results.to_parquet(pair_path, index=False)
        outputs[pair_path.name] = pair_path

        legacy_dir = output_dir / "legacy"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        curve_path = legacy_dir / "legacy_delta_sensitivity_curves.parquet"
        bundle.delta_curve_results.to_parquet(curve_path, index=False)
        outputs[f"legacy/{curve_path.name}"] = curve_path

        # Do not leave deprecated delta outputs in the primary manuscript artifact surface.
        for stale_path in (
            output_dir / "legacy_delta_sensitivity_curves.parquet",
            tables_dir / "table_k_legacy_delta_sensitivity_curve_points.csv",
        ):
            if stale_path.exists():
                stale_path.unlink()

        mnar_curve_path = output_dir / "prevalence_mnar_sensitivity_curves.parquet"
        bundle.mnar_curve_results.to_parquet(mnar_curve_path, index=False)
        outputs[mnar_curve_path.name] = mnar_curve_path

        tipping_point_path = output_dir / "prevalence_mnar_tipping_points.parquet"
        bundle.mnar_tipping_point_results.to_parquet(tipping_point_path, index=False)
        outputs[tipping_point_path.name] = tipping_point_path

        enrichment_path = output_dir / "validated_edge_enrichment.parquet"
        bundle.edge_enrichment_results.to_parquet(enrichment_path, index=False)
        outputs[enrichment_path.name] = enrichment_path

        summary_table = self._format_summary_table(results)
        summary_csv = tables_dir / "table_k_prevalence_shift.csv"
        summary_table.to_csv(summary_csv, index=False)
        outputs[summary_csv.name] = summary_csv

        diagnostics_table = self._format_diagnostics_table(results)
        diagnostics_csv = tables_dir / "table_k_prevalence_shift_diagnostics.csv"
        diagnostics_table.to_csv(diagnostics_csv, index=False)
        outputs[diagnostics_csv.name] = diagnostics_csv

        bounds_table = self._format_bounds_table(results)
        bounds_csv = tables_dir / "table_k_prevalence_shift_bounds.csv"
        bounds_table.to_csv(bounds_csv, index=False)
        outputs[bounds_csv.name] = bounds_csv

        mnar_curve_table = self._format_mnar_curve_table(bundle.mnar_curve_results)
        mnar_curve_csv = tables_dir / "table_k_mnar_sensitivity_curve_points.csv"
        mnar_curve_table.to_csv(mnar_curve_csv, index=False)
        outputs[mnar_curve_csv.name] = mnar_curve_csv

        tipping_point_table = self._format_mnar_tipping_point_table(bundle.mnar_tipping_point_results)
        tipping_point_csv = tables_dir / "table_k_mnar_tipping_points.csv"
        tipping_point_table.to_csv(tipping_point_csv, index=False)
        outputs[tipping_point_csv.name] = tipping_point_csv

        plot_stem = figures_dir / "figure_prevalence_shift_forest"
        figure_formats = request.figure_formats or self._settings.reporting.default_figure_formats
        outputs.update(self._plotter.export_forest(plot_results, plot_stem, figure_formats))

        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": request.scope,
            "site": request.site,
            "organism": request.organism,
            "row_count": int(len(results)),
            "legacy_delta_curve_row_count": int(len(bundle.delta_curve_results)),
            "mnar_curve_row_count": int(len(bundle.mnar_curve_results)),
            "mnar_tipping_point_row_count": int(len(bundle.mnar_tipping_point_results)),
            "edge_enrichment_row_count": int(len(bundle.edge_enrichment_results)),
            "top_n_plotted": int(len(plot_results)),
            "drugs_filter": list(request.drugs or ()),
            "delta_curve_points": self._settings.prevalence.delta_curve_points,
            "mnar_lambda_grid": list(self._settings.prevalence.mnar_lambda_grid),
            "decision_thresholds": list(self._settings.prevalence.decision_thresholds),
        }
        summary_path = output_dir / "prevalence_shift_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs[summary_path.name] = summary_path
        return outputs

    def _build_analyzer(self, request: PrevalenceShiftRequest) -> PrevalenceShiftAnalyzer:
        prevalence_settings = self._settings.prevalence
        if (
            request.min_eligible_support is not None
            or request.min_tested_support is not None
            or request.min_resistant_support is not None
        ):
            prevalence_settings = replace(
                prevalence_settings,
                min_eligible_support=(
                    request.min_eligible_support
                    if request.min_eligible_support is not None
                    else prevalence_settings.min_eligible_support
                ),
                min_tested_support=(
                    request.min_tested_support
                    if request.min_tested_support is not None
                    else prevalence_settings.min_tested_support
                ),
                min_resistant_support=(
                    request.min_resistant_support
                    if request.min_resistant_support is not None
                    else prevalence_settings.min_resistant_support
                ),
            )
        analysis_settings = replace(self._settings, prevalence=prevalence_settings)
        return PrevalenceShiftAnalyzer(analysis_settings)

    @staticmethod
    def _format_summary_table(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Naive prevalence",
                    "MNAR prevalence (lambda=0)",
                    "Subgroup-standardised prevalence",
                    "Shift (percentage points)",
                    "Lower bound",
                    "Upper bound",
                    "rho independent/cascade",
                    "Cascade trigger fraction",
                ]
            )
        table = results.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Naive prevalence"] = table["naive_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["MNAR prevalence (lambda=0)"] = table["mnar_lambda0_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Subgroup-standardised prevalence"] = table["standardised_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Shift (percentage points)"] = table["mnar_lambda0_shift_from_naive_pct"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["Lower bound"] = table["prevalence_lower_bound_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Upper bound"] = table["prevalence_upper_bound_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["rho independent/cascade"] = table["rho_independent_vs_cascade"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        table["Cascade trigger fraction"] = table["cascade_trigger_fraction"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "NA"
        )
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Naive prevalence",
                "MNAR prevalence (lambda=0)",
                "Subgroup-standardised prevalence",
                "Shift (percentage points)",
                "Lower bound",
                "Upper bound",
                "rho independent/cascade",
                "Cascade trigger fraction",
            ],
        ]

    @staticmethod
    def _format_diagnostics_table(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Eligible n",
                    "Observed n",
                    "Binary-evaluable tested n",
                    "Unobserved n",
                    "Observed non-evaluable n",
                    "Unknown binary outcome n",
                    "Cascade-triggered observed n",
                    "Independent observed n",
                    "Cascade-triggered tested n",
                    "Independent tested n",
                    "Cascade eligible n",
                    "Independent eligible n",
                    "Cascade eligible weight",
                    "Independent eligible weight",
                    "Mean abs SMD (observed vs unobserved)",
                    "Mean abs SMD (independent evaluable vs unobserved)",
                    "Mean abs SMD (cascade evaluable vs unobserved)",
                    "Flagged covariates (observed vs unobserved)",
                    "Denominator inflation effect",
                ]
            )
        table = results.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Eligible n"] = table["eligible_n"]
        table["Observed n"] = table["observed_n"]
        table["Binary-evaluable tested n"] = table["evaluable_tested_n"]
        table["Unobserved n"] = table["unobserved_n"]
        table["Observed non-evaluable n"] = table["non_evaluable_observed_n"]
        table["Unknown binary outcome n"] = table["unknown_binary_outcome_n"]
        table["Cascade-triggered observed n"] = table["cascade_triggered_observed_n"]
        table["Independent observed n"] = table["independent_observed_n"]
        table["Cascade-triggered tested n"] = table["cascade_triggered_tested_n"]
        table["Independent tested n"] = table["independent_tested_n"]
        table["Cascade eligible n"] = table["cascade_eligible_n"]
        table["Independent eligible n"] = table["independent_eligible_n"]
        table["Cascade eligible weight"] = table["w_casc_eligible"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else "NA"
        )
        table["Independent eligible weight"] = table["w_ind_eligible"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else "NA"
        )
        table["Mean abs SMD (observed vs unobserved)"] = table["mean_abs_smd_tested_vs_untested"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        table["Mean abs SMD (independent evaluable vs unobserved)"] = table["mean_abs_smd_independent_vs_untested"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        table["Mean abs SMD (cascade evaluable vs unobserved)"] = table["mean_abs_smd_cascade_vs_untested"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        table["Flagged covariates (observed vs unobserved)"] = table["flagged_covariate_n_tested_vs_untested"]
        table["Denominator inflation effect"] = table["denominator_inflation_effect_pct"].map(PrevalenceShiftWorkflow._format_percent)
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Eligible n",
                "Observed n",
                "Binary-evaluable tested n",
                "Unobserved n",
                "Observed non-evaluable n",
                "Unknown binary outcome n",
                "Cascade-triggered observed n",
                "Independent observed n",
                "Cascade-triggered tested n",
                "Independent tested n",
                "Cascade eligible n",
                "Independent eligible n",
                "Cascade eligible weight",
                "Independent eligible weight",
                "Mean abs SMD (observed vs unobserved)",
                "Mean abs SMD (independent evaluable vs unobserved)",
                "Mean abs SMD (cascade evaluable vs unobserved)",
                "Flagged covariates (observed vs unobserved)",
                "Denominator inflation effect",
            ],
        ]

    @staticmethod
    def _format_bounds_table(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Naive prevalence",
                    "Lower bound",
                    "Upper bound",
                    "Bound width",
                    "MNAR prevalence (lambda=0)",
                    "MNAR shift (percentage points)",
                    "MNAR feature count",
                    "MNAR status (lambda=0)",
                ]
            )
        table = results.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Naive prevalence"] = table["naive_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Lower bound"] = table["prevalence_lower_bound_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Upper bound"] = table["prevalence_upper_bound_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Bound width"] = table["prevalence_bound_width_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["MNAR prevalence (lambda=0)"] = table["mnar_lambda0_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["MNAR shift (percentage points)"] = table["mnar_lambda0_shift_from_naive_pct"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["MNAR feature count"] = table["mnar_feature_count"]
        table["MNAR status (lambda=0)"] = table["mnar_status_lambda0"]
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Naive prevalence",
                "Lower bound",
                "Upper bound",
                "Bound width",
                "MNAR prevalence (lambda=0)",
                "MNAR shift (percentage points)",
                "MNAR feature count",
                "MNAR status (lambda=0)",
            ],
        ]

    @staticmethod
    def _format_legacy_delta_curve_table(curves: pd.DataFrame) -> pd.DataFrame:
        if curves.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Delta",
                    "Eligible-scale prevalence",
                    "Shift (percentage points)",
                    "Empirical anchor",
                    "Null-selection reference",
                    "Upper bound point",
                ]
            )
        table = curves.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Delta"] = table["delta"].map(lambda value: f"{value:.3f}")
        table["Eligible-scale prevalence"] = table["shifted_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Shift (percentage points)"] = table["delta_prevalence_pct"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["Empirical anchor"] = table["is_empirical_anchor"].map(lambda value: "Yes" if bool(value) else "No")
        table["Null-selection reference"] = table["is_null_selection_reference"].map(lambda value: "Yes" if bool(value) else "No")
        table["Upper bound point"] = table["is_upper_bound"].map(lambda value: "Yes" if bool(value) else "No")
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Delta",
                "Eligible-scale prevalence",
                "Shift (percentage points)",
                "Empirical anchor",
                "Null-selection reference",
                "Upper bound point",
            ],
        ]

    @staticmethod
    def _format_mnar_curve_table(curves: pd.DataFrame) -> pd.DataFrame:
        if curves.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Lambda",
                    "MNAR prevalence",
                    "Shift from naive (percentage points)",
                    "Expected resistant unknown n",
                    "MNAR feature count",
                    "MNAR feature columns",
                    "Status",
                ]
            )
        table = curves.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Lambda"] = table["mnar_lambda"].map(lambda value: f"{value:.2f}")
        table["MNAR prevalence"] = table["mnar_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Shift from naive (percentage points)"] = table["mnar_shift_from_naive_pct"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["Expected resistant unknown n"] = table["mnar_unknown_expected_resistant_n"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["MNAR feature count"] = table["mnar_feature_count"]
        table["MNAR feature columns"] = table["mnar_feature_columns"]
        table["Status"] = table["mnar_status"]
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Lambda",
                "MNAR prevalence",
                "Shift from naive (percentage points)",
                "Expected resistant unknown n",
                "MNAR feature count",
                "MNAR feature columns",
                "Status",
            ],
        ]

    @staticmethod
    def _format_mnar_tipping_point_table(tipping_points: pd.DataFrame) -> pd.DataFrame:
        if tipping_points.empty:
            return pd.DataFrame(
                columns=[
                    "Organism",
                    "Drug",
                    "Decision threshold",
                    "Naive prevalence",
                    "MNAR prevalence at lambda=0",
                    "Shift at lambda=0",
                    "Tipping lambda",
                    "Crossing status",
                    "Eligible n",
                    "Binary-evaluable tested n",
                    "Unknown binary outcome n",
                    "Cascade trigger fraction",
                    "rho independent/cascade",
                ]
            )
        table = tipping_points.copy()
        table["Organism"] = table["organism"]
        table["Drug"] = table["drug"]
        table["Decision threshold"] = table["decision_threshold_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["Naive prevalence"] = table["naive_prevalence_pct"].map(PrevalenceShiftWorkflow._format_percent)
        table["MNAR prevalence at lambda=0"] = table["prevalence_at_lambda0_pct"].map(
            PrevalenceShiftWorkflow._format_percent
        )
        table["Shift at lambda=0"] = table["shift_at_lambda0_pct"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "NA"
        )
        table["Tipping lambda"] = table["mnar_lambda_star"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        table["Crossing status"] = table["crossing_status"]
        table["Eligible n"] = table["eligible_n"]
        table["Binary-evaluable tested n"] = table["tested_n"]
        table["Unknown binary outcome n"] = table["unknown_binary_outcome_n"]
        table["Cascade trigger fraction"] = table["cascade_trigger_fraction"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "NA"
        )
        table["rho independent/cascade"] = table["rho_independent_vs_cascade"].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "NA"
        )
        return table.loc[
            :,
            [
                "Organism",
                "Drug",
                "Decision threshold",
                "Naive prevalence",
                "MNAR prevalence at lambda=0",
                "Shift at lambda=0",
                "Tipping lambda",
                "Crossing status",
                "Eligible n",
                "Binary-evaluable tested n",
                "Unknown binary outcome n",
                "Cascade trigger fraction",
                "rho independent/cascade",
            ],
        ]

    @staticmethod
    def _format_percent(value: object) -> str:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.1f}%"
