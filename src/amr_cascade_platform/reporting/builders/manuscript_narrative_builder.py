"""Build manuscript-facing narrative summaries, captions, and notes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager


class ManuscriptNarrativeBuilder:
    """Create scope-level markdown summaries and interpretation notes."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager

    def build_scope_summary_markdown(
        self,
        scope: str,
        site: str | None,
        selected_figures: tuple[str, ...],
        figure_formats: tuple[str, ...],
    ) -> str:
        scope_dir = site if scope == "site" and site else "combined"
        cascade_summary = self._read_json(
            self._paths.paths.artifacts
            / self._settings.cascade.outputs.result_dir
            / scope_dir
            / "cascade_summary.json"
        )
        report_summary = self._read_json(
            self._paths.paths.artifacts
            / self._settings.cascade.outputs.result_dir
            / scope_dir
            / "report_summary.json"
        )
        comparison_summary = self._safe_read_parquet(
            self._paths.paths.artifacts
            / self._settings.cascade.outputs.result_dir
            / scope_dir
            / "site_vs_combined_summary.parquet"
        )
        primary_table = self._safe_read_csv(
            self._paths._project_root
            / self._settings.reporting.tables_dir
            / scope_dir
            / "table_c_primary_cascade.csv"
        )

        lines = [
            f"# Scope Summary: {scope_dir}",
            "",
            "## Overview",
            self._overview_paragraph(scope, site, cascade_summary),
            "",
            "## Key Quantitative Summary",
            self._quantitative_summary(cascade_summary, report_summary, primary_table),
            "",
            "## Interpretation Boundary",
            (
                "These results describe empirical patterns consistent with diagnostic escalation cascades. "
                "Primary analyses are restricted to directly observed AST results so that estimates reflect observed testing behavior rather than rule-based inference. "
                "Adjusted models are robustness analyses of association, not causal claims."
            ),
            "",
            "## Export Package",
            (
                f"Configured figure families: {', '.join(selected_figures)}. "
                f"Configured figure formats: {', '.join(figure_formats)}."
            ),
        ]
        if not comparison_summary.empty:
            lines.extend(
                [
                    "",
                    "## Site Comparison Snapshot",
                    self._site_comparison_paragraph(comparison_summary),
                ]
            )
        if scope == "site" and site == "armd":
            implied_delta = self._read_json(
                self._paths.paths.artifacts
                / self._settings.cascade.outputs.result_dir
                / "armd_sensitivity_implied"
                / "implied_delta_summary.json"
            )
            if implied_delta:
                lines.extend(
                    [
                        "",
                        "## Sensitivity Note",
                        self._implied_delta_paragraph(implied_delta),
                    ]
                )
        return "\n".join(lines).strip() + "\n"

    def build_figure_captions_markdown(
        self,
        scope: str,
        site: str | None,
        selected_figures: tuple[str, ...],
    ) -> str:
        scope_dir = site if scope == "site" and site else "combined"
        captions = self._figure_captions(scope_dir)
        lines = [f"# Figure Captions: {scope_dir}", ""]
        for figure_key in selected_figures:
            caption = captions.get(figure_key)
            if caption is None:
                continue
            lines.extend([f"## {caption['title']}", caption["caption"], ""])
        return "\n".join(lines).strip() + "\n"

    def build_table_notes_markdown(
        self,
        scope: str,
        site: str | None,
        exported_tables: tuple[str, ...],
    ) -> str:
        scope_dir = site if scope == "site" and site else "combined"
        notes = self._table_notes(scope_dir)
        lines = [f"# Table Notes: {scope_dir}", ""]
        for table_name in exported_tables:
            note = notes.get(table_name)
            if note is None:
                continue
            lines.extend([f"## {table_name}", note, ""])
        return "\n".join(lines).strip() + "\n"

    def _overview_paragraph(self, scope: str, site: str | None, cascade_summary: dict) -> str:
        scope_name = site if scope == "site" and site else "combined multi-site"
        rows = cascade_summary.get("rows", {})
        return (
            f"The {scope_name} cascade report summarizes observed downstream-testing structure after applying the current support thresholds "
            f"(min total support={cascade_summary.get('min_total_support')}, min result support={cascade_summary.get('min_result_support')}). "
            f"This scope contains {rows.get('retained_edges', 0)} retained directed edges from {rows.get('escalation_results', 0)} escalation-ratio candidates."
        )

    def _quantitative_summary(
        self,
        cascade_summary: dict,
        report_summary: dict,
        primary_table: pd.DataFrame,
    ) -> str:
        rows = cascade_summary.get("rows", {})
        report_rows = report_summary.get("row_counts", {})
        summary = (
            f"Conditional testing rows: {rows.get('conditional_probabilities', 0)}; "
            f"escalation-ratio rows: {rows.get('escalation_results', 0)}; "
            f"adjusted-model rows: {rows.get('adjusted_results', 0)}; "
            f"retained edges: {rows.get('retained_edges', 0)}; "
            f"two-step paths: {report_rows.get('top_paths', 0)}."
        )
        if primary_table.empty:
            return summary
        top_edge = primary_table.iloc[0]
        return (
            summary
            + " "
            + (
                f"The highest-ranked retained edge is {top_edge['upstream_drug']} -> {top_edge['downstream_drug']} "
                f"with escalation ratio {float(top_edge['escalation_ratio']):.3f} and adjusted OR {float(top_edge['adjusted_OR']):.3f}."
            )
        )

    def _site_comparison_paragraph(self, comparison_summary: pd.DataFrame) -> str:
        pivot = comparison_summary.pivot(index="site", columns="edge_presence", values="edge_count").fillna(0)
        pieces: list[str] = []
        for site, row in pivot.iterrows():
            pieces.append(
                f"{site}: shared={int(row.get('shared', 0))}, "
                f"site_only={int(row.get('site_only', 0))}, "
                f"combined_only={int(row.get('combined_only', 0))}"
            )
        return "Site-vs-combined retained-edge overlap is summarized as follows: " + "; ".join(pieces) + "."

    def _implied_delta_paragraph(self, implied_delta: dict) -> str:
        return (
            f"In the armd sensitivity analysis that adds implied susceptibility as a separate observation layer, "
            f"{implied_delta.get('shared_edge_count', 0)} retained edges are shared with the primary observed-AST cascade, "
            f"{implied_delta.get('primary_only_edge_count', 0)} are primary-only, and "
            f"{implied_delta.get('sensitivity_only_edge_count', 0)} are sensitivity-only. "
            "These results should be interpreted as the combined effect of observed testing and rule-based inference, not pure testing behavior."
        )

    def _figure_captions(self, scope_dir: str) -> dict[str, dict[str, str]]:
        return {
            "primary_cascade_forest": {
                "title": "Primary Cascade Forest",
                "caption": (
                    f"Retained directed cascade edges for {scope_dir}, ranked by escalation ratio. "
                    "Each point represents one upstream-downstream antibiotic pair. The horizontal position shows the escalation ratio on a log scale."
                ),
            },
            "upstream_contribution_forest": {
                "title": "Downstream Trigger Forests",
                "caption": (
                    f"One forest plot per downstream antibiotic for {scope_dir}, with robust/supported upstream trigger antibiotics ranked by escalation ratio. "
                    "These figures display validated downstream-testing response patterns, not raw escalation candidates."
                ),
            },
            "downstream_trigger_forest": {
                "title": "Downstream Trigger Forests",
                "caption": (
                    f"One forest plot per downstream antibiotic for {scope_dir}, with robust/supported upstream trigger antibiotics ranked by escalation ratio. "
                    "These figures display validated downstream-testing response patterns, not raw escalation candidates."
                ),
            },
            "threshold_sensitivity": {
                "title": "Threshold Sensitivity",
                "caption": (
                    f"Retained-edge counts across alternative support and escalation-ratio thresholds for {scope_dir}. "
                    "This figure is intended as a robustness summary rather than a hypothesis test."
                ),
            },
            "site_vs_combined_summary": {
                "title": "Site vs Combined Summary",
                "caption": (
                    "Stacked counts of shared, site-only, and combined-only retained edges. "
                    "This figure summarizes cross-site alignment of retained cascade structure."
                ),
            },
            "cascade_sankey": {
                "title": "Cascade Sankey",
                "caption": (
                    "Sankey-ready visualization of high-ranking cascade paths. "
                    "Flow width reflects aggregated support, while direction reflects empirical cascade ordering."
                ),
            },
            "cascade_chord": {
                "title": "Cascade Chord Diagram",
                "caption": (
                    "Chord diagram summarizing retained directed antibiotic relationships. "
                    "This figure is complementary to the directed network and should not replace direction-sensitive interpretation."
                ),
            },
            "cascade_network": {
                "title": "Cascade Network",
                "caption": (
                    "Directed retained-edge network of the cascade structure. "
                    "Node prominence reflects network position, while edges encode empirical cascade relationships."
                ),
            },
        }

    def _table_notes(self, scope_dir: str) -> dict[str, str]:
        return {
            "table_a_provenance.csv": (
                f"Provenance counts for {scope_dir}. Row counts describe staged datasets and analytical units after harmonization and gold-table construction."
            ),
            "table_b_eligibility.csv": (
                "Eligibility totals separate observed testing from structural omission using intrinsic-resistance reference knowledge."
            ),
            "table_c_primary_cascade.csv": (
                "Primary retained edges are based on directly observed AST only. "
                "n_R and n_S denote resistant and susceptible upstream support counts within the eligible denominator. "
                "The escalation ratio is the unadjusted descriptive effect measure, while adjusted OR is estimated separately from a logistic downstream-testing model. "
                "Current adjusted models include organism, source site, ICU status, prior antibiotic exposure within 90 days, calendar year/month, age bin, sex, prior same-organism infection within 90 days, and comorbidity count, with availability indicators where needed."
            ),
            "table_d_concordance_summary.csv": (
                "Concordance summarizes alignment between empirical retained edges and structural eligibility logic. "
                "It should not be interpreted as direct evidence of laboratory intent."
            ),
            "table_d_concordance_examples.csv": (
                "Examples are sorted to highlight lower structural alignment first, making potential conflicts easier to inspect."
            ),
            "table_e_threshold_sensitivity.csv": (
                "Threshold sensitivity enumerates retained-edge counts under alternative support and escalation-ratio filters. "
                "These are descriptive robustness outputs."
            ),
            "table_f_pathway_flows.csv": (
                "Two-stage pathway rows are Sankey-ready representations of high-ranking retained paths. "
                "Each path contributes one stage-1 row and one stage-2 row."
            ),
            "table_g_implied_delta.csv": (
                "This sensitivity-only table compares the primary observed-AST cascade with the armd implied-susceptibility cascade. "
                "It should not be merged into the primary cross-site interpretation."
            ),
            "site_vs_combined_edge_comparison.csv": (
                "This comparison table aligns site-level retained edges against the combined retained-edge set and labels each row as shared, site-only, or combined-only."
            ),
            "site_vs_combined_summary.csv": (
                "This summary aggregates shared, site-only, and combined-only retained-edge counts by site."
            ),
        }

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _safe_read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _safe_read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
