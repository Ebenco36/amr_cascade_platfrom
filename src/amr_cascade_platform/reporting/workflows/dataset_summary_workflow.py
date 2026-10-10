"""Describe the analysis dataset from the gold layer alone, before any cascade analysis.

On the cluster this runs as its own job right after the combined gold build, so
the cohort, its eligibility and observation structure and its covariate
coverage can be read while the cascade and validation jobs are still queued.
Every table and figure here is produced the same way by the full report later;
this job only makes them available early, with a one-page digest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.provenance import run_provenance
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.core.utils.site_labels import ALL_SITES, site_label
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.reporting.builders.manuscript_table_builder import ManuscriptTableBuilder
from amr_cascade_platform.visualization.report.figure_manager import FigureManager


class DatasetSummaryWorkflow:
    """Gold-only descriptive tables, figures and a Markdown digest for one scope."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._tables = ManuscriptTableBuilder(settings, path_manager)
        self._figures = FigureManager(settings, path_manager)
        self._store = DatasetStore(settings)

    def run(self, scope: str, site: str | None, organism: str | None, formats: tuple[str, ...], *, hide_figure_titles: bool = False) -> dict[str, Path]:
        self._figures = FigureManager(self._settings, self._paths, hide_titles=hide_figure_titles)
        scope_dir = scoped_output_dir(root=Path("."), scope=scope, site=site, organism=organism)
        root = self._paths.project_root
        tables_dir = root / self._settings.reporting.reports_dir / scope_dir / "dataset_summary"
        figures_dir = root / self._settings.reporting.figures_dir / scope_dir / "dataset_summary"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = scoped_output_dir(self._paths.paths.gold, scope, site=site, organism=organism)

        build = self._tables
        flow = build.build_data_quality_flow_table(scope, site, organism)
        exclusion_flow, reconciliation = build.build_exclusion_flow_table(scope, site, organism, flow)
        opportunity_space = build.build_eligibility_table(scope, site, organism)
        cohort_long, cohort = build.build_cohort_characteristics_tables(scope, site, organism)
        frames = build.analysis_frames(scope, site, organism)
        coverage = self._figures.observation_coverage_table(frames.opportunities) if not frames.empty else pd.DataFrame()
        exposure, exposure_strata = build.build_availability_exposure_tables(scope, site, organism)
        breadth, breadth_distribution = build.build_panel_breadth_tables(scope, site, organism)
        availability = build.build_operational_availability_table(scope, site, organism)
        coverage_by_era = build.build_coverage_by_era_table(availability, scope, site)
        antibiogram = build.build_antibiogram_table(scope, site, organism)
        episodes_per_patient = build.build_episodes_per_patient_table(scope, site, organism)

        tables = {
            "table_a_data_quality_flow.csv": flow,
            "table_a_exclusion_flow.csv": exclusion_flow,
            "table_b_eligibility.csv": opportunity_space,
            "table_v_cohort_characteristics.csv": cohort,
            "table_v_cohort_characteristics_long.csv": cohort_long,
            "table_w_observation_coverage.csv": coverage,
            "table_u_availability_exposure.csv": exposure,
            "table_u_availability_exposure_strata.csv": exposure_strata,
            "table_w_panel_breadth.csv": breadth,
            "table_w_panel_breadth_distribution.csv": breadth_distribution,
            "table_u_operational_availability.csv": availability,
            "table_w_observation_coverage_by_era.csv": coverage_by_era,
            "table_k_antibiogram.csv": antibiogram,
            "table_t_episodes_per_patient.csv": episodes_per_patient,
        }
        outputs: dict[str, Path] = {}
        for name, frame in tables.items():
            if frame is not None and not frame.empty:
                path = tables_dir / name
                frame.to_csv(path, index=False)
                outputs[name] = path

        figures = self._figures
        stem = figures_dir
        if not opportunity_space.empty:
            outputs.update(figures.export_opportunity_space(opportunity_space=opportunity_space, output_stem=stem / "figure_opportunity_space", formats=formats))
        if not breadth_distribution.empty:
            outputs.update(figures.export_panel_breadth(summary=breadth, distribution=breadth_distribution, output_stem=stem / "figure_panel_breadth", formats=formats))
        if not coverage_by_era.empty:
            outputs.update(figures.export_observation_coverage_by_era(coverage_by_era=coverage_by_era, output_stem=stem / "figure_observation_coverage_by_era", formats=formats))
        if not antibiogram.empty:
            outputs.update(figures.export_antibiogram(antibiogram=antibiogram, output_stem=stem / "figure_antibiogram", formats=formats))
        if not coverage.empty:
            outputs.update(figures.export_observation_coverage(coverage=coverage, output_stem=stem / "figure_observation_coverage", formats=formats))
            outputs.update(figures.export_observation_coverage_by_site(coverage=coverage, output_stem=stem / "figure_observation_coverage_by_site", formats=formats))
        if not availability.empty:
            outputs.update(figures.export_operational_availability_matrix(availability_table=availability, output_stem=stem / "figure_operational_availability_matrix", formats=formats))
            outputs.update(figures.export_site_era_availability_timeline(availability_table=availability, output_stem=stem / "figure_site_era_availability_timeline", formats=formats))

        eligible_pairs = self._store.read_pandas(gold_dir / "eligible_pairs.parquet")
        culture_episodes = self._store.read_pandas(gold_dir / "culture_episodes.parquet")
        drug_pair_episodes = self._store.read_pandas(gold_dir / "drug_pair_episodes.parquet")
        culture_episodes = build.culture_episodes_with_covariates(scope, site, culture_episodes, organism)
        outputs.update(
            figures.export_dataset_summary_figures(
                eligible_pairs=eligible_pairs,
                culture_episodes=culture_episodes,
                drug_pair_episodes=drug_pair_episodes,
                output_dir=stem,
                formats=formats,
            )
        )
        del eligible_pairs, drug_pair_episodes

        digest = tables_dir / "dataset_summary.md"
        digest.write_text(self._digest(organism, frames, opportunity_space, exclusion_flow, reconciliation, sorted(outputs)), encoding="utf-8")
        outputs[digest.name] = digest
        manifest = tables_dir / "dataset_summary_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "scope": scope,
                    "site": site,
                    "organism": organism,
                    "hide_figure_titles": hide_figure_titles,
                    "outputs": {name: str(path) for name, path in outputs.items()},
                    "exclusion_flow_reconciliation": reconciliation,
                    "provenance": run_provenance(root, self._settings, {"eligible_pairs": gold_dir / "eligible_pairs.parquet"}),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        outputs[manifest.name] = manifest
        return outputs

    @staticmethod
    def _digest(organism, frames, opportunity_space: pd.DataFrame, exclusion_flow: pd.DataFrame, reconciliation: list[str], files: list[str]) -> str:
        lines = [f"# Dataset summary: {organism or 'all organisms'}", ""]
        if not frames.empty:
            episodes = frames.episodes
            times = episodes["episode_time"].dropna()
            lines += [
                f"- Culture episodes: {len(episodes):,}; patients: {episodes[['source_site', 'anon_id']].drop_duplicates().shape[0]:,}",
                f"- Observed episode–antibiotic results: {len(frames.results):,} across {frames.results['antibiotic'].nunique()} antibiotics",
                f"- Study period: {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}" if not times.empty else "- Study period: no parseable culture times",
                "",
                "## By site",
                "",
                "| Site | Episodes | Patients |",
                "|---|---:|---:|",
            ]
            for site, part in episodes.groupby("source_site", observed=True):
                lines.append(f"| {site_label(str(site))} | {len(part):,} | {part['anon_id'].nunique():,} |")
            lines.append("")
        if not opportunity_space.empty and {"site", "grid_rows", "eligible_rows", "eligible_observed_rows"} <= set(opportunity_space.columns):
            lines += [
                "## Episode–antibiotic opportunity space",
                "",
                "| Scope | Grid rows | Eligible | Observed among eligible | Not available | Intrinsic |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for row in opportunity_space.itertuples(index=False):
                observed = row.eligible_observed_rows / row.eligible_rows if row.eligible_rows else float("nan")
                lines.append(
                    f"| {site_label(str(row.site))} | {row.grid_rows:,} | {row.eligible_rows:,} | {observed:.1%} | "
                    f"{getattr(row, 'not_available_rows', 0):,} | {getattr(row, 'intrinsic_rows', 0):,} |"
                )
            lines.append("")
        lines += ["## Exclusion flow", ""]
        if reconciliation:
            lines += ["Reconciliation problems:", *[f"- {problem}" for problem in reconciliation], ""]
        else:
            lines += ["The exclusion flow reconciles row for row (see table_a_exclusion_flow.csv).", ""]
        lines += ["## Files", "", *[f"- {name}" for name in files], ""]
        return "\n".join(lines)
