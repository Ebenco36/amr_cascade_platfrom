"""Package the figures and CSV tables the two manuscripts cite from one run's outputs.

Each paper's .tex files are scanned for \\includegraphics targets, \\csvreader inputs, and
CSV files cited by name. Every cited file is mapped to the run output that generates it
and copied to submission_figures/<paper>/ (\\csvreader inputs also into the paper folder
LaTeX compiles from). A cited file with no generator, a missing output, or an output
older than the run's gold layer stops the package, so nothing from an earlier run can
reach the manuscripts. Replaced files and uncited files in submission_figures are moved
aside rather than deleted, and a manifest records the source, destinations, size, and
SHA-256 of every packaged file.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PAPERS = ("paper1_detection_validation", "paper2_prevalence_surveillance")

# Cited CSV name -> path under outputs/tables/combined/organisms/<organism>/.
CSV_SOURCES = {
    "table_a_data_quality_flow.csv": "table_a_data_quality_flow.csv",
    "table_r_correction_sensitivity.csv": "table_r_correction_sensitivity.csv",
    "table_s2_antibiotic_panel_reference.csv": "supplementary/table_s2_antibiotic_panel_reference.csv",
    "table_s5_aware_transition_summary.csv": "table_m_aware_transition_summary.csv",
    "table_s6_support_threshold_sensitivity.csv": "supplementary/table_s6_support_threshold_sensitivity.csv",
    "table_s7_patient_cluster_bootstrap_sensitivity.csv": "supplementary/table_s7_patient_cluster_bootstrap_sensitivity.csv",
    "table_s8_heterogeneity_threshold_sensitivity.csv": "supplementary/table_s8_heterogeneity_threshold_sensitivity.csv",
    "table_s9_pbi_sensitivity.csv": "supplementary/table_s9_pbi_sensitivity.csv",
    "table_s10_confounding_strength_calibration.csv": "supplementary/table_s10_confounding_strength_calibration.csv",
    "table_s11_cross_site_replication.csv": "supplementary/table_s11_cross_site_replication.csv",
    "table_s12_model_comparison.csv": "supp_pred_table1_model_comparison.csv",
    "table_s13_lr_threshold_sensitivity.csv": "supp_pred_table3_lr_threshold.csv",
    "table_s14_lr_coefficients.csv": "supp_pred_table2_lr_top_features.csv",
    "table_s15_era_stratified_permutation_sensitivity.csv": "supplementary/table_s15_era_stratified_permutation_sensitivity.csv",
    "table_s16_ridge_penalty_sensitivity.csv": "supplementary/table_s16_ridge_penalty_sensitivity.csv",
    "table_s2_prevalence_diagnostics.csv": "supplementary/table_s2_prevalence_diagnostics.csv",
    "table_s3_smd_balance_summary.csv": "supplementary/table_s3_smd_balance_summary.csv",
    # Descriptive summaries of the analysis set (report step) and the availability-sensitivity comparison.
    "table_a_exclusion_flow.csv": "table_a_exclusion_flow.csv",
    "table_b_eligibility.csv": "table_b_eligibility.csv",
    "table_k_antibiogram.csv": "table_k_antibiogram.csv",
    "table_t_episodes_per_patient.csv": "table_t_episodes_per_patient.csv",
    "table_u_availability_exposure.csv": "table_u_availability_exposure.csv",
    "table_w_observation_coverage.csv": "table_w_observation_coverage.csv",
    "table_w_observation_coverage_by_era.csv": "table_w_observation_coverage_by_era.csv",
    "table_w_panel_breadth.csv": "table_w_panel_breadth.csv",
    "table_availability_sensitivity.csv": "supplementary/table_availability_sensitivity.csv",
}

# Hand-made assets: never generated, never overwritten, never moved aside.
STATIC_ASSETS = {"figure_pipeline_overview.pdf", "figure_pipeline_overview.svg"}
SUPERSEDED_DIR = "_superseded"
MANIFEST_NAME = "package_manifest.json"

_INCLUDEGRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_CSVREADER = re.compile(r"\\csvreader\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_TEXTTT_CSV = re.compile(r"\\texttt\{([^}]*?\.csv)\}")
_GENERATED_INPUT = re.compile(r"\\(?:input|InputIfFileExists)\s*\{(generated_[^}]+\.tex)\}")
# Generated LaTeX fragments live here, relative to the organism's tables directory.
GENERATED_LATEX_DIR = "supplementary/latex"


def _strip_comments(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


@dataclass
class Citations:
    figures: set[str] = field(default_factory=set)
    csvreader: set[str] = field(default_factory=set)
    csv_cited: set[str] = field(default_factory=set)
    generated_tex: set[str] = field(default_factory=set)


def scan_citations(paper_dir: Path) -> Citations:
    citations = Citations()
    for tex in sorted(paper_dir.glob("*.tex")):
        text = _strip_comments(tex.read_text(encoding="utf-8"))
        citations.figures.update(Path(target.strip()).name for target in _INCLUDEGRAPHICS.findall(text))
        citations.csvreader.update(Path(target.strip()).name for target in _CSVREADER.findall(text))
        citations.generated_tex.update(Path(target.strip()).name for target in _GENERATED_INPUT.findall(text))
        for cited in _TEXTTT_CSV.findall(text):
            name = cited.replace("\\_", "_").strip()
            if "/" not in name:
                citations.csv_cited.add(name)
    return citations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Plan:
    copies: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    static: list[str] = field(default_factory=list)
    unreferenced: list[Path] = field(default_factory=list)


def plan_package(
    project_root: Path,
    figures_dir: Path,
    tables_dir: Path,
    run_started: float,
) -> dict[str, Plan]:
    plans: dict[str, Plan] = {}
    for paper in PAPERS:
        paper_dir = project_root / paper
        submission_dir = project_root / "submission_figures" / paper
        citations = scan_citations(paper_dir)
        plan = Plan()

        def add(name: str, source: Path, destinations: list[Path]) -> None:
            if not source.exists():
                plan.errors.append(f"{paper}: {name} is cited but {source} does not exist")
            elif source.stat().st_mtime < run_started:
                plan.errors.append(f"{paper}: {name} comes from {source}, which is older than this run's gold layer")
            else:
                plan.copies.append({"name": name, "source": source, "destinations": destinations})

        for name in sorted(citations.figures):
            if name in STATIC_ASSETS:
                if not (submission_dir / name).exists():
                    plan.errors.append(f"{paper}: static asset {name} is missing from {submission_dir}")
                plan.static.append(name)
                continue
            add(name, figures_dir / name, [submission_dir / name])
        for name in sorted(citations.generated_tex):
            add(name, tables_dir / GENERATED_LATEX_DIR / name, [paper_dir / name])
        for name in sorted(citations.csvreader | citations.csv_cited):
            if name not in CSV_SOURCES:
                plan.errors.append(f"{paper}: {name} is cited but no generator is registered for it")
                continue
            destinations = [submission_dir / name]
            if name in citations.csvreader:
                destinations.append(paper_dir / name)
            add(name, tables_dir / CSV_SOURCES[name], destinations)

        packaged = {Path(str(d)).name for copy in plan.copies for d in copy["destinations"]} | set(plan.static)
        if submission_dir.exists():
            plan.unreferenced = sorted(
                path
                for path in submission_dir.iterdir()
                if path.is_file() and path.name not in packaged and path.name != MANIFEST_NAME and path.name not in STATIC_ASSETS
            )
        plans[paper] = plan
    return plans


def execute(plans: dict[str, Plan], project_root: Path, prune: bool, dry_run: bool, context: dict[str, object]) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    edit_log = project_root / "manuscript_edit_log" / datetime.now(UTC).strftime("%Y-%m-%d") / "package_superseded" / stamp
    for paper, plan in plans.items():
        submission_dir = project_root / "submission_figures" / paper
        superseded = submission_dir / SUPERSEDED_DIR / stamp
        entries = []
        for copy in plan.copies:
            source: Path = copy["source"]
            digest = _sha256(source)
            for destination in copy["destinations"]:
                if destination.exists() and _sha256(destination) != digest:
                    backup_root = superseded if destination.parent == submission_dir else edit_log / paper
                    print(f"  replace {destination} (old copy -> {backup_root})")
                    if not dry_run:
                        backup_root.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(destination), backup_root / destination.name)
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                print(f"  {'would copy' if dry_run else 'copied'} {source} -> {destination}")
            entries.append(
                {
                    "name": copy["name"],
                    "source": str(source),
                    "destinations": [str(d) for d in copy["destinations"]],
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                    "source_modified_utc": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
                }
            )
        moved = []
        for path in plan.unreferenced:
            if prune:
                print(f"  uncited {path.name} -> {superseded}")
                if not dry_run:
                    superseded.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), superseded / path.name)
                moved.append(path.name)
            else:
                print(f"  uncited file left in place: {path}")
        manifest = {
            "generated_utc": datetime.now(UTC).isoformat(),
            **context,
            "packaged": entries,
            "static_assets": sorted(plan.static),
            "uncited_moved_aside": moved,
            "uncited_left_in_place": [] if prune else [p.name for p in plan.unreferenced],
        }
        if not dry_run:
            submission_dir.mkdir(parents=True, exist_ok=True)
            (submission_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{paper}: {len(entries)} cited files packaged, {len(plan.static)} static, {len(moved)} uncited moved aside")
