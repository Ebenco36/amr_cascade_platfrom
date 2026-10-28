#!/usr/bin/env python
"""Fail closed on incomplete manuscript exports; not a substitute for visual review.

The historical filename is kept for existing Paper 1 commands. Use
--publication manuscripts to audit both papers without the paper folders on HPC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amr_cascade_platform.cli.main import build_context
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.reporting.paper1_profile import PAPER1_PREDICTION_FIGURES, PAPER1_TEMPORAL_FIGURES
from amr_cascade_platform.reporting.manuscript_profile import required_exports
from amr_cascade_platform.reporting.builders.manuscript_package import CSV_SOURCES
from amr_cascade_platform.reporting.covariate_consistency import covariate_count_errors


def audit_refresh(figures: Path, tables: Path, reports: Path, marker: Path,
                  artifacts: Path, features: Path, organism: str, sites: tuple[str, ...]) -> list[str]:
    """Additional contract for the covariate refresh; never certify an old mixed export."""
    import pandas as pd

    if not marker.is_file():
        return [f"Missing refresh marker: {marker}"]
    required = [reports / "report_manifest.json", tables / "table_v_cohort_characteristics_long.csv",
                tables / "masking_experiment/masking_results.csv", tables / "masking_experiment/masking_run.json",
                tables / "temporal_regimes/temporal_regime_years.csv", tables / "temporal_regimes/temporal_regime_summary.csv",
                tables / "temporal_regimes/temporal_regime_numbers.json",
                tables / "supplementary/supplementary_numbers.json", tables / "supplementary/latex/generated_numbers.tex"]
    profile = figures / "dataset_summary/dataset_missing_data_profile_counts.csv"
    required.append(profile)
    required.extend(figures / f"dataset_summary/dataset_missing_data_profile.{fmt}" for fmt in ("png", "pdf", "svg", "html"))
    for scope, site in [("combined", None)] + [("site", site) for site in sites]:
        directory = scoped_output_dir(artifacts / "cascade", scope, site=site, organism=organism)
        required.extend(directory / name for name in ("adjusted_results.parquet", "edge_report.parquet"))
    prevalence = scoped_output_dir(artifacts / "prevalence_shift", "combined", organism=organism)
    required.extend(prevalence / name for name in ("prevalence_shift.parquet", "prevalence_mnar_sensitivity_curves.parquet"))
    required.append(scoped_output_dir(features, "combined", organism=organism) / "model_ready_pair_features.parquet")
    errors = []
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Refresh output missing/empty: {path}")
        elif path.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            errors.append(f"Refresh output predates submission: {path}")
    try:
        counts = pd.read_csv(profile)
        # Both producers must agree: the early summary and the final manuscript report.
        for cohort in (tables / "table_v_cohort_characteristics_long.csv",
                       reports / "dataset_summary/table_v_cohort_characteristics_long.csv"):
            if not cohort.is_file() or cohort.stat().st_mtime_ns < marker.stat().st_mtime_ns:
                errors.append(f"Missing/stale cohort table for reconciliation: {cohort}")
            else:
                errors.extend(f"{cohort}: {error}" for error in covariate_count_errors(counts, pd.read_csv(cohort)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Covariate reconciliation could not run: {exc}")
    return errors


def audit(figures: Path, reports: Path, tables: Path, *, publication: str = "paper1") -> dict:
    """Check only the named publication assets, not every optional platform output."""
    errors = []
    manifest_path = reports / "report_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        manifest = {}
        errors.append(f"Report manifest unreadable: {exc}")
    if manifest.get("publication") != publication:
        errors.append(f"Report was not generated with --publication {publication}")
    if manifest.get("missing_figure_groups"):
        errors.append(f"Missing report figures: {manifest['missing_figure_groups']}")
    numbers_path = tables / "supplementary" / "supplementary_numbers.json"
    try:
        numbers = json.loads(numbers_path.read_text())
        if numbers.get("missing_inputs"):
            errors.append(f"Missing supplementary inputs: {numbers['missing_inputs']}")
        if manifest_path.exists() and numbers_path.stat().st_mtime < manifest_path.stat().st_mtime:
            errors.append("Supplementary tables predate the report; rebuild them after reporting")
    except (OSError, ValueError) as exc:
        errors.append(f"Supplementary numbers unreadable: {exc}")
    files = []
    expected = required_exports(publication, ("pdf", "png", "svg", "html")) + tuple(
        f"{stem}.{fmt}" for stem in PAPER1_PREDICTION_FIGURES for fmt in ("pdf", "png", "svg", "html")
    ) + tuple(f"{stem}.{fmt}" for stem in PAPER1_TEMPORAL_FIGURES for fmt in ("pdf", "png", "svg"))
    for filename in expected:
        path = figures / filename
        stem, fmt = path.stem, path.suffix[1:]
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing/empty export: {path}")
            continue
        with path.open("rb") as handle:
            header = handle.read(1024)
        if fmt == "pdf" and not header.startswith(b"%PDF-"):
            errors.append(f"Invalid PDF header: {path}")
        if fmt == "png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"Invalid PNG header: {path}")
        if (stem in PAPER1_PREDICTION_FIGURES + PAPER1_TEMPORAL_FIGURES and manifest_path.exists()
                and path.stat().st_mtime < manifest_path.stat().st_mtime):
            errors.append(f"Figure drawn after the report predates it: {path}")
        files.append(str(path))
    paper2_csvs = {"table_s2_prevalence_diagnostics.csv", "table_s3_smd_balance_summary.csv"}
    csv_paths = {relative for cited, relative in CSV_SOURCES.items()
                 if publication == "manuscripts" or cited not in paper2_csvs}
    for relative in sorted(csv_paths | {"supplementary/latex/generated_numbers.tex"}):
        path = tables / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing/empty manuscript table: {path}")
        else:
            files.append(str(path))
    # Also verify uncited supporting tables used to derive the main-text numbers.
    for name, filename in manifest.get("tables", {}).items():
        path = Path(filename)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing/empty report table: {name}")
    return {
        "publication": publication, "passed": not errors, "errors": errors,
        "checked_files": files,
        "scope": "Generated-asset completeness and basic file integrity; not full scientific or visual certification.",
        "manual_asset": "figure_pipeline_overview.pdf must be checked during manuscript packaging",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hpc")
    parser.add_argument("--organism", default="ESCHERICHIA COLI")
    parser.add_argument("--publication", choices=("paper1", "manuscripts"), default="paper1")
    parser.add_argument("--refresh-marker", type=Path, help="Require current covariate-refresh outputs and reconcile the availability figure with Table 1.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings, paths, _ = build_context(root, args.env)
    def scoped(directory):
        return scoped_output_dir(root / directory, "combined", organism=args.organism)
    reports = scoped(settings.reporting.reports_dir)
    result = audit(scoped(settings.reporting.figures_dir), reports, scoped(settings.reporting.tables_dir), publication=args.publication)
    if args.refresh_marker:
        result["errors"].extend(audit_refresh(
            scoped(settings.reporting.figures_dir), scoped(settings.reporting.tables_dir), reports,
            args.refresh_marker, paths.paths.artifacts, paths.paths.features, args.organism, tuple(settings.platform.sites),
        ))
        result["passed"] = not result["errors"]
        result["refresh_marker"] = str(args.refresh_marker)
    reports.mkdir(parents=True, exist_ok=True)
    destination = reports / f"{args.publication}_artifact_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.publication} artifact audit: {'PASS' if result['passed'] else 'FAIL'} ({destination})")
    for error in result["errors"]:
        print(error, file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
