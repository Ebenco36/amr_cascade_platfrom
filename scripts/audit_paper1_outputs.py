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
from amr_cascade_platform.reporting.paper1_profile import PAPER1_PREDICTION_FIGURES
from amr_cascade_platform.reporting.manuscript_profile import required_exports
from amr_cascade_platform.reporting.builders.manuscript_package import CSV_SOURCES


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
    )
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
        if stem in PAPER1_PREDICTION_FIGURES and manifest_path.exists() and path.stat().st_mtime < manifest_path.stat().st_mtime:
            errors.append(f"Prediction figure predates report: {path}")
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings, _, _ = build_context(root, args.env)
    def scoped(directory):
        return scoped_output_dir(root / directory, "combined", organism=args.organism)
    reports = scoped(settings.reporting.reports_dir)
    result = audit(scoped(settings.reporting.figures_dir), reports, scoped(settings.reporting.tables_dir), publication=args.publication)
    reports.mkdir(parents=True, exist_ok=True)
    destination = reports / f"{args.publication}_artifact_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.publication} artifact audit: {'PASS' if result['passed'] else 'FAIL'} ({destination})")
    for error in result["errors"]:
        print(error, file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
