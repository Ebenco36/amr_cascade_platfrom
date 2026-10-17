#!/usr/bin/env python
"""Export the adjusted-model SHAP figures into their own folder.

The report job archives every top-level figure it does not write, so running it for SHAP
alone would move the manuscript figures aside. This writes the SHAP figures (and the
report's tables) under <outputs>/shap_explainability/ instead and leaves the manuscript
outputs untouched. It reads the combined cascade's saved SHAP files; it fits nothing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amr_cascade_platform.cli.main import build_context  # noqa: E402
from amr_cascade_platform.core.paths.path_manager import PathManager  # noqa: E402
from amr_cascade_platform.reporting.workflows.report_export_workflow import (  # noqa: E402
    ReportExportRequest,
    ReportExportWorkflow,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hpc")
    parser.add_argument("--organism", default="ESCHERICHIA COLI")
    parser.add_argument("--hide-figure-titles", action="store_true")
    parser.add_argument("--figure-format", action="append", choices=("pdf", "png", "svg", "html"))
    args = parser.parse_args()

    settings, paths, _ = build_context(ROOT, args.env)
    outputs = Path(settings.reporting.figures_dir)
    outputs = (outputs if outputs.is_absolute() else ROOT / outputs).parent / "shap_explainability"
    settings = replace(settings, reporting=replace(
        settings.reporting,
        tables_dir=str(outputs / "tables"),
        figures_dir=str(outputs / "figures"),
        reports_dir=str(outputs / "reports"),
    ))
    ReportExportWorkflow(settings, PathManager(ROOT, settings)).run(ReportExportRequest(
        organism=args.organism,
        figure_formats=tuple(args.figure_format or ("pdf", "png", "svg")),
        figures=("adjusted_model_shap",),
        hide_figure_titles=args.hide_figure_titles,
        fail_on_missing_figures=True,
    ))
    written = sorted((outputs / "figures").rglob("figure_adjusted_model_shap*"))
    if not written:
        print("No SHAP figures were written; is the combined cascade's SHAP output empty (--skip-shap)?", file=sys.stderr)
        return 1
    print(f"{len(written)} SHAP figure files written under {outputs / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
