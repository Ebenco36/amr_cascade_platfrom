#!/usr/bin/env python
"""Copy every figure and CSV the two manuscripts cite from this run's outputs.

See amr_cascade_platform.reporting.builders.manuscript_package for the rules. Run on the
machine that holds the paper folders (they are not in git), after the run's outputs are
available there: --run-root is the directory containing the run's data/ and outputs/,
for example an extracted archive from scripts/archive_export_artifacts.sh. Paper folders
and submission_figures/ are always taken from this project.

Usage:
  python scripts/build_manuscript_package.py --env hpc --run-root amr_cascade_hpc_outputs_<stamp> --dry-run
  python scripts/build_manuscript_package.py --env hpc --run-root amr_cascade_hpc_outputs_<stamp>
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.paths.path_manager import PathManager
    from amr_cascade_platform.core.utils.scopes import scoped_output_dir
    from amr_cascade_platform.reporting.builders.manuscript_package import execute, plan_package

    parser = argparse.ArgumentParser(description="Package the manuscripts' cited figures and tables from this run.")
    parser.add_argument("--env", default=None)
    parser.add_argument("--organism", default="ESCHERICHIA COLI")
    parser.add_argument("--run-root", type=Path, default=None, help="Directory holding the run's data/ and outputs/ (default: this project).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be copied and moved without changing files.")
    parser.add_argument("--keep-uncited", action="store_true", help="Leave uncited files in submission_figures instead of moving them aside.")
    args = parser.parse_args()

    settings, _, _ = build_context(project_root, args.env)
    run_root = (args.run_root or project_root).resolve()
    scope_dir = scoped_output_dir(Path("."), "combined", organism=args.organism)
    figures_dir = run_root / settings.reporting.figures_dir / scope_dir
    tables_dir = run_root / settings.reporting.tables_dir / scope_dir
    gold = scoped_output_dir(PathManager(run_root, settings).paths.gold, "combined", organism=args.organism) / "culture_episodes.parquet"
    if not gold.exists():
        print(f"Gold layer not found: {gold}", file=sys.stderr)
        return 1
    run_started = gold.stat().st_mtime

    plans = plan_package(project_root, figures_dir, tables_dir, run_started)
    errors = [error for plan in plans.values() for error in plan.errors]
    if errors:
        print("Package not built; fix these first:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    context = {
        "organism": args.organism,
        "run_root": str(run_root),
        "gold_layer": str(gold),
        "gold_layer_modified_utc": datetime.fromtimestamp(run_started, UTC).isoformat(),
    }
    execute(plans, project_root, prune=not args.keep_uncited, dry_run=args.dry_run, context=context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
