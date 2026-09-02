#!/usr/bin/env python
"""Run ARMD ESKAPE-family cascade validation and write a summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

def _bootstrap() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return project_root


def _build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ESKAPE-family cascade validation for ARMD using predefined organism-family "
            "definitions, then write one summary table."
        ),
    )
    parser.add_argument("--env", default="mac", help="Runtime environment.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help=(
            "Optional subset of ESKAPE targets to run. Examples: Enterobacter Pseudomonas "
            "Staphylococcus"
        ),
    )
    parser.add_argument(
        "--skip-armd",
        action="store_true",
        help="Skip ARMD ESKAPE validation.",
    )
    parser.add_argument(
        "--armd-source-scope",
        choices=("site", "combined"),
        default="site",
        help="ARMD source scope to analyze.",
    )
    parser.add_argument(
        "--armd-site",
        default="armd",
        help="ARMD site when using --armd-source-scope site.",
    )
    parser.add_argument(
        "--armd-output-root",
        default=str(project_root / "data" / "artifacts" / "eskape_validation" / "armd"),
        help="Output root for ARMD ESKAPE artifacts.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(project_root / "outputs" / "tables" / "eskape"),
        help="Directory for cross-dataset summary tables.",
    )
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help="Allow absent targets and write partial comparison rows instead of failing preflight.",
    )
    return parser


def main() -> None:
    project_root = _bootstrap()
    parser = _build_parser(project_root)
    args = parser.parse_args()

    from amr_cascade_platform.cascade.workflows.eskape_family_validation import (
        ArmdEskapeCascadeRunner,
        ArmdEskapeRequest,
    )
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.utils.cascade_input_preflight import (
        require_armd_eskape_targets,
        require_path_exists,
    )
    from amr_cascade_platform.core.utils.eskape import resolve_eskape_targets

    settings, path_manager, _ = build_context(project_root, args.env)
    targets = resolve_eskape_targets(args.targets)

    summary_rows: list[dict[str, object]] = []
    missing_armd = []

    armd_cohort = None
    if not args.skip_armd:
        if args.armd_source_scope == "combined":
            armd_cohort_path = path_manager.paths.harmonized / "combined" / "cohort.parquet"
        else:
            armd_cohort_path = path_manager.paths.harmonized / "site_aligned" / args.armd_site / "cohort.parquet"
        armd_cohort_path = require_path_exists(armd_cohort_path, "ARMD harmonized cohort")
        armd_cohort = pd.read_parquet(armd_cohort_path, columns=["organism"])
        missing_armd = require_armd_eskape_targets(
            armd_cohort,
            targets,
            allow_missing=args.allow_missing_targets,
        )

    if not args.skip_armd:
        armd_runner = ArmdEskapeCascadeRunner(settings, path_manager)
        for target in targets:
            if target in missing_armd:
                summary_rows.append(
                    {
                        "dataset": "armd",
                        "source_scope": args.armd_source_scope,
                        "site": args.armd_site if args.armd_source_scope == "site" else "combined",
                        "eskape_target": target.display_name,
                        "scientific_focus": target.scientific_focus,
                        "retained_edges": 0,
                        "robust_edges": 0,
                        "supported_edges": 0,
                        "mixed_edges": 0,
                        "insufficient_edges": 0,
                        "validated_edges": 0,
                        "cascade_exists": False,
                        "decision_rule": "No usable rows for this ESKAPE family in the current ARMD input",
                    }
                )
                continue
            outputs = armd_runner.run(
                ArmdEskapeRequest(
                    target=target,
                    source_scope=args.armd_source_scope,
                    site=args.armd_site if args.armd_source_scope == "site" else None,
                    output_root=args.armd_output_root,
                )
            )
            summary = pd.read_csv(outputs["existence_summary_csv"])
            summary_rows.extend(summary.to_dict(orient="records"))

    comparison = pd.DataFrame(summary_rows)
    if not comparison.empty:
        preferred_columns = [
            "dataset",
            "source_scope",
            "site",
            "eskape_target",
            "scientific_focus",
            "organism",
            "filtered_rows",
            "culture_episode_rows",
            "drug_pair_rows",
            "retained_edges",
            "robust_edges",
            "supported_edges",
            "mixed_edges",
            "insufficient_edges",
            "validated_edges",
            "cascade_exists",
            "decision_rule",
        ]
        columns = [column for column in preferred_columns if column in comparison.columns]
        remaining = [column for column in comparison.columns if column not in columns]
        comparison = comparison.loc[:, columns + remaining].sort_values(
            by=["eskape_target", "dataset", "site"],
            na_position="last",
        ).reset_index(drop=True)

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "eskape_cascade_comparison.csv"
    json_path = summary_dir / "eskape_cascade_comparison.json"
    comparison.to_csv(csv_path, index=False)
    json_path.write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")

    payload = {
        "env": args.env,
        "targets": [target.display_name for target in targets],
        "csv": str(csv_path),
        "json": str(json_path),
        "rows": int(len(comparison)),
    }
    print("ESKAPE cascade validation completed.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
