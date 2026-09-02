#!/usr/bin/env python
"""Audit paper-facing cascade artifacts for publication readiness.

This script is intentionally artifact-facing: it checks the concrete outputs that
would feed manuscript tables and figures, not only whether pipeline commands ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _bootstrap() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return project_root


PROJECT_ROOT = _bootstrap()

from amr_cascade_platform.core.config.config_loader import ConfigLoader  # noqa: E402
from amr_cascade_platform.core.paths.path_manager import PathManager  # noqa: E402
from amr_cascade_platform.core.utils.scopes import scoped_output_dir  # noqa: E402


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    severity: str
    status: str
    message: str
    details: dict[str, Any]


def _slug(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _read_parquet(path: Path, checks: list[AuditCheck], check_id: str, label: str) -> pd.DataFrame:
    if not path.exists():
        checks.append(
            AuditCheck(
                check_id=check_id,
                severity="fatal",
                status="FAIL",
                message=f"Missing required artifact: {label}",
                details={"path": str(path)},
            )
        )
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        checks.append(
            AuditCheck(
                check_id=check_id,
                severity="fatal",
                status="FAIL",
                message=f"Could not read required artifact: {label}",
                details={"path": str(path), "error": repr(exc)},
            )
        )
        return pd.DataFrame()


def _add_check(
    checks: list[AuditCheck],
    check_id: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    severity: str = "fatal",
    warn: bool = False,
) -> None:
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    checks.append(
        AuditCheck(
            check_id=check_id,
            severity="warning" if warn else severity,
            status=status,
            message=message,
            details=details or {},
        )
    )


def _scope_dir(root: Path, scope: str, site: str | None, organism: str | None) -> Path:
    return scoped_output_dir(root=root, scope=scope, site=site, organism=organism)


def _report_scope_dir(scope: str, site: str | None, organism: str | None) -> Path:
    if scope == "site" and site:
        base = Path(site)
    else:
        base = Path("combined")
    if organism:
        base = base / "organisms" / _slug(organism)
    return base


def _static_placeholder_files(figure_dir: Path) -> list[str]:
    placeholder_markers = (
        b"Static export unavailable",
        b"Use the HTML figure for the interactive version",
    )
    offenders: list[str] = []
    for path in sorted(list(figure_dir.glob("*.pdf")) + list(figure_dir.glob("*.svg"))):
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if any(marker in content for marker in placeholder_markers):
            offenders.append(path.name)
    return offenders


def _read_report_manifest(report_dir: Path) -> dict[str, Any]:
    manifest_path = report_dir / "report_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_outputs(checks: list[AuditCheck], output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(check) for check in checks]
    (output_dir / "publication_readiness_audit.json").write_text(
        json.dumps({**payload, "checks": rows}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output_dir / "publication_readiness_audit.csv", index=False)
    lines = [
        "# Publication Readiness Audit",
        "",
        f"- Generated UTC: {payload['generated_at_utc']}",
        f"- Environment: `{payload['environment']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Site: `{payload['site']}`",
        f"- Organism: `{payload['organism']}`",
        f"- Overall status: **{payload['overall_status']}**",
        "",
        "| Check | Status | Severity | Message |",
        "|---|---:|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| `{check.check_id}` | {check.status} | {check.severity} | "
            f"{check.message.replace('|', '/')} |"
        )
    (output_dir / "publication_readiness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> int:
    settings = ConfigLoader(PROJECT_ROOT).load(args.env)
    paths = PathManager(PROJECT_ROOT, settings)
    organism = args.organism
    site = args.site
    scope = args.scope

    checks: list[AuditCheck] = []

    gold_dir = _scope_dir(paths.paths.gold, scope, site, organism)
    cascade_dir = _scope_dir(
        paths.paths.artifacts / settings.cascade.outputs.result_dir,
        scope,
        site,
        organism,
    )
    prevalence_dir = _scope_dir(
        paths.paths.artifacts / settings.prevalence.output_dir,
        scope,
        site,
        organism,
    )
    feature_dir = _scope_dir(paths.paths.features, scope, site, organism)
    model_dir = _scope_dir(
        paths.paths.artifacts / settings.modeling.output_dir / settings.modeling.task_name,
        scope,
        site,
        organism,
    )
    report_scope = _report_scope_dir(scope, site, organism)
    table_dir = PROJECT_ROOT / settings.reporting.tables_dir / report_scope
    figure_dir = PROJECT_ROOT / settings.reporting.figures_dir / report_scope
    report_dir = PROJECT_ROOT / settings.reporting.reports_dir / report_scope
    audit_dir = report_dir / "audits"

    required_gold = [
        "culture_episodes.parquet",
        "culture_drug_episodes.parquet",
        "eligible_pairs.parquet",
        "drug_pair_episodes.parquet",
        "testing_matrix.parquet",
    ]
    for filename in required_gold:
        _add_check(
            checks,
            f"gold:{filename}",
            (gold_dir / filename).exists(),
            f"Gold artifact present: {filename}",
            {"path": str(gold_dir / filename)},
        )

    drug_pairs = _read_parquet(gold_dir / "drug_pair_episodes.parquet", checks, "gold:read-drug-pairs", "drug_pair_episodes")
    if not drug_pairs.empty:
        key_cols = [
            col
            for col in (*settings.gold.episode_key_columns, "upstream_antibiotic", "downstream_antibiotic")
            if col in drug_pairs.columns
        ]
        expected_key_len = len(settings.gold.episode_key_columns) + 2
        duplicate_rows = int(drug_pairs.duplicated(key_cols).sum()) if key_cols else len(drug_pairs)
        max_multiplicity = int(drug_pairs.groupby(key_cols, dropna=False).size().max()) if key_cols else None
        _add_check(
            checks,
            "gold:episode-pair-unique",
            bool(key_cols) and len(key_cols) == expected_key_len and duplicate_rows == 0 and max_multiplicity == 1,
            "Drug-pair table has one row per configured episode/upstream/downstream key.",
            {
                "key_columns": key_cols,
                "rows": int(len(drug_pairs)),
                "duplicate_rows": duplicate_rows,
                "max_multiplicity": max_multiplicity,
            },
        )
        if organism and "organism" in drug_pairs.columns:
            organisms = sorted(str(value) for value in drug_pairs["organism"].dropna().unique())
            _add_check(
                checks,
                "gold:organism-scoped",
                len(organisms) == 1,
                "Drug-pair artifact is organism-scoped.",
                {"organisms": organisms},
            )

    required_cascade = [
        "retained_edges.parquet",
        "validation_results.parquet",
        "edge_report.parquet",
        "network_edges.parquet",
        "network_nodes.parquet",
        settings.cascade.outputs.summary_filename,
    ]
    for filename in required_cascade:
        _add_check(
            checks,
            f"cascade:{filename}",
            (cascade_dir / filename).exists(),
            f"Cascade artifact present: {filename}",
            {"path": str(cascade_dir / filename)},
        )

    retained_edges = _read_parquet(cascade_dir / "retained_edges.parquet", checks, "cascade:read-retained", "retained_edges")
    validation = _read_parquet(cascade_dir / "validation_results.parquet", checks, "cascade:read-validation", "validation_results")
    edge_report = _read_parquet(cascade_dir / "edge_report.parquet", checks, "cascade:read-edge-report", "edge_report")

    if not retained_edges.empty and not validation.empty:
        retained_keys = retained_edges[["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates()
        validation_keys = validation[["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates()
        missing_validation = len(
            retained_keys.merge(
                validation_keys,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
                indicator=True,
            ).query("_merge == 'left_only'")
        )
        _add_check(
            checks,
            "cascade:retained-validation-coverage",
            missing_validation == 0 and len(validation_keys) >= len(retained_keys),
            "Every retained edge has validation results.",
            {
                "retained_edge_n": int(len(retained_keys)),
                "validation_edge_n": int(len(validation_keys)),
                "missing_validation_n": int(missing_validation),
            },
        )

    if not validation.empty:
        status_counts = validation["validation_status"].value_counts(dropna=False).to_dict()
        primary_statuses = {"robust", "supported"}
        primary_n = int(validation["validation_status"].isin(primary_statuses).sum())
        _add_check(
            checks,
            "cascade:primary-validated-edge-count",
            primary_n >= args.min_primary_validated_edges,
            "Primary validated edge count meets the requested minimum.",
            {
                "minimum": args.min_primary_validated_edges,
                "primary_validated_edge_n": primary_n,
                "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            },
        )
        required_validation_cols = [
            "permutation_p_value",
            "permutation_fdr_q_value",
            "bootstrap_sign_stability",
            "site_direction_agreement_rate",
            "temporal_direction_agreement",
            "cascade_direction",
            "validation_status",
        ]
        missing_cols = [col for col in required_validation_cols if col not in validation.columns]
        _add_check(
            checks,
            "cascade:validation-schema",
            not missing_cols,
            "Validation output contains the expected publication-facing diagnostic columns.",
            {"missing_columns": missing_cols},
        )

    if not edge_report.empty:
        missing_direction = (
            int(edge_report["cascade_direction"].isna().sum())
            if "cascade_direction" in edge_report.columns
            else len(edge_report)
        )
        selection_weighted_cols = [col for col in edge_report.columns if "selection_weighted" in col]
        _add_check(
            checks,
            "cascade:edge-report-direction",
            missing_direction == 0,
            "Edge report preserves cascade_direction after merges.",
            {"missing_cascade_direction_n": missing_direction},
        )
        _add_check(
            checks,
            "cascade:no-selection-weighted-reporting",
            len(selection_weighted_cols) == 0,
            "Edge report does not expose stale selection-weighted/IPW-era columns.",
            {"selection_weighted_columns": selection_weighted_cols},
        )

    required_prevalence = [
        "prevalence_shift.parquet",
        "prevalence_mnar_sensitivity_curves.parquet",
        "prevalence_mnar_tipping_points.parquet",
        "validated_edge_enrichment.parquet",
        "prevalence_shift_summary.json",
    ]
    for filename in required_prevalence:
        _add_check(
            checks,
            f"prevalence:{filename}",
            (prevalence_dir / filename).exists(),
            f"Prevalence artifact present: {filename}",
            {"path": str(prevalence_dir / filename)},
        )

    prevalence = _read_parquet(prevalence_dir / "prevalence_shift.parquet", checks, "prevalence:read-summary", "prevalence_shift")
    if not prevalence.empty:
        required_prevalence_cols = [
            "naive_prevalence",
            "prevalence_lower_bound",
            "prevalence_upper_bound",
            "prevalence_bound_width",
            "cascade_trigger_fraction",
            "rho_independent_vs_cascade",
            "mnar_lambda0_prevalence",
            "mnar_lambda0_shift_from_naive",
        ]
        missing_cols = [col for col in required_prevalence_cols if col not in prevalence.columns]
        _add_check(
            checks,
            "prevalence:mnar-schema",
            not missing_cols,
            "Prevalence output uses the MNAR/bounds-era publication schema.",
            {"missing_columns": missing_cols},
        )
        legacy_cols = [col for col in prevalence.columns if col.startswith("legacy_")]
        _add_check(
            checks,
            "prevalence:legacy-columns-contained",
            len(legacy_cols) == 0,
            "Legacy delta columns should be absent from the primary prevalence artifact.",
            {"legacy_columns": legacy_cols},
            warn=True,
        )
        if {"prevalence_lower_bound", "naive_prevalence", "prevalence_upper_bound"}.issubset(prevalence.columns):
            bounds_ok = bool(
                (
                    (prevalence["prevalence_lower_bound"] <= prevalence["naive_prevalence"])
                    & (prevalence["naive_prevalence"] <= prevalence["prevalence_upper_bound"])
                )
                .fillna(False)
                .all()
            )
            _add_check(
                checks,
                "prevalence:bounds-contain-naive",
                bounds_ok,
                "Eligible-denominator bounds contain naive tested-only prevalence.",
                {"row_count": int(len(prevalence))},
            )

    tipping_points = _read_parquet(
        prevalence_dir / "prevalence_mnar_tipping_points.parquet",
        checks,
        "prevalence:read-tipping-points",
        "prevalence_mnar_tipping_points",
    )
    if not tipping_points.empty:
        required_tipping_cols = [
            "organism",
            "drug",
            "decision_threshold",
            "decision_threshold_pct",
            "mnar_lambda_star",
            "crossing_status",
            "naive_prevalence",
            "prevalence_at_lambda0",
            "shift_at_lambda0",
        ]
        missing_tipping_cols = [col for col in required_tipping_cols if col not in tipping_points.columns]
        valid_status_values = {"crosses_threshold", "always_above_threshold", "always_below_threshold", "not_evaluable"}
        invalid_status_values = (
            sorted(set(tipping_points["crossing_status"].dropna().astype(str)) - valid_status_values)
            if "crossing_status" in tipping_points.columns
            else []
        )
        _add_check(
            checks,
            "prevalence:tipping-point-schema",
            not missing_tipping_cols and not invalid_status_values,
            "MNAR tipping-point output has the publication schema and recognised crossing statuses.",
            {
                "missing_columns": missing_tipping_cols,
                "invalid_crossing_status_values": invalid_status_values,
            },
        )

    if args.require_training:
        required_model = [
            "feature_dataset.parquet",
            "metrics.parquet",
            "threshold_metrics.parquet",
            "selected_thresholds.parquet",
            "modeling_summary.json",
            "logistic_regression_predictions.parquet",
            "random_forest_predictions.parquet",
            "xgboost_predictions.parquet",
        ]
        for filename in required_model:
            _add_check(
                checks,
                f"model:{filename}",
                (model_dir / filename).exists(),
                f"Model artifact present: {filename}",
                {"path": str(model_dir / filename)},
            )
    else:
        _add_check(
            checks,
            "model:optional",
            True,
            "Model artifacts were not required for this audit.",
            {"require_training": False},
        )

    _add_check(
        checks,
        "features:model-ready",
        (feature_dir / "model_ready_pair_features.parquet").exists(),
        "Model-ready feature matrix is present.",
        {"path": str(feature_dir / "model_ready_pair_features.parquet")},
        warn=not args.require_training,
    )

    _add_check(
        checks,
        "report:manifest",
        (report_dir / "report_manifest.json").exists(),
        "Report manifest is present.",
        {"path": str(report_dir / "report_manifest.json")},
    )
    report_manifest = _read_report_manifest(report_dir)

    csv_tables = sorted(table_dir.glob("*.csv"))
    legacy_tables = [path.name for path in csv_tables if "legacy" in path.name.lower()]
    _add_check(
        checks,
        "report:tables",
        len(csv_tables) >= args.min_table_count,
        "Manuscript table CSV count meets the requested minimum.",
        {"minimum": args.min_table_count, "table_count": len(csv_tables)},
    )
    _add_check(
        checks,
        "report:no-legacy-tables",
        not legacy_tables,
        "Manuscript table directory should not expose legacy delta tables.",
        {"legacy_tables": legacy_tables},
        warn=True,
    )

    manifest_figure_exports = set((report_manifest.get("figure_exports") or {}).keys())
    allow_unmanifested = {"_contact_sheet.png"}
    figure_files = sorted(
        path
        for suffix in [".pdf", ".png", ".svg", ".html"]
        for path in figure_dir.glob(f"*{suffix}")
    )
    unmanifested_figures = [
        path.name for path in figure_files
        if manifest_figure_exports and path.name not in manifest_figure_exports and path.name not in allow_unmanifested
    ]
    _add_check(
        checks,
        "figures:no-unmanifested-figure-files",
        not unmanifested_figures,
        "Figure directory should not contain stale files outside the current report manifest.",
        {"unmanifested_figures": unmanifested_figures},
    )

    if manifest_figure_exports:
        counted_figures = [figure_dir / name for name in manifest_figure_exports if (figure_dir / name).exists()]
    else:
        counted_figures = figure_files
    figure_counts = {
        suffix: sum(1 for path in counted_figures if path.suffix == suffix)
        for suffix in [".pdf", ".png", ".svg", ".html"]
    }
    for suffix, minimum in {
        ".pdf": args.min_pdf_figures,
        ".png": args.min_png_figures,
        ".svg": args.min_svg_figures,
        ".html": args.min_html_figures,
    }.items():
        _add_check(
            checks,
            f"figures:{suffix.lstrip('.')}",
            figure_counts[suffix] >= minimum,
            f"Figure export count for {suffix} meets the requested minimum.",
            {"minimum": minimum, "count": figure_counts[suffix]},
        )

    png_dimensions: dict[str, Any] = {}
    if figure_counts[".png"] > 0:
        try:
            from PIL import Image

            for path in sorted(path for path in counted_figures if path.suffix == ".png"):
                with Image.open(path) as image:
                    png_dimensions[path.name] = {"width": image.size[0], "height": image.size[1]}
            min_width = min(item["width"] for item in png_dimensions.values())
            min_height = min(item["height"] for item in png_dimensions.values())
            _add_check(
                checks,
                "figures:png-resolution",
                min_width >= args.min_png_width and min_height >= args.min_png_height,
                "PNG figures meet minimum resolution.",
                {
                    "min_width_observed": min_width,
                    "min_height_observed": min_height,
                    "min_width_required": args.min_png_width,
                    "min_height_required": args.min_png_height,
                },
            )
        except Exception as exc:  # pragma: no cover - optional image validation
            _add_check(
                checks,
                "figures:png-resolution",
                False,
                "PNG resolution could not be inspected.",
                {"error": repr(exc)},
                warn=True,
            )

    placeholder_static_files = _static_placeholder_files(figure_dir)
    _add_check(
        checks,
        "figures:no-placeholder-static-exports",
        not placeholder_static_files,
        "Static PDF/SVG exports must contain real figure content, not fallback placeholder text.",
        {"placeholder_static_files": placeholder_static_files},
    )

    fail_count = sum(1 for check in checks if check.status == "FAIL")
    warn_count = sum(1 for check in checks if check.status == "WARN")
    overall = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": args.env,
        "scope": scope,
        "site": site,
        "organism": organism,
        "overall_status": overall,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "gold_dir": str(gold_dir),
        "cascade_dir": str(cascade_dir),
        "prevalence_dir": str(prevalence_dir),
        "feature_dir": str(feature_dir),
        "model_dir": str(model_dir),
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir),
        "report_dir": str(report_dir),
        "figure_counts": figure_counts,
        "png_dimensions": png_dimensions,
    }
    _write_outputs(checks, audit_dir, payload)
    print(json.dumps({key: payload[key] for key in ["overall_status", "fail_count", "warn_count", "figure_counts"]}, indent=2))
    print(f"Audit written to: {audit_dir}")
    return 1 if fail_count else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper-facing artifacts for publication readiness.")
    parser.add_argument("--env", default="mac")
    parser.add_argument("--scope", choices=("combined", "site"), default="combined")
    parser.add_argument("--site", default=None)
    parser.add_argument("--organism", required=True)
    parser.add_argument("--require-training", action="store_true")
    parser.add_argument("--min-primary-validated-edges", type=int, default=1)
    parser.add_argument("--min-table-count", type=int, default=10)
    parser.add_argument("--min-pdf-figures", type=int, default=1)
    parser.add_argument("--min-png-figures", type=int, default=1)
    parser.add_argument("--min-svg-figures", type=int, default=1)
    parser.add_argument("--min-html-figures", type=int, default=1)
    parser.add_argument("--min-png-width", type=int, default=1200)
    parser.add_argument("--min-png-height", type=int, default=628)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_audit(parse_args(sys.argv[1:])))
