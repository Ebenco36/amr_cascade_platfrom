"""Explicit output contract for the detection-and-validation manuscript."""

import json
from pathlib import Path

PAPER1_FIGURE_GROUPS = (
    "cascade_directional", "cascade_evidence_scatter", "observation_coverage",
    "descriptive_summaries",
)
PAPER1_REPORT_FIGURES = (
    "figure_escalation_matrix_validated", "figure_suppression_matrix_validated",
    "figure_escalation_forest_validated", "figure_suppression_forest_validated",
    "figure_escalation_sankey_validated", "figure_suppression_sankey_validated",
    "figure_escalation_network_validated", "figure_suppression_network_validated",
    "figure_escalation_aware_validated", "figure_suppression_aware_validated",
    "figure_das_vs_pbi_validated", "figure_raw_vs_adjusted_validated",
    "figure_adjustment_concordance_summary_validated",
    "figure_opportunity_space", "figure_observation_coverage",
    "figure_observation_coverage_by_era", "figure_panel_breadth",
)
PAPER1_PREDICTION_FIGURES = (
    "supp_pred_fig1_roc_curves", "supp_pred_fig2_pr_curves",
    "supp_pred_fig3_calibration", "supp_pred_fig4_cross_split_roc",
    "supp_pred_fig5_threshold_analysis", "supp_pred_fig6_lr_coefficients",
)
# Drawn by scripts/run_temporal_regimes.py after the report step (static formats only).
PAPER1_TEMPORAL_FIGURES = ("figure_temporal_regimes",)


def report_ready(manifest_path: Path, edge_report_path: Path, *, hide_titles: bool = False,
                 publication: str = "paper1", additional_inputs: tuple[Path, ...] = (),
                 project_root: Path | None = None) -> bool:
    """A manifest is not completion evidence when its exports have disappeared."""
    try:
        manifest = json.loads(manifest_path.read_text())
        if project_root is not None:
            from amr_cascade_platform.core.utils.provenance import file_fingerprint, git_state
            provenance = manifest.get("provenance", {})
            if provenance.get("git") != git_state(project_root):
                return False
            recorded = provenance.get("inputs", {})
            if not recorded or any(file_fingerprint(Path(value["path"])) != value for value in recorded.values()):
                return False
        if bool(manifest.get("hide_figure_titles", False)) != hide_titles:
            return False
        if any(manifest_path.stat().st_mtime <= path.stat().st_mtime
               for path in (edge_report_path, *additional_inputs)):
            return False
        if manifest.get("publication") != publication or manifest.get("missing_figure_groups") != {}:
            return False
        from amr_cascade_platform.reporting.manuscript_profile import required_exports
        exports = manifest["figure_exports"]
        required = [exports[name] for name in required_exports(publication, ("pdf", "png", "svg", "html"))]
        tables = manifest["tables"]
        return bool(tables) and all(
            Path(name).is_file() and Path(name).stat().st_size > 0
            for name in required + list(tables.values())
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
