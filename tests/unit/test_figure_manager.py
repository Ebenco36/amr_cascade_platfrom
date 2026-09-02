from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.visualization.report.figure_manager import FigureManager
from amr_cascade_platform.visualization.report.plotly_forest_plotter import EdgeReportNormalizer


def test_export_forest_summary_creates_png(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "total_support_n": 50,
            }
        ]
    )
    path = manager.export_forest_summary(edge_report, tmp_path / "forest.png")
    assert path.exists()


def test_export_downstream_trigger_forest_creates_suite(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.5,
                "total_support_n": 50,
                "positive_probability": 0.5,
                "negative_probability": 0.2,
                "positive_support_n": 20,
                "negative_support_n": 30,
            }
        ]
    )
    outputs = manager.export_all_downstream_trigger_forests(
        escalation_results,
        tmp_path,
        ("png",),
    )
    assert "figure_downstream_trigger_forest__b.png" in outputs
    assert outputs["figure_downstream_trigger_forest__b.png"].exists()


def test_export_validated_primary_forest_creates_png(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "ALPHA_TRIGGER_ONLY",
                "downstream_antibiotic": "BETA_DOWNSTREAM_ONLY",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 50,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_probability": 0.4,
                "negative_probability": 0.2,
                "validation_status": "robust",
            },
            {
                "upstream_antibiotic": "CHARLIE_UNRELATED_ONLY",
                "downstream_antibiotic": "DELTA_DOWNSTREAM_ONLY",
                "escalation_ratio": 3.0,
                "adjusted_odds_ratio": 1.8,
                "total_support_n": 50,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_probability": 0.6,
                "negative_probability": 0.2,
                "validation_status": "mixed",
            },
        ]
    )
    outputs = manager.export_validated_primary_forest(
        edge_report,
        tmp_path / "validated_forest",
        ("png",),
    )
    assert "validated_forest.png" in outputs
    assert outputs["validated_forest.png"].exists()


def test_export_aware_transition_heatmap_creates_png(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    transition_summary = pd.DataFrame(
        [
            {
                "upstream_aware_category": "Access",
                "downstream_aware_category": "Watch",
                "aware_transition": "Access -> Watch",
                "aware_direction": "upward",
                "aware_step": 1,
                "validated_edge_n": 3,
                "robust_edge_n": 2,
                "supported_edge_n": 1,
                "total_support_sum": 100,
                "support_weighted_mean_escalation_ratio": 2.3,
            },
            {
                "upstream_aware_category": "Watch",
                "downstream_aware_category": "Reserve",
                "aware_transition": "Watch -> Reserve",
                "aware_direction": "upward",
                "aware_step": 1,
                "validated_edge_n": 2,
                "robust_edge_n": 1,
                "supported_edge_n": 1,
                "total_support_sum": 80,
                "support_weighted_mean_escalation_ratio": 1.8,
            },
        ]
    )
    outputs = manager.export_aware_transition_heatmap(
        transition_summary,
        tmp_path / "aware_heatmap",
        ("png",),
    )
    assert "aware_heatmap.png" in outputs
    assert outputs["aware_heatmap.png"].exists()


def test_export_mnar_tipping_point_creates_html_and_png(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    curves = pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "MEROPENEM",
                "eligible_n": 100,
                "tested_n": 40,
                "resistant_n": 20,
                "unknown_binary_outcome_n": 60,
                "naive_prevalence_pct": 50.0,
                "mnar_lambda": -1.0,
                "mnar_prevalence_pct": 65.0,
                "mnar_shift_from_naive_pct": -15.0,
                "mnar_status": "estimated",
            },
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "MEROPENEM",
                "eligible_n": 100,
                "tested_n": 40,
                "resistant_n": 20,
                "unknown_binary_outcome_n": 60,
                "naive_prevalence_pct": 50.0,
                "mnar_lambda": 0.0,
                "mnar_prevalence_pct": 40.0,
                "mnar_shift_from_naive_pct": 10.0,
                "mnar_status": "estimated",
            },
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "MEROPENEM",
                "eligible_n": 100,
                "tested_n": 40,
                "resistant_n": 20,
                "unknown_binary_outcome_n": 60,
                "naive_prevalence_pct": 50.0,
                "mnar_lambda": 1.0,
                "mnar_prevalence_pct": 15.0,
                "mnar_shift_from_naive_pct": 35.0,
                "mnar_status": "estimated",
            },
        ]
    )
    tipping_points = pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "MEROPENEM",
                "decision_threshold_pct": 20.0,
                "mnar_lambda_star": 0.8,
                "crossing_status": "crosses_threshold",
                "naive_prevalence_pct": 50.0,
                "prevalence_at_lambda0_pct": 40.0,
                "shift_at_lambda0_pct": 10.0,
                "eligible_n": 100,
                "tested_n": 40,
                "unknown_binary_outcome_n": 60,
                "cascade_trigger_fraction": 0.75,
                "rho_independent_vs_cascade": 0.50,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "MEROPENEM",
                "prevalence_lower_bound_pct": 20.0,
                "prevalence_upper_bound_pct": 80.0,
                "cascade_trigger_fraction": 0.75,
                "rho_independent_vs_cascade": 0.50,
                "naive_prevalence_pct": 50.0,
            }
        ]
    )

    outputs = manager.export_mnar_tipping_point(
        curves,
        tipping_points,
        summary,
        tmp_path / "mnar_tipping",
        ("html", "png"),
        organism="ESCHERICHIA COLI",
    )
    assert "mnar_tipping.html" in outputs
    assert "mnar_tipping.png" in outputs
    assert outputs["mnar_tipping.html"].exists()
    assert outputs["mnar_tipping.png"].exists()
    html = outputs["mnar_tipping.html"].read_text(encoding="utf-8")
    assert "Tipping point" in html
    assert "Interactive what-if simulator" in html
    assert 'id="simTested"' in html
    assert 'id="simResistant"' in html
    assert 'id="simUnknown"' in html
    assert 'id="simThreshold"' in html
    assert 'id="simLambda"' in html
    assert 'id="simChart"' in html


def test_downstream_trigger_suite_excludes_downstream_drug_from_upstream_rows(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    escalation_results = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "AZTREONAM",
                "downstream_antibiotic": "AZTREONAM",
                "escalation_ratio": 2.5,
                "total_support_n": 50,
                "positive_probability": 0.5,
                "negative_probability": 0.2,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_tested_n": 10,
                "negative_tested_n": 6,
            },
            {
                "upstream_antibiotic": "CEFEPIME",
                "downstream_antibiotic": "AZTREONAM",
                "escalation_ratio": 3.0,
                "total_support_n": 50,
                "positive_probability": 0.5,
                "negative_probability": 0.2,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_tested_n": 10,
                "negative_tested_n": 6,
            },
        ]
    )
    plotter = manager._forest_plotter
    canonical = plotter._canonicalize_pair_table(EdgeReportNormalizer.normalize(escalation_results))
    upstream_universe = sorted(canonical["upstream_antibiotic"].dropna().unique().tolist())
    filtered_universe = [drug for drug in upstream_universe if drug != "AZTREONAM"]
    prepared = plotter._prepare_downstream_plot_data("AZTREONAM", canonical[canonical["downstream_antibiotic"] == "AZTREONAM"], filtered_universe)
    assert "AZTREONAM" not in prepared["upstream_antibiotic"].tolist()


def test_downstream_trigger_suite_does_not_fill_unrelated_validated_upstreams(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "ALPHA_TRIGGER_ONLY",
                "downstream_antibiotic": "BETA_DOWNSTREAM_ONLY",
                "escalation_ratio": 2.5,
                "total_support_n": 50,
                "positive_probability": 0.5,
                "negative_probability": 0.2,
                "positive_support_n": 20,
                "negative_support_n": 30,
                "positive_tested_n": 10,
                "negative_tested_n": 6,
            },
            {
                "upstream_antibiotic": "CHARLIE_UNRELATED_ONLY",
                "downstream_antibiotic": "DELTA_DOWNSTREAM_ONLY",
                "escalation_ratio": 3.0,
                "total_support_n": 60,
                "positive_probability": 0.6,
                "negative_probability": 0.2,
                "positive_support_n": 20,
                "negative_support_n": 40,
                "positive_tested_n": 12,
                "negative_tested_n": 8,
            },
        ]
    )

    outputs = manager.export_all_downstream_trigger_forests(edge_report, tmp_path, ("html",))
    b_html = outputs["figure_downstream_trigger_forest__beta_downstream_only.html"].read_text(encoding="utf-8")

    assert "ALPHA_TRIGGER_ONLY" in b_html
    assert "CHARLIE_UNRELATED_ONLY" not in b_html


# These three cover the exact FigureManager.export_* call path (not the underlying
# plotter classes directly) with tier_label, because that distinction is exactly
# what let a real bug through: export_chord's tier_label fix was applied to
# PyCirclizeChordPlotter, a legacy class -- but FigureManager actually wires
# _chord_plotter to PlotlyCascadeMatrixPlotter, a different class entirely, which
# didn't accept the new argument. A unit test against PyCirclizeChordPlotter in
# isolation passed cleanly and never caught it; only calling through FigureManager
# itself, as report_export_workflow.py actually does, exercises the real wiring.
def test_export_sankey_accepts_tier_label(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    pathway_flows = pd.DataFrame(
        [
            {
                "path_rank": 1,
                "source": "CEFTAZIDIM",
                "target": "CO-TRIMOXAZOL",
                "stage": 1,
                "path_min_support_n": 10,
                "edge_escalation_ratio": 2.0,
                "path_escalation_ratio": 2.0,
            }
        ]
    )
    outputs = manager.export_sankey(pathway_flows, tmp_path / "sankey", ("html", "png"), tier_label="Robust")
    html = outputs["sankey.html"].read_text(encoding="utf-8")

    assert "CEFTAZIDIME" in html
    assert "SULFAMETHOXAZOLE\\u002fTRIMETHOPRIM" in html
    assert outputs["sankey.png"].exists()


def test_export_chord_accepts_tier_label(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CO-TRIMOXAZOL",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.4,
                "total_support_n": 50,
            }
        ]
    )
    outputs = manager.export_chord(retained_edges, tmp_path / "chord", ("html", "png"), tier_label="Robust")
    html = outputs["chord.html"].read_text(encoding="utf-8")

    assert "SULFAMETHOXAZOLE\\u002fTRIMETHOPRIM" in html
    assert "table_supplementary_antibiotic_abbreviations.csv" in html
    assert outputs["chord.png"].exists()


def test_export_network_accepts_tier_label(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    from amr_cascade_platform.core.paths.path_manager import PathManager

    manager = FigureManager(settings, PathManager(project_root, settings))
    network_edges = pd.DataFrame(
        [{"source": "CEFTAZIDIM", "target": "CO-TRIMOXAZOL", "escalation_ratio": 2.0, "adjusted_odds_ratio": 1.4, "edge_weight": 0.3}]
    )
    network_nodes = pd.DataFrame(
        [
            {"antibiotic": "CEFTAZIDIM", "out_degree": 1, "in_degree": 0, "pagerank": 0.5, "betweenness": 0.0},
            {"antibiotic": "CO-TRIMOXAZOL", "out_degree": 0, "in_degree": 1, "pagerank": 0.5, "betweenness": 0.0},
        ]
    )
    outputs = manager.export_network(network_edges, network_nodes, tmp_path / "network", ("html", "png"), tier_label="Robust")
    html = outputs["network.html"].read_text(encoding="utf-8")

    assert "CEFTAZIDIME" in html
    assert "SULFAMETHOXAZOLE\\u002fTRIMETHOPRIM" in html
    assert outputs["network.png"].exists()
