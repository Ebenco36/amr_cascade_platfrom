import pandas as pd

from amr_cascade_platform.reporting.workflows.report_export_workflow import ReportExportWorkflow


def test_archive_existing_figure_exports_moves_top_level_static_outputs(tmp_path) -> None:
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    stale_html = figure_dir / "figure_cascade_network.html"
    stale_png = figure_dir / "figure_cascade_network.png"
    contact_sheet = figure_dir / "_contact_sheet.png"
    nested = figure_dir / "dataset_characterization"
    nested.mkdir()
    nested_figure = nested / "figure_nested.png"
    stale_html.write_text("<html></html>", encoding="utf-8")
    stale_png.write_bytes(b"png")
    contact_sheet.write_bytes(b"contact")
    nested_figure.write_bytes(b"nested")

    ReportExportWorkflow._archive_existing_figure_exports(figure_dir)

    archive = figure_dir / "_stale_figure_archive"
    assert (archive / stale_html.name).exists()
    assert (archive / stale_png.name).exists()
    assert not stale_html.exists()
    assert not stale_png.exists()
    assert contact_sheet.exists()
    assert nested_figure.exists()


def _pathway_flows() -> pd.DataFrame:
    # path_rank 1: both hops robust -> should land in "robust" and "validated" only.
    # path_rank 2: hop1 robust, hop2 supported (weaker) -> "supported" and "validated" only,
    #   not "robust" -- a path's tier is its weakest hop.
    # path_rank 3: hop1 robust, hop2 mixed -> excluded from all three tiers.
    # path_rank 4: single-hop, supported -> "supported" and "validated".
    # path_rank 5: hop1 robust, hop2 has no matching validation_results row at all ->
    #   excluded from all three tiers (unmatched stages are treated as unvalidated).
    return pd.DataFrame(
        [
            {"path_rank": 1, "source": "A", "target": "B", "stage": 1},
            {"path_rank": 1, "source": "B", "target": "C", "stage": 2},
            {"path_rank": 2, "source": "D", "target": "E", "stage": 1},
            {"path_rank": 2, "source": "E", "target": "F", "stage": 2},
            {"path_rank": 3, "source": "G", "target": "H", "stage": 1},
            {"path_rank": 3, "source": "H", "target": "I", "stage": 2},
            {"path_rank": 4, "source": "J", "target": "K", "stage": 1},
            {"path_rank": 5, "source": "L", "target": "M", "stage": 1},
            {"path_rank": 5, "source": "M", "target": "N", "stage": 2},
        ]
    )


def _validation_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "validation_status": "robust"},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "C", "validation_status": "robust"},
            {"upstream_antibiotic": "D", "downstream_antibiotic": "E", "validation_status": "robust"},
            {"upstream_antibiotic": "E", "downstream_antibiotic": "F", "validation_status": "supported"},
            {"upstream_antibiotic": "G", "downstream_antibiotic": "H", "validation_status": "robust"},
            {"upstream_antibiotic": "H", "downstream_antibiotic": "I", "validation_status": "mixed"},
            {"upstream_antibiotic": "J", "downstream_antibiotic": "K", "validation_status": "supported"},
            {"upstream_antibiotic": "L", "downstream_antibiotic": "M", "validation_status": "robust"},
            # M -> N deliberately absent.
        ]
    )


def test_tiered_pathway_flows_splits_by_weakest_hop() -> None:
    tiers = ReportExportWorkflow._tiered_pathway_flows(_pathway_flows(), _validation_results())

    assert set(tiers["robust"]["path_rank"]) == {1}
    assert set(tiers["supported"]["path_rank"]) == {2, 4}
    assert set(tiers["validated"]["path_rank"]) == {1, 2, 4}

    # A path with any mixed/insufficient/unmatched hop must never appear in any tier.
    for tier in tiers.values():
        assert 3 not in set(tier["path_rank"])
        assert 5 not in set(tier["path_rank"])

    # robust and supported are disjoint by construction (weakest-hop rule).
    assert set(tiers["robust"]["path_rank"]).isdisjoint(set(tiers["supported"]["path_rank"]))


def test_tiered_pathway_flows_handles_empty_inputs() -> None:
    pathway_flows = _pathway_flows()
    validation_results = _validation_results()

    empty_flows_result = ReportExportWorkflow._tiered_pathway_flows(pathway_flows.head(0), validation_results)
    assert all(df.empty for df in empty_flows_result.values())

    empty_validation_result = ReportExportWorkflow._tiered_pathway_flows(pathway_flows, validation_results.head(0))
    assert all(df.empty for df in empty_validation_result.values())


def test_validated_edge_report_filters_to_robust_and_supported() -> None:
    edge_report = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "validation_status": "robust"},
            {"upstream_antibiotic": "C", "downstream_antibiotic": "D", "validation_status": "supported"},
            {"upstream_antibiotic": "E", "downstream_antibiotic": "F", "validation_status": "mixed"},
            {"upstream_antibiotic": "G", "downstream_antibiotic": "H", "validation_status": "insufficient"},
        ]
    )

    filtered = ReportExportWorkflow._validated_edge_report(edge_report)

    assert set(filtered["validation_status"]) == {"robust", "supported"}
    assert len(filtered) == 2


def _retained_edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "total_support_n": 100},
            {"upstream_antibiotic": "D", "downstream_antibiotic": "E", "total_support_n": 80},
            {"upstream_antibiotic": "G", "downstream_antibiotic": "H", "total_support_n": 60},
            {"upstream_antibiotic": "X", "downstream_antibiotic": "Y", "total_support_n": 40},
        ]
    )


def _edge_validation_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B", "validation_status": "robust"},
            {"upstream_antibiotic": "D", "downstream_antibiotic": "E", "validation_status": "supported"},
            {"upstream_antibiotic": "G", "downstream_antibiotic": "H", "validation_status": "mixed"},
            # X -> Y deliberately absent from validation_results.
        ]
    )


def test_tiered_retained_edges_splits_by_status() -> None:
    tiers = ReportExportWorkflow._tiered_retained_edges(_retained_edges(), _edge_validation_results())

    assert set(tiers["robust"]["upstream_antibiotic"]) == {"A"}
    assert set(tiers["supported"]["upstream_antibiotic"]) == {"D"}
    assert set(tiers["validated"]["upstream_antibiotic"]) == {"A", "D"}
    # "validation_status" is a join artifact, not part of the retained_edges contract --
    # downstream consumers (chord/network plotters) shouldn't see it.
    assert "validation_status" not in tiers["robust"].columns

    for tier in tiers.values():
        assert "G" not in set(tier["upstream_antibiotic"])  # mixed
        assert "X" not in set(tier["upstream_antibiotic"])  # unmatched


def test_tiered_retained_edges_handles_empty_inputs() -> None:
    retained_edges = _retained_edges()
    validation_results = _edge_validation_results()

    empty_edges_result = ReportExportWorkflow._tiered_retained_edges(retained_edges.head(0), validation_results)
    assert all(df.empty for df in empty_edges_result.values())

    empty_validation_result = ReportExportWorkflow._tiered_retained_edges(retained_edges, validation_results.head(0))
    assert all(df.empty for df in empty_validation_result.values())
