"""The direction-pure output contract: figures may only draw what these tables say."""

from __future__ import annotations

import pandas as pd
import pytest

from amr_cascade_platform.reporting.builders import directional_views as dv


def _edges() -> pd.DataFrame:
    rows = [
        # upstream, downstream, ER, support, status
        ("A", "B", 4.0, 1000, "robust"),
        ("A", "C", 2.0, 500, "supported"),
        ("B", "C", 8.0, 200, "robust"),
        ("C", "A", 0.5, 800, "robust"),
        ("B", "A", 0.25, 300, "supported"),
        ("C", "B", 0.9, 5000, "robust"),
        ("A", "D", 1.5, 50, "mixed"),          # not validated
        ("D", "A", 0.2, 60, "insufficient"),   # not validated
    ]
    return pd.DataFrame(rows, columns=[
        "upstream_antibiotic", "downstream_antibiotic", "escalation_ratio", "total_support_n", "validation_status",
    ])


AWARE = {"A": "Access", "B": "Watch", "C": "Reserve", "D": "Access"}
RANK = {"Access": 1, "Watch": 2, "Reserve": 3}


def test_direction_is_derived_from_the_ratio_and_partitions_edges():
    val = dv.validated_edges(_edges())
    parts = dv.split_by_direction(val)
    assert len(val) == 6
    assert set(parts["escalation"].escalation_ratio) == {4.0, 2.0, 8.0}
    assert set(parts["suppression"].escalation_ratio) == {0.5, 0.25, 0.9}
    assert (parts["escalation"].escalation_ratio > 1).all()
    assert (parts["suppression"].escalation_ratio < 1).all()


def test_non_validated_edges_never_reach_any_direction():
    val = dv.validated_edges(_edges())
    assert set(val.validation_status) <= {"robust", "supported"}
    assert "D" not in set(val.upstream_antibiotic) | set(val.downstream_antibiotic)


def test_edge_with_ratio_of_exactly_one_is_dropped_and_counted_not_hidden():
    edges = pd.concat([_edges(), pd.DataFrame([{
        "upstream_antibiotic": "A", "downstream_antibiotic": "E", "escalation_ratio": 1.0,
        "total_support_n": 10, "validation_status": "robust",
    }])], ignore_index=True)
    val = dv.validated_edges(edges)
    assert val.attrs["n_undirected"] == 1
    assert len(val) == 6


def test_contradictory_cascade_direction_column_is_refused():
    edges = _edges()
    edges["cascade_direction"] = ["escalation"] * len(edges)  # wrong for the ER<1 rows
    with pytest.raises(ValueError, match="contradicts"):
        dv.edge_directions(edges)


def test_matching_cascade_direction_column_is_accepted():
    edges = _edges()
    edges["cascade_direction"] = dv.edge_directions(edges)
    assert dv.edge_directions(edges).equals(edges["cascade_direction"].astype("object"))


def test_bipartite_link_width_is_the_pairs_own_support_taken_once():
    val = dv.validated_edges(_edges())
    links = dv.bipartite_links(val, "escalation")
    assert dict(zip(zip(links.upstream_antibiotic, links.downstream_antibiotic), links.support_n)) == {
        ("A", "B"): 1000, ("A", "C"): 500, ("B", "C"): 200,
    }
    # A feeds two downstream drugs: A's outgoing width is the sum of two distinct pairs,
    # never the same pair counted per path.
    assert links.loc[links.upstream_antibiotic == "A", "support_n"].sum() == 1500


def test_bipartite_links_contain_only_the_requested_direction():
    val = dv.validated_edges(_edges())
    assert (dv.bipartite_links(val, "escalation").escalation_ratio > 1).all()
    assert (dv.bipartite_links(val, "suppression").escalation_ratio < 1).all()


def test_top_n_keeps_the_highest_support_pairs_and_reports_coverage():
    val = dv.validated_edges(_edges())
    links = dv.bipartite_links(val, "escalation", top_n=2)
    assert list(links.support_n) == [1000, 500]
    cov = dv.coverage_summary(links, val, "escalation")
    assert cov["n_shown"] == 2 and cov["n_total"] == 3
    assert cov["support_fraction"] == pytest.approx(1500 / 1700)


def test_matrix_cells_are_direction_pure_with_positive_strength_in_both_directions():
    val = dv.validated_edges(_edges())
    order = dv.shared_drug_order(val, AWARE.get, lambda c: RANK[c])
    esc = dv.matrix_frame(val, "escalation", order)
    sup = dv.matrix_frame(val, "suppression", order)
    assert (esc.strength > 0).all() and (sup.strength > 0).all()
    # ER=0.25 is a stronger suppression (2 doublings) than ER=0.5 (1 doubling)
    by_er = dict(zip(sup.escalation_ratio, sup.strength))
    assert by_er[0.25] == pytest.approx(2.0) and by_er[0.5] == pytest.approx(1.0)
    assert by_er[0.25] > by_er[0.5] > by_er[0.9]
    # identical row/column order in both directions
    assert len(order) == 3 and order == sorted(order, key=lambda d: (RANK[AWARE[d]], d))


def test_two_edges_in_one_matrix_cell_are_an_error_not_a_silent_overwrite():
    val = dv.validated_edges(_edges())
    dup = pd.concat([val, val.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="same matrix cell"):
        dv.matrix_frame(dup, "escalation", ["A", "B", "C"])


def test_node_statistics_are_computed_inside_each_direction_only():
    val = dv.validated_edges(_edges())
    _, esc_nodes = dv.directional_graph_tables(val, "escalation")
    _, sup_nodes = dv.directional_graph_tables(val, "suppression")
    esc = esc_nodes.set_index("antibiotic")
    sup = sup_nodes.set_index("antibiotic")
    assert esc.loc["A", "out_degree"] == 2 and esc.loc["C", "in_degree"] == 2
    assert sup.loc["C", "out_degree"] == 2 and sup.loc["A", "in_degree"] == 2
    assert esc.out_degree.sum() == 3 and sup.out_degree.sum() == 3


def test_node_roles_columns_keep_directions_separate_and_sum_to_edge_counts():
    val = dv.validated_edges(_edges())
    roles = dv.node_roles(val)
    assert roles.escalation_out_degree.sum() == 3
    assert roles.suppression_out_degree.sum() == 3
    assert roles.escalation_in_degree.sum() == roles.escalation_out_degree.sum()
    assert roles.set_index("antibiotic").loc["A", "escalation_out_robust"] == 1  # A->B robust, A->C supported


def test_direction_by_aware_matrix_counts_every_directed_pattern_once():
    val = dv.validated_edges(_edges())

    def step(up: str, down: str) -> str:
        d = RANK[down] - RANK[up]
        return "upward" if d > 0 else "downward" if d < 0 else "lateral"

    matrix = dv.direction_by_aware_matrix(val, AWARE.get, step)
    assert int(matrix.values.sum()) == len(val)
    assert int(matrix.loc["escalation"].sum()) == 3 and int(matrix.loc["suppression"].sum()) == 3


def test_aware_table_reports_within_direction_median_not_a_pooled_mean():
    val = dv.validated_edges(_edges())

    def step(up: str, down: str) -> str:
        return "lateral"

    esc = dv.aware_direction_table(val, "escalation", AWARE.get, step)
    sup = dv.aware_direction_table(val, "suppression", AWARE.get, step)
    assert (esc.median_log2_er > 0).all() and (sup.median_log2_er < 0).all()


def test_circular_layout_is_deterministic_and_groups_aware_tiers_contiguously():
    drugs = ["A", "B", "C", "D"]
    first = dv.circular_layout(drugs, AWARE.get, lambda c: RANK[c])
    again = dv.circular_layout(list(reversed(drugs)), AWARE.get, lambda c: RANK[c])
    assert first == again
    assert all(abs(x * x + y * y - 1.0) < 1e-9 for x, y in first.values())


def test_canonicalisation_never_double_counts_a_pair_that_collapses():
    edges = pd.DataFrame([
        {"upstream_antibiotic": "cefotaxim", "downstream_antibiotic": "B", "escalation_ratio": 2.0, "total_support_n": 10, "validation_status": "robust"},
        {"upstream_antibiotic": "CEFOTAXIME", "downstream_antibiotic": "B", "escalation_ratio": 2.0, "total_support_n": 99, "validation_status": "robust"},
    ])
    out = dv.canonicalize_edges(edges, lambda s: "CEFOTAXIME" if str(s).lower().startswith("cefotaxim") else str(s))
    assert len(out) == 1 and out.attrs["n_collapsed_duplicates"] == 1
    assert int(out.total_support_n.iloc[0]) == 99


def test_unknown_direction_and_tier_are_errors():
    val = dv.validated_edges(_edges())
    with pytest.raises(ValueError):
        dv.bipartite_links(val, "both")
    with pytest.raises(ValueError):
        dv.tier_edges(val, "everything")
    assert set(dv.tier_edges(val, "robust").validation_status) == {"robust"}


def test_display_floor_is_symmetric_in_fold_change_and_never_mixes_directions():
    val = dv.validated_edges(_edges())
    esc = dv.display_selection(val, "escalation", min_fold=3.0)   # keeps ER >= 3
    sup = dv.display_selection(val, "suppression", min_fold=3.0)  # keeps ER <= 1/3
    assert sorted(esc.escalation_ratio) == [4.0, 8.0]
    assert sorted(sup.escalation_ratio) == [0.25]
    assert len(dv.display_selection(val, "escalation", min_fold=1.0)) == 3   # floor off
    assert len(dv.display_selection(val, "suppression", min_fold=1.0)) == 3
    boundary = dv.display_selection(val, "escalation", min_fold=2.0)         # ER == 2.0 is kept
    assert 2.0 in set(boundary.escalation_ratio)


def _adjustment_edges() -> pd.DataFrame:
    rows = [
        # label,                escalation_ratio, adjusted_OR, ci_lower, ci_upper
        ("concordant_escalation",   4.0,  2.5, 1.2, 5.0),
        ("attenuated_escalation",   4.0,  1.8, 0.7, 3.0),   # same side, CI includes 1
        ("attenuated_no_ci",        4.0,  1.8, None, None),  # same side, CI missing entirely
        ("reversed_escalation",     4.0,  0.6, 0.3, 0.9),   # crosses to the other side of 1
        ("concordant_suppression",  0.25, 0.4, 0.2, 0.8),
        ("attenuated_suppression",  0.25, 0.7, 0.4, 1.3),
        ("reversed_suppression",    0.25, 1.9, 1.1, 3.0),
        ("no_adjusted_or",          4.0,  None, None, None),
        ("neutral_raw_er",          1.0,  2.0, 1.1, 3.0),   # raw ER == 1: no direction to compare against
    ]
    return pd.DataFrame(rows, columns=[
        "label", "escalation_ratio", "adjusted_odds_ratio", "adjusted_odds_ratio_ci_lower", "adjusted_odds_ratio_ci_upper",
    ])


def test_adjustment_concordance_matches_each_defined_category():
    result = dv.with_adjustment_concordance(_adjustment_edges()).set_index("label")["adjustment_concordance"]
    assert result["concordant_escalation"] == "concordant"
    assert result["attenuated_escalation"] == "attenuated"
    assert result["attenuated_no_ci"] == "attenuated"
    assert result["reversed_escalation"] == "reversed"
    assert result["concordant_suppression"] == "concordant"
    assert result["attenuated_suppression"] == "attenuated"
    assert result["reversed_suppression"] == "reversed"
    assert result["no_adjusted_or"] == "unavailable"
    assert result["neutral_raw_er"] == "unavailable"


def test_adjustment_concordance_is_idempotent_and_never_overwrites_an_existing_column():
    edges = _adjustment_edges()
    edges["adjustment_concordance"] = "precomputed_upstream"
    result = dv.with_adjustment_concordance(edges)
    assert (result["adjustment_concordance"] == "precomputed_upstream").all()


def test_adjustment_concordance_only_uses_categories_the_plotter_expects():
    result = dv.with_adjustment_concordance(_adjustment_edges())
    assert set(result["adjustment_concordance"]) <= set(dv.CONCORDANCE_CATEGORIES)
