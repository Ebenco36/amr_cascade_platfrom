from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_pair_dependence_analyzer import CascadePairDependenceAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_cascade_pair_dependence_analyzer_reports_asymmetry_metrics() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadePairDependenceAnalyzer(settings)

    drug_pairs = pd.DataFrame(
        [
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2024-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "B", "downstream_tested": 1, "downstream_eligible": 1},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2024-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "B", "downstream_tested": 1, "downstream_eligible": 1},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2024-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "B", "downstream_tested": 0, "downstream_eligible": 1},
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2024-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "B", "downstream_antibiotic": "A", "downstream_tested": 0, "downstream_eligible": 1},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2024-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "B", "downstream_antibiotic": "A", "downstream_tested": 0, "downstream_eligible": 1},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2024-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "B", "downstream_antibiotic": "A", "downstream_tested": 1, "downstream_eligible": 1},
            {"anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "2024-01-01T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "C", "downstream_tested": 1, "downstream_eligible": 1},
            {"anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "2024-01-02T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "C", "downstream_tested": 1, "downstream_eligible": 1},
            {"anon_id": "p3", "pat_enc_csn_id_coded": "e3", "order_proc_id_coded": "o3", "order_time_jittered": "2024-01-03T00:00:00Z", "organism": "ESCHERICHIA COLI", "source_site": "armd", "upstream_antibiotic": "A", "downstream_antibiotic": "C", "downstream_tested": 1, "downstream_eligible": 1},
        ]
    )
    retained_edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B"},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "A"},
            {"upstream_antibiotic": "A", "downstream_antibiotic": "C"},
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)

    assert len(result) == 3
    assert {"smoothed_pair_test_rate", "smoothed_reverse_test_rate", "testing_asymmetry_score", "testing_asymmetry_bin"}.issubset(result.columns)
    row = result.loc[
        result["upstream_antibiotic"].eq("A") & result["downstream_antibiotic"].eq("B")
    ].iloc[0]
    assert row["smoothed_pair_test_rate"] > 0
    assert row["smoothed_reverse_test_rate"] > 0
    assert row["testing_asymmetry_score"] >= 0
    assert row["testing_asymmetry_bin"] in {"low_asymmetry", "medium_asymmetry", "high_asymmetry", "undifferentiated"}

    # A<->B has 3 rows of support in each direction: perfectly balanced denominators.
    assert "directional_support_ratio" in result.columns
    assert row["directional_support_ratio"] == 1.0

    # A->C has no C->A rows in this fixture at all, so the reverse denominator
    # does not exist -- the ratio must be undefined (NaN), not silently 0 or 1.
    ac_row = result.loc[
        result["upstream_antibiotic"].eq("A") & result["downstream_antibiotic"].eq("C")
    ].iloc[0]
    assert pd.isna(ac_row["directional_support_ratio"])

    # A<->B rows 1-3 and 4-6 share the exact same three episodes (p1/e1/o1 etc.),
    # so overlap should agree with the support ratio here: both 1.0.
    assert row["directional_episode_overlap"] == 1.0
    assert pd.isna(ac_row["directional_episode_overlap"])


def test_directional_episode_overlap_catches_what_support_ratio_cannot() -> None:
    """Reviewer's exact motivating example: two directions can have identical
    support counts (DSR = 1.0) while being built from completely different
    episodes (overlap = 0.0). The size ratio alone cannot see this; the episode
    overlap must.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadePairDependenceAnalyzer(settings)

    def _row(anon_id: str, upstream: str, downstream: str, tested: int) -> dict:
        return {
            "anon_id": anon_id,
            "pat_enc_csn_id_coded": f"e_{anon_id}",
            "order_proc_id_coded": f"o_{anon_id}",
            "order_time_jittered": "2024-01-01T00:00:00Z",
            "organism": "ESCHERICHIA COLI",
            "source_site": "armd",
            "upstream_antibiotic": upstream,
            "downstream_antibiotic": downstream,
            "downstream_tested": tested,
            "downstream_eligible": 1,
        }

    drug_pairs = pd.DataFrame(
        [
            # A->B: 3 episodes, p1/p2/p3.
            _row("p1", "A", "B", 1),
            _row("p2", "A", "B", 1),
            _row("p3", "A", "B", 0),
            # B->A: also 3 episodes (identical support count), but p4/p5/p6 --
            # completely disjoint from p1/p2/p3.
            _row("p4", "B", "A", 1),
            _row("p5", "B", "A", 0),
            _row("p6", "B", "A", 1),
        ]
    )
    retained_edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B"},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "A"},
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)
    row = result.loc[result["upstream_antibiotic"].eq("A") & result["downstream_antibiotic"].eq("B")].iloc[0]

    assert row["directional_support_ratio"] == 1.0
    assert row["directional_episode_overlap"] == 0.0


def test_mutual_eligible_das_restricts_both_directions_to_common_opportunity_universe() -> None:
    """DAS recomputed within a common, mutually-eligible opportunity universe

    must differ from the primary DAS when one direction's upstream drug is
    frequently ineligible in episodes where the reverse direction's upstream
    drug is eligible -- exactly the support-population asymmetry concern DAS
    alone cannot detect, per the reviewer's corrected argument: mutual
    eligibility does not require mutual observation, so this is a well-posed
    restriction distinct from (and much less restrictive than) requiring both
    drugs to be tested.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadePairDependenceAnalyzer(settings)

    def _row(anon_id, upstream, downstream, tested, upstream_eligible):
        return {
            "anon_id": anon_id,
            "pat_enc_csn_id_coded": f"e_{anon_id}",
            "order_proc_id_coded": f"o_{anon_id}",
            "order_time_jittered": "2024-01-01T00:00:00Z",
            "organism": "ESCHERICHIA COLI",
            "source_site": "armd",
            "upstream_antibiotic": upstream,
            "downstream_antibiotic": downstream,
            "downstream_tested": tested,
            "downstream_eligible": 1,
            "upstream_eligible": upstream_eligible,
        }

    drug_pairs = pd.DataFrame(
        [
            # A->B: 4 rows, only 2 have A (the upstream drug) itself eligible.
            _row("p1", "A", "B", 1, upstream_eligible=1),
            _row("p2", "A", "B", 1, upstream_eligible=1),
            _row("p3", "A", "B", 0, upstream_eligible=0),
            _row("p4", "A", "B", 0, upstream_eligible=0),
            # B->A: 4 rows, all with B (the upstream drug) eligible.
            _row("p5", "B", "A", 1, upstream_eligible=1),
            _row("p6", "B", "A", 1, upstream_eligible=1),
            _row("p7", "B", "A", 0, upstream_eligible=1),
            _row("p8", "B", "A", 0, upstream_eligible=1),
        ]
    )
    retained_edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "A", "downstream_antibiotic": "B"},
            {"upstream_antibiotic": "B", "downstream_antibiotic": "A"},
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)
    row = result.loc[result["upstream_antibiotic"].eq("A") & result["downstream_antibiotic"].eq("B")].iloc[0]

    assert "directional_asymmetry_score_mutual_eligible" in result.columns
    assert row["mutual_eligible_pair_support_n"] == 2.0
    assert row["mutual_eligible_reverse_support_n"] == 4.0
    assert pd.notna(row["directional_asymmetry_score_mutual_eligible"])


def test_mutual_eligible_diagnostics_are_nan_without_upstream_eligible_column() -> None:
    """Backward compatibility: drug_pairs materialised before PairGenerator

    started carrying upstream_eligible must not crash this analyzer -- the
    mutual-eligibility diagnostics simply become unavailable (NaN), same as
    every other "column not present yet" degradation pattern in this codebase.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadePairDependenceAnalyzer(settings)

    drug_pairs = pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2024-01-01T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "downstream_tested": 1,
                "downstream_eligible": 1,
                # no upstream_eligible column at all
            }
        ]
    )
    retained_edges = pd.DataFrame([{"upstream_antibiotic": "A", "downstream_antibiotic": "B"}])

    result = analyzer.analyze(drug_pairs, retained_edges)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["directional_asymmetry_score_mutual_eligible"])
