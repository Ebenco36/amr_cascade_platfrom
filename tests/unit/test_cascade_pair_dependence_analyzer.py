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
