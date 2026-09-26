from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.cotesting_filter_analyzer import CoTestingFilterAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_cotesting_filter_removes_symmetric_near_deterministic_pairs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CoTestingFilterAnalyzer(settings)

    rows = []
    for idx in range(30):
        rows.append(
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "downstream_tested": 1,
                "downstream_eligible": 1,
                "source_site": "armd",
                "order_proc_id_coded": f"o{idx}",
            }
        )
        rows.append(
            {
                "upstream_antibiotic": "B",
                "downstream_antibiotic": "A",
                "downstream_tested": 1,
                "downstream_eligible": 1,
                "source_site": "armd",
                "order_proc_id_coded": f"o{idx}",
            }
        )
    rows.append(
        {
            "upstream_antibiotic": "A",
            "downstream_antibiotic": "C",
            "downstream_tested": 0,
            "downstream_eligible": 1,
            "source_site": "armd",
            "order_proc_id_coded": "spare",
        }
    )
    drug_pairs = pd.DataFrame(rows)

    filtered, flagged = analyzer.filter(drug_pairs)

    assert not flagged.empty
    assert {("A", "B"), ("B", "A")} <= set(zip(flagged["upstream_antibiotic"], flagged["downstream_antibiotic"]))
    assert ("A", "B") not in set(zip(filtered["upstream_antibiotic"], filtered["downstream_antibiotic"]))
    assert ("B", "A") not in set(zip(filtered["upstream_antibiotic"], filtered["downstream_antibiotic"]))
    assert ("A", "C") in set(zip(filtered["upstream_antibiotic"], filtered["downstream_antibiotic"]))


def test_filter_with_probabilities_reports_every_assessed_pair() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CoTestingFilterAnalyzer(settings)
    rows = []
    for idx in range(30):
        for up, down, tested in (("A", "B", 1), ("B", "A", 1), ("A", "C", int(idx < 10)), ("C", "A", 1)):
            rows.append(
                {
                    "upstream_antibiotic": up,
                    "downstream_antibiotic": down,
                    "downstream_tested": tested,
                    "downstream_eligible": 1,
                    "order_proc_id_coded": f"o{idx}",
                }
            )
    rows.append({"upstream_antibiotic": "A", "downstream_antibiotic": "D", "downstream_tested": 1, "downstream_eligible": 1, "order_proc_id_coded": "o0"})
    pairs = pd.DataFrame(rows)

    filtered, flagged, assessed = analyzer.filter_with_probabilities(pairs)
    filtered_only, flagged_only = analyzer.filter(pairs)

    assert filtered.equals(filtered_only) and flagged.equals(flagged_only)
    probabilities = assessed.set_index(["upstream_antibiotic", "downstream_antibiotic"])
    # A->D has no reverse direction: it is recorded, with empty reverse columns, and cannot be flagged.
    assert set(probabilities.index) == {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("A", "D")}
    assert pd.isna(probabilities.loc[("A", "D"), "p_upstream_given_downstream"]) and not probabilities.loc[("A", "D"), "flagged"]
    assert probabilities.loc[("A", "B"), "flagged"] and probabilities.loc[("B", "A"), "flagged"]
    assert not probabilities.loc[("A", "C"), "flagged"]
    assert probabilities.loc[("A", "C"), "p_downstream_given_upstream"] == 10 / 30
    assert probabilities.loc[("A", "C"), "p_upstream_given_downstream"] == 1.0
