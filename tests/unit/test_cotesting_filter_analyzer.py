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
