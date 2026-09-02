from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.guideline_concordance_analyzer import GuidelineConcordanceAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager


def test_guideline_concordance_analyzer_reports_structural_and_documented_matches() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = GuidelineConcordanceAnalyzer(settings, PathManager(project_root, settings))

    drug_pairs = pd.DataFrame(
        [
            {
                "organism": "ACINETOBACTER",
                "source_site": "armd",
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "CEFOTAXIME",
                "downstream_eligible": 1,
                "downstream_intrinsic_resistance": 0,
                "downstream_tested": 1,
            },
            {
                "organism": "ACINETOBACTER",
                "source_site": "armd",
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "CEFOTAXIME",
                "downstream_eligible": 1,
                "downstream_intrinsic_resistance": 0,
                "downstream_tested": 0,
            },
        ]
    )
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "CEFOTAXIME",
                "escalation_ratio": 2.0,
            }
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["structural_concordance"] == "aligned_with_eligibility_logic"
    assert bool(row["documented_rule_match"]) is True
    assert row["documented_rule_concordance"] == "matches_documented_rule"
    assert int(row["matching_rule_n"]) >= 1


def test_guideline_concordance_analyzer_collapses_all_organism_runs_to_edge_level() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = GuidelineConcordanceAnalyzer(settings, PathManager(project_root, settings))

    drug_pairs = pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "FOSFOMYCIN",
                "downstream_eligible": 1,
                "downstream_intrinsic_resistance": 0,
                "downstream_tested": 1,
            },
            {
                "organism": "KLEBSIELLA PNEUMONIAE",
                "source_site": "armd",
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "FOSFOMYCIN",
                "downstream_eligible": 1,
                "downstream_intrinsic_resistance": 0,
                "downstream_tested": 0,
            },
        ]
    )
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "FOSFOMYCIN",
                "escalation_ratio": 4.0,
            }
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["organism"] == "ALL_ORGANISMS"
    assert row["documented_rule_concordance"] == "not_evaluated_for_all_organism_aggregate"
