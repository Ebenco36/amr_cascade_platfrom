from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.guideline_concordance_analyzer import GuidelineConcordanceAnalyzer
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager


def test_guideline_concordance_analyzer_reports_structural_and_documented_matches(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    # The analyzer reads each site's laboratory rules file from the data folder,
    # which is not in git; give it its own, so the test runs on a clean checkout.
    rules_dir = tmp_path / settings.environment.data_root / "raw" / "armd"
    rules_dir.mkdir(parents=True)
    pd.DataFrame(
        {"Organism": ["ACINETOBACTER"], "Antibiotic": ["CEFOTAXIME"], "Rule": ["SAME AS CEFTRIAXONE IF REPORTED"]}
    ).to_csv(rules_dir / "implied_susceptibility_rules.csv", index=False)
    analyzer = GuidelineConcordanceAnalyzer(settings, PathManager(tmp_path, settings))

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
