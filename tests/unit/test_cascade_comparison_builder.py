from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.outputs.cascade_comparison_builder import CascadeComparisonBuilder


def test_cascade_comparison_builder_labels_presence(tmp_path: Path) -> None:
    artifact_root = tmp_path / "cascade"
    combined_dir = artifact_root / "combined"
    site_dir = artifact_root / "armd"
    combined_dir.mkdir(parents=True)
    site_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "C"],
            "downstream_antibiotic": ["B", "D"],
            "escalation_ratio": [2.0, 3.0],
            "adjusted_odds_ratio": [1.2, 1.4],
            "total_support_n": [30, 40],
        }
    ).to_parquet(combined_dir / "edge_report.parquet", index=False)
    pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "X"],
            "downstream_antibiotic": ["B", "Y"],
            "escalation_ratio": [2.5, 4.0],
            "adjusted_odds_ratio": [1.3, 1.8],
            "total_support_n": [31, 22],
        }
    ).to_parquet(site_dir / "edge_report.parquet", index=False)

    outputs = CascadeComparisonBuilder().export(artifact_root=artifact_root, scope_name="armd")
    comparison = pd.read_parquet(outputs["site_vs_combined_edge_comparison_path"])
    assert set(comparison["edge_presence"]) == {"shared", "site_only", "combined_only"}


def test_cascade_comparison_builder_supports_organism_scoped_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "cascade"
    organism_suffix = Path("organisms") / "escherichia_coli"
    combined_dir = artifact_root / "combined" / organism_suffix
    site_dir = artifact_root / "armd" / organism_suffix
    combined_dir.mkdir(parents=True)
    site_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
            "escalation_ratio": [2.0],
            "adjusted_odds_ratio": [1.2],
            "total_support_n": [30],
        }
    ).to_parquet(combined_dir / "edge_report.parquet", index=False)
    pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "X"],
            "downstream_antibiotic": ["B", "Y"],
            "escalation_ratio": [2.5, 4.0],
            "adjusted_odds_ratio": [1.3, 1.8],
            "total_support_n": [31, 22],
        }
    ).to_parquet(site_dir / "edge_report.parquet", index=False)

    outputs = CascadeComparisonBuilder().export(
        artifact_root=artifact_root,
        scope_name="combined/organisms/escherichia_coli",
    )

    summary_path = combined_dir / "site_vs_combined_summary.parquet"
    assert outputs["site_vs_combined_summary_path"] == summary_path
    assert summary_path.exists()
    comparison = pd.read_parquet(outputs["site_vs_combined_edge_comparison_path"])
    assert set(comparison["edge_presence"]) == {"shared", "site_only"}
