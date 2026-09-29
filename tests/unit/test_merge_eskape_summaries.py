import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_merge_accepts_an_absolute_glob(tmp_path: Path) -> None:
    """The DAGs pass environment-resolved absolute paths (reporting.tables_dir)."""
    for target in ("klebsiella", "pseudomonas"):
        directory = tmp_path / "_per_target" / "combined" / target
        directory.mkdir(parents=True)
        pd.DataFrame([{"eskape_target": target, "dataset": "armd", "site": "combined"}]).to_csv(
            directory / "eskape_cascade_comparison.csv", index=False
        )
    output_csv = tmp_path / "combined_armd" / "eskape_cascade_comparison.csv"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "merge_eskape_summaries.py"),
            "--input-glob", str(tmp_path / "_per_target" / "combined" / "*" / "eskape_cascade_comparison.csv"),
            "--output-csv", str(output_csv),
            "--output-json", str(output_csv.with_suffix(".json")),
        ],
        check=True,
        cwd=tmp_path,
    )

    assert pd.read_csv(output_csv)["eskape_target"].tolist() == ["klebsiella", "pseudomonas"]
