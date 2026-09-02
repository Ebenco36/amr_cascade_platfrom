#!/usr/bin/env python
"""Merge per-target ESKAPE cascade-validation summary CSVs into one combined table.

Each target's validation runs as an independent SLURM job (see
submit_pipeline_dag_hpc.sh), writing its own eskape_cascade_comparison.csv.
This script concatenates those per-target files back into the single combined
comparison table the original monolithic eskape workflow produced in one run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern (relative to the current directory) matching per-target eskape_cascade_comparison.csv files.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched glob: {args.input_glob}")

    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ("eskape_target", "dataset", "site") if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(by=sort_columns, na_position="last").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    Path(args.output_json).write_text(combined.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Merged {len(paths)} file(s) -> {output_csv} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
