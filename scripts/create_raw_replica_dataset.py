#!/usr/bin/env python
"""Build a synthetic replica of data/raw for end-to-end pipeline tests.

Two steps, runnable separately or together:

  profile   Read data/raw once (streamed, column batches; the ~20 GB Stanford
            comorbidity extract only from its head) and write an aggregate
            profile: file layouts, value spellings, vocabularies, quantiles and
            fitted AST models. No record-level data is kept.
  generate  Write synthetic CSVs with the raw file names, column order and
            spellings into <output-root>/raw/<site>/, copy the reference files
            into <output-root>/reference/, and write a manifest with row counts,
            checksums and a fidelity comparison against the real data.

Examples:
  python scripts/create_raw_replica_dataset.py profile
  python scripts/create_raw_replica_dataset.py generate --scale 0.03
  python scripts/create_raw_replica_dataset.py all --scale 0.03

Then run the pipeline on it with --env test_replica.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_replica"
DEFAULT_PROFILE = DEFAULT_OUTPUT_ROOT / "profile" / "raw_profile.json"


def _profile(args: argparse.Namespace) -> Path:
    from amr_cascade_platform.data.replica.raw_profiler import ProfileOptions, RawProfiler, load_table_map

    options = ProfileOptions(
        min_cell=args.min_cell,
        chain_min_episodes=args.chain_min_episodes,
        sites=tuple(args.sites) if args.sites else None,
    )
    started = time.monotonic()
    table_map = load_table_map(args.reference_dir)
    profile = RawProfiler(args.raw_root, table_map, options).profile()
    profile["source"] = {"raw_root": str(args.raw_root), "reference_dir": str(args.reference_dir)}
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.profile.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile), encoding="utf-8")
    temporary.replace(args.profile)
    logging.info(
        "Profile written to %s (%.1f MB) in %.0fs",
        args.profile, args.profile.stat().st_size / 1e6, time.monotonic() - started,
    )
    return args.profile


def _generate(args: argparse.Namespace) -> None:
    from amr_cascade_platform.data.replica.replica_generator import GenerationOptions, ReplicaGenerator

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    options = GenerationOptions(
        scale=args.scale,
        seed=args.seed,
        comorbidity_rows_cap=args.comorbidity_rows_cap,
        sites=tuple(args.sites) if args.sites else None,
    )
    started = time.monotonic()
    manifest = ReplicaGenerator(profile, options).write(
        output_root=args.output_root,
        reference_source=args.reference_dir,
        force=args.force,
    )
    logging.info("Replica written to %s in %.0fs", args.output_root, time.monotonic() - started)
    for site, tables in manifest["row_counts"].items():
        logging.info("  %s: %s", site, ", ".join(f"{name}={count:,}" for name, count in tables.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("profile", "generate", "all"))
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--reference-dir", type=Path, default=PROJECT_ROOT / "data" / "reference")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--min-cell", type=int, default=11, help="Small-cell threshold for any category kept in the profile.")
    parser.add_argument(
        "--chain-min-episodes", type=int, default=1000,
        help="Organisms with at least this many episodes at a site get the full AST chain model.",
    )
    parser.add_argument("--scale", type=float, default=0.03, help="Fraction of each site's real culture orders to generate.")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--comorbidity-rows-cap", type=int, default=60,
        help="Maximum comorbidity rows per culture order (0 = no cap). The Stanford extract averages several hundred.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing replica under --output-root/raw.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command in {"profile", "all"}:
        _profile(args)
    if args.command in {"generate", "all"}:
        _generate(args)


if __name__ == "__main__":
    main()
