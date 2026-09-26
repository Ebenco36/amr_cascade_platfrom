#!/usr/bin/env python
"""Write every generated supplementary table and the numbers the manuscripts quote.

Reads the combined-scope cascade, gold, and report outputs for one organism (plus the
per-site edge reports) and writes CSV tables and supplementary_numbers.json to
outputs/tables/combined/organisms/<organism>/supplementary/. Exits non-zero when an
input is missing unless --allow-missing is given, so a partial table set cannot pass
for a complete one.

Usage:
  python scripts/build_supplementary_tables.py --env hpc --organism "ESCHERICHIA COLI"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    _bootstrap()
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.reporting.builders.supplementary_table_builder import SupplementaryTableBuilder

    parser = argparse.ArgumentParser(description="Build the generated supplementary tables.")
    parser.add_argument("--env", default=None, help="Runtime environment (e.g. mac, hpc).")
    parser.add_argument("--organism", default="ESCHERICHIA COLI")
    parser.add_argument("--allow-missing", action="store_true", help="Write what can be built and exit 0 even if inputs are missing.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)
    result = SupplementaryTableBuilder(settings, path_manager).build(args.organism)

    for name in sorted(result.written):
        print(f"wrote {result.written[name]}")
    if result.missing_inputs:
        print("\nMissing inputs (their tables were not written):", file=sys.stderr)
        for path in result.missing_inputs:
            print(f"  {path}", file=sys.stderr)
        if not args.allow_missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
