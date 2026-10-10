#!/usr/bin/env python
"""Read-only check that saved validation can be resumed under this checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_cascade_platform.cli.main import build_context
from amr_cascade_platform.core.utils.provenance import read_json, sidecar_path, validation_fingerprint
from amr_cascade_platform.core.utils.scopes import scoped_output_dir


def check(root: Path, settings, paths, sites, organisms, *, include_sites: bool) -> list[str]:
    errors = []
    scopes = [("combined", None)] + ([("site", site) for site in sites] if include_sites else [])
    for organism in organisms:
        for scope, site in scopes:
            directory = scoped_output_dir(paths.paths.artifacts / settings.cascade.outputs.result_dir, scope, site=site, organism=organism)
            saved = list((directory / "validation_shards").glob("shard_*.parquet"))
            merged = directory / "validation_results.parquet"
            if merged.exists():
                saved.append(merged)
            if not saved:
                continue
            pairs = scoped_output_dir(paths.paths.gold, scope, site=site, organism=organism) / "drug_pair_episodes.parquet"
            expected, _ = validation_fingerprint(root, settings, pairs)
            mismatched = [path for path in saved if (read_json(sidecar_path(path)) or {}).get("fingerprint") != expected]
            if mismatched:
                record = read_json(sidecar_path(mismatched[0])) or {}
                errors.append(f"{directory}: {len(mismatched)}/{len(saved)} validation artifacts are incompatible; "
                              f"recorded commit={record.get('git', {}).get('commit', 'unrecorded')}; "
                              f"first={mismatched[0].name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hpc")
    parser.add_argument("--sites", default="armd,armd_ecuh,armd_utsw")
    parser.add_argument("--organisms", default="ESCHERICHIA COLI")
    parser.add_argument("--include-sites", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings, paths, _ = build_context(root, args.env)
    errors = check(root, settings, paths, args.sites.split(","), args.organisms.split(","), include_sites=args.include_sites)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print("STOP: no files were changed. Finish validation in the recorded frozen checkout, "
              "or plan an explicit rebuild. Do not edit provenance sidecars to force reuse.", file=sys.stderr)
        return 1
    print(f"Validation reuse check PASS ({args.env}); absent results still need computing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
