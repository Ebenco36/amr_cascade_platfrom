#!/usr/bin/env python
"""Read-only check that saved validation can be resumed under this checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amr_cascade_platform.cli.main import build_context
from amr_cascade_platform.core.utils.provenance import read_json, sidecar_path, validation_fingerprint
from amr_cascade_platform.core.utils.scopes import scoped_output_dir


_COMPONENT_KEYS = ("git", "config_sha256", "inputs", "parameters")
_MISSING = object()


def _component_differences(record: dict, expected: dict) -> list[tuple[str, object, object]]:
    """Explain the fingerprint comparison, without changing its acceptance rule."""
    differences = []

    def compare(key, recorded_value, expected_value):
        if isinstance(recorded_value, dict) and isinstance(expected_value, dict):
            for child in sorted(recorded_value.keys() | expected_value.keys()):
                compare(
                    f"{key}.{child}",
                    recorded_value.get(child, _MISSING),
                    expected_value.get(child, _MISSING),
                )
        elif recorded_value != expected_value:
            differences.append((key, recorded_value, expected_value))

    for key in _COMPONENT_KEYS:
        compare(key, record.get(key, _MISSING), expected.get(key, _MISSING))
    return differences


def _display(value: object) -> str:
    return "<not recorded>" if value is _MISSING else json.dumps(value, sort_keys=True)


def check(root: Path, settings, paths, sites, organisms, *, include_sites: bool, explain: bool = False) -> list[str]:
    errors = []
    scopes = [("combined", None)] + ([("site", site) for site in sites] if include_sites else [])
    for organism in organisms:
        for scope, site in scopes:
            directory = scoped_output_dir(paths.paths.artifacts / settings.cascade.outputs.result_dir, scope, site=site, organism=organism)
            saved = sorted((directory / "validation_shards").glob("shard_*.parquet"))
            merged = directory / "validation_results.parquet"
            if merged.exists():
                saved.append(merged)
            if not saved:
                continue
            pairs = scoped_output_dir(paths.paths.gold, scope, site=site, organism=organism) / "drug_pair_episodes.parquet"
            expected, components = validation_fingerprint(root, settings, pairs)
            mismatched = []
            for path in saved:
                record = read_json(sidecar_path(path))
                record = record if isinstance(record, dict) else {}
                if record.get("fingerprint") != expected:
                    mismatched.append((path, record))
            if mismatched:
                first_path, record = mismatched[0]
                git_record = record.get("git") or {}
                recorded_commit = git_record.get("commit", "unrecorded") if isinstance(git_record, dict) else "unrecorded"
                message = (
                    f"{directory}: {len(mismatched)}/{len(saved)} validation artifacts have a provenance mismatch; "
                    f"recorded commit={recorded_commit}; first={first_path.name}"
                )
                if explain:
                    # Group identical diagnostics, but do not hide a shard with a
                    # different input/config mismatch behind the first shard.
                    reasons = {}
                    for path, saved_record in mismatched:
                        details = tuple(
                            f"    {key}: recorded={_display(old)}; current={_display(new)}"
                            for key, old, new in _component_differences(saved_record, components)
                        )
                        if not details:
                            details = (
                                "    Fingerprint differs although the recorded components match; "
                                "inspect the sidecar format or integrity.",
                            )
                        reasons.setdefault(details, []).append(path.name)
                    for details, names in reasons.items():
                        message += f"\n  {len(names)} artifact(s), example={names[0]}:\n" + "\n".join(details)
                errors.append(message)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hpc")
    parser.add_argument("--sites", default="armd,armd_ecuh,armd_utsw")
    parser.add_argument("--organisms", default="ESCHERICHIA COLI")
    parser.add_argument("--include-sites", action="store_true")
    parser.add_argument("--explain", action="store_true", help="Show recorded/current fingerprint differences; read-only, no jobs submitted.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings, paths, _ = build_context(root, args.env)
    errors = check(root, settings, paths, args.sites.split(","), args.organisms.split(","), include_sites=args.include_sites, explain=args.explain)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print("STOP: no files were changed. A provenance mismatch does not by itself prove the statistical results invalid. "
              "Use --explain to distinguish code, configuration and input identity changes. "
              "Code identity covers the validation code and configs/ (provenance._VALIDATION_CODE_PATHS), "
              "not reporting, figures, modelling or the submit scripts. "
              "A code mismatch means one of those files differs from the recorded commit; otherwise plan an explicit rebuild. "
              "Do not edit provenance sidecars to force reuse.", file=sys.stderr)
        return 1
    print(f"Validation reuse check PASS ({args.env}); absent results still need computing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
