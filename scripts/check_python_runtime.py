#!/usr/bin/env python3
"""Validate the Python runtime used by SLURM jobs.

This script intentionally imports the packages that every pipeline stage needs.
On failure it prints enough diagnostics to distinguish a missing package from a
wrong virtualenv, wrong project root, or broken PYTHONPATH on a compute node.
"""

from __future__ import annotations

import importlib
import os
import site
import sys
import sysconfig


REQUIRED_MODULES = (
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "amr_cascade_platform",
)


def main() -> int:
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Python version: {sys.version.split()[0]}", file=sys.stderr)
    print(f"sys.prefix: {sys.prefix}", file=sys.stderr)
    print(f"sys.base_prefix: {sys.base_prefix}", file=sys.stderr)
    print(f"purelib: {sysconfig.get_path('purelib')}", file=sys.stderr)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}", file=sys.stderr)
    print(f"PYTHONNOUSERSITE: {os.environ.get('PYTHONNOUSERSITE', '')}", file=sys.stderr)
    print(f"site.ENABLE_USER_SITE: {site.ENABLE_USER_SITE}", file=sys.stderr)

    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - diagnostics should catch all import failures.
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "version unavailable")
        location = getattr(module, "__file__", "location unavailable")
        print(f"OK {module_name} {version} ({location})", file=sys.stderr)

    if missing:
        print("Runtime import check failed:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
