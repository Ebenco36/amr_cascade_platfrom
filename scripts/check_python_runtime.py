#!/usr/bin/env python3
"""Validate the Python runtime used by SLURM jobs.

This script intentionally imports the packages that every pipeline stage needs.
On failure it prints enough diagnostics to distinguish a missing package from a
wrong virtualenv, wrong project root, or broken PYTHONPATH on a compute node.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import site
import sys
import sysconfig
import tempfile


# SHAP imports Matplotlib. Compute-node home directories may be read-only, so
# force its font/config cache into the job's writable temporary directory before
# importing any runtime module.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amr_cascade_matplotlib"))


REQUIRED_MODULES = (
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "shap",
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
    return _check_code_identity()


def _check_code_identity() -> int:
    """Stop a job whose checkout is not the code its submission recorded.

    HPC submission exports AMR_CASCADE_EXPECTED_CODE (provenance.code_identity at
    submit time). If src/, scripts/ or configs/ changed while the job waited in
    the queue, running would mix code versions within one run, so the job stops
    here instead. Unset (local runs): nothing to compare.
    """
    expected = os.environ.get("AMR_CASCADE_EXPECTED_CODE", "").strip()
    if not expected:
        return 0
    from amr_cascade_platform.core.utils.provenance import code_identity

    found = code_identity(Path(__file__).resolve().parents[1])
    if found == expected:
        print(f"OK code identity {found}", file=sys.stderr)
        return 0
    print(
        "Code identity check failed: this job was submitted for code "
        f"{expected} but the checkout is now {found}. Restore the submitted "
        "checkout, or resubmit from the current one.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
