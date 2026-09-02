#!/usr/bin/env python
"""Wrapper for manuscript report exports."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _inject_subcommand(argv: list[str], subcommand: str) -> list[str]:
    insert_at = 1
    if len(argv) > 2 and argv[1] == "--env":
        insert_at = 3
    return argv[:insert_at] + [subcommand] + argv[insert_at:]


if __name__ == "__main__":
    _bootstrap()
    from amr_cascade_platform.cli.main import main

    sys.argv = _inject_subcommand(sys.argv, "export-report")
    main()
