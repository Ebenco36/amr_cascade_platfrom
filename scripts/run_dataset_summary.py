#!/usr/bin/env python
"""Describe the analysis dataset right after the gold build (tables, figures, digest).

Needs only the gold layer, so the cluster DAG runs it as soon as the combined
gold build finishes, before the cascade and validation jobs. Outputs:
outputs/reports/<scope>/dataset_summary/ (tables, dataset_summary.md) and
outputs/figures/<scope>/dataset_summary/ (figures).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    _bootstrap()
    from amr_cascade_platform.cli.main import build_context
    from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
    from amr_cascade_platform.reporting.workflows.dataset_summary_workflow import DatasetSummaryWorkflow

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default=None)
    parser.add_argument("--scope", choices=("combined", "site"), default="combined")
    parser.add_argument("--site", default=None)
    parser.add_argument("--organism", default=None)
    parser.add_argument("--hide-figure-titles", action="store_true", help="Omit global titles/subtitles while retaining axis and panel labels.")
    parser.add_argument("--figure-format", action="append", choices=("html", "png", "svg", "pdf"), dest="formats")
    args = parser.parse_args()

    LoggerFactory.configure()
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).resolve().parents[1]
    settings, path_manager, _ = build_context(project_root, args.env)
    formats = tuple(args.formats or ("png", "pdf", "html"))
    outputs = DatasetSummaryWorkflow(settings, path_manager).run(args.scope, args.site, args.organism, formats, hide_figure_titles=args.hide_figure_titles)
    digest = next((path for name, path in outputs.items() if name == "dataset_summary.md"), None)
    logger.info("Dataset summary written: %d files; digest %s", len(outputs), digest)
    if digest is not None:
        print(Path(digest).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
