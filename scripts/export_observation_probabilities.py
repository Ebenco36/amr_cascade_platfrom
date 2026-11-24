#!/usr/bin/env python
"""Render paired observation plots from a saved edge report; no models are fitted."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from amr_cascade_platform.reporting.builders import directional_views as dv  # noqa: E402
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver  # noqa: E402
from amr_cascade_platform.visualization.report.paired_observation_plotter import PairedObservationPlotter  # noqa: E402
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--tier", choices=("validated", "robust"), default="validated")
    parser.add_argument("--hide-figure-titles", action="store_true")
    parser.add_argument("--figure-format", action="append", choices=("html", "pdf", "png", "svg"))
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    source = args.edge_report.resolve(strict=True)
    classification_path = ROOT / "data" / "antibiotic_classification_complete.csv"
    classification = AntibioticClassificationResolver(classification_path)
    edges = pd.read_parquet(source)
    canonical = dv.canonicalize_edges(edges, classification.canonical_display_label)
    selected = dv.tier_edges(dv.validated_edges(canonical), args.tier)
    exporter = PlotlyFigureExporter(1600, 900, allow_generic_fallback=False, hide_titles=args.hide_figure_titles)
    plotter = PairedObservationPlotter(exporter)
    outputs = {}
    for direction in dv.DIRECTIONS:
        outputs.update(plotter.export(
            selected, direction, args.output_dir / f"figure_{direction}_observation_probabilities_{args.tier}",
            tuple(args.figure_format or ("html", "pdf", "png", "svg")), top_n=args.top_n,
        ))
    manifest = {
        "edge_report": str(source), "edge_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "classification_file": str(classification_path),
        "classification_sha256": hashlib.sha256(classification_path.read_bytes()).hexdigest(),
        "tier": args.tier, "top_n_per_direction": args.top_n,
        "ranking": "absolute raw R-minus-S observation difference; alphabetical tie-break",
        "er_magnitude_filter": None, "input_rows": len(edges), "validated_rows": len(selected),
        "canonical_duplicate_rows_collapsed": canonical.attrs.get("n_collapsed_duplicates", 0),
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    path = args.output_dir / "observation_probabilities_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Read {source}\nValidated pairs: {len(selected)}\nWrote {len(outputs)} files and {path}")


if __name__ == "__main__":
    main()
