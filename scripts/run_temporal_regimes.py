#!/usr/bin/env python
"""Year-by-year course of the validated directional observation patterns within each site (post hoc, descriptive).

Writes, under the organism's tables directory, temporal_regimes/temporal_regime_years.csv (pattern x site x year),
temporal_regimes/temporal_regime_summary.csv (pattern x site) and temporal_regimes/temporal_regime_numbers.json, and
figure_temporal_regimes.{pdf,png,svg} in the organism's figures directory (the five patterns the paper names).
Run it after the report step: the report moves every figure already in that folder aside before drawing.

    .venv/bin/python scripts/run_temporal_regimes.py --env hpc --organism 'ESCHERICHIA COLI'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from amr_cascade_platform.cli.main import build_context  # noqa: E402
from amr_cascade_platform.core.utils.scopes import scoped_output_dir  # noqa: E402
from amr_cascade_platform.core.utils.site_labels import site_label  # noqa: E402
from amr_cascade_platform.reporting.builders import temporal_regimes as tr  # noqa: E402
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver  # noqa: E402
from amr_cascade_platform.visualization.report.figure_presentation import save_matplotlib  # noqa: E402

FIGURE_PATTERNS = tr.EXAMPLE_PATTERNS
FIGURE_STEM = "figure_temporal_regimes"
RESISTANT, SUSCEPTIBLE, GRID = "#2a78d6", "#eb6834", "#e6e9ee"  # validated categorical slots 1 and 2


def _display(resolver: AntibioticClassificationResolver, label: str) -> str:
    text = str(resolver.canonical_display_label(label)).strip().lower().replace("/", "–")
    return text[:1].upper() + text[1:]


def draw(years: pd.DataFrame, sites: list[str], resolver: AntibioticClassificationResolver, stem: Path) -> list[Path]:
    present = set(zip(years["upstream_antibiotic"], years["downstream_antibiotic"]))
    missing = [pair for pair in FIGURE_PATTERNS if pair not in present]
    if missing:
        raise SystemExit(f"Figure patterns not among the validated patterns: {missing}")
    evaluable = years.loc[years["evaluable"]]
    span = range(int(evaluable["year"].min()), int(evaluable["year"].max()) + 1)
    fig, axes = plt.subplots(len(FIGURE_PATTERNS), len(sites), figsize=(7.2, 1.55 * len(FIGURE_PATTERNS) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for row, (up, down) in enumerate(FIGURE_PATTERNS):
        for col, site in enumerate(sites):
            ax = axes[row][col]
            cells = evaluable.loc[evaluable["upstream_antibiotic"].eq(up) & evaluable["downstream_antibiotic"].eq(down)
                                  & evaluable["source_site"].eq(site)].set_index("year").reindex(span)
            ax.plot(cells.index, cells["pR"], color=RESISTANT, lw=1.6, marker="o", ms=3.2)
            ax.plot(cells.index, cells["pS"], color=SUSCEPTIBLE, lw=1.6, ls="--", marker="o", ms=3.2, mfc="white")
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0%", "50%", "100%"], fontsize=7.5)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.tick_params(axis="x", labelsize=7.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.text(0.5, 1.30, site_label(site), transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
            if col == 0:
                ax.text(0.0, 1.06, f"{_display(resolver, up)} → {_display(resolver, down)}", transform=ax.transAxes,
                        ha="left", fontsize=8.5)
            if cells["pR"].isna().all():
                ax.text(0.5, 0.5, "no evaluable year", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#6b7280")
    for ax in axes[-1]:
        ax.set_xticks([y for y in span if y % 5 == 0])
    fig.supylabel("Share of episodes with the downstream antibiotic observed", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=RESISTANT, lw=1.6, marker="o", ms=3.2, label="Upstream result resistant"),
                        Line2D([], [], color=SUSCEPTIBLE, lw=1.6, ls="--", marker="o", ms=3.2, mfc="white", label="Upstream result susceptible")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96), h_pad=2.0)
    paths = []
    for fmt in ("pdf", "png", "svg"):
        path = stem.with_suffix(f".{fmt}")
        save_matplotlib(fig, path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="hpc")
    parser.add_argument("--organism", default="ESCHERICHIA COLI")
    parser.add_argument("--pairs", type=Path, help="drug_pair_episodes.parquet (default: the combined gold layer)")
    parser.add_argument("--validation", type=Path, help="validation_results.parquet (default: the combined cascade artifacts)")
    parser.add_argument("--tables-dir", type=Path, help="organism tables directory (default: from the settings)")
    parser.add_argument("--figures-dir", type=Path, help="organism figures directory (default: from the settings)")
    args = parser.parse_args()
    settings, paths, _ = build_context(ROOT, args.env)
    organism = args.organism
    pairs = args.pairs or scoped_output_dir(paths.paths.gold, "combined", organism=organism) / "drug_pair_episodes.parquet"
    validation = args.validation or scoped_output_dir(paths.paths.artifacts / settings.cascade.outputs.result_dir, "combined", organism=organism) / "validation_results.parquet"
    tables = args.tables_dir or scoped_output_dir(ROOT / settings.reporting.tables_dir, "combined", organism=organism)
    figures = args.figures_dir or scoped_output_dir(ROOT / settings.reporting.figures_dir, "combined", organism=organism)
    out = tables / "temporal_regimes"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    patterns = tr.validated_patterns(pd.read_parquet(validation))
    counts = tr.yearly_counts(pairs, patterns)
    years = tr.classify_years(counts, patterns, float(settings.cascade.continuity_correction))
    summary = tr.summarise(years)
    numbers = tr.headline_numbers(summary)
    years.to_csv(out / "temporal_regime_years.csv", index=False)
    summary.to_csv(out / "temporal_regime_summary.csv", index=False)
    (out / "temporal_regime_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    resolver = AntibioticClassificationResolver(ROOT / "data" / "antibiotic_classification_complete.csv")
    written = draw(years, [site for site in settings.platform.sites], resolver, figures / FIGURE_STEM)
    print(f"{len(patterns)} validated patterns, {len(years)} pattern-site-years ({int(years['evaluable'].sum())} evaluable)")
    print(json.dumps(numbers, indent=2))
    print("wrote", out, "and", ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
