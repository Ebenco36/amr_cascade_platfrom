"""
Legacy diagnostic for the archived empirical-delta projection approach.

The active prevalence workflow uses eligible-denominator bounds plus
cascade-aware MNAR sensitivity curves. This script is retained only for audit
of the older empirical-delta projection diagnostic.

Usage
-----
# Sample data (placeholder until HPC completes):
    python scripts/legacy/generate_projection_audit_figure.py --sample

# Legacy production audit:
    python scripts/legacy/generate_projection_audit_figure.py \
        --parquet outputs/reports/combined/organisms/escherichia_coli/prevalence_shift_pair_results.parquet \
        --organism "Escherichia coli"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_DRUGS = [
    "AMOXICILLIN/CLAVULANIC ACID", "AMPICILLIN/SULBACTAM", "CEFAZOLIN",
    "CEFEPIME", "CEFTAZIDIME", "CEFTRIAXONE", "CIPROFLOXACIN",
    "ERTAPENEM", "GENTAMICIN", "IMIPENEM", "LEVOFLOXACIN",
    "MEROPENEM", "MOXIFLOXACIN", "NITROFURANTOIN", "PIPERACILLIN/TAZOBACTAM",
    "TOBRAMYCIN", "TRIMETHOPRIM/SULFAMETHOXAZOLE", "AZTREONAM",
    "CEFOXITIN", "CEPHALEXIN",
]


def make_sample_data(rng: np.random.Generator) -> pd.DataFrame:
    """Generate realistic-looking placeholder data for the projection audit."""
    n = len(SAMPLE_DRUGS)
    naive_prevalence = rng.uniform(0.06, 0.58, size=n).round(3)
    delta_max = (1.0 / naive_prevalence).round(4)

    # Most empirical anchors fall well within the admissible interval;
    # a realistic minority exceed it (simulates δ_raw > δ_max).
    delta_raw = rng.exponential(scale=1.2, size=n).round(4)
    # Force ~4 pairs to be projected (raw > max)
    projected_idx = rng.choice(n, size=4, replace=False)
    for idx in projected_idx:
        delta_raw[idx] = delta_max[idx] * rng.uniform(1.05, 1.45)

    delta_used = np.minimum(delta_raw, delta_max).round(4)
    was_projected = ~np.isclose(delta_raw, delta_used, rtol=0, atol=1e-9)

    return pd.DataFrame({
        "drug": SAMPLE_DRUGS,
        "naive_prevalence": naive_prevalence,
        "delta_empirical_raw": delta_raw,
        "delta_max": delta_max,
        "delta_empirical": delta_used,
        "delta_empirical_was_projected": was_projected,
    })


# ── Scatter plot ──────────────────────────────────────────────────────────────
def plot_projection_scatter(df: pd.DataFrame, organism: str, out_path: Path, sample: bool) -> None:
    evaluable = df[df["delta_empirical_raw"].notna()].copy()
    proj = evaluable[evaluable["delta_empirical_was_projected"]]
    ok   = evaluable[~evaluable["delta_empirical_was_projected"]]

    n_proj  = len(proj)
    n_total = len(evaluable)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Empirical Anchor Projection Audit — {organism}"
        + ("\n[SAMPLE DATA — replace with production artefacts before submission]" if sample else ""),
        fontsize=11, fontweight="bold", y=1.01,
    )

    # ── Left: scatter (raw δ vs admissible upper) ──────────────────────────
    ax = axes[0]
    ax.scatter(ok["delta_empirical_raw"],   ok["delta_max"],
               color="#3A86FF", alpha=0.75, s=55, label="Not projected", zorder=3)
    ax.scatter(proj["delta_empirical_raw"], proj["delta_max"],
               color="#FF5A5F", marker="D", s=70, alpha=0.95,
               label="Projected to boundary", zorder=4)

    lim = max(evaluable["delta_empirical_raw"].max(), evaluable["delta_max"].max()) * 1.08
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.1, label="$y = x$ boundary", zorder=2)

    # Annotation arrows for projected points
    for _, row in proj.iterrows():
        ax.annotate(
            "",
            xy=(row["delta_max"], row["delta_max"]),
            xytext=(row["delta_empirical_raw"], row["delta_max"]),
            arrowprops=dict(arrowstyle="->", color="#FF5A5F", lw=1.2),
            zorder=5,
        )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Raw empirical $\delta_a$ (data-derived anchor)", fontsize=10)
    ax.set_ylabel(r"Admissible upper bound $(1/\pi^{naive}_a)$", fontsize=10)
    ax.set_title("Raw anchor vs admissible ceiling\n(arrows show clipping direction)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate(
        f"Projected: {n_proj} / {n_total} pairs ({100 * n_proj / max(n_total, 1):.0f}%)",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFFACD", ec="grey"),
    )
    ax.set_aspect("equal")

    # ── Right: strip / dot plot per drug ──────────────────────────────────
    ax2 = axes[1]
    sorted_df = evaluable.sort_values("delta_empirical_raw", ascending=True).reset_index(drop=True)
    y_pos = range(len(sorted_df))

    colors = ["#FF5A5F" if p else "#3A86FF" for p in sorted_df["delta_empirical_was_projected"]]

    ax2.barh(list(y_pos), sorted_df["delta_max"], height=0.55,
             color="#E8E8E8", label=r"Admissible interval $[0,\,1/\pi^{naive}]$", zorder=1)
    ax2.scatter(sorted_df["delta_empirical_raw"], list(y_pos),
                color=colors, s=55, zorder=3, label="Raw $\\delta^{emp}$ (projected=red ◆)",
                marker="D")
    ax2.scatter(sorted_df["delta_empirical"], list(y_pos),
                color="black", s=20, zorder=4, marker="|",
                label=r"$\delta$ used (after projection)")

    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(
        [d[:28] + "…" if len(d) > 28 else d for d in sorted_df["drug"]],
        fontsize=7,
    )
    ax2.set_xlabel(r"$\delta_a$ value", fontsize=10)
    ax2.set_title(
        "Per-drug anchor audit\n(grey bar = admissible range, tick = value used)",
        fontsize=10,
    )
    ax2.axvline(1.0, color="darkgreen", linestyle=":", linewidth=1.2,
                label=r"$\delta_a = 1$ (MAR, no correction)")
    ax2.legend(fontsize=8, loc="lower right")

    # Vertical reference
    ax2.annotate("MAR", xy=(1.0, len(sorted_df) - 0.5), fontsize=7,
                 color="darkgreen", ha="center")

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Table ─────────────────────────────────────────────────────────────────────
def print_latex_table(df: pd.DataFrame) -> None:
    """Print a LaTeX-ready tabular for the supplement."""
    evaluable = df[df["delta_empirical_raw"].notna()].copy()
    evaluable = evaluable.sort_values(
        ["delta_empirical_was_projected", "delta_empirical_raw"],
        ascending=[False, False],
    )
    print()
    print(r"\begin{tabular}{lrrrrl}")
    print(r"\toprule")
    print(r"\textbf{Drug} & $\boldsymbol{\pi^{naive}}$ & $\boldsymbol{\delta^{emp}_{\text{raw}}}$"
          r" & \textbf{Admissible upper} & $\boldsymbol{\delta^{emp}_{\text{used}}}$ & \textbf{Projected?} \\")
    print(r"\midrule")
    for _, row in evaluable.iterrows():
        flag = r"\textbf{Yes}" if row["delta_empirical_was_projected"] else "No"
        print(
            f"{row['drug']} & "
            f"{row['naive_prevalence']:.3f} & "
            f"{row['delta_empirical_raw']:.3f} & "
            f"{row['delta_max']:.3f} & "
            f"{row['delta_empirical']:.3f} & "
            f"{flag} \\\\"
        )
    n_proj  = evaluable["delta_empirical_was_projected"].sum()
    n_total = len(evaluable)
    print(r"\midrule")
    print(rf"\multicolumn{{6}}{{l}}{{\textit{{Projection occurred in {n_proj} of {n_total}"
          rf" pairs ({100*n_proj/max(n_total,1):.0f}\%).}}}} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate projection audit figure.")
    parser.add_argument("--sample", action="store_true",
                        help="Use synthetic sample data (default when no --parquet given).")
    parser.add_argument("--parquet", type=Path, default=None,
                        help="Path to prevalence_shift_pair_results.parquet from HPC run.")
    parser.add_argument("--organism", default="Escherichia coli",
                        help="Organism display name for figure title.")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/figures/combined/organisms/escherichia_coli/"
                                    "figure_supp_projection_audit.png"),
                        help="Output figure path.")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=42)

    if args.parquet and args.parquet.exists():
        df = pd.read_parquet(args.parquet)
        sample = False
        print(f"Using production data from: {args.parquet}")
    else:
        df = make_sample_data(rng)
        sample = True
        print("Using SAMPLE DATA — replace with production artefacts before submission.")

    plot_projection_scatter(df, args.organism, args.out, sample=sample)
    print_latex_table(df)


if __name__ == "__main__":
    main()
