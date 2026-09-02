"""Sensitivity curve figures for prevalence-shift analysis."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from amr_cascade_platform.visualization.report.organism_labels import format_organism_label


class PrevalenceDeltaCurvePlotter:
    """Export multi-panel prevalence sensitivity curves."""

    _DPI = 300
    _FONT = "DejaVu Sans"
    _COLS = 4
    # (tier key, panel label, lower bound inclusive, upper bound exclusive) in
    # percentage points of prevalence swing across the full lambda grid. Drugs whose
    # MNAR estimate barely moves land in "stable" regardless of their absolute
    # prevalence level; drugs where the lambda assumption changes the clinical
    # picture land in "sensitive". Boundaries are a round-number tertile split of
    # the observed swing distribution, not a discovered natural break.
    _SENSITIVITY_TIERS = [
        ("stable", "Stable (Δ<2pp)", 0.0, 2.0),
        ("moderate", "Moderate (Δ 2–10pp)", 2.0, 10.0),
        ("sensitive", "Sensitive (Δ≥10pp)", 10.0, math.inf),
    ]
    # Traffic-light semantics: green/amber/red map directly onto the tier's own
    # meaning (safe to trust the naive estimate -> worth a second look), so the
    # color is informative rather than an arbitrary per-series label.
    _TIER_COLORS = {
        "stable": "#2E7D32",
        "moderate": "#B8860B",
        "sensitive": "#C0392B",
        "not_evaluable": "#667085",
    }

    def export(
        self,
        curves: pd.DataFrame,
        prevalence_summary: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        organism: str = "",
    ) -> dict[str, Path]:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        static_formats = [fmt for fmt in formats if fmt in {"png", "svg", "pdf", "tiff"}]
        if not static_formats:
            return outputs

        if not curves.empty and {"mnar_lambda", "mnar_prevalence_pct"}.issubset(curves.columns):
            tier_figures = self._build_mnar_tier_figures(curves, prevalence_summary, organism)
        else:
            tier_figures = [(None, self._build_legacy_or_empty_figure(curves, prevalence_summary, organism))]

        for suffix, fig in tier_figures:
            stem = output_stem if suffix is None else output_stem.with_name(f"{output_stem.name}_{suffix}")
            for fmt in static_formats:
                path = stem.with_suffix(f".{fmt}")
                dpi = self._DPI if fmt in {"png", "tiff"} else 220
                fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
                outputs[path.name] = path
            plt.close(fig)
        return outputs

    def _build_legacy_or_empty_figure(self, curves: pd.DataFrame, prevalence_summary: pd.DataFrame, organism: str = "") -> plt.Figure:
        if curves.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "No prevalence sensitivity curve data available.",
                    ha="center", va="center", fontsize=13, fontfamily=self._FONT)
            return fig
        return self._build_legacy_delta_figure(curves, prevalence_summary, organism)

    def _build_legacy_delta_figure(self, curves: pd.DataFrame, prevalence_summary: pd.DataFrame, organism: str = "") -> plt.Figure:
        # Build per-drug empirical anchor lookup from summary
        emp_lookup: dict[str, float | None] = {}
        naive_lookup: dict[str, float] = {}
        if not prevalence_summary.empty:
            for _, row in prevalence_summary.iterrows():
                drug = str(row.get("drug", ""))
                emp = row.get("legacy_delta_empirical")
                emp_lookup[drug] = float(emp) if pd.notna(emp) else None
                naive_pct = row.get("naive_prevalence_pct")
                if pd.notna(naive_pct):
                    naive_lookup[drug] = float(naive_pct)

        drugs = curves["drug"].dropna().unique().tolist()
        drugs = sorted(drugs, key=lambda d: d)
        n_drugs = len(drugs)
        cols = min(self._COLS, n_drugs)
        rows = math.ceil(n_drugs / cols)

        fig_w = cols * 4.8
        fig_h = rows * 4.2 + 1.5
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                                  squeeze=False, constrained_layout=True)
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))

        for idx, drug in enumerate(drugs):
            row_idx = idx // cols
            col_idx = idx % cols
            ax = axes[row_idx][col_idx]
            drug_curves = curves[curves["drug"] == drug].sort_values("delta")

            if drug_curves.empty:
                ax.axis("off")
                continue

            x = drug_curves["delta"].values
            y = drug_curves["shifted_prevalence_pct"].values
            ax.plot(x, y, color="#1F77B4", linewidth=2.0, label=r"$\hat{\pi}^{shift}(\delta)$")

            # Naive prevalence horizontal reference
            naive = naive_lookup.get(drug)
            if naive is not None:
                ax.axhline(naive, color="#344054", linewidth=1.2, linestyle="--",
                           label=f"Naïve ({naive:.1f}%)", alpha=0.8)

            # Empirical anchor vertical line
            emp = emp_lookup.get(drug)
            if emp is not None and len(x) > 0 and emp <= x[-1]:
                ax.axvline(emp, color="#C0392B", linewidth=1.2, linestyle=":",
                           label=r"$\delta^{emp}$", alpha=0.9)

            # Reference points at δ=0, 1
            # Each tuple: (delta_value, label, color, linestyle)
            for delta_ref, label_ref, ref_color, ref_ls in [
                (0.0, "δ=0", "k", ":"),
                (1.0, "δ=1", "gray", "--"),
            ]:
                if len(x) > 0 and x[-1] >= delta_ref:
                    ax.axvline(delta_ref, linewidth=0.9, linestyle=ref_ls,
                               color=ref_color, alpha=0.5)

            ax.set_title(drug, fontsize=13, fontfamily=self._FONT, fontweight="bold",
                          pad=6, wrap=True)
            ax.set_xlabel("δ", fontsize=11, fontfamily=self._FONT)
            ax.set_ylabel("Prevalence (%)", fontsize=11, fontfamily=self._FONT)
            ax.tick_params(labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#E8EEF5", linewidth=0.7)
            ax.legend(fontsize=9, loc="upper right")

        # Hide unused subplots
        for idx in range(n_drugs, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        title = "Prevalence-Shift Delta Sensitivity Curves"
        if organism:
            title += f" — {format_organism_label(organism)}"
        fig.suptitle(
            title,
            fontsize=16, fontfamily=self._FONT, fontweight="bold", y=0.98,
        )
        fig.text(
            0.0, -0.01,
            "Each panel shows the eligible-scale prevalence estimator across the full admissible δ interval. "
            "Dashed line = naïve tested-row prevalence; dotted red = empirical anchor δ when available.",
            fontsize=11, color="#475467", fontfamily=self._FONT, wrap=True,
        )
        return fig

    def _build_mnar_tier_figures(
        self, curves: pd.DataFrame, prevalence_summary: pd.DataFrame, organism: str = ""
    ) -> list[tuple[str | None, plt.Figure]]:
        """One figure per sensitivity tier, so each can be sized/captioned/placed independently
        in the manuscript instead of living inside one very tall multi-section image."""
        naive_lookup: dict[str, float] = {}
        if not prevalence_summary.empty:
            for _, row in prevalence_summary.iterrows():
                drug = str(row.get("drug", ""))
                naive_pct = row.get("naive_prevalence_pct")
                if pd.notna(naive_pct):
                    naive_lookup[drug] = float(naive_pct)

        estimated = curves[curves["mnar_status"] == "estimated"]
        span_by_drug = estimated.groupby("drug")["mnar_prevalence_pct"].agg(lambda s: s.max() - s.min())

        tier_drugs: dict[str, list[str]] = {key: [] for key, _, _, _ in self._SENSITIVITY_TIERS}
        tier_drugs["not_evaluable"] = []
        for drug in curves["drug"].dropna().unique().tolist():
            span = span_by_drug.get(drug)
            if span is None or pd.isna(span):
                tier_drugs["not_evaluable"].append(drug)
                continue
            for key, _, lower, upper in self._SENSITIVITY_TIERS:
                if lower <= span < upper:
                    tier_drugs[key].append(drug)
                    break
        for key in ("stable", "moderate", "sensitive"):
            tier_drugs[key] = sorted(tier_drugs[key], key=lambda d: -span_by_drug[d])
        tier_drugs["not_evaluable"] = sorted(tier_drugs["not_evaluable"])

        tier_definitions = [*self._SENSITIVITY_TIERS, ("not_evaluable", "Not Evaluable", None, None)]
        populated = [
            (key, label, self._TIER_COLORS[key], tier_drugs[key])
            for key, label, _, _ in tier_definitions
            if tier_drugs[key]
        ]
        if not populated:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "No organism-drug pairs met the prevalence-shift support thresholds.",
                    ha="center", va="center", fontsize=13, fontfamily=self._FONT)
            return [(None, fig)]

        return [
            (tier_key, self._build_tier_figure(tier_key, tier_label, color, drugs, estimated,
                                                span_by_drug, naive_lookup, organism))
            for tier_key, tier_label, color, drugs in populated
        ]

    def _build_tier_figure(
        self,
        tier_key: str,
        tier_label: str,
        color: str,
        drugs: list[str],
        estimated: pd.DataFrame,
        span_by_drug: pd.Series,
        naive_lookup: dict[str, float],
        organism: str,
    ) -> plt.Figure:
        cols = min(self._COLS, len(drugs))
        rows = math.ceil(len(drugs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.5 + 1.3),
                                  squeeze=False, constrained_layout=True)
        # Reserve explicit headroom above the top row for the title and legend so
        # neither crowds the panels' own colored top spine; constrained_layout does
        # not know to leave room for fig.suptitle()/fig.legend() on its own.
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.90))

        for idx, drug in enumerate(drugs):
            ax = axes[idx // cols][idx % cols]
            if tier_key == "not_evaluable":
                ax.axis("off")
                ax.text(0.5, 0.5, f"{drug}\nnot evaluable", ha="center", va="center",
                        fontsize=10, fontfamily=self._FONT, color="#667085")
                continue
            drug_curves = estimated[estimated["drug"] == drug].sort_values("mnar_lambda")
            ax.plot(drug_curves["mnar_lambda"], drug_curves["mnar_prevalence_pct"],
                    color="#1F77B4", linewidth=2.0, marker="o", markersize=3.5)
            naive = naive_lookup.get(drug)
            if naive is not None:
                ax.axhline(naive, color="#344054", linewidth=1.1, linestyle="--", alpha=0.8)
            ax.axvline(0.0, color="#C0392B", linewidth=1.0, linestyle=":", alpha=0.7)
            ax.set_title(f"{drug} (Δ{span_by_drug[drug]:.1f}pp)", fontsize=11,
                         fontfamily=self._FONT, fontweight="bold", pad=5, wrap=True)
            ax.set_xlabel("λ", fontsize=9.5, fontfamily=self._FONT)
            if idx % cols == 0:
                ax.set_ylabel("Prevalence (%)", fontsize=9.5, fontfamily=self._FONT)
            ax.tick_params(labelsize=9)
            for spine_name in ("left", "bottom", "right"):
                ax.spines[spine_name].set_visible(spine_name in ("left", "bottom"))
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_color(color)
            ax.spines["top"].set_linewidth(3.0)
            ax.grid(axis="y", color="#E8EEF5", linewidth=0.6)

        for idx in range(len(drugs), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plural = "s" if len(drugs) != 1 else ""
        title = f"Cascade-Aware MNAR Prevalence Sensitivity Curves — {tier_label} ({len(drugs)} drug{plural})"
        if organism:
            title += f" — {format_organism_label(organism)}"
        fig.suptitle(title, fontsize=15, fontfamily=self._FONT, fontweight="bold", color=color, y=0.99)
        if tier_key == "not_evaluable":
            caption = "No MNAR model could be fit for these drugs at any λ; drugs are listed alphabetically."
        else:
            legend_handles = [
                Line2D([0], [0], color="#1F77B4", linewidth=2.0, marker="o", markersize=4, label="MNAR estimate"),
                Line2D([0], [0], color="#344054", linewidth=1.3, linestyle="--", label="Naive prevalence"),
                Line2D([0], [0], color="#C0392B", linewidth=1.2, linestyle=":", label="λ=0 (no-tilt reference)"),
            ]
            fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                       ncol=3, fontsize=10, frameon=False)
            caption = "Δ = the percentage-point range each drug's MNAR estimate spans across the λ grid; drugs are sorted by descending Δ."
        fig.text(
            0.0, -0.01,
            caption,
            fontsize=10, color="#475467", fontfamily=self._FONT,
        )
        return fig
