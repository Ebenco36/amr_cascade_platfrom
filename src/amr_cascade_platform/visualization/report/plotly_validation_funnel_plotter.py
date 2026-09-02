"""Candidate attrition funnel through each validation gate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

_PASS_COLOR = "#2ca02c"
_FAIL_COLOR = "#d62728"
_BAR_COLOR  = "#1f77b4"
_GRID       = "#E8EEF5"


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper", font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


class PlotlyValidationFunnelPlotter:
    """
    Horizontal funnel showing candidate pair attrition through each validation gate.

    Gates (derived from edge_report columns):
        1. All pairs in report (any escalation_ratio)
        2. Support criteria  (passes_support_threshold == True)
        3. Permutation test  (permutation_fdr_supported == True  OR  permutation_p_value ≤ 0.05)
        4. Bootstrap stable  (bootstrap_sign_stability ≥ configured threshold)
        5. Replicated  (site-level or temporal direction agreement)
        6. Validated  (validation_status in {"robust", "supported"})

    Two sub-charts:
        Left  — funnel bar showing count at each gate (absolute)
        Right — step-loss bar showing how many pairs are removed at each gate
    """

    _VALIDATED_STATUSES = CascadeValidationAnalyzer.VALIDATED_STATUSES

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        bootstrap_stability_threshold: float = 0.80,
        min_replicated_sites: int = 2,
        min_site_direction_agreement: float = 0.50,
        permutation_p_threshold: float = 0.05,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._boot_threshold  = bootstrap_stability_threshold
        self._min_sites       = min_replicated_sites
        self._min_agree_rate  = min_site_direction_agreement
        self._perm_threshold  = permutation_p_threshold

    def export(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        if edge_report.empty or "escalation_ratio" not in edge_report.columns:
            fig = _empty(self._template, self._width, self._height,
                         "Validation funnel unavailable — edge report is empty or missing escalation_ratio")
            return self._exporter.write(fig, output_stem, formats)

        stages, counts, n_supported, n_robust = self._compute_funnel(edge_report)
        if counts[0] == 0:
            fig = _empty(
                self._template,
                self._width,
                self._height,
                "Validation funnel unavailable — no candidate pairs with an escalation ratio",
            )
            return self._exporter.write(fig, output_stem, formats)
        fig = self._build_figure(stages, counts, n_supported, n_robust)
        return self._exporter.write(
            fig,
            output_stem,
            formats,
            static_fallback=lambda fmt, path: self._write_static(stages, counts, n_supported, n_robust, fmt, path),
            prefer_static_fallback=True,
        )

    # ── Funnel computation ────────────────────────────────────────────────

    def _compute_funnel(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[int], int, int]:
        """Return (stages, counts, n_supported, n_robust).

        The funnel reflects the actual pipeline logic:
        • "Supported" pairs pass permutation FDR + bootstrap stability but do NOT
          require cross-site replication — they exit the funnel at Stage 3.
        • "Robust" pairs additionally pass site or temporal replication — they are a
          strict subset of Stage 3 and appear at Stage 4.
        Showing replication as a sequential gate before "validated"
        would overstate how many pairs passed it and misrepresent "supported" pairs
        as having failed a gate they were never required to pass.
        """
        df = df.copy()

        # Stage 0: all rows with any ER data
        n_all = int(df["escalation_ratio"].notna().sum())

        # Stage 1: support criteria
        if "passes_support_threshold" in df.columns:
            support_mask = df["passes_support_threshold"].eq(True)
        else:
            support_mask = pd.Series(True, index=df.index)
        n_support = int(support_mask.sum())
        df_s = df[support_mask]

        # Stage 2: permutation FDR
        if "permutation_fdr_supported" in df_s.columns:
            perm_mask = df_s["permutation_fdr_supported"].eq(True)
        elif "permutation_p_value" in df_s.columns:
            perm_mask = pd.to_numeric(df_s["permutation_p_value"], errors="coerce").le(self._perm_threshold)
        else:
            perm_mask = pd.Series(True, index=df_s.index)
        n_perm = int(perm_mask.sum())
        df_p = df_s[perm_mask]

        # Stage 3: bootstrap stability.
        # Pairs passing both permutation FDR and bootstrap stability form the
        # "validated floor" — every pair here is at minimum "supported".
        if "bootstrap_sign_stability" in df_p.columns:
            boot_mask = pd.to_numeric(df_p["bootstrap_sign_stability"], errors="coerce").ge(self._boot_threshold)
        else:
            boot_mask = pd.Series(True, index=df_p.index)
        n_boot = int(boot_mask.sum())
        df_b = df_p[boot_mask]

        # Stage 4: site or temporal replication (robust only).
        # Read directly from validation_status when available — most accurate because
        # temporal_direction_agreement can also confer "robust" status and is not
        # re-derivable from the site columns alone.
        if "validation_status" in df.columns:
            n_robust = int(df["validation_status"].eq("robust").sum())
            n_supported = int(df["validation_status"].eq("supported").sum())
        else:
            # Fallback: apply replication gate to perm+boot passers.
            if "site_direction_agreement_rate" in df_b.columns and "site_replication_n" in df_b.columns:
                rep_mask = (
                    pd.to_numeric(df_b["site_direction_agreement_rate"], errors="coerce").ge(self._min_agree_rate)
                    & pd.to_numeric(df_b["site_replication_n"], errors="coerce").ge(self._min_sites)
                )
                n_robust = int(rep_mask.sum())
            else:
                n_robust = n_boot
            n_supported = max(0, n_boot - n_robust)

        stages = [
            "All candidate pairs",
            "Pass support criteria<br>(n≥25, n_R≥5, n_S≥5)",
            "Pass permutation FDR<br>(q ≤ 0.05)",
            "Validated: perm + bootstrap<br>(supported ∪ robust)",
            "Robust: + replication support<br>(site or temporal)",
        ]
        counts = [n_all, n_support, n_perm, n_boot, n_robust]
        return stages, counts, n_supported, n_robust

    # ── Figure assembly ───────────────────────────────────────────────────

    def _build_figure(
        self,
        stages: list[str],
        counts: list[int],
        n_supported: int,
        n_robust: int,
    ) -> go.Figure:
        n_all = counts[0] or 1  # avoid division by zero
        pct   = [c / n_all * 100 for c in counts]
        lost  = [counts[i - 1] - counts[i] for i in range(1, len(counts))]

        fig = go.Figure()

        # ── Main funnel bars ─────────────────────────────────────────────
        # Five stages: all, support, perm, validated-floor, robust
        bar_colors = ["#4e79a7", "#59a14f", "#f28e2b", "#76b7b2", "#2ca02c"]

        fig.add_trace(
            go.Bar(
                x=counts,
                y=stages,
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="white", width=0.6),
                ),
                text=[f"<b>{c:,}</b>  ({p:.1f}%)" for c, p in zip(counts, pct)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=14, color="white"),
                customdata=list(zip(stages, counts, [f"{p:.1f}" for p in pct])),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Pairs remaining: %{customdata[1]:,}<br>"
                    "Retention: %{customdata[2]}%<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # ── Dropout annotations (right of each bar except last) ──────────
        # The final step (validated-floor → robust) is special: the "drop" is
        # supported pairs, which ARE fully validated — they just don't have
        # enough replication support for the robust label. Label them in green,
        # not red.
        last_idx = len(stages) - 2
        for i, (stage, count, drop) in enumerate(zip(stages[1:], counts[1:], lost)):
            if drop <= 0:
                continue
            if i == last_idx:
                # Supported pairs: validated but insufficient replication support
                fig.add_annotation(
                    x=counts[i],
                    y=i + 1,
                    xref="x",
                    yref="y",
                    text=f"Supported: {drop:,}<br>(perm+boot, replication insufficient)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowcolor=_PASS_COLOR,
                    arrowwidth=1.5,
                    ax=60,
                    ay=0,
                    font=dict(size=13, color=_PASS_COLOR),
                    xanchor="left",
                )
            else:
                fig.add_annotation(
                    x=counts[i],
                    y=i + 1,
                    xref="x",
                    yref="y",
                    text=f"−{drop:,} excluded",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowcolor=_FAIL_COLOR,
                    arrowwidth=1.5,
                    ax=60,
                    ay=0,
                    font=dict(size=13, color=_FAIL_COLOR),
                    xanchor="left",
                )

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            title=dict(
                text=(
                    "Validation Funnel — Candidate Pair Attrition Through Each Gate"
                    f"<br><sup>Supported={n_supported:,}; Robust={n_robust:,}</sup>"
                ),
                x=0.5,
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Number of candidate pairs",
                showgrid=True,
                gridcolor=_GRID,
                range=[0, n_all * 1.30],
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=13),
            ),
            margin=dict(t=90, b=60, l=240, r=120),
        )
        return fig

    def _write_static(
        self,
        stages: list[str],
        counts: list[int],
        n_supported: int,
        n_robust: int,
        fmt: str,
        path: Path,
    ) -> None:
        """Write a static funnel without relying on Plotly's horizontal-bar export."""
        labels = [stage.replace("<br>", "\n") for stage in stages]
        n_all = max(counts[0], 1)
        y_pos = list(range(len(labels)))
        colors = ["#4e79a7", "#59a14f", "#f28e2b", "#76b7b2", "#2ca02c"]

        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        ax.barh(y_pos, counts, color=colors, alpha=0.92, edgecolor="white", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Number of candidate pairs", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", alpha=0.18)
        ax.set_axisbelow(True)

        x_offset = max(n_all * 0.015, 1)
        for y, count in zip(y_pos, counts):
            pct = count / n_all * 100
            ax.text(
                count + x_offset,
                y,
                f"{count:,} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#1D2939",
            )

        lost = [counts[i - 1] - counts[i] for i in range(1, len(counts))]
        loss_lines = []
        for stage, drop in zip(labels[1:], lost):
            if drop > 0:
                loss_lines.append(f"{drop:,} excluded before: {stage.splitlines()[0]}")
        if n_supported > 0:
            loss_lines.append(f"{n_supported:,} supported: permutation + bootstrap; replication insufficient")
        if loss_lines:
            ax.text(
                0.99,
                0.03,
                "\n".join(loss_lines[:4]),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#475467",
            )

        ax.set_xlim(0, max(counts) * 1.28)
        ax.set_title(
            f"Validation Funnel — Candidate Pair Attrition Through Each Gate\n"
            f"Supported={n_supported:,}; Robust={n_robust:,}",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
