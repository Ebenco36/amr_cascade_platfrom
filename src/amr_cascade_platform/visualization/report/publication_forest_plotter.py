"""Publication-grade forest plots — pure matplotlib, no forestplot dependency."""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter, NullFormatter, FixedLocator
import matplotlib.font_manager as _fm

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver


def _select_font(*candidates: str) -> str:
    """Return the first font family that matplotlib can resolve on this system.

    Tries candidates in order — Arial for local/Windows/Mac, Liberation Sans
    (metric-compatible Arial substitute common on Linux), then DejaVu Sans
    which is always bundled with matplotlib as an absolute fallback.
    """
    available = {f.name for f in _fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return candidates[-1]


# ── publication style constants ────────────────────────────────────────────
_DPI = 300
_SAVEFIG = {"dpi": _DPI, "bbox_inches": "tight", "facecolor": "white"}
_FONT = _select_font("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans")
_ALT_ROW_COLOR = "#F8FAFC"
_HEADER_LINE_COLOR = "#CBD5E1"
_GRID_COLOR = "#E2E8F0"
_REF_LINE_COLOR = "#B91C1C"
_BODY_TEXT = "#1E293B"
_MUTED_TEXT = "#64748B"
_STATUS_MARKER: dict[str, str] = {"robust": "D", "supported": "s"}
_STATUS_EDGE: dict[str, str] = {"robust": "#0F172A", "supported": "#9A3412"}
_STATUS_LW: dict[str, float] = {"robust": 1.4, "supported": 1.0}
_STATUS_LABEL_COLOR: dict[str, str] = {"Robust": "#1D4ED8", "Supported": "#C2410C"}
_AWARE_COLORS = AntibioticClassificationResolver.CATEGORY_COLORS


class PublicationForestPlotter:
    """Render manuscript-facing forest plots with pure matplotlib."""

    def __init__(
        self,
        width: int,
        height: int,
        top_n: int,
        classification_resolver: AntibioticClassificationResolver,
    ) -> None:
        self._width = width
        self._height = height
        self._top_n = top_n
        self._classification = classification_resolver

    # ── public API (identical signatures to old implementation) ────────────

    def export_primary_forest(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        prepared = self._prepare_primary_dataframe(edge_report)
        return self._export(prepared, output_stem, formats, validated=False)

    def export_validated_primary_forest(
        self,
        edge_report: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        allowed_statuses: tuple[str, ...] = ("robust", "supported"),
    ) -> dict[str, Path]:
        prepared = self._prepare_validated_dataframe(edge_report, allowed_statuses)
        return self._export(prepared, output_stem, formats, validated=True)

    def export_forest_summary(self, edge_report: pd.DataFrame, output_path: Path) -> Path:
        fmt = output_path.suffix.lstrip(".") or "png"
        outputs = self.export_primary_forest(edge_report, output_path.with_suffix(""), (fmt,))
        return outputs[output_path.name]

    # ── data preparation ───────────────────────────────────────────────────

    def _prepare_primary_dataframe(self, edge_report: pd.DataFrame) -> pd.DataFrame:
        canonical = self._canonicalize_pair_table(edge_report)
        if canonical.empty:
            return canonical
        canonical = self._attach_ci(canonical)
        canonical = self._annotate_rows(canonical, validated=False)
        rank_map = {"robust": 0, "supported": 1, "mixed": 2, "insufficient": 3}
        if "validation_status" in canonical.columns:
            canonical["__vrank"] = canonical["validation_status"].map(rank_map).fillna(9)
        else:
            canonical["__vrank"] = 9
        canonical["__or_trustworthy"] = self._or_trustworthy(canonical)
        ordered = canonical.sort_values(
            ["__vrank", "__or_trustworthy", "estimate", "total_support_n"],
            ascending=[True, False, False, False],
        ).head(self._top_n)
        return ordered.drop(columns=["__vrank", "__or_trustworthy"]).reset_index(drop=True)

    def _prepare_validated_dataframe(
        self,
        edge_report: pd.DataFrame,
        allowed_statuses: tuple[str, ...],
    ) -> pd.DataFrame:
        canonical = self._canonicalize_pair_table(edge_report)
        if canonical.empty or "validation_status" not in canonical.columns:
            return canonical.iloc[0:0].copy()
        canonical = canonical.loc[canonical["validation_status"].isin(allowed_statuses)].copy()
        if canonical.empty:
            return canonical
        canonical = self._attach_ci(canonical)
        canonical = self._annotate_rows(canonical, validated=True)
        rank_map = {"robust": 0, "supported": 1}
        canonical["__vrank"] = canonical["validation_status"].map(rank_map).fillna(9)
        canonical["__or_trustworthy"] = self._or_trustworthy(canonical)
        ordered = canonical.sort_values(
            ["__vrank", "__or_trustworthy", "estimate", "total_support_n"],
            ascending=[True, False, False, False],
        ).head(self._top_n)
        return ordered.drop(columns=["__vrank", "__or_trustworthy"]).reset_index(drop=True)

    @staticmethod
    def _or_trustworthy(df: pd.DataFrame) -> pd.Series:
        """Adjusted OR is present AND agrees in direction with the raw ER.

        Escalation ratio and adjusted-OR convergence are anti-correlated in
        this data: an extreme ER is the signature of quasi-complete
        separation, which is exactly what makes the covariate-adjusted
        logistic model either fail to converge or -- more insidiously --
        "converge" to a near-zero/near-infinite coefficient that contradicts
        the raw ER's own direction (raw_adjusted_direction_agreement ==
        "directionally_misaligned"). Ranking by estimate alone still
        surfaces those rows, just with a confident-looking wrong number
        instead of an honest N/A. Requiring direction agreement keeps
        "top retained" limited to pairs where the raw and adjusted signals
        actually corroborate each other.
        """
        present = df["adjusted_odds_ratio"].notna()
        if "raw_adjusted_direction_agreement" not in df.columns:
            return present
        return present & df["raw_adjusted_direction_agreement"].eq("directionally_aligned")

    def _annotate_rows(self, df: pd.DataFrame, *, validated: bool) -> pd.DataFrame:
        df = df.copy()
        df = self._classification.add_metadata(df, "downstream_antibiotic", "downstream")
        df["edge_label"] = df.apply(
            lambda r: self._format_edge_label(
                str(r["upstream_antibiotic"]), str(r["downstream_antibiotic"])
            ),
            axis=1,
        )
        df["aware_label"] = df["downstream_aware_category"].fillna("Unclassified")
        df["support_label"] = df["total_support_n"].map(lambda v: f"{int(v):,}")
        df["adjusted_or_label"] = pd.to_numeric(df["adjusted_odds_ratio"], errors="coerce").map(
            lambda v: (
                f"{v:.2f}"
                if pd.notna(v) and abs(v) < 1000
                else (f"{v:.1e}" if pd.notna(v) else "N/A")
            )
        )
        df["estimate"] = pd.to_numeric(df["escalation_ratio"], errors="coerce")
        df["downstream_aware_color"] = df["downstream_aware_category"].map(
            lambda c: _AWARE_COLORS.get(str(c), _AWARE_COLORS["Unclassified"])
        )
        if validated and "validation_status" in df.columns:
            df["group_label"] = (
                df["validation_status"].fillna("screened").astype(str).str.replace("_", " ").str.title()
            )
        else:
            df["group_label"] = df["downstream_aware_category"].fillna("Unclassified")
        return df

    # ── CI attachment ──────────────────────────────────────────────────────

    def _attach_ci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prefer bootstrap CI (available even for quasi-separated pairs); fall back to Wald."""
        df = df.copy()
        has_boot = "bootstrap_er_ci_lower" in df.columns and "bootstrap_er_ci_upper" in df.columns
        has_wald = "er_ci_lower" in df.columns and "er_ci_upper" in df.columns

        if has_boot:
            df["ll"] = pd.to_numeric(df["bootstrap_er_ci_lower"], errors="coerce")
            df["hl"] = pd.to_numeric(df["bootstrap_er_ci_upper"], errors="coerce")
            df["_ci_type"] = "bootstrap"
        elif has_wald:
            df["ll"] = pd.to_numeric(df["er_ci_lower"], errors="coerce")
            df["hl"] = pd.to_numeric(df["er_ci_upper"], errors="coerce")
            df["_ci_type"] = "wald"
        else:
            # Compute Wald CI from counts (legacy path)
            df = self._compute_wald_ci(df)
            df["_ci_type"] = "wald_computed"

        # Ensure ll/hl are positive; collapse to point-only when missing
        bad = df["ll"].isna() | df["hl"].isna() | (df["ll"] <= 0) | (df["hl"] <= 0)
        est = pd.to_numeric(df.get("escalation_ratio", df.get("estimate", pd.Series(dtype=float))), errors="coerce")
        df.loc[bad, "ll"] = est[bad]
        df.loc[bad, "hl"] = est[bad]
        return df

    def _compute_wald_ci(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        a = df.get("resistant_tested_n", pd.Series(0, index=df.index)).astype(float)
        n1 = df.get("resistant_support_n", pd.Series(0, index=df.index)).astype(float)
        c = df.get("susceptible_tested_n", pd.Series(0, index=df.index)).astype(float)
        n0 = df.get("susceptible_support_n", pd.Series(0, index=df.index)).astype(float)
        er = pd.to_numeric(df.get("escalation_ratio", pd.Series(dtype=float)), errors="coerce")
        valid = er.notna() & (er > 0) & (a > 0) & (c > 0) & (n1 > 0) & (n0 > 0)
        se = np.where(
            valid,
            np.sqrt(np.where(valid, 1 / a.clip(lower=1e-9) - 1 / n1.clip(lower=1e-9) + 1 / c.clip(lower=1e-9) - 1 / n0.clip(lower=1e-9), np.nan)),
            np.nan,
        )
        log_er = np.where(valid, np.log(er.clip(lower=1e-9)), np.nan)
        df["ll"] = np.where(valid, np.exp(log_er - 1.96 * se), np.nan)
        df["hl"] = np.where(valid, np.exp(log_er + 1.96 * se), np.nan)
        return df

    # ── figure building ────────────────────────────────────────────────────

    def _build_figure(self, plot_data: pd.DataFrame, *, validated: bool) -> plt.Figure:
        if plot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
            ax.axis("off")
            msg = "No validated retained edges available" if validated else "No retained edges available"
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=13, color=_MUTED_TEXT)
            return fig

        n = len(plot_data)
        fig_h = max(5.5, 0.70 * n + 3.0)
        fig_w = 16.0

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

        # Title + subtitle above the grid
        title = (
            "Validated Antibiotic Cascade: Forest Plot of Robust and Supported Directed Pairs"
            if validated
            else "Primary Antibiotic Cascade: Forest Plot of Top Retained Directed Pairs"
        )
        ci_note = "Bootstrap 95% CI" if (not plot_data.empty and plot_data["_ci_type"].iloc[0] == "bootstrap") else "Approximate 95% CI"
        subtitle = (
            f"{ci_note}  ·  Diamond = robust validation  ·  Square = supported validation  ·  Colour = downstream AWaRe tier"
            if validated
            else f"{ci_note}  ·  Colour = downstream AWaRe tier"
        )
        fig.text(0.5, 0.978, title, ha="center", va="top", fontsize=16, fontweight="bold", color=_BODY_TEXT, fontfamily=_FONT)
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=11, color=_MUTED_TEXT, fontfamily=_FONT)

        gs = GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[3.8, 4.2, 1.2],
            wspace=0.02,
            left=0.02, right=0.98,
            top=0.90, bottom=0.11,
        )
        ax_lbl = fig.add_subplot(gs[0])
        ax_fp = fig.add_subplot(gs[1])
        ax_or = fig.add_subplot(gs[2])

        # Row y-positions: row 0 of df = top of figure → highest y value
        y_pos = list(range(n - 1, -1, -1))

        # ── determine x-axis range ─────────────────────────────────────────
        ests = pd.to_numeric(plot_data["estimate"], errors="coerce").dropna()
        hls = pd.to_numeric(plot_data["hl"], errors="coerce").dropna()
        lls = pd.to_numeric(plot_data["ll"], errors="coerce").dropna()

        raw_max = float(hls.max()) if not hls.empty else float(ests.max()) if not ests.empty else 20.0
        raw_min = float(lls.min()) if not lls.empty else 1.0
        x_max = min(600.0, max(raw_max * 1.8, float(ests.max()) * 2.2 if not ests.empty else 20.0, 20.0))
        x_min = max(0.5, min(raw_min * 0.55, 1.5))

        # ── forest panel ──────────────────────────────────────────────────
        ax_fp.set_xscale("log")
        ax_fp.set_xlim(x_min, x_max)
        ax_fp.set_ylim(-0.80, n - 0.20)

        # Alternating row shading
        for i in range(n):
            if i % 2 == 0:
                ax_fp.axhspan(y_pos[i] - 0.48, y_pos[i] + 0.48, color=_ALT_ROW_COLOR, zorder=0, lw=0)

        # Group separator lines (for validated plot)
        if validated and "group_label" in plot_data.columns:
            groups = plot_data["group_label"].tolist()
            for i in range(1, n):
                if groups[i] != groups[i - 1]:
                    sep_y = (y_pos[i - 1] + y_pos[i]) / 2
                    ax_fp.axhline(sep_y, color=_HEADER_LINE_COLOR, lw=0.9, zorder=1)

        # Reference line at ER = 1
        ax_fp.axvline(1.0, color=_REF_LINE_COLOR, linestyle="--", lw=1.6, zorder=2, alpha=0.80)

        # x-axis tick marks: clean log ticks
        nice_ticks = [t for t in [1, 2, 5, 10, 20, 50, 100, 200, 500] if x_min <= t <= x_max]
        ax_fp.set_xticks(nice_ticks)
        ax_fp.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_fp.xaxis.set_minor_formatter(NullFormatter())

        ax_fp.tick_params(axis="x", which="major", labelsize=11, direction="out", length=4)
        ax_fp.tick_params(axis="x", which="minor", length=2)
        ax_fp.set_yticks([])
        ax_fp.spines[["top", "left", "right"]].set_visible(False)
        ax_fp.spines["bottom"].set_color(_HEADER_LINE_COLOR)
        ax_fp.set_xlabel("Escalation Ratio  (log scale)", fontsize=12, labelpad=7, fontfamily=_FONT)
        ax_fp.xaxis.grid(True, which="major", color=_GRID_COLOR, lw=0.7, zorder=0)
        ax_fp.set_axisbelow(True)

        # Draw each row
        for idx, row in plot_data.iterrows():
            y = y_pos[idx]
            est = pd.to_numeric(row["estimate"], errors="coerce")
            if pd.isna(est) or est <= 0:
                continue

            ll_raw = pd.to_numeric(row["ll"], errors="coerce")
            hl_raw = pd.to_numeric(row["hl"], errors="coerce")
            is_point_only = (pd.isna(ll_raw) or pd.isna(hl_raw) or ll_raw <= 0 or hl_raw <= 0 or (ll_raw == est and hl_raw == est))

            color = row.get("downstream_aware_color", _AWARE_COLORS["Unclassified"])
            status = str(row.get("validation_status", "robust"))
            marker = _STATUS_MARKER.get(status, "D")
            ec = _STATUS_EDGE.get(status, _STATUS_EDGE["robust"])
            lw = _STATUS_LW.get(status, 1.4)

            if not is_point_only:
                ll_clip = max(float(ll_raw), x_min * 1.01)
                hl_clip = min(float(hl_raw), x_max * 0.99)
                # CI bar
                ax_fp.plot(
                    [ll_clip, hl_clip], [y, y],
                    color="#334155", lw=1.8, solid_capstyle="butt", zorder=3,
                )
                # Arrowhead if lower CI clipped by axis
                if float(ll_raw) < x_min * 1.01:
                    ax_fp.annotate(
                        "", xy=(x_min * 1.06, y), xytext=(x_min * 1.30, y),
                        arrowprops=dict(arrowstyle="<-", color="#334155", lw=1.1),
                        zorder=4,
                    )
                # Arrowhead if upper CI clipped by axis
                if float(hl_raw) > x_max * 0.99:
                    ax_fp.annotate(
                        "", xy=(x_max * 0.94, y), xytext=(x_max * 0.70, y),
                        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.1),
                        zorder=4,
                    )
                # Serifs (end caps) on CI bar
                for cap_x in [ll_clip, hl_clip]:
                    ax_fp.plot([cap_x, cap_x], [y - 0.14, y + 0.14], color="#334155", lw=1.4, zorder=3)
            else:
                # Quasi-separation: dotted span to convey uncertainty
                span_lo = max(est * 0.70, x_min * 1.01)
                span_hi = min(est * 1.42, x_max * 0.99)
                ax_fp.plot([span_lo, span_hi], [y, y], color="#94A3B8", lw=1.2, linestyle=":", zorder=3)

            # Point estimate marker (clipped to axis if extreme)
            est_display = min(float(est), x_max * 0.97)
            ax_fp.scatter(
                [est_display], [y],
                c=[color], marker=marker, s=68, zorder=5,
                edgecolors=ec, linewidths=lw,
            )

        # ── label panel (left) ────────────────────────────────────────────
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(-0.80, n - 0.20)
        ax_lbl.axis("off")

        header_y = n - 0.04
        ax_lbl.text(0.02, header_y, "Directed Antibiotic Pair", fontweight="bold", fontsize=12,
                    va="bottom", color=_BODY_TEXT, fontfamily=_FONT)
        ax_lbl.text(0.77, header_y, "AWaRe", fontweight="bold", fontsize=11, va="bottom",
                    ha="center", color=_BODY_TEXT, fontfamily=_FONT)
        ax_lbl.text(0.97, header_y, "N", fontweight="bold", fontsize=11, va="bottom",
                    ha="right", color=_BODY_TEXT, fontfamily=_FONT)
        ax_lbl.axhline(n - 0.22, color=_HEADER_LINE_COLOR, lw=0.9, xmin=0.0, xmax=1.0)

        # Group subheaders
        if validated and "group_label" in plot_data.columns:
            groups = plot_data["group_label"].tolist()
            current_group: str | None = None
            for i in range(n):
                g = groups[i]
                if g != current_group:
                    current_group = g
                    label_color = _STATUS_LABEL_COLOR.get(g, _MUTED_TEXT)
                    ax_lbl.text(
                        0.02, y_pos[i] + 0.42,
                        f">> {g}",
                        fontsize=11, fontweight="bold",
                        color=label_color, va="center", fontfamily=_FONT,
                    )
                    if i > 0:
                        sep_y = (y_pos[i - 1] + y_pos[i]) / 2
                        ax_lbl.axhline(sep_y, color=_HEADER_LINE_COLOR, lw=0.7, xmin=0.0, xmax=1.0)

        # Row alternating shading in label panel (match forest panel)
        for i in range(n):
            if i % 2 == 0:
                ax_lbl.axhspan(y_pos[i] - 0.48, y_pos[i] + 0.48, color=_ALT_ROW_COLOR, zorder=0, lw=0)

        for idx, row in plot_data.iterrows():
            y = y_pos[idx]
            ax_lbl.text(
                0.02, y, row["edge_label"],
                fontsize=11, va="center", color=_BODY_TEXT,
                multialignment="left", fontfamily=_FONT,
            )
            aware_color = row.get("downstream_aware_color", _AWARE_COLORS["Unclassified"])
            ax_lbl.text(
                0.77, y, row.get("aware_label", ""),
                fontsize=10, va="center", ha="center",
                color=aware_color, fontweight="bold", fontfamily=_FONT,
            )
            ax_lbl.text(
                0.97, y, row.get("support_label", ""),
                fontsize=11, va="center", ha="right",
                color=_MUTED_TEXT, fontfamily=_FONT,
            )

        # ── OR annotation panel (right) ───────────────────────────────────
        ax_or.set_xlim(0, 1)
        ax_or.set_ylim(-0.80, n - 0.20)
        ax_or.axis("off")

        ax_or.text(0.5, n - 0.04, "Adj. OR", fontweight="bold", fontsize=12,
                   va="bottom", ha="center", color=_BODY_TEXT, fontfamily=_FONT)
        ax_or.axhline(n - 0.22, color=_HEADER_LINE_COLOR, lw=0.9)

        # Row alternating shading
        for i in range(n):
            if i % 2 == 0:
                ax_or.axhspan(y_pos[i] - 0.48, y_pos[i] + 0.48, color=_ALT_ROW_COLOR, zorder=0, lw=0)

        for idx, row in plot_data.iterrows():
            y = y_pos[idx]
            ax_or.text(
                0.5, y, row.get("adjusted_or_label", "N/A"),
                fontsize=11, va="center", ha="center",
                color=_BODY_TEXT, fontfamily=_FONT,
            )

        # ── legend ────────────────────────────────────────────────────────
        legend_elements: list[Line2D] = []
        if validated:
            legend_elements += [
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#64748B",
                       markeredgecolor=_STATUS_EDGE["robust"], markeredgewidth=1.3,
                       markersize=7, label="Robust"),
                Line2D([0], [0], marker="s", color="none", markerfacecolor="#64748B",
                       markeredgecolor=_STATUS_EDGE["supported"], markeredgewidth=1.0,
                       markersize=7, label="Supported"),
            ]
        for cat in ("Access", "Watch", "Reserve"):
            legend_elements.append(
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=_AWARE_COLORS[cat],
                       markeredgecolor="#374151", markeredgewidth=0.6,
                       markersize=7, label=f"{cat} (AWaRe)")
            )
        legend_elements.append(
            Line2D([0], [0], color=_REF_LINE_COLOR, linestyle="--", lw=1.4, label="ER = 1")
        )
        ax_fp.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=11,
            frameon=True,
            framealpha=0.92,
            edgecolor=_HEADER_LINE_COLOR,
            ncol=2,
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=0.8,
        )

        # ── footer note ───────────────────────────────────────────────────
        fig.text(
            0.02, 0.022,
            (
                "CI bars show bootstrap 95% CI on the escalation ratio (ER) where available; dotted spans indicate pairs where the ER's own CI could not be "
                "estimated (typically near-zero susceptible-arm counts), independent of whether the covariate-adjusted OR converged.  "
                "ER = escalation ratio; OR = adjusted odds ratio; N = total pair support."
            ),
            fontsize=11.0, color=_MUTED_TEXT, fontfamily=_FONT, va="bottom",
        )

        return fig

    # ── export helpers ─────────────────────────────────────────────────────

    def _export(
        self,
        plot_data: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        validated: bool,
    ) -> dict[str, Path]:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        figure = self._build_figure(plot_data, validated=validated)
        png_bytes: bytes | None = None
        for fmt in formats:
            path = output_stem.with_suffix(f".{fmt}")
            if fmt == "html":
                png_bytes = png_bytes or self._figure_png_bytes(figure)
                title = output_stem.name.replace("_", " ").title()
                self._write_html(path, png_bytes, title=title)
            else:
                figure.savefig(path, format=fmt, **_SAVEFIG)
            outputs[path.name] = path
        plt.close(figure)
        return outputs

    # ── static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_edge_label(upstream: str, downstream: str) -> str:
        def _wrap(text: str, width: int = 24) -> str:
            if len(text) <= width:
                return text
            for sep in ("/", " ", "-"):
                positions = [i for i, c in enumerate(text) if c == sep]
                if not positions:
                    continue
                best = min(positions, key=lambda i: abs(i - width))
                if 0 < best < len(text) - 1:
                    return f"{text[:best + 1].rstrip()}\n{text[best + 1:].lstrip()}"
            return f"{text[:width]}\n{text[width:]}"

        return f"{_wrap(upstream)} →\n  {_wrap(downstream)}"

    @staticmethod
    def _figure_png_bytes(figure: plt.Figure) -> bytes:
        buf = io.BytesIO()
        figure.savefig(buf, format="png", **_SAVEFIG)
        return buf.getvalue()

    @staticmethod
    def _write_html(path: Path, png_bytes: bytes, *, title: str) -> None:
        encoded = base64.b64encode(png_bytes).decode("ascii")
        path.write_text(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #ffffff; }}
    img {{ width: 100%; height: auto; border: 1px solid #E2E8F0; box-shadow: 0 4px 16px rgba(15,23,42,.08); }}
  </style>
</head>
<body>
  <img alt="{title}" src="data:image/png;base64,{encoded}">
</body>
</html>""",
            encoding="utf-8",
        )

    # ── canonicalization (preserved from original) ─────────────────────────

    def _canonicalize_pair_table(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        canonical = dataframe.copy()
        if canonical.empty:
            return canonical
        rename_map = {
            "positive_support_n": "resistant_support_n",
            "positive_tested_n": "resistant_tested_n",
            "negative_support_n": "susceptible_support_n",
            "negative_tested_n": "susceptible_tested_n",
            "positive_probability": "resistant_downstream_test_probability",
            "negative_probability": "susceptible_downstream_test_probability",
            "is_retained_edge": "retained_edge",
        }
        canonical = canonical.rename(columns=rename_map)
        defaults: dict[str, object] = {
            "adjusted_odds_ratio": np.nan,
            "total_support_n": 0,
            "resistant_support_n": 0,
            "resistant_tested_n": 0,
            "susceptible_support_n": 0,
            "susceptible_tested_n": 0,
            "resistant_downstream_test_probability": 0.0,
            "susceptible_downstream_test_probability": 0.0,
        }
        for col, default in defaults.items():
            if col not in canonical.columns:
                canonical[col] = default

        for col in ("upstream_antibiotic", "downstream_antibiotic"):
            if col not in canonical.columns:
                continue
            key_col = f"__{col}_key"
            raw_col = f"__{col}_raw"
            canonical[raw_col] = canonical[col].astype("string").fillna("Unknown").str.upper()
            canonical[key_col] = canonical[col].map(self._classification.normalize_label)
            display_map = (
                canonical.groupby(key_col, observed=True)[raw_col]
                .apply(
                    lambda vals: (
                        self._classification.canonical_display_label(vals.name)
                        or self._choose_preferred_display_label(vals, vals.name)
                    )
                )
                .to_dict()
            )
            canonical[col] = canonical[key_col].map(display_map)

        sort_cols = []
        ascending: list[bool] = []
        if "escalation_ratio" in canonical.columns:
            canonical["__has_er"] = (
                canonical["escalation_ratio"].notna() & canonical["escalation_ratio"].gt(1.0)
            )
            sort_cols.append("__has_er")
            ascending.append(False)
        if "total_support_n" in canonical.columns:
            sort_cols.append("total_support_n")
            ascending.append(False)
        sort_cols += ["__upstream_antibiotic_key", "__downstream_antibiotic_key"]
        ascending += [True, True]

        canonical = canonical.sort_values(sort_cols, ascending=ascending)
        canonical = canonical.drop_duplicates(
            ["__upstream_antibiotic_key", "__downstream_antibiotic_key"], keep="first"
        )
        drop = [
            "__has_er",
            "__upstream_antibiotic_key",
            "__downstream_antibiotic_key",
            "__upstream_antibiotic_raw",
            "__downstream_antibiotic_raw",
        ]
        return canonical.drop(columns=drop, errors="ignore").reset_index(drop=True)

    @staticmethod
    def _choose_preferred_display_label(values: pd.Series, canonical_key: str | None = None) -> str:
        candidates = sorted({str(v).strip().upper() for v in values if str(v).strip()})
        if not candidates:
            return "UNKNOWN"
        if canonical_key and canonical_key in candidates:
            return canonical_key

        def _score(label: str) -> tuple[int, int, int, int]:
            alpha = sum(c.isalpha() for c in label)
            sep_bonus = label.count("/") + label.count(" ")
            rm_pen = 0 if " RM" not in label else -1
            abbr_pen = 0 if len(label) > 4 else -1
            return (abbr_pen, rm_pen, alpha + sep_bonus, len(label))

        return max(candidates, key=_score)

    # kept for any external callers that still use the old name
    def _attach_rr_confidence_intervals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self._attach_ci(dataframe)
