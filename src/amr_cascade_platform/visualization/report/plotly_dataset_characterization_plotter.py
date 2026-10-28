"""Dataset characterisation figures — eligible space vs. observed space and cohort summaries.

Generates a complete, self-consistent figure set describing the dataset before
any cascade modelling: what was eligible, what was observed, who was tested vs.
not tested, and how those groups differ on every measured covariate.

All figures degrade gracefully when a column is absent — they render a labelled
placeholder rather than raising, so the suite can be called even on partial data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amr_cascade_platform.core.utils.site_labels import ALL_SITES, scope_order, site_label
from amr_cascade_platform.visualization.report.plotly_exporter import PRINT_FONT_SCALE, PlotlyFigureExporter
from amr_cascade_platform.visualization.report.print_layout import Geometry, export_both_media

# ── Palette ───────────────────────────────────────────────────────────────────
_OBS   = "#2C6FAC"   # observed / tested
_UNOBS = "#E07B39"   # unobserved / untested
_SITE  = ["#2C6FAC", "#E07B39", "#27AE60", "#8E44AD", "#C0392B"]
_GRID  = "#E8EEF5"
_TEXT  = "#1A1A2E"
_REF   = "#888888"
_HIGH  = "#C0392B"   # high imbalance (|SMD| ≥ 0.1)
_LOW   = "#27AE60"   # acceptable balance
_BG    = "white"  # static PNG/PDF exports composite transparency onto black in some viewers/renderers -- explicit white matches every other figure module in this codebase
# Opportunity-space colours shared with figure_opportunity_space (validated as a set).
_ELIGIBLE      = "#1c5cab"
_NOT_AVAILABLE = "#a8861a"
_INTRINSIC     = "#7b68b5"
_FUNNEL        = ["#86b6ef", "#3987e5", "#1c5cab"]  # one hue, darker as the space narrows
_BLUE_RAMP     = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]

# Covariate and source-record coverage (not all are primary model inputs), using
# definitions as Table V): "known" = a category other than unknown; "flag" =
# the covariate builder's record indicator; "adi" = the ADI record indicator
# with a positive score (the builder writes 0 where the source lacks it);
# "dated" = comorbidity data in which every named condition has a start date.
_COVARIATE_RECORDS = (
    ("Age", "cov_age_bin", "known"),
    ("Sex", "cov_sex", "known"),
    ("Ordering context", "cov_ordering_mode", "known"),
    ("Specimen type", "cov_specimen_type", "known"),
    ("Emergency-department status", "cov_er_available", "flag"),
    ("Intensive-care status", "cov_icu_available", "flag"),
    ("Medication record before culture", "cov_prior_abx_available", "flag"),
    ("Earlier culture record", "cov_prior_organism_available", "flag"),
    ("Nursing-home record", "cov_nursing_home_available", "flag"),
    ("Procedure record", "cov_prior_procedure_available", "flag"),
    ("Comorbidity data", "cov_comorbidity_available", "flag"),
    ("Comorbidity data, all starts dated", "cov_comorbidity_available", "dated"),
    ("Area Deprivation Index", "cov_adi_available", "adi"),
    ("Area Deprivation Index state rank", "cov_adi_state_rank_available", "flag"),
    ("Laboratory values", "cov_labs_available", "flag"),
    ("Vital signs", "cov_vitals_available", "flag"),
)

_MISSING_CATEGORY_TOKENS = {"", "unknown", "null", "none", "nan", "na", "n/a", "<na>", "nat", "missing", "not recorded"}
_RECORD_INDICATORS = {
    "cov_prior_abx_available", "cov_prior_organism_available", "cov_nursing_home_available",
    "cov_prior_procedure_available", "cov_comorbidity_available", "cov_labs_available", "cov_vitals_available",
}
_PROFILE_LABELS = {
    "Medication record before culture": "Prior antibiotic-exposure record",
    "Comorbidity data, all starts dated": "Comorbidity: all starts dated",
    "Area Deprivation Index": "Deprivation index score",
    "Area Deprivation Index state rank": "Deprivation index state rank",
    "Laboratory values": "Laboratory record*",
    "Vital signs": "Vital-sign record*",
}


def _empty(template: str, w: int, h: int, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       xref="paper", yref="paper",
                       font=dict(size=13, color="#666666"))
    fig.update_layout(template=template, width=w, height=h,
                      paper_bgcolor=_BG, plot_bgcolor=_BG,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def _clean_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return _clean_num(frame[column]).fillna(0).astype(int).eq(1)


def _recorded(frame: pd.DataFrame, column: str, kind: str) -> pd.Series | None:
    """Which episodes have the covariate recorded (see _COVARIATE_RECORDS); None when the column is absent."""
    if column not in frame.columns:
        return None
    if kind == "known":
        values = frame[column].astype("string").fillna("").str.strip().str.lower()
        return ~values.isin(_MISSING_CATEGORY_TOKENS)
    recorded = _clean_num(frame[column]).eq(1).fillna(False)
    if kind == "adi":
        if "cov_adi_score" not in frame.columns:
            return None
        recorded &= _clean_num(frame["cov_adi_score"]).gt(0)
    if kind == "dated":
        if "cov_comorbidity_start_unrecorded" not in frame.columns:
            return None
        recorded &= _clean_num(frame["cov_comorbidity_start_unrecorded"]).eq(0).fillna(False)
    return recorded


def _availability_text(recorded: pd.Series | None) -> str:
    """Cell text that never rounds a gap away: 100% only when nothing is missing,
    0% only when nothing is recorded, otherwise the share rounded down to one decimal."""
    if recorded is None or not len(recorded):
        return "—"
    missing = int((~recorded.astype(bool)).sum())
    if missing == 0:
        return "100%"
    if missing == len(recorded):
        return "0%"
    share = math.floor(1000 * (len(recorded) - missing) / len(recorded)) / 10
    return "<0.1%" if share == 0 else f"{share:.1f}%"


def _availability_hover(recorded: pd.Series | None) -> str:
    """Exact counts for the hover text: recorded and without the covariate."""
    if recorded is None or not len(recorded):
        return "no data"
    present = int(recorded.astype(bool).sum())
    return f"{present:,} of {len(recorded):,} episodes; {len(recorded) - present:,} without it"


def _scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(display label, rows) for all sites, then each site present, in the configured order."""
    sites = frame["source_site"].astype(str) if "source_site" in frame.columns else pd.Series("", index=frame.index)
    scopes = [(site_label(ALL_SITES), frame)]
    for site in scope_order((), sites.unique())[1:]:
        part = frame.loc[sites.eq(site).to_numpy()]
        if not part.empty:
            scopes.append((site_label(site), part))
    return scopes


def _box_stats(values: pd.Series) -> dict[str, float] | None:
    """Tukey box statistics (whiskers at the most extreme values within 1.5 IQR), precomputed so the figure stays small."""
    values = _clean_num(values).dropna()
    if values.empty:
        return None
    q1, median, q3 = (float(v) for v in values.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return {"q1": q1, "median": median, "q3": q3, "mean": float(values.mean()),
            "lowerfence": float(inside.min()), "upperfence": float(inside.max()), "n": int(len(values))}


class DatasetCharacterizationPlotter:
    """Publication and presentation figures for dataset characterisation.

    Each ``export_*`` method is self-contained: it accepts only the data it
    needs, writes one or more files, and returns a ``{key: Path}`` dict.

    ``export_all`` is the convenience entry-point; it calls every method and
    merges the output dicts.

    Parameters
    ----------
    exporter:
        Shared :class:`PlotlyFigureExporter` instance.
    template:
        Plotly template name (e.g. ``"plotly_white"``).
    width, height:
        Output pixel dimensions.
    """

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
    ) -> None:
        self._exporter  = exporter
        self._template  = template
        self._width     = width
        self._height    = height

    # ── Public convenience entry-point ────────────────────────────────────────

    def export_all(
        self,
        eligible_pairs: pd.DataFrame,
        culture_episodes: pd.DataFrame,
        drug_pair_episodes: pd.DataFrame,
        escalation_results: pd.DataFrame,
        upstream_balance_table: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...] = ("png", "html"),
    ) -> dict[str, Path]:
        """Export the full characterisation suite into *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        outputs.update(self.export_eligibility_funnel(
            eligible_pairs,
            output_dir / "dataset_eligibility_funnel",
            formats,
        ))
        outputs.update(self.export_eligibility_by_site(
            eligible_pairs,
            output_dir / "dataset_eligibility_by_site",
            formats,
        ))
        outputs.update(self.export_eligible_vs_observed_by_drug(
            eligible_pairs,
            output_dir / "dataset_eligible_vs_observed",
            formats,
        ))
        outputs.update(self.export_observation_rate_heatmap(
            eligible_pairs,
            output_dir / "dataset_observation_rate_heatmap",
            formats,
        ))
        outputs.update(self.export_testing_rate_by_upstream_result(
            drug_pair_episodes,
            output_dir / "dataset_testing_rate_by_upstream_result",
            formats,
        ))
        outputs.update(self.export_selection_imbalance_love_plot(
            upstream_balance_table,
            output_dir / "dataset_selection_imbalance_love_plot",
            formats,
        ))
        outputs.update(self.export_episode_temporal_heatmap(
            culture_episodes,
            output_dir / "dataset_episode_temporal_heatmap",
            formats,
        ))
        outputs.update(self.export_covariate_comparison(
            eligible_pairs,
            culture_episodes,
            output_dir / "dataset_covariate_comparison",
            formats,
        ))
        outputs.update(self.export_pair_support_landscape(
            escalation_results,
            output_dir / "dataset_pair_support_landscape",
            formats,
        ))
        outputs.update(self.export_missing_data_profile(
            culture_episodes,
            output_dir / "dataset_missing_data_profile",
            formats,
        ))
        return outputs

    # ── Figure 0: Eligibility funnel and its split by site ────────────────────

    _ELIGIBILITY_COLUMNS = {"is_eligible", "is_intrinsic_resistance", "is_operationally_available", "source_site"}

    @staticmethod
    def _eligibility_segments(frame: pd.DataFrame) -> tuple[int, int, int]:
        """(intrinsic, operationally unavailable, eligible) rows.

        is_eligible is definitionally (is_intrinsic_resistance == 0) AND
        (is_operationally_available == 1) under the primary denominator
        (Methods, Denominator construction), so the three counts partition the
        candidate space exactly -- by construction, not by rounding.
        """
        total = len(frame)
        intrinsic = int(frame["is_intrinsic_resistance"].eq(1).sum())
        biologically_eligible = frame.loc[frame["is_intrinsic_resistance"].eq(0)]
        operationally_unavailable = int(biologically_eligible["is_operationally_available"].eq(0).sum())
        eligible = int(frame["is_eligible"].eq(1).sum())
        assert intrinsic + operationally_unavailable + eligible == total, (
            "eligibility segments must partition the candidate space exactly"
        )
        return intrinsic, operationally_unavailable, eligible

    def export_eligibility_funnel(
        self,
        eligible_pairs: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Funnel of the two sequential gates from the candidate episode-antibiotic space
        to the eligible opportunity space: biological (intrinsic resistance), then
        operational (the drug was reported at the site in that era).

        The split by site is export_eligibility_by_site, a figure of its own.
        """
        if eligible_pairs.empty or not self._ELIGIBILITY_COLUMNS.issubset(eligible_pairs.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Eligibility funnel unavailable — required columns missing"),
                output_stem, formats,
            )
        total_n = len(eligible_pairs)
        intrinsic_n, _, eligible_n = self._eligibility_segments(eligible_pairs)
        stages = [
            "Candidate episode–antibiotic pairs",
            "Biologically eligible<br>(no intrinsic resistance)",
            "Eligible<br>(also reported at the site in that era)",
        ]
        fig = go.Figure(go.Funnel(
            y=stages,
            x=[total_n, total_n - intrinsic_n, eligible_n],
            textinfo="value+percent initial",
            texttemplate="%{value:,}<br>%{percentInitial:.1%} of candidates",
            textposition="inside",
            textfont=dict(size=15, color="white"),
            marker=dict(color=_FUNNEL),
            connector=dict(line=dict(color=_REF, width=1)),
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>N = %{x:,}<extra></extra>",
        ))
        fig.update_layout(
            template=self._template,
            width=1400,
            height=620,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="<b>Eligible opportunity space</b><br>"
                     "<sup>Biological exclusion is fixed by organism–drug identity; "
                     "operational exclusion varies by site and era</sup>",
                font=dict(size=16, color=_TEXT),
                x=0.01, xanchor="left",
            ),
            yaxis=dict(title="", tickfont=dict(size=13)),
            margin=dict(l=40, r=40, t=110, b=30),
        )
        return self._exporter.write(fig, output_stem, formats)

    def export_eligibility_by_site(
        self,
        eligible_pairs: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Share of each site's candidate space that is eligible, not operationally
        available, or intrinsically resistant; all sites first.

        Biological exclusion is essentially fixed across sites while operational
        exclusion is not -- the actual driver of any site-level eligible-rate gap.
        """
        if eligible_pairs.empty or not self._ELIGIBILITY_COLUMNS.issubset(eligible_pairs.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Eligibility by site unavailable — required columns missing"),
                output_stem, formats,
            )
        rows = []
        for label, group in _scopes(eligible_pairs):
            intrinsic, unavailable, eligible = self._eligibility_segments(group)
            total = len(group)
            rows.append({"scope": label, "total": total, "eligible_n": eligible,
                         "eligible_pct": 100 * eligible / total,
                         "operational_pct": 100 * unavailable / total,
                         "intrinsic_pct": 100 * intrinsic / total})
        table = pd.DataFrame(rows)
        fig = go.Figure()
        for name, color, key in [
            ("Eligible", _ELIGIBLE, "eligible_pct"),
            ("Not operationally available", _NOT_AVAILABLE, "operational_pct"),
            ("Intrinsic resistance", _INTRINSIC, "intrinsic_pct"),
        ]:
            fig.add_trace(go.Bar(
                name=name,
                y=table["scope"],
                x=table[key],
                orientation="h",
                marker=dict(color=color, line=dict(color=_BG, width=2)),
                hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1f}}%<extra></extra>",
            ))
        for _, row in table.iterrows():
            fig.add_annotation(
                x=100, y=row["scope"], xref="x", yref="y", xanchor="left", xshift=10,
                text=f"{row['eligible_pct']:.1f}% eligible<br>{row['eligible_n']:,} of {row['total']:,}",
                showarrow=False, align="left", font=dict(size=12, color=_TEXT),
            )
        fig.update_layout(
            template=self._template,
            width=1600,
            height=max(460, 110 * len(table) + 220),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            barmode="stack",
            bargap=0.35,
            title=dict(
                text="<b>Eligible share of the candidate space, by site</b><br>"
                     "<sup>Candidate episode–antibiotic pairs split into the three mutually exclusive categories</sup>",
                font=dict(size=16, color=_TEXT),
                x=0.01, xanchor="left",
            ),
            xaxis=dict(title="Share of candidate pairs (%)", range=[0, 100], gridcolor=_GRID, ticksuffix="%"),
            yaxis=dict(title="", autorange="reversed", tickfont=dict(size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
            margin=dict(l=40, r=160, t=120, b=60),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 1: Eligible vs. Observed space by drug ─────────────────────────

    def export_eligible_vs_observed_by_drug(
        self,
        eligible_pairs: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Horizontal stacked bar: eligible N vs. observed N per drug.

        This is the primary eligible-space vs. observed-space comparison.
        Drugs are sorted by observation rate (ascending) so the most-missing
        drugs appear at the top.
        """
        required = {"antibiotic", "is_eligible", "is_observed_tested"}
        if eligible_pairs.empty or not required.issubset(eligible_pairs.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Eligible vs. observed figure unavailable — required columns missing"),
                output_stem, formats,
            )

        df = (
            eligible_pairs[eligible_pairs["is_eligible"].eq(1)]
            .groupby("antibiotic", observed=True)
            .agg(
                eligible_n=("is_eligible", "sum"),
                observed_n=("is_observed_tested", "sum"),
            )
            .reset_index()
        )
        if df.empty:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "No eligible pairs found"),
                output_stem, formats,
            )

        df["unobserved_n"]    = df["eligible_n"] - df["observed_n"]
        df["obs_rate"]        = df["observed_n"] / df["eligible_n"].clip(lower=1)
        df                    = df.sort_values("obs_rate", ascending=True).reset_index(drop=True)
        drug_labels           = df["antibiotic"].tolist()

        h = max(self._height, 60 * len(drug_labels))
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Observed (tested)",
            y=drug_labels,
            x=df["observed_n"],
            orientation="h",
            marker_color=_OBS,
            hovertemplate="<b>%{y}</b><br>Observed: %{x:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Unobserved (missing)",
            y=drug_labels,
            x=df["unobserved_n"],
            orientation="h",
            marker_color=_UNOBS,
            hovertemplate="<b>%{y}</b><br>Unobserved: %{x:,}<extra></extra>",
        ))

        # Observation rate annotation on the right
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row["eligible_n"],
                y=row["antibiotic"],
                text=f"  {row['obs_rate']:.0%}",
                showarrow=False,
                font=dict(size=10, color=_TEXT),
                xanchor="left",
            )

        # This figure's height h is dynamic (grows with drug count -- 900px for a
        # short list, 2500px+ for the full antibiotic set). A title/legend
        # positioned via fixed *fractions* of paper space (e.g. y=0.99) lands at a
        # fixed fraction of a wildly different total height each time and was
        # empirically confirmed (see plotly 5.24 + kaleido 0.2.1 isolation test)
        # to NOT track the plot's actual top edge -- the legend rendered on top of
        # row 2 of the data regardless of the y fraction given. Anchoring the
        # legend's bottom edge to y=1.0 (the plot domain's own top edge, via
        # yanchor="bottom") instead of a paper fraction reliably places it in the
        # top margin immediately above the plot, independent of h. The title is
        # left at its default position (just x-aligned), which also renders
        # correctly above the legend once the margin gives both enough room.
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            barmode="stack",
            title=dict(text="<b>Eligible vs. Observed Antibiotic-Episode Pairs</b>",
                       font=dict(size=16, color=_TEXT),
                       x=0.01, xanchor="left"),
            xaxis=dict(title="Episode-drug pair count", gridcolor=_GRID),
            # df is sorted obs_rate ascending (most-missing drug first) so that
            # drug is meant to lead the chart -- Plotly's default y-axis for a
            # horizontal bar plots the first category at the BOTTOM, which put
            # the best-observed drugs on top instead. autorange="reversed" makes
            # the visual order match the documented, intended reading order.
            yaxis=dict(title="", tickfont=dict(size=11), autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            margin=dict(l=180, r=80, t=130, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 2: Observation rate heatmap (drug × site) ─────────────────────

    def export_observation_rate_heatmap(
        self,
        eligible_pairs: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Heatmap of testing rate (%) per drug × site.

        Reveals site-level variation in cascade protocols: the same drug may be
        almost always tested at one site and rarely at another.
        """
        required = {"antibiotic", "source_site", "is_eligible", "is_observed_tested"}
        if eligible_pairs.empty or not required.issubset(eligible_pairs.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Observation rate heatmap unavailable — required columns missing"),
                output_stem, formats,
            )

        df = (
            eligible_pairs[eligible_pairs["is_eligible"].eq(1)]
            .groupby(["antibiotic", "source_site"], observed=True)
            .agg(eligible_n=("is_eligible", "sum"),
                 observed_n=("is_observed_tested", "sum"))
            .reset_index()
        )
        df["obs_rate"] = df["observed_n"] / df["eligible_n"].clip(lower=1)

        pivot = df.pivot(index="antibiotic", columns="source_site", values="obs_rate")
        # Sort rows by mean observation rate ascending
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=[site_label(str(site)) for site in pivot.columns],
            y=pivot.index.tolist(),
            # One hue, light to dark: a share is a magnitude, and red-green is not colour-blind safe.
            colorscale=[[stop, colour] for stop, colour in _BLUE_RAMP],
            zmin=0, zmax=100,
            text=[[f"{v * 100:.0f}%" if not math.isnan(v) else "—"
                   for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            colorbar=dict(title="Testing<br>rate (%)", ticksuffix="%"),
        ))
        h = max(self._height, 50 * len(pivot))  # was 28 -- taller rows needed once PRINT_FONT_SCALE enlarges tick labels
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Observation Rate by Drug and Site</b>",
                       font=dict(size=16, color=_TEXT)),
            xaxis=dict(title="Site", side="bottom"),
            yaxis=dict(title="", tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=200, r=60, t=60, b=60),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure: Operational availability matrix (drug x site, era collapsed) ──

    def export_operational_availability_matrix(
        self,
        availability_table: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Heatmap: share of eras each drug was operationally available, per site.

        Collapses the era dimension of the site x era x drug availability
        table into one summary statistic per drug x site cell -- "what could
        plausibly be observed" as a stable snapshot. See
        export_site_era_availability_timeline for the temporal view of the
        same underlying table.
        """
        required = {"site", "era", "antibiotic", "is_operationally_available"}
        if availability_table.empty or not required.issubset(availability_table.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Operational availability matrix unavailable — required columns missing"),
                output_stem, formats,
            )

        df = (
            availability_table.groupby(["antibiotic", "site"], observed=True)["is_operationally_available"]
            .mean()
            .reset_index(name="available_fraction")
        )
        pivot = df.pivot(index="antibiotic", columns="site", values="available_fraction")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=[site_label(str(site)) for site in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[stop, colour] for stop, colour in _BLUE_RAMP],
            zmin=0, zmax=100,
            text=[[f"{v * 100:.0f}%" if not math.isnan(v) else "—"
                   for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            colorbar=dict(title="Eras<br>available", ticksuffix="%"),
        ))
        h = max(self._height, 50 * len(pivot))  # was 28 -- taller rows needed once PRINT_FONT_SCALE enlarges tick labels
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Operational Availability by Drug and Site</b><br>"
                            "<sup>Share of 5-year eras each drug met the site's operational-availability rule</sup>",
                       font=dict(size=16, color=_TEXT)),
            xaxis=dict(title="Site", side="bottom"),
            yaxis=dict(title="", tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=200, r=60, t=80, b=60),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure: Site-era availability timeline ────────────────────────────────

    def export_site_era_availability_timeline(
        self,
        availability_table: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Faceted heatmap: drug x era operational availability, one panel per site.

        Era runs chronologically left to right within each panel, so a drug
        entering observability partway through the study window shows as
        blank cells on the left (cold start), and one leaving shows as blank
        cells on the right (panel discontinuation) -- read directly off the
        grid rather than inferred from a summary statistic.
        """
        required = {"site", "era", "antibiotic", "is_operationally_available"}
        if availability_table.empty or not required.issubset(availability_table.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Site-era availability timeline unavailable — required columns missing"),
                output_stem, formats,
            )

        sites = sorted(availability_table["site"].dropna().unique().tolist())
        eras = sorted(availability_table["era"].dropna().unique().tolist())
        totals = availability_table.groupby("antibiotic", observed=True)["is_operationally_available"].sum()
        drugs = sorted(availability_table["antibiotic"].dropna().unique().tolist(), key=lambda d: -totals.get(d, 0))

        fig = make_subplots(
            rows=1, cols=len(sites), shared_yaxes=True,
            subplot_titles=[site_label(str(site)) for site in sites], horizontal_spacing=0.03,
        )
        for annotation in fig.layout.annotations:
            annotation.font = {"size": 12}
        for col, current_site in enumerate(sites, start=1):
            site_df = availability_table[availability_table.site == current_site]
            pivot = site_df.pivot_table(index="antibiotic", columns="era", values="is_operationally_available", aggfunc="first")
            pivot = pivot.reindex(index=drugs, columns=eras)
            heatmap_kwargs = dict(
                z=pivot.values, x=eras, y=drugs,
                colorscale=[[0, "#F2F2F2"], [1, "#2C6FAC"]], zmin=0, zmax=1,
                showscale=(col == 1),
                hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
            )
            if col == 1:
                heatmap_kwargs["colorbar"] = dict(title="Available")
            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=1, col=col)
        h = max(self._height, 36 * len(drugs))  # was 20 -- taller rows needed once PRINT_FONT_SCALE enlarges tick labels
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=8))
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Site-Era Availability Timeline</b><br>"
                            "<sup>Blank left of a drug's first solid cell = cold start; blank right of its last = panel discontinuation</sup>",
                       font=dict(size=16, color=_TEXT)),
            margin=dict(l=180, r=40, t=90, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 3: Testing rate by upstream result ─────────────────────────────

    def export_testing_rate_by_upstream_result(
        self,
        drug_pair_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Scatter: P(downstream tested | upstream resistant) vs. P(downstream tested | upstream sensitive).

        Points above the diagonal are cascade-pattern pairs — upstream resistance
        predicts MORE downstream testing. This is the aggregate cascade signal
        before any modelling.
        """
        required = {"upstream_antibiotic", "downstream_antibiotic",
                    "upstream_susceptibility", "downstream_tested"}
        if drug_pair_episodes.empty or not required.issubset(drug_pair_episodes.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Testing-rate-by-upstream-result figure unavailable — required columns missing"),
                output_stem, formats,
            )

        # Infer positive/negative labels from the data (most common non-null values)
        sus_vals = drug_pair_episodes["upstream_susceptibility"].dropna().astype(str).str.upper()
        pos_label = sus_vals[sus_vals.isin(["RESISTANT", "R", "1"])].value_counts().idxmax() \
            if sus_vals.isin(["RESISTANT", "R", "1"]).any() else None
        neg_label = sus_vals[sus_vals.isin(["SUSCEPTIBLE", "S", "0"])].value_counts().idxmax() \
            if sus_vals.isin(["SUSCEPTIBLE", "S", "0"]).any() else None

        if pos_label is None or neg_label is None:
            # Fall back to whatever two values have the most rows
            top2 = sus_vals.value_counts().head(2).index.tolist()
            if len(top2) < 2:
                return self._exporter.write(
                    _empty(self._template, self._width, self._height,
                           "Cannot determine upstream result categories"),
                    output_stem, formats,
                )
            pos_label, neg_label = top2[0], top2[1]

        up_col = drug_pair_episodes["upstream_susceptibility"].astype(str).str.upper()
        df = drug_pair_episodes.copy()
        df["_result_group"] = "other"
        df.loc[up_col.eq(pos_label), "_result_group"] = "positive"
        df.loc[up_col.eq(neg_label), "_result_group"] = "negative"
        df = df[df["_result_group"].isin(["positive", "negative"])]

        agg = (
            df.groupby(
                ["upstream_antibiotic", "downstream_antibiotic", "_result_group"],
                observed=True,
            )
            .agg(support_n=("downstream_tested", "size"),
                 tested_n=("downstream_tested", "sum"))
            .reset_index()
        )
        agg["rate"] = agg["tested_n"] / agg["support_n"].clip(lower=1)

        pos_df = agg[agg["_result_group"] == "positive"].rename(
            columns={"rate": "pos_rate", "support_n": "pos_n"})
        neg_df = agg[agg["_result_group"] == "negative"].rename(
            columns={"rate": "neg_rate", "support_n": "neg_n"})
        merged = pos_df[["upstream_antibiotic", "downstream_antibiotic",
                          "pos_rate", "pos_n"]].merge(
            neg_df[["upstream_antibiotic", "downstream_antibiotic",
                     "neg_rate", "neg_n"]],
            on=["upstream_antibiotic", "downstream_antibiotic"],
        )
        merged["total_n"] = merged["pos_n"] + merged["neg_n"]
        merged = merged[merged["total_n"] >= 20]
        if merged.empty:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Insufficient data for testing-rate scatter (min support 20)"),
                output_stem, formats,
            )

        merged["er"] = merged["pos_rate"] / merged["neg_rate"].clip(lower=1e-4)
        merged["label"] = (merged["upstream_antibiotic"].astype(str)
                           + " → " + merged["downstream_antibiotic"].astype(str))

        er_clipped = merged["er"].clip(upper=10)
        size_norm   = np.clip(np.log1p(merged["total_n"]), 1, None)
        marker_size = 6 + 12 * (size_norm / size_norm.max())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["neg_rate"],
            y=merged["pos_rate"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=er_clipped,
                colorscale=[[0, "#27AE60"], [0.3, "#F5CBA7"], [1, "#C0392B"]],
                cmin=1, cmax=5,
                showscale=True,
                colorbar=dict(title="Escalation<br>ratio (clipped ≤10)",
                              tickvals=[1, 2, 3, 4, 5],
                              ticktext=["1×", "2×", "3×", "4×", "≥5×"]),
                line=dict(color="white", width=0.5),
                opacity=0.85,
            ),
            text=merged["label"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "P(tested | sensitive): %{x:.1%}<br>"
                "P(tested | resistant): %{y:.1%}<br>"
                "<extra></extra>"
            ),
        ))
        # Diagonal reference (no cascade)
        axis_max = max(merged["pos_rate"].max(), merged["neg_rate"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
                      line=dict(color=_REF, dash="dash", width=1.2))
        fig.add_annotation(
            x=axis_max * 0.72, y=axis_max * 0.68,
            text="No cascade<br>(equal testing)", showarrow=False,
            font=dict(size=11, color=_REF), textangle=-35,
        )

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="<b>Downstream Testing Rate: Upstream Resistant vs. Sensitive</b>",
                font=dict(size=15, color=_TEXT),
            ),
            xaxis=dict(title="P(downstream tested | upstream sensitive)",
                       tickformat=".0%", gridcolor=_GRID,
                       range=[0, axis_max]),
            yaxis=dict(title="P(downstream tested | upstream resistant)",
                       tickformat=".0%", gridcolor=_GRID,
                       range=[0, axis_max]),
            margin=dict(l=70, r=60, t=60, b=70),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 4: Selection imbalance love plot ───────────────────────────────

    def export_selection_imbalance_love_plot(
        self,
        upstream_balance_table: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Love plot (dot plot) of standardised mean differences, tested vs. untested.

        Expects the output of ManuscriptTableBuilder.build_upstream_selection_balance_table().
        Required columns: covariate, standardised_mean_difference.
        """
        required = {"covariate", "standardised_mean_difference"}
        if upstream_balance_table.empty or not required.issubset(upstream_balance_table.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Love plot unavailable — run build_upstream_selection_balance_table() first"),
                output_stem, formats,
            )

        df = upstream_balance_table.copy()
        df["smd"] = _clean_num(df["standardised_mean_difference"])
        df = df.dropna(subset=["smd"])

        # Summarise across sites / drugs: take the mean |SMD| per covariate
        summary = (
            df.groupby("covariate")["smd"]
            .apply(lambda s: s.abs().mean())
            .reset_index()
            .rename(columns={"smd": "mean_abs_smd"})
            .sort_values("mean_abs_smd", ascending=True)
        )
        if summary.empty:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "No SMD data available"),
                output_stem, formats,
            )

        colors = [_HIGH if v >= 0.1 else _LOW for v in summary["mean_abs_smd"]]
        h = max(self._height, 40 * len(summary))

        fig = go.Figure()
        fig.add_vline(x=0,    line=dict(color=_REF, width=1))
        fig.add_vline(x=0.1,  line=dict(color=_HIGH, dash="dash", width=1))
        fig.add_vline(x=-0.1, line=dict(color=_HIGH, dash="dash", width=1))

        fig.add_trace(go.Scatter(
            x=summary["mean_abs_smd"],
            y=summary["covariate"],
            mode="markers",
            marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
            hovertemplate="<b>%{y}</b><br>Mean |SMD|: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="<b>Covariate Imbalance: Tested vs. Untested Episodes</b><br>"
                     "<sup>Mean absolute SMD across drug-site strata | "
                     "dashed line = 0.10 threshold</sup>",
                font=dict(size=15, color=_TEXT),
            ),
            xaxis=dict(title="Mean absolute standardised mean difference",
                       gridcolor=_GRID, range=[-0.02, None]),
            yaxis=dict(title="", tickfont=dict(size=11)),
            margin=dict(l=220, r=40, t=80, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 5: Episode temporal heatmap ───────────────────────────────────

    def export_episode_temporal_heatmap(
        self,
        culture_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Calendar heatmap: episode volume by year × month, faceted by site.

        Reveals data sparsity, temporal gaps, and enrolment ramp-up.
        """
        year_col  = next((c for c in culture_episodes.columns
                          if "year" in c.lower() and c not in ("order_time_jittered",)), None)
        month_col = next((c for c in culture_episodes.columns
                          if "month" in c.lower() and c not in ("order_time_jittered",)), None)
        site_col  = "source_site" if "source_site" in culture_episodes.columns else None

        # Fall back to any datetime column when no explicit year/month columns exist
        datetime_col = next(
            (c for c in culture_episodes.columns
             if pd.api.types.is_datetime64_any_dtype(culture_episodes[c])
             or c.lower().endswith(("_date", "_time", "_dt", "_jittered"))),
            None,
        )

        if culture_episodes.empty or (year_col is None and month_col is None and datetime_col is None):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Temporal heatmap unavailable — year/month columns not found"),
                output_stem, formats,
            )

        df = culture_episodes.copy()
        if year_col is None or month_col is None:
            # format="mixed": this column is combined across sites, and raw source
            # timestamp strings differ in format per site (e.g. "2017-12-11
            # 01:16:00+00:00" vs "2022-02-11T09:37:00.000Z" vs a naive
            # "2009-09-03 08:51:00"). Without format="mixed", pandas infers a
            # single format from the array and silently coerces every row that
            # doesn't match it to NaT -- which previously dropped 2 of 3 sites
            # entirely (their rows failed the subsequent dropna) with no error.
            # utc=True is required alongside it since the mix includes both
            # tz-aware and tz-naive strings.
            dt = pd.to_datetime(df[datetime_col], errors="coerce", format="mixed", utc=True)
            df["_year"]  = dt.dt.year
            df["_month"] = dt.dt.month
        else:
            df["_year"]  = _clean_num(df[year_col])
            df["_month"] = _clean_num(df[month_col])
        df = df.dropna(subset=["_year", "_month"])
        df["_year"]  = df["_year"].astype(int)
        df["_month"] = df["_month"].astype(int)

        group_cols = ["_year", "_month"] + (["source_site"] if site_col else [])
        counts = df.groupby(group_cols, observed=True).size().reset_index(name="n")

        sites = sorted(counts["source_site"].unique()) if site_col else ["all"]
        n_sites = len(sites)

        fig = make_subplots(
            rows=1, cols=n_sites,
            subplot_titles=[site_label(str(s)) for s in sites],
            horizontal_spacing=0.08,
        )
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        global_max = counts["n"].max()

        for col_idx, site in enumerate(sites, start=1):
            sub = counts[counts["source_site"].eq(site)] if site_col else counts
            pivot = (
                sub.pivot(index="_month", columns="_year", values="n")
                .reindex(index=range(1, 13))
            )
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(y) for y in pivot.columns],
                    y=months,
                    colorscale=[[0, "#F0F4F8"], [1, _OBS]],
                    zmin=0, zmax=global_max,
                    showscale=(col_idx == n_sites),
                    colorbar=dict(title="Episodes"),
                    hovertemplate="Year %{x}, %{y}: %{z:,} episodes<extra></extra>",
                ),
                row=1, col=col_idx,
            )

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=max(720, self._height),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Episode Volume by Calendar Period and Site</b>",
                       font=dict(size=15, color=_TEXT)),
            margin=dict(l=60, r=60, t=80, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 6: Comorbidity and deprivation by site ─────────────────────────

    def export_covariate_comparison(
        self,
        eligible_pairs: pd.DataFrame,
        culture_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Comorbidity count and ADI score by site, among episodes with the covariate recorded.

        Expects the adjusted model's covariates (CascadeCovariateBuilder) merged onto
        culture_episodes. The earlier tested-versus-untested split was dropped:
        nearly every culture episode has at least one reported result, so its
        "untested" group was empty. eligible_pairs is accepted for interface
        compatibility.
        """
        panels = [
            ("Comorbidity count", "cov_comorbidity_count", "cov_comorbidity_available", "flag"),
            ("Area Deprivation Index score", "cov_adi_score", "cov_adi_available", "adi"),
        ]
        panels = [panel for panel in panels if panel[1] in culture_episodes.columns and panel[2] in culture_episodes.columns]
        if culture_episodes.empty or not panels:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Covariate comparison unavailable — the adjusted model's covariates were not supplied"),
                output_stem, formats,
            )
        fig = make_subplots(rows=len(panels), cols=1, subplot_titles=[label for label, *_ in panels], vertical_spacing=0.24)
        for row, (label, value_column, record_column, kind) in enumerate(panels, start=1):
            for scope_label, part in _scopes(culture_episodes):
                recorded = _recorded(part, record_column, kind)
                stats = _box_stats(part.loc[recorded, value_column]) if recorded is not None else None
                if stats is None:
                    continue
                pooled = scope_label == site_label(ALL_SITES)
                fig.add_trace(go.Box(
                    y=[f"{scope_label} (n = {stats['n']:,})"], q1=[stats["q1"]], median=[stats["median"]], q3=[stats["q3"]],
                    lowerfence=[stats["lowerfence"]], upperfence=[stats["upperfence"]], mean=[stats["mean"]],
                    orientation="h", boxmean=True, boxpoints=False, name=scope_label, showlegend=False,
                    marker_color=_TEXT if pooled else _OBS, line=dict(width=2),
                    fillcolor="rgba(26,26,46,0.12)" if pooled else "rgba(44,111,172,0.18)",
                    hovertemplate=f"<b>{scope_label}</b><br>{label}: median %{{median}}<extra></extra>",
                ), row=row, col=1)
            fig.update_yaxes(autorange="reversed", row=row, col=1)
            fig.update_xaxes(gridcolor=_GRID, zeroline=False, row=row, col=1)
        fig.update_layout(
            template=self._template,
            width=1600,
            height=260 + 300 * len(panels),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="<b>Comorbidity and deprivation by site</b><br>"
                     "<sup>Episodes with the covariate recorded (n per row); box = quartiles, "
                     "whiskers = most extreme values within 1.5 IQR, dashed line = mean</sup>",
                font=dict(size=16, color=_TEXT), x=0.01, xanchor="left",
            ),
            margin=dict(l=40, r=40, t=110, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 7: Pair support landscape ─────────────────────────────────────

    def export_pair_support_landscape(
        self,
        escalation_results: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Scatter: total support N vs. escalation ratio, coloured by retention.

        Shows the joint distribution of data volume and cascade signal across
        all candidate drug pairs, and where the support and ER thresholds sit.
        """
        required = {"upstream_antibiotic", "downstream_antibiotic",
                    "total_support_n", "escalation_ratio", "passes_support_threshold"}
        if escalation_results.empty or not required.issubset(escalation_results.columns):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Pair support landscape unavailable — required columns missing"),
                output_stem, formats,
            )

        df = escalation_results.copy()
        df["escalation_ratio"] = _clean_num(df["escalation_ratio"])
        df["total_support_n"]  = _clean_num(df["total_support_n"])
        df = df[df["escalation_ratio"].notna() & df["total_support_n"].notna()]
        df = df[~df["escalation_ratio"].isin([math.inf, -math.inf])]

        retained_col = "is_retained_edge" if "is_retained_edge" in df.columns else None
        if retained_col:
            df["_status"] = df[retained_col].map({True: "Retained", False: "Not retained"})
        else:
            df["_status"] = df["passes_support_threshold"].map(
                {True: "Passes support", False: "Below support"})

        status_colors = {
            "Retained": _OBS,
            "Not retained": _UNOBS,
            "Passes support": _OBS,
            "Below support": "#AAAAAA",
        }
        df["_label"] = (df["upstream_antibiotic"].astype(str)
                        + " → " + df["downstream_antibiotic"].astype(str))

        fig = go.Figure()
        for status, color in status_colors.items():
            sub = df[df["_status"].eq(status)]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["total_support_n"],
                y=sub["escalation_ratio"],
                mode="markers",
                name=status,
                marker=dict(size=7, color=color, opacity=0.75,
                            line=dict(color="white", width=0.4)),
                text=sub["_label"],
                hovertemplate="<b>%{text}</b><br>N: %{x:,}<br>ER: %{y:.2f}<extra></extra>",
            ))

        fig.add_hline(y=1.0, line=dict(color=_REF, dash="dash", width=1.2),
                      annotation_text="ER = 1 (no cascade)", annotation_position="top left")

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Drug-Pair Support vs. Escalation Ratio</b>",
                       font=dict(size=15, color=_TEXT)),
            xaxis=dict(title="Total episode pairs (support N)", type="log", dtick=1,
                       gridcolor=_GRID),
            yaxis=dict(title="Escalation ratio", gridcolor=_GRID),
            legend=dict(title=""),
            margin=dict(l=70, r=40, t=60, b=70),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 8: Missing data profile ───────────────────────────────────────

    def export_missing_data_profile(
        self,
        culture_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Export value/record coverage, with exact counts and explicit unavailable inputs.

        Availability flags for history, labs and vitals describe source records,
        not completeness of every field or confirmed absence of an event.
        """
        if culture_episodes.empty:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Missing data profile unavailable — culture_episodes is empty"),
                output_stem, formats,
            )
        scopes = _scopes(culture_episodes)
        rows, shares, texts, hovers, audit = [], [], [], [], []
        for label, column, kind in _COVARIATE_RECORDS:
            row_shares, row_texts, row_hovers = [], [], []
            for scope, part in scopes:
                recorded = _recorded(part, column, kind)
                available = recorded is not None and len(part) > 0
                share = float(recorded.mean()) if available else math.nan
                present = int(recorded.sum()) if available else None
                absent = len(part) - present if available else None
                absence_label = "no record" if column in _RECORD_INDICATORS and kind != "dated" else "not recorded"
                if kind == "dated":
                    absence_label = "not fully dated"
                row_shares.append(share)
                percentage = _availability_text(recorded) if available else "N/A"
                cell = f"<b>{percentage}</b>"
                if available and absent:
                    cell += f"<br>{absent:,} {absence_label}"
                elif not available:
                    cell += "<br>input unavailable"
                row_texts.append(cell)
                row_hovers.append(
                    f"{present:,} / {len(part):,} recorded; {absent:,} {absence_label}"
                    if available else "Required input was not supplied; coverage cannot be calculated."
                )
                audit.append({
                    "scope": scope, "covariate": label, "source_column": column,
                    "definition": kind, "episodes": len(part), "recorded_n": present,
                    "not_recorded_n": absent, "recorded_fraction": share,
                    "display_percentage": percentage, "absence_label": absence_label,
                    "evaluable": available,
                })
            rows.append(label)
            shares.append(row_shares)
            texts.append(row_texts)
            hovers.append(row_hovers)
        columns = [f"<b>{label}</b><br>n = {len(part):,}" for label, part in scopes]
        data = (rows, shares, texts, hovers, columns)
        outputs = export_both_media(self._exporter, self._missing_data_profile_figure, data, output_stem, formats)
        audit_path = output_stem.with_name(output_stem.name + "_counts.csv")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(audit_path, index=False)
        outputs[audit_path.name] = audit_path
        return outputs

    def _missing_data_profile_figure(self, data: tuple, *, medium: str) -> go.Figure:
        rows, shares, texts, hovers, columns = data
        width = max(1060, 350 + 180 * len(columns))
        g = Geometry.for_medium(medium, screen_width=width, print_width=round(width * PRINT_FONT_SCALE))
        height = round(g.px(48 * len(rows) + 220))
        # Pale cells support dark, two-line labels in both HTML and vector exports.
        # A neutral background cell also makes N/A visibly different from zero.
        fig = go.Figure(go.Heatmap(
            z=[[value if math.isfinite(value) else -1 for value in row] for row in shares],
            x=columns, y=rows, zmin=-1, zmax=1,
            colorscale=[[0, "#eeeeee"], [0.4999, "#eeeeee"], [0.5, "#f3f7fb"], [1, "#9fc5e8"]],
            showscale=False, text=texts, texttemplate="%{text}",
            textfont=dict(size=12, color=_TEXT), customdata=hovers,
            xgap=g.px(3), ygap=g.px(3), hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
        ))
        fig.update_layout(
            template=self._template,
            width=g.width, height=height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            font=dict(family="Arial", size=14, color=_TEXT),
            meta=dict(print_geometry=True, geometry_scale=g.margin_scale, heading_band=48),
            title=dict(
                text="<b>Covariate and source-record availability</b>",
                font=dict(size=20, color=_TEXT), x=0.015, xanchor="left",
                y=1 - g.px(14) / height, yanchor="top", yref="container",
            ),
            xaxis=dict(title="", side="top", tickfont=dict(size=14), tickangle=0,
                       showgrid=False, zeroline=False, fixedrange=True),
            yaxis=dict(title="", autorange="reversed", tickfont=dict(size=14),
                       tickmode="array", tickvals=rows, ticktext=[_PROFILE_LABELS.get(row, row) for row in rows],
                       showgrid=False, zeroline=False, fixedrange=True),
            margin=dict(l=320, r=20, t=110, b=110),
        )
        for boundary in (3.5, 5.5, 9.5, 13.5):
            fig.add_hline(y=boundary, line=dict(color="white", width=g.px(7)))
        note = (
            "Percent recorded; second line counts episodes without the value or record. N/A = input unavailable.<br>"
            "Percentages are rounded down to 0.1%; 100% means complete. No record does not prove no event.<br>"
            "* Labs/vitals show source-record coverage, not completeness of individual measurements; not primary adjustment inputs.<br>"
            "Fully dated comorbidity requires a record and no undated starts. All percentages use the full column denominator."
        )
        fig.add_annotation(text=note, x=0, y=0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                           yshift=-g.px(20), align="left", showarrow=False, font=dict(size=10, color="#4b5563"))
        return fig
