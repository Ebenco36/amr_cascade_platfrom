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

from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

# ── Palette ───────────────────────────────────────────────────────────────────
_OBS   = "#2C6FAC"   # observed / tested
_UNOBS = "#E07B39"   # unobserved / untested
_SITE  = ["#2C6FAC", "#E07B39", "#27AE60", "#8E44AD", "#C0392B"]
_GRID  = "#E8EEF5"
_TEXT  = "#1A1A2E"
_REF   = "#888888"
_HIGH  = "#C0392B"   # high imbalance (|SMD| ≥ 0.1)
_LOW   = "#27AE60"   # acceptable balance
_BG    = "rgba(0,0,0,0)"


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
            yaxis=dict(title="", tickfont=dict(size=11)),
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
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#C0392B"], [0.5, "#F5CBA7"], [1, "#27AE60"]],
            zmin=0, zmax=100,
            text=[[f"{v * 100:.0f}%" if not math.isnan(v) else "—"
                   for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            colorbar=dict(title="Testing<br>rate (%)", ticksuffix="%"),
        ))
        h = max(self._height, 28 * len(pivot))
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
            subplot_titles=[str(s) for s in sites],
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
            height=max(400, self._height // 2),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Episode Volume by Calendar Period and Site</b>",
                       font=dict(size=15, color=_TEXT)),
            margin=dict(l=60, r=60, t=80, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)

    # ── Figure 6: Covariate comparison (tested vs. untested) ─────────────────

    def export_covariate_comparison(
        self,
        eligible_pairs: pd.DataFrame,
        culture_episodes: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """Two-panel figure: comorbidity count and ADI score, tested vs. untested.

        Both panels show the distribution separately for tested (blue) and
        untested (orange) episodes so the severity-driven selection into testing
        is immediately visible.
        """
        ep_keys = [c for c in ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]
                   if c in eligible_pairs.columns and c in culture_episodes.columns]
        comorb_col = next((c for c in culture_episodes.columns
                           if "comorbidity_count" in c.lower()), None)
        adi_col    = next((c for c in culture_episodes.columns
                           if "adi_score" in c.lower() or "adi_state_rank" in c.lower()), None)

        if not ep_keys or (comorb_col is None and adi_col is None):
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Covariate comparison unavailable — episode key or covariate columns not found"),
                output_stem, formats,
            )

        # Determine tested status per episode: tested if ANY drug was tested
        tested_episodes = (
            eligible_pairs[eligible_pairs["is_observed_tested"].eq(1)][ep_keys]
            .drop_duplicates()
            .assign(_tested=1)
        )
        episodes = culture_episodes[ep_keys + [c for c in [comorb_col, adi_col]
                                               if c is not None]].drop_duplicates()
        episodes = episodes.merge(tested_episodes, on=ep_keys, how="left")
        episodes["_tested"] = episodes["_tested"].fillna(0).astype(int)
        episodes["_group"] = episodes["_tested"].map({1: "Tested", 0: "Untested"})

        cols_to_plot = [(c, label) for c, label in [
            (comorb_col, "Comorbidity count"),
            (adi_col, "ADI score"),
        ] if c is not None]
        n_panels = len(cols_to_plot)

        fig = make_subplots(rows=1, cols=n_panels,
                            subplot_titles=[label for _, label in cols_to_plot],
                            horizontal_spacing=0.12)

        for idx, (col, label) in enumerate(cols_to_plot, start=1):
            for group, color in [("Tested", _OBS), ("Untested", _UNOBS)]:
                vals = _clean_num(episodes.loc[episodes["_group"].eq(group), col]).dropna()
                fig.add_trace(
                    go.Box(
                        y=vals,
                        name=group,
                        marker_color=color,
                        boxmean=True,
                        showlegend=(idx == 1),
                        legendgroup=group,
                        hovertemplate=f"<b>{group}</b><br>{label}: %{{y:.1f}}<extra></extra>",
                    ),
                    row=1, col=idx,
                )

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(
                text="<b>Severity Covariates: Tested vs. Untested Episodes</b>",
                font=dict(size=15, color=_TEXT),
            ),
            boxmode="group",
            legend=dict(title=""),
            margin=dict(l=60, r=40, t=80, b=50),
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
                      annotation_text="ER = 1 (no cascade)", annotation_position="right")

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=self._height,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Drug-Pair Support vs. Escalation Ratio</b>",
                       font=dict(size=15, color=_TEXT)),
            xaxis=dict(title="Total episode pairs (support N)", type="log",
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
        """Horizontal bar chart: % of episodes with each covariate available.

        Uses the ``cov_*_available`` indicator columns where present; falls back
        to non-null rate otherwise. Sorted ascending so least-available covariates
        appear at top.
        """
        if culture_episodes.empty:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "Missing data profile unavailable — culture_episodes is empty"),
                output_stem, formats,
            )

        rows: list[dict[str, object]] = []
        cov_cols = [c for c in culture_episodes.columns if c.startswith("cov_")]

        # Prefer _available flags; fall back to non-null rate of the raw covariate
        availability_flags = {c for c in cov_cols if c.endswith("_available")}
        base_covariates    = {c for c in cov_cols if not c.endswith("_available")}

        for col in sorted(base_covariates):
            flag = col + "_available"
            label = col.replace("cov_", "").replace("_", " ").title()
            if flag in availability_flags:
                rate = _clean_num(culture_episodes[flag]).mean()
            else:
                rate = culture_episodes[col].notna().mean()
            if pd.notna(rate):
                rows.append({"covariate": label, "availability": float(rate)})

        if not rows:
            return self._exporter.write(
                _empty(self._template, self._width, self._height,
                       "No cov_* columns found in culture_episodes"),
                output_stem, formats,
            )

        summary = (
            pd.DataFrame(rows)
            .sort_values("availability", ascending=True)
            .reset_index(drop=True)
        )
        colors = [_HIGH if v < 0.80 else _OBS for v in summary["availability"]]
        h = max(self._height, 35 * len(summary))

        fig = go.Figure(go.Bar(
            y=summary["covariate"],
            x=summary["availability"] * 100,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0%}" for v in summary["availability"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Available: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=80, line=dict(color=_HIGH, dash="dash", width=1.2),
                      annotation_text="80% threshold", annotation_position="top right")

        fig.update_layout(
            template=self._template,
            width=self._width,
            height=h,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            title=dict(text="<b>Covariate Availability Profile</b>",
                       font=dict(size=15, color=_TEXT)),
            xaxis=dict(title="Episodes with covariate available (%)",
                       range=[0, 110], gridcolor=_GRID),
            yaxis=dict(title="", tickfont=dict(size=11)),
            margin=dict(l=200, r=80, t=60, b=50),
        )
        return self._exporter.write(fig, output_stem, formats)
