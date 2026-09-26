"""Figures for the descriptive summaries of the analysis set.

* ``figure_opportunity_space`` -- per scope, the episode x antibiotic grid as one
  100% bar split into eligible observed, eligible not observed, not
  operationally available, and intrinsic resistance.
* ``figure_panel_breadth`` -- small multiples (specimen group x scope) of the
  number of antibiotics with an interpretable result per culture episode.
* ``figure_observation_coverage_by_era`` -- small multiples (one heatmap per
  scope) of the observed share of the eligible space per antibiotic and era.
* ``figure_antibiogram`` -- small multiples (one column per scope) of the
  resistant and intermediate shares among tested episodes per antibiotic.

Colours: observed/eligible reuse the observation-coverage figures' ink and
track; the two ineligible categories are a validated two-hue pair (all-pairs
colour-vision checks pass, both >= 3:1 on white); resistant/intermediate are a
validated one-hue ordinal ramp; the heatmap uses the reference sequential blue
ramp, with a neutral grey for cells outside the eligible space so "not
eligible" never reads as "0% observed". Every figure is built for screen and
print geometry (see print_layout).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amr_cascade_platform.core.utils.site_labels import ALL_SITES, site_label
from amr_cascade_platform.reporting.builders.descriptive_summaries import (
    ALL_ANTIBIOTICS,
    ALL_SPECIMENS,
    SPECIMEN_GROUPS,
    era_display,
)
from amr_cascade_platform.visualization.report.print_layout import (
    FONT,
    FONT_HEADING,
    FONT_LABEL,
    FONT_VALUE,
    GRID,
    INK,
    INK_SPARSE,
    LABEL_GAP,
    MUTED,
    RULE,
    TEXT,
    TRACK,
    VALUE_GAP,
    Geometry,
    empty_figure,
    export_both_media,
    frame,
    key,
)

NOT_AVAILABLE = "#a8861a"
INTRINSIC = "#7b68b5"
RESISTANT = "#b8322f"
INTERMEDIATE = "#ec8f86"
ABSENT = "#e9ebee"
# Reference sequential blue, steps 100 / 250 / 400 / 550 / 700.
BLUE_RAMP = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
AWARE_ORDER = ("Access", "Watch", "Reserve", "Not Set", "Unclassified")
AWARE_DISPLAY = {"Not Set": "Not classified", "Unclassified": "Not classified"}
MIN_CELL_SUPPORT = 20

_VALUE_X = 104  # data units on a 0-130 axis: every cell's value sits beside its 0-100 track


def _scope_order(values: pd.Series) -> list[str]:
    seen = list(dict.fromkeys(values.astype(str)))
    return ([ALL_SITES] if ALL_SITES in seen else []) + [scope for scope in seen if scope != ALL_SITES]


def _drug_rows(frame_: pd.DataFrame, share_column: str, pooled_label: str | None = None) -> list[tuple[str, str | None]]:
    """(tick text, antibiotic) rows: AWaRe heading rows (antibiotic None), drugs by the all-sites value."""
    drugs = frame_.loc[frame_["site"].eq(ALL_SITES) & frame_["antibiotic"].ne(ALL_ANTIBIOTICS)]
    drugs = drugs.drop_duplicates("antibiotic")
    rank = {name: index for index, name in enumerate(AWARE_ORDER)}
    # Drugs whose pooled value rests on too few results sort last within their group, so a
    # 1-of-2 resistance share cannot head the list.
    sparse = drugs["sparse"].astype(bool) if "sparse" in drugs.columns else pd.Series(False, index=drugs.index)
    drugs = drugs.assign(_aware=drugs["aware_category"].map(rank).fillna(len(AWARE_ORDER)), _sparse=sparse)
    drugs = drugs.sort_values(["_aware", "_sparse", share_column, "antibiotic_display"], ascending=[True, True, False, True], kind="mergesort")
    rows: list[tuple[str, str | None]] = [(f"<b>{pooled_label}</b>", ALL_ANTIBIOTICS)] if pooled_label else []
    for category, group in drugs.groupby("aware_category", sort=False):
        rows.append((f"<b>{AWARE_DISPLAY.get(category, category)}</b>", None))
        rows.extend(zip(group["antibiotic_display"], group["antibiotic"], strict=True))
    return rows


class PlotlyDescriptivePlotter:
    """The four descriptive-summary figures (see module docstring)."""

    def __init__(self, exporter, template: str, min_tested: int = 30) -> None:
        self._exporter = exporter
        self._template = template
        self._min_tested = min_tested

    # -------------------------------------------------------------- exports
    def export_opportunity_space(self, opportunity_space: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return export_both_media(self._exporter, self.opportunity_space_figure, opportunity_space, output_stem, formats)

    def export_panel_breadth(self, summary: pd.DataFrame, distribution: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return export_both_media(self._exporter, self.panel_breadth_figure, (summary, distribution), output_stem, formats)

    def export_coverage_by_era(self, coverage_by_era: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return export_both_media(self._exporter, self.coverage_by_era_figure, coverage_by_era, output_stem, formats)

    def export_antibiogram(self, antibiogram: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        return export_both_media(self._exporter, self.antibiogram_figure, antibiogram, output_stem, formats)

    # ---------------------------------------------------- opportunity space
    _SEGMENTS = (
        ("eligible_observed_rows", "Eligible, observed", INK, INK, "white"),
        ("eligible_unobserved_rows", "Eligible, not observed", TRACK, RULE, TEXT),
        ("not_available_rows", "Not operationally available", NOT_AVAILABLE, NOT_AVAILABLE, "white"),
        ("intrinsic_rows", "Intrinsic resistance", INTRINSIC, INTRINSIC, "white"),
    )
    _OPPORTUNITY_MARGIN = {"l": 165, "r": 185, "t": 118, "b": 62}

    def opportunity_space_figure(self, table: pd.DataFrame, medium: str = "screen") -> go.Figure:
        if table.empty:
            return empty_figure("No eligibility grid to summarise", self._template)
        g = Geometry.for_medium(medium, screen_width=1400, print_width=2000)
        table = table.set_index("site").loc[_scope_order(table["site"])].reset_index()
        labels = [f"<b>{site_label(site)}</b>" if site == ALL_SITES else site_label(site) for site in table["site"]]
        grid = table["grid_rows"].astype(float).replace(0, np.nan)
        plot_height = g.px(52) * len(table)
        height = int(round(sum(self._OPPORTUNITY_MARGIN[side] for side in ("t", "b")) * g.margin_scale + plot_height))
        fig = go.Figure()
        base = np.zeros(len(table))
        for column, name, colour, outline, text_colour in self._SEGMENTS:
            share = (100 * table[column] / grid).fillna(0).to_numpy()
            fig.add_trace(go.Bar(
                x=share, y=labels, base=base, orientation="h", width=0.6, name=name, showlegend=False,
                marker={"color": colour, "line": {"color": "white", "width": g.px(1.5)}},
                customdata=np.column_stack([table[column].to_numpy(), table["grid_rows"].to_numpy()]),
                hovertemplate=f"%{{y}}<br>{name}: %{{x:.1f}}%<br>%{{customdata[0]:,}} of %{{customdata[1]:,}} opportunities<extra></extra>",
            ))
            for label, start, width in zip(labels, base, share, strict=True):
                if width >= 7:
                    fig.add_annotation(x=start + width / 2, y=label, text=f"<b>{width:.0f}%</b>", showarrow=False,
                                       font={"size": FONT_VALUE, "color": text_colour, "family": FONT})
            base = base + share
        for label, row in zip(labels, table.itertuples(index=False), strict=True):
            fig.add_annotation(
                x=1, xref="x domain", y=label, xanchor="left", xshift=g.px(VALUE_GAP), showarrow=False,
                text=f"<span style='color:{MUTED}'>n = {int(row.grid_rows):,}</span>",
                font={"size": FONT_VALUE, "color": TEXT, "family": FONT},
            )
        # A category with no rows at any scope (intrinsic drugs rarely enter the grid) gets no key entry.
        key(fig, g, [(name, "square", colour, outline) for column, name, colour, outline, _ in self._SEGMENTS if table[column].sum() > 0])
        fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont={"size": FONT_LABEL, "color": TEXT},
                         ticklabelstandoff=round(g.px(LABEL_GAP)))
        fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                         showgrid=True, gridcolor=GRID, gridwidth=g.px(1.0), zeroline=False,
                         tickfont={"size": FONT_LABEL, "color": MUTED},
                         title={"text": "Share of the episode–antibiotic opportunity grid", "font": {"size": FONT_LABEL, "color": TEXT}})
        frame(fig, g, height, "Eligibility and observation of episode–antibiotic opportunities",
              self._OPPORTUNITY_MARGIN, {"legend": None}, self._template)
        return fig

    # --------------------------------------------------------- panel breadth
    _PANEL_MARGIN = {"l": 150, "r": 20, "t": 88, "b": 86}

    def panel_breadth_figure(self, data: tuple[pd.DataFrame, pd.DataFrame], medium: str = "screen") -> go.Figure:
        summary, distribution = data
        if distribution.empty:
            return empty_figure("No culture episodes to summarise", self._template)
        g = Geometry.for_medium(medium, screen_width=1500, print_width=2250)
        scopes = _scope_order(distribution["site"])
        groups = [group for group in SPECIMEN_GROUPS if group in set(distribution["specimen_group"])]
        rows_n, cols_n = len(groups), len(scopes)
        cell_height = g.px(150)
        plot_height = cell_height * rows_n
        height = int(round(sum(self._PANEL_MARGIN[side] for side in ("t", "b")) * g.margin_scale + plot_height))
        fig = make_subplots(
            rows=rows_n, cols=cols_n, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.02,
            vertical_spacing=min(0.08, g.px(34) / plot_height),
            column_titles=[f"<b>{site_label(scope)}</b>" for scope in scopes],
        )
        x_max = int(distribution["observed_antibiotics"].max())
        y_max = float(100 * distribution["share"].max()) if not distribution.empty else 1.0
        stats = summary.set_index(["site", "specimen_group"]) if not summary.empty else pd.DataFrame()
        for row, group in enumerate(groups, start=1):
            for col, scope in enumerate(scopes, start=1):
                cell = distribution.loc[distribution["site"].eq(scope) & distribution["specimen_group"].eq(group)]
                if cell.empty:
                    continue
                episodes = int(cell["episodes"].sum())
                colour = INK if episodes >= MIN_CELL_SUPPORT else INK_SPARSE
                fig.add_trace(go.Bar(
                    x=cell["observed_antibiotics"], y=100 * cell["share"], marker={"color": colour, "line": {"width": 0}},
                    width=0.78, showlegend=False,
                    customdata=cell[["episodes"]].to_numpy(),
                    hovertemplate=(f"{site_label(scope)}, {group}<br>%{{x}} antibiotics observed<br>"
                                   "%{y:.1f}% of episodes (%{customdata[0]:,})<extra></extra>"),
                ), row=row, col=col)
                label = f"n = {episodes:,}"
                if (scope, group) in stats.index:
                    median = float(stats.loc[(scope, group), "observed_median"])
                    fig.add_vline(x=median, line={"color": MUTED, "width": g.px(1.4), "dash": "dot"}, row=row, col=col)
                    label += f" · median {median:.0f}"
                axis = (row - 1) * cols_n + col
                fig.add_annotation(
                    x=0, y=1, xref=f"x{'' if axis == 1 else axis} domain", yref=f"y{'' if axis == 1 else axis} domain",
                    xanchor="left", xshift=g.px(8), yanchor="top", showarrow=False, text=label, bgcolor="rgba(255,255,255,0.85)",
                    font={"size": FONT_VALUE, "color": MUTED, "family": FONT},
                )
            fig.update_yaxes(title={"text": f"<b>{group}</b><br>episodes (%)", "font": {"size": FONT_LABEL, "color": TEXT}}, row=row, col=1)
        fig.update_xaxes(range=[-0.6, x_max + 0.6], dtick=5, showgrid=False, zeroline=False, ticks="outside",
                         ticklen=g.px(4), tickcolor=RULE, tickfont={"size": FONT_LABEL, "color": MUTED}, linecolor=RULE,
                         linewidth=g.px(1.0), showline=True)
        fig.update_yaxes(range=[0, y_max * 1.22], showgrid=True, gridcolor=GRID, gridwidth=g.px(1.0), zeroline=False,
                         tickfont={"size": FONT_LABEL, "color": MUTED}, ticksuffix="%")
        for annotation in fig.layout.annotations:
            if annotation.text and annotation.text.startswith("<b>") and annotation.yref == "paper":
                annotation.update(font={"size": FONT_HEADING, "color": TEXT, "family": FONT}, yshift=g.px(6))
        fig.add_annotation(
            x=0.5, xref="paper", y=0, yref="paper", yanchor="top", yshift=-g.px(34), showarrow=False,
            text="Antibiotics with an interpretable result (R, S or I) per culture episode",
            font={"size": FONT_LABEL, "color": TEXT, "family": FONT},
        )
        frame(fig, g, height, "Antibiotics reported per culture episode, by specimen and site", self._PANEL_MARGIN,
              {"legend": None}, self._template)
        fig.update_layout(bargap=0.1, showlegend=False)
        return fig

    # ------------------------------------------------------- coverage by era
    _ERA_MARGIN = {"l": 240, "r": 20, "t": 146, "b": 84}

    def coverage_by_era_figure(self, coverage: pd.DataFrame, medium: str = "screen") -> go.Figure:
        if coverage.empty:
            return empty_figure("No eligible opportunities to summarise by era", self._template)
        g = Geometry.for_medium(medium, screen_width=1500, print_width=2250)
        scopes = _scope_order(coverage["site"])
        drugs = coverage.loc[coverage["antibiotic"].ne(ALL_ANTIBIOTICS)]
        overall = (
            drugs.loc[drugs["site"].eq(ALL_SITES)]
            .groupby(["antibiotic", "antibiotic_display", "aware_category"], dropna=False)[["eligible_n", "observed_n"]]
            .sum()
            .reset_index()
        )
        overall["share"] = overall["observed_n"] / overall["eligible_n"]
        overall["site"] = ALL_SITES
        rows = _drug_rows(overall, "share", pooled_label="All antibiotics")
        row_y = {antibiotic: position for position, (_, antibiotic) in enumerate(rows) if antibiotic is not None}
        eras = sorted(coverage["era"].astype(str).unique())
        era_labels = [era_display(era) for era in eras]
        plot_height = g.px(19) * len(rows)
        height = int(round(sum(self._ERA_MARGIN[side] for side in ("t", "b")) * g.margin_scale + plot_height))
        fig = make_subplots(rows=1, cols=len(scopes), shared_yaxes=True, horizontal_spacing=0.018,
                            subplot_titles=[f"<b>{site_label(scope)}</b>" for scope in scopes])
        for index, scope in enumerate(scopes, start=1):
            cells = coverage.loc[coverage["site"].eq(scope)]
            share = np.full((len(rows), len(eras)), np.nan)
            eligible = np.zeros((len(rows), len(eras)))
            for item in cells.itertuples(index=False):
                if item.antibiotic in row_y and str(item.era) in eras:
                    position, era_index = row_y[item.antibiotic], eras.index(str(item.era))
                    share[position, era_index] = item.observed_share
                    eligible[position, era_index] = item.eligible_n
            drug_row = np.array([antibiotic is not None for _, antibiotic in rows])
            absent = np.where(np.isnan(share) & drug_row[:, None], 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                z=absent, x=era_labels, y=list(range(len(rows))), colorscale=[[0, ABSENT], [1, ABSENT]], zmin=0, zmax=1,
                showscale=False, hoverinfo="skip", xgap=g.px(2), ygap=g.px(2),
            ), row=1, col=index)
            text = np.where(np.arange(len(rows))[:, None] == row_y[ALL_ANTIBIOTICS], np.vectorize(
                lambda value: "" if np.isnan(value) else f"{100 * value:.0f}")(share), "")
            fig.add_trace(go.Heatmap(
                z=share, x=era_labels, y=list(range(len(rows))), colorscale=BLUE_RAMP, zmin=0, zmax=1,
                xgap=g.px(2), ygap=g.px(2), text=text, texttemplate="%{text}",
                textfont={"size": FONT_VALUE - 1, "family": FONT},
                customdata=eligible,
                hovertemplate=f"{site_label(scope)}, %{{x}}<br>%{{z:.1%}} observed of %{{customdata:,.0f}} eligible<extra></extra>",
                showscale=index == 1,
                colorbar={
                    "orientation": "h", "x": 0.5, "xanchor": "center", "y": 0, "yref": "paper", "yanchor": "top",
                    "ypad": g.px(18), "len": 0.42, "thickness": g.px(12), "outlinewidth": 0,
                    "tickvals": [0, 0.25, 0.5, 0.75, 1], "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                    "tickfont": {"size": FONT_LABEL, "color": MUTED},
                    "title": {"text": "Observed share of eligible opportunities", "side": "top", "font": {"size": FONT_LABEL, "color": TEXT}},
                },
            ), row=1, col=index)
        for position, (_, antibiotic) in enumerate(rows):
            if antibiotic is None or antibiotic == ALL_ANTIBIOTICS:
                if position > 0:
                    fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=position - 0.5, y1=position - 0.5,
                                  line={"color": RULE, "width": g.px(1.0)})
        fig.update_yaxes(
            tickmode="array", tickvals=list(range(len(rows))), ticktext=[text for text, _ in rows],
            tickfont={"size": FONT_LABEL, "color": TEXT}, range=[len(rows) - 0.5, -0.5], showgrid=False, ticks="",
            ticklabelstandoff=round(g.px(LABEL_GAP)), zeroline=False,
        )
        fig.update_xaxes(side="top", tickfont={"size": FONT_VALUE, "color": MUTED}, tickangle=-45, showgrid=False, ticks="")
        for annotation in fig.layout.annotations:  # subplot titles
            annotation.update(font={"size": FONT_HEADING, "color": TEXT, "family": FONT}, yshift=g.px(66))
        key(fig, g, [("Outside the site's eligible space", "square", ABSENT, RULE)])
        frame(fig, g, height, "Observation coverage of the eligible opportunity space, by availability era",
              self._ERA_MARGIN, {"legend": None}, self._template)
        return fig

    # ------------------------------------------------------------ antibiogram
    _ANTIBIOGRAM_MARGIN = {"l": 240, "r": 20, "t": 142, "b": 76}

    def antibiogram_figure(self, antibiogram: pd.DataFrame, medium: str = "screen") -> go.Figure:
        if antibiogram.empty:
            return empty_figure("No observed results to summarise", self._template)
        g = Geometry.for_medium(medium, screen_width=1500, print_width=2400)
        scopes = _scope_order(antibiogram["site"])
        rows = _drug_rows(antibiogram, "resistant_share")
        row_y = {antibiotic: position for position, (_, antibiotic) in enumerate(rows) if antibiotic is not None}
        plot_height = g.px(21) * len(rows)
        height = int(round(sum(self._ANTIBIOGRAM_MARGIN[side] for side in ("t", "b")) * g.margin_scale + plot_height))
        fig = make_subplots(rows=1, cols=len(scopes), shared_yaxes=True, horizontal_spacing=0.025,
                            subplot_titles=[f"<b>{site_label(scope)}</b>" for scope in scopes])
        any_sparse = any_absent = False
        for index, scope in enumerate(scopes, start=1):
            cells = antibiogram.loc[antibiogram["site"].eq(scope)].copy()
            cells["y"] = cells["antibiotic"].map(row_y)
            fig.add_trace(go.Bar(x=[100] * len(cells), y=cells["y"], orientation="h", width=0.58, marker_color=TRACK,
                                 showlegend=False, hoverinfo="skip"), row=1, col=index)
            for sparse in (False, True):
                part = cells.loc[cells["sparse"].astype(bool).eq(sparse)]
                if part.empty:
                    continue
                any_sparse = any_sparse or sparse
                opacity = 0.42 if sparse else 1.0
                custom = part[["antibiotic_display", "resistant_n", "intermediate_n", "tested_n"]].to_numpy()
                fig.add_trace(go.Bar(
                    x=100 * part["resistant_share"], y=part["y"], orientation="h", width=0.58, opacity=opacity,
                    marker_color=RESISTANT, showlegend=False, customdata=custom,
                    hovertemplate=(f"%{{customdata[0]}}, {site_label(scope)}<br>Resistant %{{x:.1f}}% "
                                   "(%{customdata[1]:,} of %{customdata[3]:,} tested)<extra></extra>"),
                ), row=1, col=index)
                fig.add_trace(go.Bar(
                    x=100 * part["intermediate_share"], base=100 * part["resistant_share"], y=part["y"], orientation="h",
                    width=0.58, opacity=opacity, marker_color=INTERMEDIATE, showlegend=False, customdata=custom,
                    hovertemplate=(f"%{{customdata[0]}}, {site_label(scope)}<br>Intermediate %{{x:.1f}}% "
                                   "(%{customdata[2]:,} of %{customdata[3]:,} tested)<extra></extra>"),
                ), row=1, col=index)
            absent = [antibiotic for antibiotic in row_y if antibiotic not in set(cells["antibiotic"])]
            any_absent = any_absent or bool(absent)
            values = [f"{100 * share:.0f}%" for share in cells["resistant_share"]]
            colours = [MUTED if sparse else TEXT for sparse in cells["sparse"].astype(bool)]
            fig.add_trace(go.Scatter(
                x=[_VALUE_X] * (len(cells) + len(absent)), y=list(cells["y"]) + [row_y[a] for a in absent],
                mode="text", textposition="middle right", showlegend=False, hoverinfo="skip", cliponaxis=False,
                text=values + ["–"] * len(absent),
                textfont={"size": FONT_VALUE, "color": colours + [MUTED] * len(absent), "family": FONT},
            ), row=1, col=index)
            fig.update_xaxes(range=[0, 130], tickvals=[0, 50, 100], ticktext=["0", "50", "100%"], showgrid=True,
                             gridcolor=GRID, gridwidth=g.px(1.0), zeroline=False, tickfont={"size": FONT_LABEL, "color": MUTED},
                             row=1, col=index)
        for position, (_, antibiotic) in enumerate(rows):
            if antibiotic is None and position > 0:
                fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=position - 0.5, y1=position - 0.5,
                              line={"color": RULE, "width": g.px(1.0)})
        fig.update_yaxes(
            showgrid=False, ticks="", tickmode="array", tickvals=list(range(len(rows))), ticktext=[text for text, _ in rows],
            tickfont={"size": FONT_LABEL, "color": TEXT}, range=[len(rows) - 0.5, -0.5], ticklabelstandoff=round(g.px(LABEL_GAP)),
        )
        entries = [("Resistant", "square", RESISTANT, RESISTANT), ("Intermediate", "square", INTERMEDIATE, INTERMEDIATE),
                   ("Susceptible", "square", TRACK, RULE)]
        key(fig, g, entries, row=1, col=1)
        # Text-only notes on a second row: a lighter swatch would read as a fourth result category.
        notes = []
        if any_sparse:
            notes.append((f"lighter bar: fewer than {self._min_tested} tested", "square", "rgba(0,0,0,0)", "rgba(0,0,0,0)"))
        if any_absent:
            notes.append(("–  not tested at the site", "square", "rgba(0,0,0,0)", "rgba(0,0,0,0)"))
        key(fig, g, notes, legend="legend2", row=1, col=1)
        for annotation in fig.layout.annotations:
            if annotation.text and annotation.text.startswith("<b>"):
                annotation.update(font={"size": FONT_HEADING, "color": TEXT, "family": FONT}, yshift=g.px(6))
        fig.add_annotation(
            x=0.5, xref="paper", y=0, yref="paper", yanchor="top", yshift=-g.px(30), showarrow=False,
            text="Share of culture episodes tested for the antibiotic (value: % resistant)", font={"size": FONT_LABEL, "color": TEXT, "family": FONT},
        )
        frame(fig, g, height, "Resistance among tested culture episodes, by site", self._ANTIBIOGRAM_MARGIN,
              {"legend": None, "legend2": None} if notes else {"legend": None}, self._template)
        return fig
