"""Observation coverage of the eligible opportunity space.

An eligible opportunity is one culture episode and one antibiotic that is
biologically interpretable for the organism and operationally available at that
site in that era (``is_eligible`` in the gold eligibility table). It is observed
when the episode carries an interpretive AST result (S, I or R) for the drug.
The share observed is what tested-only summaries rest on, and one minus it is
the width of the model-free prevalence bounds.

Two layouts of the same table, both on one 0-100% scale, drugs grouped by WHO
AWaRe category and ordered by the all-sites share:

* ``figure_observation_coverage`` -- two panels: (A) the share observed across
  all antibiotics, for all sites and each site; (B) per antibiotic, a light
  track for the drug's eligible space, a dark fill for the all-sites share and
  one marker per site for the site's own share.
* ``figure_observation_coverage_by_site`` -- small multiples: one column per
  scope (all sites, then each site), one row per antibiotic after an
  all-antibiotics total row, each cell a track, a fill and its percentage.
  Identity comes from the column heading, so no colour key is needed.

A site with fewer than ``MIN_SITE_SUPPORT`` eligible opportunities for a drug is
marked (open marker, lighter fill); a drug outside a site's eligible space has
no track there. Site colours are categorical slots 1-3 of the validated
reference palette (all-pairs colour-vision checks pass) and every site also has
its own marker shape, so identity never rests on colour alone.

Each layout is built for two media from one specification. The interactive
HTML uses screen geometry; the static files use print geometry, because the
house exporter enlarges every font and margin by ``PRINT_FONT_SCALE`` for PNG,
SVG and PDF but not the canvas, marker sizes, line widths or pixel offsets.
Margins are sized for the labels at the base font, so they still fit once the
exporter scales labels and margins together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amr_cascade_platform.core.utils.site_labels import ALL_SITES, SITE_DISPLAY
from amr_cascade_platform.visualization.report.plotly_exporter import PRINT_FONT_SCALE

SITE_STYLE = {
    "armd": ("#2a78d6", "circle"),
    "armd_ecuh": ("#eb6834", "square"),
    "armd_utsw": ("#1baf7a", "diamond"),
}
_FALLBACK_SHAPES = ("triangle-up", "triangle-down", "cross", "x")
AWARE_ORDER = ("Access", "Watch", "Reserve", "Not Set", "Unclassified")
AWARE_DISPLAY = {"Not Set": "Not classified", "Unclassified": "Not classified"}
MIN_SITE_SUPPORT = 20
TITLE = "Observation coverage of the eligible opportunity space"
X_TITLE = "Observed share of eligible episode-drug opportunities"

_INK = "#2f3b4c"
_INK_SPARSE = "#a3acb9"
_TRACK = "#e4e7ec"
_TEXT = "#1a1a2e"
_MUTED = "#5b6472"
_GRID = "#eef1f5"
_RULE = "#c9ced6"
_FONT = "Arial"
_TOTAL = "__all_antibiotics__"

# Sized so that labels print at about 7 pt once the figure is placed at full width or full page height.
_FONT_TITLE, _FONT_HEADING, _FONT_LABEL, _FONT_VALUE, _FONT_LEGEND = 20, 15, 14, 13, 14
# Screen-pixel gaps: row label to axis, track end (or a marker drawn at 100%) to its value, legend placement.
_LABEL_GAP, _VALUE_GAP, _LEGEND_TOP, _LEGEND_ROW = 10, 14, 50, 30


@dataclass(frozen=True)
class _Geometry:
    """Pixel sizes the exporter does not scale: canvas, pitches, marks, offsets."""

    width: int
    unit: float  # multiplier for every other pixel size below
    margin_scale: float  # what the exporter multiplies margins by

    @classmethod
    def for_medium(cls, medium: str, screen_width: int, print_width: int) -> _Geometry:
        if medium == "screen":
            return cls(width=screen_width, unit=1.0, margin_scale=1.0)
        if medium == "print":
            return cls(width=print_width, unit=PRINT_FONT_SCALE, margin_scale=PRINT_FONT_SCALE)
        raise ValueError(f"medium must be 'screen' or 'print', not {medium!r}")

    def px(self, value: float) -> float:
        return value * self.unit


def _sentence_case(label: str) -> str:
    """"AMOXICILLIN/CLAVULANIC ACID" -> "Amoxicillin/clavulanic acid"."""
    text = str(label).strip().lower()
    return text[:1].upper() + text[1:]


def observation_coverage_table(eligible_pairs: pd.DataFrame, resolver, sites: tuple[str, ...] = ()) -> pd.DataFrame:
    """Eligible and observed episode-drug opportunities per antibiotic, by site and for all sites.

    Returns one row per antibiotic and scope (``all_sites`` first, then each
    site in ``sites`` order), ordered as the figures draw them: WHO AWaRe
    category, then all-sites observed share (highest first).
    """
    required = {"antibiotic", "source_site", "is_eligible", "is_observed_tested"}
    missing = sorted(required - set(eligible_pairs.columns))
    if missing:
        raise ValueError(f"eligible_pairs lacks {missing}")
    columns = [
        "aware_category", "antibiotic_display", "antibiotic", "site", "site_display",
        "eligible_n", "observed_n", "observed_share",
    ]
    eligible = eligible_pairs.loc[
        pd.to_numeric(eligible_pairs["is_eligible"], errors="coerce").eq(1),
        ["antibiotic", "source_site", "is_observed_tested"],
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    eligible["antibiotic"] = eligible["antibiotic"].astype(str)
    eligible["site"] = eligible["source_site"].astype(str)
    eligible["observed"] = pd.to_numeric(eligible["is_observed_tested"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    per_site = (
        eligible.groupby(["antibiotic", "site"], observed=True)["observed"]
        .agg(eligible_n="size", observed_n="sum")
        .reset_index()
    )
    combined = per_site.groupby("antibiotic", as_index=False)[["eligible_n", "observed_n"]].sum().assign(site=ALL_SITES)
    table = pd.concat([combined, per_site], ignore_index=True)
    table["observed_share"] = table["observed_n"] / table["eligible_n"]

    resolved = {label: resolver.resolve(label) for label in table["antibiotic"].unique()}
    table["antibiotic_display"] = table["antibiotic"].map(lambda label: _sentence_case(resolved[label].display_label))
    table["aware_category"] = table["antibiotic"].map(lambda label: resolved[label].aware_category)
    table["site_display"] = table["site"].map(lambda site: SITE_DISPLAY.get(site, site))

    site_order = [ALL_SITES, *sites, *sorted(set(table["site"]) - {ALL_SITES, *sites})]
    combined_share = table.loc[table["site"].eq(ALL_SITES)].set_index("antibiotic")["observed_share"]
    aware_rank = {name: rank for rank, name in enumerate(AWARE_ORDER)}
    table["_aware_rank"] = table["aware_category"].map(aware_rank).fillna(len(AWARE_ORDER))
    table["_drug_share"] = table["antibiotic"].map(combined_share)
    table["_site_rank"] = table["site"].map({site: rank for rank, site in enumerate(site_order)})
    table = table.sort_values(
        ["_aware_rank", "_drug_share", "antibiotic_display", "_site_rank"],
        ascending=[True, False, True, True],
    )
    return table.loc[:, columns].reset_index(drop=True)


def site_totals(coverage: pd.DataFrame) -> pd.DataFrame:
    """Eligible and observed opportunities over all antibiotics, per scope."""
    totals = coverage.groupby(["site", "site_display"], as_index=False, sort=False)[["eligible_n", "observed_n"]].sum()
    totals["observed_share"] = totals["observed_n"] / totals["eligible_n"]
    return totals


def _drug_rows(coverage: pd.DataFrame) -> list[tuple[str, str | None]]:
    """(tick text, antibiotic) per row; a ``None`` antibiotic is an AWaRe group heading."""
    drugs = coverage.loc[coverage["site"].eq(ALL_SITES)]
    rows: list[tuple[str, str | None]] = []
    for category, group in drugs.groupby("aware_category", sort=False):
        rows.append((f"<b>{AWARE_DISPLAY.get(category, category)}</b>", None))
        rows.extend(zip(group["antibiotic_display"], group["antibiotic"], strict=True))
    return rows


def _scopes(coverage: pd.DataFrame) -> tuple[list[str], bool]:
    sites = [site for site in dict.fromkeys(coverage["site"]) if site != ALL_SITES]
    return sites, len(sites) > 1


class PlotlyObservationCoveragePlotter:
    """Both observation-coverage layouts (see module docstring)."""

    def __init__(self, exporter, template: str, width: int | None = None) -> None:
        self._exporter = exporter
        self._template = template

    # -------------------------------------------------------------- exports
    def export(self, coverage: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        """Two-panel layout (A totals, B per antibiotic with site markers)."""
        return self._export(self.figure, coverage, output_stem, formats)

    def export_by_site(self, coverage: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        """Small multiples: one column per scope."""
        return self._export(self.small_multiples_figure, coverage, output_stem, formats)

    def _export(self, build, coverage: pd.DataFrame, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        screen = tuple(fmt for fmt in formats if fmt == "html")
        static = tuple(fmt for fmt in formats if fmt != "html")
        if screen:
            outputs.update(self._exporter.write(build(coverage, medium="screen"), output_stem, screen))
        if static:
            outputs.update(self._exporter.write(build(coverage, medium="print"), output_stem, static))
        return outputs

    # -------------------------------------------------------- two-panel layout
    _MARGIN = {"l": 240, "r": 160, "t": 140, "b": 76}

    def figure(self, coverage: pd.DataFrame, medium: str = "screen") -> go.Figure:
        if coverage.empty:
            return self._empty("No eligible episode-drug opportunities to summarise")
        g = _Geometry.for_medium(medium, screen_width=1400, print_width=2000)
        sites, show_sites = _scopes(coverage)
        styles = self._site_styles(sites)
        rows = _drug_rows(coverage)
        row_y = {antibiotic: position for position, (_, antibiotic) in enumerate(rows) if antibiotic is not None}
        drugs = coverage.loc[coverage["site"].eq(ALL_SITES)].reset_index(drop=True)

        scopes = ([ALL_SITES] if show_sites else []) + sites
        height_a, gap, height_b = g.px(27) * len(scopes), g.px(52), g.px(19) * len(rows)
        plot_height = height_a + gap + height_b
        height = int(round((self._MARGIN["t"] + self._MARGIN["b"]) * g.margin_scale + plot_height))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=gap / plot_height, row_heights=[height_a, height_b])

        # Panel A: all antibiotics, per scope; a site's bar wears its colour.
        totals = site_totals(coverage).set_index("site").loc[scopes].reset_index()
        labels = [f"<b>{SITE_DISPLAY[s]}</b>" if s == ALL_SITES else SITE_DISPLAY.get(s, s) for s in totals["site"]]
        fig.add_trace(go.Bar(x=[100] * len(totals), y=labels, orientation="h", width=0.62, marker_color=_TRACK,
                             hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(
            x=100 * totals["observed_share"], y=labels, orientation="h", width=0.62, showlegend=False,
            marker_color=[_INK if s == ALL_SITES else styles[s][0] for s in totals["site"]],
            customdata=totals[["observed_n", "eligible_n"]].to_numpy(),
            hovertemplate="%{y}<br>%{x:.1f}% observed<br>%{customdata[0]:,} of %{customdata[1]:,} eligible<extra></extra>",
        ), row=1, col=1)
        for label, row in zip(labels, totals.itertuples(index=False), strict=True):
            fig.add_annotation(
                x=1, xref="x domain", y=label, yref="y", xanchor="left", xshift=g.px(_VALUE_GAP), showarrow=False,
                text=f"<b>{row.observed_share:.1%}</b>  <span style='color:{_MUTED}'>n = {row.eligible_n:,}</span>",
                font={"size": _FONT_VALUE, "color": _TEXT, "family": _FONT},
            )

        # Panel B: per antibiotic -- track, all-sites fill, site markers.
        ys = [row_y[a] for a in drugs["antibiotic"]]
        fig.add_trace(go.Bar(x=[100] * len(drugs), y=ys, orientation="h", width=0.56, marker_color=_TRACK,
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Bar(
            x=100 * drugs["observed_share"], y=ys, orientation="h", width=0.56, marker_color=_INK,
            showlegend=False, name="Observed, all sites",
            customdata=drugs[["antibiotic_display", "observed_n", "eligible_n"]].to_numpy(),
            hovertemplate="%{customdata[0]}, all sites<br>%{x:.1f}% observed<br>%{customdata[1]:,} of %{customdata[2]:,}<extra></extra>",
        ), row=2, col=1)
        any_sparse = False
        if show_sites:
            for site in sites:
                colour, shape = styles[site]
                subset = coverage.loc[coverage["site"].eq(site)]
                sparse = subset["eligible_n"].lt(MIN_SITE_SUPPORT).to_list()
                any_sparse = any_sparse or any(sparse)
                name = SITE_DISPLAY.get(site, site)
                fig.add_trace(go.Scatter(
                    x=100 * subset["observed_share"], y=[row_y[a] for a in subset["antibiotic"]],
                    mode="markers", name=name, showlegend=False, cliponaxis=False,
                    marker={
                        "symbol": [f"{shape}-open" if flag else shape for flag in sparse],
                        "size": g.px(10), "color": colour,
                        "line": {"color": [colour if flag else "#ffffff" for flag in sparse],
                                 "width": [g.px(2.0) if flag else g.px(1.4) for flag in sparse]},
                    },
                    customdata=subset[["antibiotic_display", "observed_n", "eligible_n"]].to_numpy(),
                    hovertemplate=f"%{{customdata[0]}}, {name}<br>%{{x:.1f}}% observed<br>%{{customdata[1]:,}} of %{{customdata[2]:,}}<extra></extra>",
                ), row=2, col=1)
        key = [("Observed, all sites", "square", _INK, _INK), ("Eligible, not observed", "square", _TRACK, _RULE)]
        if any_sparse:
            key.append((f"Fewer than {MIN_SITE_SUPPORT} eligible at a site", "circle-open", _MUTED, _MUTED))
        self._key(fig, g, key, row=2, col=1)
        legends: dict[str, str | None] = {"legend": None}
        if show_sites:
            # Outline in the site colour: the white ring that separates a marker from the bar would
            # swallow most of a legend symbol, which Plotly caps at 16 px.
            site_key = [(SITE_DISPLAY.get(site, site), styles[site][1], styles[site][0], styles[site][0]) for site in sites]
            self._key(fig, g, site_key, legend="legend2", row=2, col=1)
            legends["legend2"] = "Observed share at each site:"
        for row in drugs.itertuples(index=False):
            fig.add_annotation(
                x=1, xref="x2 domain", y=row_y[row.antibiotic], yref="y2", xanchor="left", xshift=g.px(_VALUE_GAP),
                showarrow=False,
                text=f"<b>{row.observed_share:.0%}</b>  <span style='color:{_MUTED}'>n = {row.eligible_n:,}</span>",
                font={"size": _FONT_VALUE, "color": _TEXT, "family": _FONT},
            )
        for position, (_, antibiotic) in enumerate(rows):
            if antibiotic is None and position > 0:
                fig.add_shape(type="line", xref="x2 domain", yref="y2", x0=0, x1=1, y0=position - 0.5, y1=position - 0.5,
                              line={"color": _RULE, "width": g.px(1.0)})

        fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont={"size": _FONT_LABEL, "color": _TEXT},
                         ticklabelstandoff=round(g.px(_LABEL_GAP)), row=1, col=1)
        fig.update_yaxes(showgrid=False, ticks="", tickmode="array", tickvals=list(range(len(rows))),
                         ticklabelstandoff=round(g.px(_LABEL_GAP)),
                         ticktext=[text for text, _ in rows], tickfont={"size": _FONT_LABEL, "color": _TEXT},
                         range=[len(rows) - 0.5, -0.5], row=2, col=1)
        fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                         showgrid=True, gridcolor=_GRID, gridwidth=g.px(1.0), zeroline=False,
                         tickfont={"size": _FONT_LABEL, "color": _MUTED})
        fig.update_xaxes(title={"text": X_TITLE, "font": {"size": _FONT_LABEL, "color": _TEXT}}, row=2, col=1)
        for text, yref in (("<b>A</b>  All antibiotics", "y domain"),
                           ("<b>B</b>  By antibiotic, grouped by WHO AWaRe category", "y2 domain")):
            fig.add_annotation(x=0, xref="paper", y=1.0, yref=yref, yanchor="bottom", xanchor="left", yshift=g.px(8),
                               showarrow=False, text=text, font={"size": _FONT_HEADING, "color": _TEXT, "family": _FONT})
        self._frame(fig, g, height, TITLE, self._MARGIN, legends)
        return fig

    # ----------------------------------------------------- small multiples
    _FACET_MARGIN = {"l": 240, "r": 20, "t": 118, "b": 76}
    _VALUE_X = 104  # data units; the axis runs to 130 so every cell's value fits beside its track

    def small_multiples_figure(self, coverage: pd.DataFrame, medium: str = "screen") -> go.Figure:
        if coverage.empty:
            return self._empty("No eligible episode-drug opportunities to summarise")
        g = _Geometry.for_medium(medium, screen_width=1500, print_width=2400)
        sites, show_sites = _scopes(coverage)
        columns = ([ALL_SITES] if show_sites else []) + sites
        rows = [("<b>All antibiotics</b>", _TOTAL), *_drug_rows(coverage)]
        row_y = {key: position for position, (_, key) in enumerate(rows) if key is not None}
        plot_height = g.px(21) * len(rows)
        height = int(round((self._FACET_MARGIN["t"] + self._FACET_MARGIN["b"]) * g.margin_scale + plot_height))
        fig = make_subplots(
            rows=1, cols=len(columns), shared_yaxes=True, horizontal_spacing=0.025,
            subplot_titles=[f"<b>{SITE_DISPLAY.get(column, column)}</b>" for column in columns],
        )
        totals = site_totals(coverage).set_index("site")
        any_sparse = any_absent = False
        for index, column in enumerate(columns, start=1):
            cells = coverage.loc[coverage["site"].eq(column), ["antibiotic", "antibiotic_display", "eligible_n", "observed_n", "observed_share"]]
            total = totals.loc[column]
            frame = pd.concat(
                [
                    pd.DataFrame([{"antibiotic": _TOTAL, "antibiotic_display": "All antibiotics", "eligible_n": int(total["eligible_n"]),
                                   "observed_n": int(total["observed_n"]), "observed_share": float(total["observed_share"])}]),
                    cells,
                ],
                ignore_index=True,
            )
            frame["y"] = frame["antibiotic"].map(row_y)
            sparse = frame["eligible_n"].lt(MIN_SITE_SUPPORT) & frame["antibiotic"].ne(_TOTAL)
            any_sparse = any_sparse or bool(sparse.any())
            fig.add_trace(go.Bar(
                x=[100] * len(frame), y=frame["y"], orientation="h", width=0.58, marker_color=_TRACK,
                showlegend=False, hoverinfo="skip",
            ), row=1, col=index)
            for is_sparse, colour, name in (
                (False, _INK, "Observed"),
                (True, _INK_SPARSE, f"Observed, fewer than {MIN_SITE_SUPPORT} eligible at the site"),
            ):
                part = frame.loc[sparse.eq(is_sparse)]
                if part.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=100 * part["observed_share"], y=part["y"], orientation="h", width=0.58, marker_color=colour,
                    name=name, showlegend=False,
                    customdata=part[["antibiotic_display", "observed_n", "eligible_n"]].to_numpy(),
                    hovertemplate=(f"%{{customdata[0]}}, {SITE_DISPLAY.get(column, column)}"
                                   "<br>%{x:.1f}% observed<br>%{customdata[1]:,} of %{customdata[2]:,}<extra></extra>"),
                ), row=1, col=index)
            # The cell's value beside its track; a dash where the drug is outside this scope's eligible space.
            absent = [key for key in row_y if key not in set(frame["antibiotic"])]
            any_absent = any_absent or bool(absent)
            fig.add_trace(go.Scatter(
                x=[self._VALUE_X] * (len(frame) + len(absent)),
                y=list(frame["y"]) + [row_y[key] for key in absent],
                mode="text", textposition="middle right", showlegend=False, hoverinfo="skip", cliponaxis=False,
                text=[f"<b>{v:.0%}</b>" if key == _TOTAL else f"{v:.0%}" for key, v in zip(frame["antibiotic"], frame["observed_share"], strict=True)]
                + ["–"] * len(absent),
                textfont={"size": _FONT_VALUE, "color": _MUTED, "family": _FONT},
            ), row=1, col=index)
            fig.update_xaxes(
                range=[0, 130], tickvals=[0, 50, 100], ticktext=["0", "50", "100%"], showgrid=True, gridcolor=_GRID,
                gridwidth=g.px(1.0), zeroline=False, tickfont={"size": _FONT_LABEL, "color": _MUTED}, row=1, col=index,
            )
        # Group rules run across every column: the all-antibiotics total, then each AWaRe block.
        for position, (_, key) in enumerate(rows):
            if position > 0 and (key is None or position == 1):
                fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=position - 0.5, y1=position - 0.5,
                              line={"color": _RULE, "width": g.px(1.0)})
        fig.update_yaxes(
            showgrid=False, ticks="", tickmode="array", tickvals=list(range(len(rows))), ticktext=[text for text, _ in rows],
            tickfont={"size": _FONT_LABEL, "color": _TEXT}, range=[len(rows) - 0.5, -0.5], ticklabelstandoff=round(g.px(_LABEL_GAP)),
        )
        key = [("Observed", "square", _INK, _INK)]
        if any_sparse:
            key.append((f"Observed, fewer than {MIN_SITE_SUPPORT} eligible at the site", "square", _INK_SPARSE, _INK_SPARSE))
        key.append(("Eligible, not observed", "square", _TRACK, _RULE))
        if any_absent:
            key.append(("–  not in the site's eligible space", "square", "rgba(0,0,0,0)", "rgba(0,0,0,0)"))
        self._key(fig, g, key, row=1, col=1)
        for annotation in fig.layout.annotations:  # the subplot titles
            annotation.update(font={"size": _FONT_HEADING, "color": _TEXT, "family": _FONT}, yshift=g.px(6))
        fig.add_annotation(
            x=0.5, xref="paper", y=0, yref="paper", yanchor="top", yshift=-g.px(30), showarrow=False,
            text=X_TITLE, font={"size": _FONT_LABEL, "color": _TEXT, "family": _FONT},
        )
        self._frame(fig, g, height, f"{TITLE}, by site", self._FACET_MARGIN, {"legend": None})
        return fig

    # ----------------------------------------------------------- shared frame
    def _frame(self, fig: go.Figure, g: _Geometry, height: int, title: str, margin: dict, legends: dict[str, str | None]) -> None:
        """Shared layout; ``legends`` maps each legend id to its row title (or ``None``), top row first."""
        rows = {}
        for index, (name, heading) in enumerate(legends.items()):
            rows[name] = {
                "orientation": "h", "x": g.px(16) / g.width, "xref": "container", "xanchor": "left",
                "y": 1 - g.px(_LEGEND_TOP + index * _LEGEND_ROW) / height, "yref": "container", "yanchor": "top",
                "font": {"size": _FONT_LEGEND, "color": _TEXT}, "traceorder": "normal", "itemsizing": "trace",
                "itemwidth": max(30, int(g.px(22))),
            }
            if heading:
                rows[name]["title"] = {"text": heading, "side": "left", "font": {"size": _FONT_LEGEND, "color": _MUTED}}
        fig.update_layout(**rows)
        fig.update_layout(
            template=self._template, width=g.width, height=height, barmode="overlay",
            plot_bgcolor="white", paper_bgcolor="white", font={"family": _FONT, "size": _FONT_LABEL, "color": _TEXT},
            title={"text": title, "x": g.px(16) / g.width, "xref": "container", "xanchor": "left",
                   "y": 1 - g.px(14) / height, "yref": "container", "yanchor": "top",
                   "font": {"size": _FONT_TITLE, "color": _TEXT}},
            margin=dict(margin),
            hoverlabel={"font": {"family": _FONT}},
            meta={"print_geometry": True},  # laid out for print already: the exporter only scales its text
        )

    @staticmethod
    def _key(fig: go.Figure, g: _Geometry, entries: list[tuple[str, str, str, str]], legend: str = "legend", **position) -> None:
        """Legend-only entries (name, symbol, fill, outline) drawn at the medium's marker size.

        Plotly sizes bar swatches and constant-size legend symbols in pixels the
        print exporter does not scale, so the key is built from marker traces
        sized for the medium and every data trace stays out of the legend.
        """
        for rank, (name, symbol, fill, outline) in enumerate(entries, start=1):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name, legend=legend, legendrank=rank, hoverinfo="skip",
                marker={"symbol": symbol, "size": g.px(13), "color": fill, "line": {"color": outline, "width": g.px(1.4)}},
            ), **position)

    @staticmethod
    def _site_styles(sites: list[str]) -> dict[str, tuple[str, str]]:
        styles: dict[str, tuple[str, str]] = {}
        extra = iter(_FALLBACK_SHAPES)
        for site in sites:
            styles[site] = SITE_STYLE.get(site) or (_MUTED, next(extra, "circle"))
        return styles

    def _empty(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           font={"size": 14, "color": _MUTED})
        fig.update_layout(template=self._template, width=1400, height=400, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis={"visible": False}, yaxis={"visible": False})
        return fig
