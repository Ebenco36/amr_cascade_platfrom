"""Observed branch probabilities, with saved ER evidence alongside each pair."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from amr_cascade_platform.reporting.builders import directional_views as dv
from amr_cascade_platform.visualization.report.print_layout import Geometry, export_both_media


class PairedObservationPlotter:
    """Draw a support-auditable alternative to a support-weighted Sankey."""

    SUSCEPTIBLE = "#247b9b"
    RESISTANT = "#bc4b24"

    def __init__(self, exporter, template: str = "plotly_white") -> None:
        self._exporter = exporter
        self._template = template

    def export(self, edges: pd.DataFrame, direction: str, output_stem: Path,
               formats: tuple[str, ...], *, top_n: int = 10) -> dict[str, Path]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        rows = dv.observation_probability_rows(edges, direction)
        rows["shown_in_figure"] = rows.rank_by_absolute_difference <= top_n
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        csv_path = output_stem.with_suffix(".csv")
        rows.to_csv(csv_path, index=False)
        selected = rows.loc[rows.shown_in_figure].copy()
        outputs = export_both_media(
            self._exporter,
            lambda data, medium: self.build(data, direction, len(rows), medium=medium),
            selected, output_stem, formats,
        )
        outputs[csv_path.name] = csv_path
        return outputs

    def build(self, rows: pd.DataFrame, direction: str, n_available: int,
              *, medium: str = "screen") -> go.Figure:
        g = Geometry.for_medium(medium, screen_width=1600, print_width=3400)
        n = len(rows)
        height = int(g.px(265 + max(n, 1) * 66))
        fig = go.Figure()
        label = ("Increased observation with upstream resistance" if direction == dv.ESCALATION
                 else "Decreased observation with upstream resistance")
        fig.update_layout(
            template=self._template, width=g.width, height=height,
            margin={"l": 20, "r": 20, "t": 115, "b": 150},
            font={"family": "Arial", "size": 13, "color": "#233142"},
            paper_bgcolor="white", plot_bgcolor="white", showlegend=True,
            meta={"print_geometry": True, "geometry_scale": g.margin_scale, "heading_band": 0},
            xaxis={"domain": [0.30, 0.60], "range": [-3, 103], "tickvals": [0, 25, 50, 75, 100],
                   "ticksuffix": "%", "title": {"text": "Downstream result observed (%)"},
                   "showgrid": True, "gridcolor": "#e8edf2", "zeroline": False, "fixedrange": True},
            yaxis={"range": [max(n, 1) - 0.5, -0.5], "visible": False, "fixedrange": True},
            legend={"orientation": "h", "x": 0.30, "y": 1.07, "xanchor": "left", "yanchor": "bottom",
                    "font": {"size": 13}, "itemsizing": "trace"},
        )

        def annotation(text, x, y, *, paper_y=False, size=13, anchor="center", **kwargs):
            fig.add_annotation(text=text, x=x, y=y, xref="paper", yref="paper" if paper_y else "y",
                               showarrow=False, xanchor=anchor, align="left" if anchor == "left" else "center",
                               font={"size": size}, **kwargs)

        # These are semantic panel/column labels, not removable figure titles.
        annotation(f"<b>{label}</b>", 0, 1.17, paper_y=True, size=17, anchor="left")
        headers = [(0, "Upstream → downstream", "left"), (0.65, "S / R observed", "center"),
                   (0.76, "ER [95% CI]", "center"), (0.88, "N (S / R)", "center"),
                   (0.98, "Δ pp", "right")]
        for x, text, anchor in headers:
            annotation(f"<b>{text}</b>", x, 1.02, paper_y=True, anchor=anchor, size=12)
        if rows.empty:
            annotation("No validated pairs in this direction", 0.5, 0, size=16)

        for i, row in enumerate(rows.itertuples(index=False)):
            if i % 2 == 0:
                fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=i - 0.48, y1=i + 0.48,
                              fillcolor="#f5f7fa", line_width=0, layer="below")
            fig.add_trace(go.Scatter(
                x=[row.susceptible_observed_pct, row.resistant_observed_pct], y=[i, i],
                mode="lines", line={"color": "#a5afba", "width": g.px(2)},
                showlegend=False, hoverinfo="skip",
            ))
            pair = f"{escape(row.upstream_antibiotic.capitalize())}<br>→ {escape(row.downstream_antibiotic.capitalize())}"
            annotation(pair, 0, i, anchor="left", size=13)
            annotation(f"{row.susceptible_observed_pct:.2f}% / {row.resistant_observed_pct:.2f}%",
                       0.65, i, size=12)
            interval = (f"[{row.er_ci_lower:.3g}, {row.er_ci_upper:.3g}]"
                        if np.isfinite(row.er_ci_lower) and np.isfinite(row.er_ci_upper) else "CI unavailable")
            annotation(f"<b>{row.escalation_ratio:.3g}</b> {interval}<br>{escape(row.validation_status)}",
                       0.76, i, size=12)
            annotation(f"{int(row.susceptible_support_n):,} / {int(row.resistant_support_n):,}", 0.88, i, size=12)
            annotation(f"{row.difference_pp:+.1f}", 0.98, i, anchor="right", size=13)
        for branch, color, symbol, legend in (
            ("susceptible", self.SUSCEPTIBLE, "circle", "Upstream susceptible (S)"),
            ("resistant", self.RESISTANT, "diamond", "Upstream resistant (R)"),
        ):
            details = [f"{escape(r.upstream_antibiotic)} → {escape(r.downstream_antibiotic)}<br>"
                       f"Observed {int(getattr(r, branch + '_tested_n')):,} / "
                       f"{int(getattr(r, branch + '_support_n')):,} eligible branch episodes"
                       for r in rows.itertuples(index=False)]
            fig.add_trace(go.Scatter(
                x=rows[f"{branch}_observed_pct"], y=list(range(n)), mode="markers", name=legend,
                marker={"color": color, "symbol": symbol, "size": g.px(9),
                        "line": {"width": g.px(1), "color": "white"}},
                customdata=details, hovertemplate="%{customdata}<br>Observation: %{x:.2f}%<extra>%{fullData.name}</extra>",
            ))
        annotation(
            f"Showing {n} of {n_available} validated pairs, ranked by |R − S|; no ER magnitude filter.<br>"
            "Dots: raw observed proportions. ER: saved continuity-corrected ratio and 95% CI. "
            "N: eligible branch episodes. Δ: R − S, percentage points.",
            0, -0.10, paper_y=True, size=11, anchor="left", yanchor="top",
        )
        return fig
