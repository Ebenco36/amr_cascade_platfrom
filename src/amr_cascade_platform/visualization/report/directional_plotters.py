"""Direction-pure cascade figures: one figure per direction, never both.

Escalation (ER > 1) is always drawn in blue, suppression (ER < 1) in purple, in
every figure family, so a reader can tell the direction from colour alone even
if a panel is cropped out of context. Both directions of a family share drug
order, node positions and scales so the two figures can be compared by eye.
Every quantity drawn is taken verbatim from `reporting.builders.directional_views`.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amr_cascade_platform.reporting.builders import directional_views as dv
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter

DIRECTION_STYLE: dict[str, dict[str, str]] = {
    dv.ESCALATION: {
        "title": "Escalation", "rule": "ER > 1", "hue": "#1F4E9C", "scale": "Blues",
        "meaning": "upstream resistance is followed by more downstream observation",
    },
    dv.SUPPRESSION: {
        "title": "Suppression", "rule": "ER < 1", "hue": "#6A3D9A", "scale": "Purples",
        "meaning": "upstream resistance is followed by less downstream observation",
    },
}
TIER_TITLE = {"validated": "robust + supported", "robust": "robust only"}


def fold_label(strength_log2: float, direction: str) -> str:
    """Human-readable ER for a |log2 ER| value: 2, 4, 8 (escalation) or 1/2, 1/4 (suppression)."""
    fold = 2.0 ** strength_log2
    text = f"{fold:g}" if fold < 1000 else f"{fold:,.0f}"
    return text if direction == dv.ESCALATION else f"1/{text}"


def selection_note(direction: str, n_shown: int, n_eligible: int, n_direction: int, min_fold: float) -> str:
    rule = f"ER {'≥' if direction == dv.ESCALATION else '≤'} {min_fold:g}{'' if direction == dv.ESCALATION else '⁻¹'}" if min_fold > 1 else "all effect sizes"
    if direction == dv.SUPPRESSION and min_fold > 1:
        rule = f"ER ≤ {1 / min_fold:.2g}"
    return f"showing {n_shown} of {n_eligible} patterns with {rule} ({n_direction} validated in this direction)"


class DirectionalSankeyPlotter:
    """Bipartite flow: upstream drug (left) -> downstream drug (right), one direction per figure."""

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, classification: AntibioticClassificationResolver) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._classification = classification

    def export(
        self,
        links: pd.DataFrame,
        direction: str,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        order: list[str],
        tier: str,
        coverage: dict[str, float],
        n_eligible: int,
        min_fold: float,
    ) -> dict[str, Path]:
        style = DIRECTION_STYLE[direction]
        if links.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No {tier} {style['title'].lower()} patterns to draw", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            fig.update_layout(template=self._template, width=self._width, height=400, xaxis={"visible": False}, yaxis={"visible": False})
            return self._exporter.write(fig, output_stem, formats)

        rank_of = {drug: i for i, drug in enumerate(order)}
        left = sorted(set(links.upstream_antibiotic), key=rank_of.get)
        right = sorted(set(links.downstream_antibiotic), key=rank_of.get)
        left_index = {drug: i for i, drug in enumerate(left)}
        right_index = {drug: len(left) + i for i, drug in enumerate(right)}
        labels = list(left) + list(right)
        node_colors = [self._classification.resolve(drug).color for drug in labels]
        aware = [self._classification.resolve(drug).aware_category for drug in labels]

        link_colors, hovers = [], []
        for row in links.itertuples(index=False):
            base = self._classification.resolve(row.upstream_antibiotic).color
            link_colors.append(AntibioticClassificationResolver.with_alpha_rgba(base, 0.55 if row.validation_status == "robust" else 0.25))
            ci = (
                f"{row.er_ci_lower:.2f}–{row.er_ci_upper:.2f}" if pd.notna(row.er_ci_lower) and pd.notna(row.er_ci_upper) else "not estimable"
            )
            adj = f"{row.adjusted_odds_ratio:.2f}" if pd.notna(row.adjusted_odds_ratio) else "not estimable"
            hovers.append(
                f"<b>{row.upstream_antibiotic} → {row.downstream_antibiotic}</b><br>"
                f"ER {row.escalation_ratio:.3g} (95% CI {ci})<br>"
                f"Episode–pair rows: {int(row.support_n):,}"
                f" (resistant {int(row.resistant_support_n) if pd.notna(row.resistant_support_n) else 'n/a'}, "
                f"susceptible {int(row.susceptible_support_n) if pd.notna(row.susceptible_support_n) else 'n/a'})<br>"
                f"Adjusted OR {adj}<br>Validation: {row.validation_status}<extra></extra>"
            )

        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            textfont={"size": 10},  # node-label font, independent of the layout font the title/legend use
            node={
                "label": labels, "color": node_colors, "pad": 26, "thickness": 14,
                "line": {"color": "#22313F", "width": 0.6},
                "customdata": aware, "hovertemplate": "<b>%{label}</b><br>AWaRe %{customdata}<extra></extra>",
            },
            link={
                "source": [left_index[d] for d in links.upstream_antibiotic],
                "target": [right_index[d] for d in links.downstream_antibiotic],
                "value": links.support_n.tolist(),
                "color": link_colors, "hovertemplate": "%{customdata}", "customdata": hovers,
            },
        ))
        for name, color in (("Access", "Access"), ("Watch", "Watch"), ("Reserve", "Reserve")):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{name} (AWaRe)", showlegend=True,
                                     marker={"size": 11, "color": self._classification.CATEGORY_COLORS[color]}))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="darker flow = robust, lighter = supported", showlegend=True,
                                 marker={"size": 11, "color": "rgba(90,90,90,0.55)", "symbol": "square"}))
        height = max(900, 55 * max(len(left), len(right)) + 260)  # 30->55 px/node: PRINT_FONT_SCALE enlarges node-label text, so the band each node gets must grow too or adjacent labels overlap
        fig.update_layout(
            template=self._template, width=self._width, height=height,
            title={
                "text": (
                    f"<b style='color:{style['hue']}'>{style['title']} patterns ({style['rule']})</b> — {TIER_TITLE[tier]}, upstream → downstream"
                    f"<br><sub>{selection_note(direction, coverage['n_shown'], n_eligible, coverage['n_total'], min_fold)}; "
                    f"together {coverage['support_fraction']:.0%} of this direction's pair-support. "
                    f"Link width = that pair's own eligible episode–pair rows (never summed across pairs).</sub>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 13},
            },
            xaxis={"visible": False}, yaxis={"visible": False},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.02},
            margin={"l": 30, "r": 30, "t": 200, "b": 50}, font={"size": 13, "family": "Arial"},
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig.add_annotation(text="<b>Upstream drug (resistant result)</b>", x=0.0, y=1.14, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"size": 12, "color": "#475467"})
        fig.add_annotation(text="<b>Downstream drug (observed?)</b>", x=1.0, y=1.14, xref="paper", yref="paper", showarrow=False, xanchor="right", font={"size": 12, "color": "#475467"})
        return self._exporter.write(fig, output_stem, formats)


class DirectionalMatrixPlotter:
    """Upstream x downstream matrix of one direction on the shared drug grid."""

    def __init__(self, exporter: PlotlyFigureExporter, template: str, classification: AntibioticClassificationResolver) -> None:
        self._exporter = exporter
        self._template = template
        self._classification = classification

    def export(
        self,
        cells: pd.DataFrame,
        direction: str,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        order: list[str],
        tier: str,
        n_direction: int,
    ) -> dict[str, Path]:
        style = DIRECTION_STYLE[direction]
        n = len(order)
        axis_map = self._classification.abbreviation_axis_label_map(order)
        tick_labels = [axis_map[drug] for drug in order]
        z = np.full((n, n), np.nan)
        hover = np.full((n, n), "", dtype=object)
        for row in cells.itertuples(index=False):
            z[row.row, row.col] = row.strength
            ci = f"{row.er_ci_lower:.2f}–{row.er_ci_upper:.2f}" if pd.notna(row.er_ci_lower) and pd.notna(row.er_ci_upper) else "not estimable"
            hover[row.row, row.col] = (
                f"<b>{row.upstream_antibiotic} → {row.downstream_antibiotic}</b><br>ER {row.escalation_ratio:.3g} (95% CI {ci})<br>"
                f"Episode–pair rows {int(row.total_support_n):,}<br>Validation: {row.validation_status}"
            )
        max_strength = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
        tick_values = [v for v in range(0, int(math.ceil(max_strength)) + 1) if v <= max_strength + 0.5]
        step = max(1, len(tick_values) // 6)
        tick_values = tick_values[::step]
        fig = go.Figure(go.Heatmap(
            z=z, x=tick_labels, y=tick_labels, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            colorscale=style["scale"], zmin=0, zmax=max(max_strength, 1.0), hoverongaps=False, xgap=1, ygap=1,
            colorbar={"title": {"text": f"ER<br>({style['rule']})"}, "tickvals": tick_values,
                      "ticktext": [fold_label(v, direction) for v in tick_values], "x": 1.02, "len": 0.6},
        ))
        for status, symbol, size, color, label in (
            ("robust", "diamond", 6, "#111827", "robust"), ("supported", "circle-open", 6, "#111827", "supported"),
        ):
            subset = cells.loc[cells.validation_status == status]
            fig.add_trace(go.Scatter(
                x=[tick_labels[c] for c in subset.col], y=[tick_labels[r] for r in subset.row], mode="markers",
                marker={"symbol": symbol, "size": size, "color": color, "line": {"width": 1.2, "color": color}},
                name=label, hoverinfo="skip", showlegend=True,
            ))
        category = [self._classification.resolve(drug).aware_category for drug in order]
        boundaries, midpoints, start = [], [], 0
        for tier_name in self._classification.CATEGORY_ORDER:
            members = [i for i, c in enumerate(category) if c == tier_name]
            if not members:
                continue
            midpoints.append((tier_name, (members[0] + members[-1]) / 2))
            if members[0] > 0:
                boundaries.append(members[0] - 0.5)
        for b in boundaries:
            fig.add_shape(type="line", x0=b, x1=b, y0=-0.5, y1=n - 0.5, line={"color": "#1E293B", "width": 1})
            fig.add_shape(type="line", x0=-0.5, x1=n - 0.5, y0=b, y1=b, line={"color": "#1E293B", "width": 1})
        for tier_name, mid in midpoints:
            color = self._classification.CATEGORY_COLORS.get(tier_name, "#94A3B8")
            fig.add_annotation(x=mid, y=1.0, xref="x", yref="paper", text=f"<b>{tier_name}</b>", showarrow=False, font={"size": 8, "color": color}, yanchor="bottom", yshift=4)
        side = max(1100, 24 * n + 700)  # 380->700: the now-larger (auto-scaled) margins need more canvas left over for the matrix itself
        fig.update_layout(
            template=self._template, width=side, height=side,
            title={
                "text": (
                    f"<b style='color:{style['hue']}'>{style['title']} matrix ({style['rule']})</b> — {TIER_TITLE[tier]}: {len(cells)} of {n_direction} validated patterns"
                    f"<br><sub>Same drug order and grid as the {'suppression' if direction == dv.ESCALATION else 'escalation'} matrix. Colour = |log2 ER| within this direction; "
                    f"◆ robust, ○ supported. Blank = no validated pattern in this direction.</sub>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 13}, "y": 0.99,
            },
            xaxis={"tickangle": 90, "tickfont": {"size": 12}, "side": "bottom", "constrain": "domain", "title": {"text": "Downstream drug"}},
            yaxis={"tickfont": {"size": 12}, "autorange": "reversed", "scaleanchor": "x", "title": {"text": "Upstream drug (resistant result)"}},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.1},
            margin={"l": 110, "r": 90, "t": 160, "b": 140}, plot_bgcolor="white", paper_bgcolor="white", font={"family": "Arial"},
        )
        return self._exporter.write(fig, output_stem, formats)


class DirectionalNetworkPlotter:
    """Directed network of one direction on positions shared by both directions."""

    _DPI = 300

    def __init__(self, classification: AntibioticClassificationResolver, size_inches: float = 9.0) -> None:
        self._classification = classification
        self._size = size_inches

    def export(
        self,
        edges: pd.DataFrame,
        nodes: pd.DataFrame,
        positions: dict[str, tuple[float, float]],
        direction: str,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
        n_eligible: int,
        n_direction: int,
        min_fold: float,
    ) -> dict[str, Path]:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch

        style = DIRECTION_STYLE[direction]
        fig, ax = plt.subplots(figsize=(self._size, self._size), facecolor="white")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.axis("off")

        active = set(nodes.antibiotic) if not nodes.empty else set()
        weighted = dict(zip(nodes.antibiotic, nodes.weighted_degree)) if not nodes.empty else {}
        top_weight = max(weighted.values(), default=1.0) or 1.0
        max_strength = float(edges.effect_magnitude.max()) if not edges.empty else 1.0

        for row in edges.sort_values("effect_magnitude").itertuples(index=False):
            (x0, y0), (x1, y1) = positions[row.upstream_antibiotic], positions[row.downstream_antibiotic]
            robust = row.validation_status == "robust"
            width = 0.7 + 4.3 * (row.effect_magnitude / max_strength if max_strength else 0.0)
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), connectionstyle="arc3,rad=0.22", arrowstyle="-|>", mutation_scale=11 + 2 * width,
                linewidth=width, color=style["hue"], alpha=0.78 if robust else 0.34, shrinkA=9, shrinkB=11, zorder=2,
            ))

        for drug, (x, y) in positions.items():
            on = drug in active
            color = self._classification.resolve(drug).color if on else "#D6DAE0"
            size = 60 + 640 * (weighted.get(drug, 0.0) / top_weight) if on else 26
            ax.scatter([x], [y], s=size, color=color, edgecolor="#17324d" if on else "#B8BEC8", linewidth=0.9 if on else 0.5, zorder=4)
            angle = math.degrees(math.atan2(y, x))
            left_side = x < 0
            ax.text(
                1.11 * x, 1.11 * y, drug.title() if len(drug) > 4 else drug, fontsize=9.5 if on else 8, color="#0F172A" if on else "#A0A7B3",
                ha="right" if left_side else "left", va="center", rotation=angle + 180 if left_side else angle, rotation_mode="anchor", zorder=5,
            )

        ax.text(0, 0.06, f"{style['title']}", ha="center", va="center", fontsize=26, fontweight="bold", color=style["hue"], alpha=0.16, zorder=0)
        ax.text(0, -0.08, style["rule"], ha="center", va="center", fontsize=16, color=style["hue"], alpha=0.16, zorder=0)
        title = f"{style['title']} network ({style['rule']}) — {TIER_TITLE[tier]}"
        subtitle = (
            f"{selection_note(direction, len(edges), n_eligible, n_direction, min_fold)}. Arrow: upstream resistant result → downstream drug observed "
            f"{'more' if direction == dv.ESCALATION else 'less'} often. Line width = |log2 ER|; solid = robust, faint = supported. "
            f"Node size = weighted degree within this direction; grey = no pattern in this direction. Node positions identical in both directions."
        )
        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985, color="#0F172A")
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9.6, color="#475467", wrap=True)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=self._classification.CATEGORY_COLORS[t], markeredgecolor="#17324d", markersize=10, label=f"{t} (AWaRe)") for t in ("Access", "Watch", "Reserve")]
        handles += [Line2D([0], [0], color=style["hue"], lw=3, alpha=0.78, label="robust"), Line2D([0], [0], color=style["hue"], lw=3, alpha=0.34, label="supported")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.02))
        return self._write(fig, output_stem, formats)

    def _write(self, fig, output_stem: Path, formats: tuple[str, ...]) -> dict[str, Path]:
        import matplotlib.pyplot as plt

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        png_bytes = None
        for fmt in formats:
            path = output_stem.with_suffix(f".{fmt}")
            if fmt == "html":
                if png_bytes is None:
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=150, facecolor="white", bbox_inches="tight")
                    png_bytes = buffer.getvalue()
                encoded = base64.b64encode(png_bytes).decode("ascii")
                path.write_text(
                    f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{output_stem.name}</title></head>"
                    f"<body style='margin:0;padding:24px;background:#fff'><img alt='{output_stem.name}' style='width:100%;height:auto' "
                    f"src='data:image/png;base64,{encoded}'></body></html>",
                    encoding="utf-8",
                )
            else:
                fig.savefig(path, format=fmt, dpi=self._DPI, facecolor="white", bbox_inches="tight")
            outputs[path.name] = path
        plt.close(fig)
        return outputs


class DirectionalForestPlotter:
    """Forest of one direction: ER with its own 95% CI, strongest effects first.

    Never plots both directions on one axis -- an escalation-only forest and a
    suppression-only forest, each centred on ER=1 with its own log-scale range,
    so a suppression pattern's CI is never dwarfed by an escalation pattern's.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, classification: AntibioticClassificationResolver) -> None:
        self._exporter = exporter
        self._template = template
        self._classification = classification

    def export(
        self,
        edges: pd.DataFrame,
        direction: str,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
        top_n: int,
        n_direction: int,
    ) -> dict[str, Path]:
        style = DIRECTION_STYLE[direction]
        pool = dv.with_effect_columns(edges.loc[edges.direction == direction]).copy()
        # A forest plot's whole point is estimate *with* uncertainty. Ranking by
        # |log2 ER| alone lets the sparsest, most extreme pairs -- exactly the ones
        # whose CI is not estimable -- dominate the "strongest effects" list with a
        # lone point and no error bar. Prefer pairs with an estimable CI first, and
        # only fill remaining slots with CI-less pairs once those run out.
        lo_all = pd.to_numeric(pool.get("er_ci_lower"), errors="coerce")
        hi_all = pd.to_numeric(pool.get("er_ci_upper"), errors="coerce")
        pool["_has_ci"] = lo_all.notna() & hi_all.notna() & (lo_all > 0) & (hi_all > 0)
        data = pool.sort_values(
            ["_has_ci", "effect_magnitude", "total_support_n"], ascending=[False, False, False], kind="mergesort"
        ).head(top_n).drop(columns="_has_ci").reset_index(drop=True)
        n = len(data)
        if n == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"No {TIER_TITLE[tier]} {direction} patterns to draw", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": "#475467"})
            fig.update_layout(template=self._template, width=900, height=300, xaxis={"visible": False}, yaxis={"visible": False}, plot_bgcolor="white", paper_bgcolor="white")
            return self._exporter.write(fig, output_stem, formats)

        est = pd.to_numeric(data.escalation_ratio, errors="coerce").to_numpy()
        lo = pd.to_numeric(data.get("er_ci_lower"), errors="coerce").to_numpy() if "er_ci_lower" in data else np.full(n, np.nan)
        hi = pd.to_numeric(data.get("er_ci_upper"), errors="coerce").to_numpy() if "er_ci_upper" in data else np.full(n, np.nan)
        has_ci = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
        x_lo = min(np.nanmin(np.where(has_ci, lo, est)), np.nanmin(est), 0.95) * 0.85
        x_hi = max(np.nanmax(np.where(has_ci, hi, est)), np.nanmax(est), 1.05) * 1.18

        y_pos = list(range(n - 1, -1, -1))  # row 0 at top, matching the original np.arange(n)[::-1]

        fig = make_subplots(
            rows=1, cols=3, shared_yaxes=True,
            column_widths=[0.64, 0.26, 0.10], horizontal_spacing=0.006,
        )

        # Alternating row shading, drawn full-width via paper xref so it spans
        # all three columns from one shape instead of three.
        for i in range(n):
            if i % 2 == 0:
                fig.add_shape(
                    type="rect", xref="paper", x0=0.0, x1=1.0, yref="y2",
                    y0=y_pos[i] - 0.48, y1=y_pos[i] + 0.48,
                    fillcolor="#F8FAFC", line={"width": 0}, layer="below",
                )

        # Category order/colour comes from the shared classification resolver so
        # this matches every other AWaRe-coloured figure in the platform.
        category_order = list(getattr(self._classification, "CATEGORY_ORDER", ("Access", "Watch", "Reserve", "Not Set")))
        data["_aware_category"] = [self._classification.resolve(d).aware_category for d in data.downstream_antibiotic]
        data["_colour"] = [self._classification.resolve(d).color for d in data.downstream_antibiotic]

        # CI whiskers: one neutral-slate line per row, no legend entry of its own
        # (the point marker already carries the legend-relevant colour/symbol).
        for i, row in data.iterrows():
            if has_ci[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[lo[i], hi[i]], y=[y_pos[i], y_pos[i]], mode="lines",
                        line={"color": "#334155", "width": 1.8}, hoverinfo="skip", showlegend=False,
                    ),
                    row=1, col=2,
                )

        hover = [
            f"<b>{str(row.upstream_antibiotic).title()} → {str(row.downstream_antibiotic).title()}</b><br>"
            f"ER {row.escalation_ratio:.3g}"
            + (f" (95% CI {lo[i]:.3g}–{hi[i]:.3g})" if has_ci[i] else " (CI not estimable)")
            + f"<br>Episode–pair rows: {int(row.total_support_n):,}<br>Validation: {row.validation_status}"
            for i, row in data.iterrows()
        ]
        for category in category_order:
            mask = data["_aware_category"] == category
            if not mask.any():
                continue
            subset = data.loc[mask]
            idx = subset.index.tolist()
            fig.add_trace(
                go.Scatter(
                    x=est[idx], y=[y_pos[i] for i in idx], mode="markers",
                    marker={
                        "size": 11,
                        "symbol": ["diamond" if r == "robust" else "square" for r in subset.validation_status],
                        "color": self._classification.CATEGORY_COLORS.get(category, "#94A3B8"),
                        "line": {
                            "width": [1.6 if r == "robust" else 1.2 for r in subset.validation_status],
                            "color": ["#0F172A" if r == "robust" else "#9A3412" for r in subset.validation_status],
                        },
                    },
                    name=f"{category} (AWaRe)", legendgroup=f"aware-{category}",
                    text=[hover[i] for i in idx], hovertemplate="%{text}<extra></extra>",
                ),
                row=1, col=2,
            )
        # Symbol-meaning legend entries (robust/supported), grey and data-less --
        # colour already carries AWaRe tier, these two only explain the shape.
        for symbol, edge, label in (("diamond", "#0F172A", "Robust"), ("square", "#9A3412", "Supported")):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker={"symbol": symbol, "size": 11, "color": "#64748B", "line": {"width": 1.4, "color": edge}},
                    name=label, hoverinfo="skip",
                ),
                row=1, col=2,
            )

        not_estimable = [i for i in range(n) if not has_ci[i]]
        for i in not_estimable:
            fig.add_annotation(
                x=est[i], y=y_pos[i], xref="x2", yref="y2", text="n.e.", showarrow=False,
                xanchor="left", xshift=9, font={"size": 10.5, "color": "#94A3B8"}, row=1, col=2,
            )

        for i, row in data.iterrows():
            up_full, down_full = str(row.upstream_antibiotic).title(), str(row.downstream_antibiotic).title()
            up_abbr = self._classification.resolve(row.upstream_antibiotic).abbreviation
            down_abbr = self._classification.resolve(row.downstream_antibiotic).abbreviation
            up = f"{up_full} ({up_abbr})" if up_abbr and up_abbr != up_full else up_full
            down = f"{down_full} ({down_abbr})" if down_abbr and down_abbr != down_full else down_full
            fig.add_annotation(x=0.015, y=y_pos[i], xref="x1", yref="y1", text=f"{up}  →  {down}",
                                showarrow=False, xanchor="left", font={"size": 13, "color": "#0F172A"}, row=1, col=1)
            fig.add_annotation(x=0.985, y=y_pos[i], xref="x1", yref="y1", text=f"{int(row.total_support_n):,}",
                                showarrow=False, xanchor="right", font={"size": 13, "color": "#475467"}, row=1, col=1)
            adj = row.get("adjusted_odds_ratio", np.nan)
            adj_text = f"{adj:.2f}" if pd.notna(adj) and abs(adj) < 1000 else "n/e"
            fig.add_annotation(x=0.5, y=y_pos[i], xref="x3", yref="y3", text=adj_text,
                                showarrow=False, xanchor="center", font={"size": 13, "color": "#0F172A"}, row=1, col=3)

        fig.add_annotation(x=0.015, y=n - 0.05, xref="x1", yref="y1", text="<b>Upstream → downstream</b>",
                            showarrow=False, xanchor="left", yanchor="bottom", font={"size": 12, "color": "#0F172A"}, row=1, col=1)
        fig.add_annotation(x=0.985, y=n - 0.05, xref="x1", yref="y1", text="<b>Episode–pair rows</b>",
                            showarrow=False, xanchor="right", yanchor="bottom", font={"size": 12, "color": "#0F172A"}, row=1, col=1)
        fig.add_annotation(x=0.5, y=n - 0.05, xref="x3", yref="y3", text="<b>Adj. OR</b>",
                            showarrow=False, xanchor="center", yanchor="bottom", font={"size": 12, "color": "#0F172A"}, row=1, col=3)

        ticks = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if x_lo <= t <= x_hi]
        fig.add_vline(x=1.0, line={"color": "#B91C1C", "dash": "dash", "width": 1.6}, opacity=0.85, row=1, col=2)
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", line={"color": "#B91C1C", "dash": "dash", "width": 1.6}, name="ER = 1 (no effect)", hoverinfo="skip"),
            row=1, col=2,
        )

        height = max(1050, 72 * n + 600)
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=3)
        fig.update_xaxes(
            type="log", range=[math.log10(x_lo), math.log10(x_hi)],
            tickvals=ticks, ticktext=[f"{t:g}" for t in ticks], tickangle=0,
            title={"text": f"Escalation ratio, ER (log scale) — {style['rule']}", "font": {"size": 13}},
            tickfont={"size": 12}, gridcolor="#E2E8F0", showline=True, linecolor="#CBD5E1", zeroline=False,
            row=1, col=2,
        )
        fig.update_yaxes(visible=False, range=[-0.8, n - 0.2], row=1, col=1)
        fig.update_yaxes(visible=False, range=[-0.8, n - 0.2], row=1, col=2)
        fig.update_yaxes(visible=False, range=[-0.8, n - 0.2], row=1, col=3)
        fig.update_layout(
            template=self._template,
            width=2450, height=height,
            title={
                "text": (
                    f"<b style='color:{style['hue']}'>{style['title']} patterns ({style['rule']})</b> — {TIER_TITLE[tier]}: strongest {n} of {n_direction}"
                    f"<br><sup>Point = ER; bar = 95% CI (log-scale, continuity-corrected). ◆ robust, ■ supported.</sup>"
                    f"<br><sup>Marker colour = downstream drug's AWaRe tier. Ranked by |log2 ER|, largest effect first.</sup>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 16},
            },
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.16, "font": {"size": 12}},
            margin={"l": 20, "r": 80, "t": 160, "b": 240},
            plot_bgcolor="white", paper_bgcolor="white", font={"family": "Arial"},
            annotations=list(fig.layout.annotations) + [
                {
                    "text": "n/e = adjusted OR not estimable (separation or sparse cells).<br>"
                            "n.e. = ER's own CI not estimable (an upstream branch has no downstream-observed rows).<br>"
                            "Pairs with an estimable CI are ranked first.",
                    "showarrow": False, "x": 0.0, "y": -0.32, "xref": "paper", "yref": "paper",
                    "xanchor": "left", "yanchor": "top", "align": "left", "font": {"size": 10.5, "color": "#475467"},
                }
            ],
        )
        return self._exporter.write(fig, output_stem, formats)



class DirectionalAwarePlotter:
    """AWaRe upstream-tier x downstream-tier validated-pattern counts, one direction per figure.

    Kept separate from the escalation/suppression Sankey and network: this
    figure answers a stewardship-tier question (upward/lateral/downward AWaRe
    movement), which is not the same axis as the observation-direction question
    those figures answer, even though both are split by direction here too.
    """

    def __init__(self, classification: AntibioticClassificationResolver) -> None:
        self._classification = classification

    def export(
        self,
        table: pd.DataFrame,
        direction: str,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
        n_direction: int,
    ) -> dict[str, Path]:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        style = DIRECTION_STYLE[direction]
        tiers = [t for t in self._classification.CATEGORY_ORDER if t != "Unclassified"]
        index = {t: i for i, t in enumerate(tiers)}
        counts = np.zeros((len(tiers), len(tiers)))
        robust = np.zeros_like(counts)
        for row in table.itertuples(index=False):
            if row.upstream_aware in index and row.downstream_aware in index:
                counts[index[row.upstream_aware], index[row.downstream_aware]] = row.n_patterns
                robust[index[row.upstream_aware], index[row.downstream_aware]] = row.n_robust
        fig, ax = plt.subplots(figsize=(8.2, 7.4), facecolor="white")
        mesh = ax.pcolormesh(counts, cmap=style["scale"], vmin=0, vmax=max(counts.max(), 1), edgecolors="white", linewidth=2)
        ax.set_xticks(np.arange(len(tiers)) + 0.5, tiers)
        ax.set_yticks(np.arange(len(tiers)) + 0.5, tiers)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Downstream drug AWaRe tier", fontsize=11, labelpad=10)
        ax.set_ylabel("Upstream drug AWaRe tier", fontsize=11)
        for (i, j), value in np.ndenumerate(counts):
            if value:
                dark = value > 0.55 * counts.max()
                ax.text(j + 0.5, i + 0.5, f"{int(value)}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if dark else "#0F172A")
                ax.text(j + 0.5, i + 0.72, f"{int(robust[i, j])} robust", ha="center", va="center", fontsize=8.5, color="white" if dark else "#475467")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03)
        bar.set_label("Validated patterns")
        if not table.empty:
            ups, lats, downs = (int(table.loc[table.aware_direction == d, "n_patterns"].sum()) for d in ("upward", "lateral", "downward"))
        else:
            ups = lats = downs = 0
        fig.suptitle(f"{style['title']} patterns ({style['rule']}) by AWaRe tier — {TIER_TITLE[tier]}", fontsize=14, fontweight="bold", color=style["hue"], y=0.985)
        fig.text(
            0.5, 0.925,
            f"{n_direction} validated {direction} patterns. AWaRe upward {ups}, lateral {lats}, downward {downs} "
            "(stewardship-tier movement, a different axis from the escalation/suppression direction this figure is already split on).",
            ha="center", fontsize=9.2, color="#475467",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        png_bytes = None
        for fmt in formats:
            path = output_stem.with_suffix(f".{fmt}")
            if fmt == "html":
                if png_bytes is None:
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=150, facecolor="white", bbox_inches="tight")
                    png_bytes = buffer.getvalue()
                encoded = base64.b64encode(png_bytes).decode("ascii")
                path.write_text(
                    f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{output_stem.name}</title></head>"
                    f"<body style='margin:0;padding:24px;background:#fff'><img alt='{output_stem.name}' style='width:100%;height:auto' "
                    f"src='data:image/png;base64,{encoded}'></body></html>",
                    encoding="utf-8",
                )
            else:
                fig.savefig(path, format=fmt, dpi=300, facecolor="white", bbox_inches="tight")
            outputs[path.name] = path
        plt.close(fig)
        return outputs


class AntiArtefactScatterPlotter:
    """DAS vs. PBI: distinguishes real directional asymmetry from fixed lab panels.

    Both directions share one plot here (unlike the five families above) --
    DAS and PBI are pair-level diagnostics, not a flow, ranking, or network
    whose meaning depends on treating "up" as one direction throughout, so
    showing escalation and suppression together with colour is the correct
    reading, the same way er_landscape does.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        validated: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
    ) -> dict[str, Path]:
        required = {"panel_bundling_index", "directional_asymmetry_score", "total_support_n", "direction", "validation_status"}
        if validated.empty or not required.issubset(validated.columns):
            fig = go.Figure()
            fig.add_annotation(text="DAS/PBI figure unavailable — required columns missing", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            fig.update_layout(template=self._template, width=self._width, height=self._height)
            return self._exporter.write(fig, output_stem, formats)

        fig = go.Figure()
        for direction in dv.DIRECTIONS:
            style = DIRECTION_STYLE[direction]
            subset = validated.loc[validated.direction == direction]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset.panel_bundling_index, y=subset.directional_asymmetry_score,
                mode="markers", name=f"{style['title']} ({style['rule']})",
                marker={
                    "size": np.clip(np.sqrt(subset.total_support_n.clip(lower=1)) * 0.9, 5, 32),
                    "color": style["hue"],
                    "opacity": np.where(subset.validation_status == "robust", 0.85, 0.45),
                    "line": {"width": 0.6, "color": "white"},
                },
                customdata=np.stack([subset.upstream_antibiotic, subset.downstream_antibiotic, subset.total_support_n, subset.validation_status], axis=-1),
                hovertemplate="<b>%{customdata[0]} → %{customdata[1]}</b><br>PBI %{x:.3f}<br>DAS %{y:.3f}<br>Support %{customdata[2]:,}<br>%{customdata[3]}<extra></extra>",
            ))
        fig.add_vline(x=0.95, line={"dash": "dash", "color": "#94A3B8", "width": 1.2}, annotation_text="PBI=0.95", annotation_font={"size": 9}, annotation_position="top left")
        fig.update_layout(
            template=self._template, width=self._width, height=self._height,
            title={
                "text": (
                    "<b>Directional asymmetry vs. panel-bundling index</b> — "
                    f"{TIER_TITLE[tier]}"
                    "<br><sup>High PBI = fixed lab panel, not a directional signal. Point size = episode-pair support.</sup>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 13},
            },
            xaxis={"title": {"text": "Panel-bundling index (PBI)"}, "range": [-0.02, 1.02]},
            yaxis={"title": {"text": "Directional asymmetry score (DAS)"}},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.14},
            margin={"l": 90, "r": 40, "t": 110, "b": 90}, plot_bgcolor="white", paper_bgcolor="white", font={"family": "Arial"},
        )
        return self._exporter.write(fig, output_stem, formats)


class AdjustmentConcordancePlotter:
    """Raw ER vs. adjusted OR: does covariate adjustment change the story.

    Also direction-combined for the same reason as AntiArtefactScatterPlotter:
    the quadrant reading (concordant / attenuated / reversed) is the same
    diagnostic regardless of which direction a given pair validated in.
    """

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        validated: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
    ) -> dict[str, Path]:
        required = {"escalation_ratio", "adjusted_odds_ratio", "direction", "validation_status"}
        if validated.empty or not required.issubset(validated.columns):
            fig = go.Figure()
            fig.add_annotation(text="Raw-vs-adjusted figure unavailable — required columns missing", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            fig.update_layout(template=self._template, width=self._width, height=self._height)
            return self._exporter.write(fig, output_stem, formats)

        estimable = validated.loc[pd.to_numeric(validated.adjusted_odds_ratio, errors="coerce").notna()].copy()
        estimable["log_er"] = np.log(pd.to_numeric(estimable.escalation_ratio, errors="coerce"))
        estimable["log_or"] = np.log(pd.to_numeric(estimable.adjusted_odds_ratio, errors="coerce"))
        n_estimable, n_total = len(estimable), len(validated)

        fig = go.Figure()
        if estimable.empty:
            fig.add_annotation(text="No validated pairs have an estimable adjusted OR", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        for direction in dv.DIRECTIONS:
            style = DIRECTION_STYLE[direction]
            subset = estimable.loc[estimable.direction == direction]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset.log_er, y=subset.log_or, mode="markers", name=f"{style['title']} ({style['rule']})",
                marker={
                    "size": 9, "color": style["hue"],
                    "opacity": np.where(subset.validation_status == "robust", 0.85, 0.45),
                    "line": {"width": 0.6, "color": "white"},
                },
                customdata=np.stack([subset.upstream_antibiotic, subset.downstream_antibiotic, subset.escalation_ratio, subset.adjusted_odds_ratio, subset.validation_status], axis=-1),
                hovertemplate="<b>%{customdata[0]} → %{customdata[1]}</b><br>ER %{customdata[2]:.3g}<br>Adj. OR %{customdata[3]:.3g}<br>%{customdata[4]}<extra></extra>",
            ))
        fig.add_hline(y=0, line={"color": "#94A3B8", "width": 1})
        fig.add_vline(x=0, line={"color": "#94A3B8", "width": 1})
        fig.add_shape(type="line", x0=-8, y0=-8, x1=8, y1=8, line={"color": "#CBD5E1", "width": 1, "dash": "dot"}, layer="below")
        fig.update_layout(
            template=self._template, width=self._width, height=self._height,
            title={
                "text": (
                    "<b>Raw escalation ratio vs. adjusted odds ratio</b> — "
                    f"{TIER_TITLE[tier]}: {n_estimable} of {n_total} validated patterns estimable"
                    "<br><sup>Dotted diagonal = raw and adjusted agree. Same-quadrant-as-origin = concordant; opposite = reversed.</sup>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 13},
            },
            xaxis={"title": {"text": "log(escalation ratio)"}, "zeroline": False},
            yaxis={"title": {"text": "log(adjusted odds ratio)"}, "zeroline": False},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.14},
            margin={"l": 90, "r": 40, "t": 110, "b": 90}, plot_bgcolor="white", paper_bgcolor="white", font={"family": "Arial"},
        )
        return self._exporter.write(fig, output_stem, formats)


class AdjustmentConcordanceSummaryPlotter:
    """Concordant / attenuated / reversed / non-estimable counts, escalation vs. suppression.

    Direction-combined by design, like AntiArtefactScatterPlotter and
    AdjustmentConcordancePlotter above: this is a category count, not a flow
    or ranking, so putting both directions on one chart (grouped, not
    stacked, so a reader can compare bar heights directly) is the correct
    reading rather than a violation of the never-pool-directions rule.
    """

    _CATEGORY_ORDER = ("concordant", "attenuated", "reversed", "unavailable")
    _CATEGORY_LABELS = {
        "concordant": "Concordant<br><span style='font-size:0.8em'>(persists, CI excl. 1)</span>",
        "attenuated": "Attenuated<br><span style='font-size:0.8em'>(same side, CI incl. 1)</span>",
        "reversed": "Reversed<br><span style='font-size:0.8em'>(opposite side of 1)</span>",
        "unavailable": "Non-estimable<br><span style='font-size:0.8em'>(no adjusted OR)</span>",
    }

    def __init__(self, exporter: PlotlyFigureExporter, template: str, width: int, height: int) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height

    def export(
        self,
        validated: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        *,
        tier: str,
    ) -> dict[str, Path]:
        required = {"direction", "adjustment_concordance"}
        if validated.empty or not required.issubset(validated.columns):
            fig = go.Figure()
            fig.add_annotation(text="Adjustment-concordance summary unavailable — required columns missing", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            fig.update_layout(template=self._template, width=self._width, height=self._height)
            return self._exporter.write(fig, output_stem, formats)

        counts = (
            validated.groupby(["adjustment_concordance", "direction"]).size()
            .reindex(
                pd.MultiIndex.from_product(
                    [self._CATEGORY_ORDER, dv.DIRECTIONS], names=["adjustment_concordance", "direction"]
                ),
                fill_value=0,
            )
            .reset_index(name="n")
        )
        n_total = len(validated)
        n_estimable = int((validated["adjustment_concordance"] != "unavailable").sum())

        fig = go.Figure()
        for direction in dv.DIRECTIONS:
            style = DIRECTION_STYLE[direction]
            subset = counts.loc[counts["direction"] == direction].set_index("adjustment_concordance").loc[list(self._CATEGORY_ORDER)]
            fig.add_trace(go.Bar(
                x=[self._CATEGORY_LABELS[c] for c in self._CATEGORY_ORDER], y=subset["n"],
                name=f"{style['title']} ({style['rule']})", marker={"color": style["hue"]},
                text=subset["n"], textposition="outside",
                hovertemplate=f"<b>{style['title']}</b><br>" + "%{x}: %{y}<extra></extra>",
            ))
        fig.update_layout(
            barmode="group",
            template=self._template, width=self._width, height=self._height,
            title={
                "text": (
                    f"<b>Adjustment concordance</b> — {TIER_TITLE[tier]}: {n_estimable}/{n_total} estimable"
                    "<br><sup>Does adjustment leave direction intact, weaken it to the null, or reverse it? Validation status does not depend on this label.</sup>"
                ),
                "x": 0.5, "xanchor": "center", "font": {"size": 14},
            },
            xaxis={"title": ""}, yaxis={"title": {"text": "Validated patterns, N"}},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.12},
            margin={"l": 70, "r": 40, "t": 110, "b": 90}, plot_bgcolor="white", paper_bgcolor="white", font={"family": "Arial"},
        )
        return self._exporter.write(fig, output_stem, formats)
