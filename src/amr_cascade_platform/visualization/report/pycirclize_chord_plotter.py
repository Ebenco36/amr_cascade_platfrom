"""Directional chord diagram export via pycirclize — uses standard abbreviations for readability."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pandas as pd

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver


class PyCirclizeChordPlotter:
    """Export directional chord diagrams from retained cascade edges."""

    def __init__(self, classification_resolver: AntibioticClassificationResolver) -> None:
        self._classification = classification_resolver

    def export(
        self,
        retained_edges: pd.DataFrame,
        output_stem: Path,
        formats: tuple[str, ...],
        tier_label: str = "",
    ) -> dict[str, Path]:
        circos = self._build_chord_diagram(retained_edges, tier_label)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        title_prefix = f"{tier_label} " if tier_label else ""
        static_formats = [fmt for fmt in formats if fmt in {"png", "svg", "pdf", "tiff"}]
        for fmt in static_formats:
            path = output_stem.with_suffix(f".{fmt}")
            dpi = 300 if fmt in {"png", "tiff"} else 220
            circos.savefig(path, dpi=dpi, figsize=(13, 13))
            outputs[path.name] = path
        if "html" in formats:
            html_path = output_stem.with_suffix(".html")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_svg = Path(tmp) / "chord.svg"
                circos.savefig(tmp_svg, dpi=300, figsize=(13, 13))
                svg_b64 = base64.b64encode(tmp_svg.read_bytes()).decode("ascii")
            html_path.write_text(
                (
                    f"<html><head><meta charset='utf-8'><title>{title_prefix}Antibiotic Cascade Chord Diagram</title>"
                    "<style>body{margin:0;background:#fff;display:flex;flex-direction:column;"
                    "align-items:center;font-family:Arial,sans-serif;padding:20px}"
                    "h2{font-size:18px;color:#1E293B;margin-bottom:8px}"
                    "p{font-size:12px;color:#64748B;margin-bottom:16px}"
                    "img{max-width:100%;height:auto}</style></head>"
                    "<body>"
                    f"<h2>{title_prefix}Antibiotic Cascade Chord Diagram</h2>"
                    "<p>Node colours reflect WHO AWaRe category. Arc width proportional to pair support. "
                    "Directed arcs run from upstream (resistant group) to downstream antibiotic.</p>"
                    f"<img alt='Cascade chord diagram' src='data:image/svg+xml;base64,{svg_b64}'/>"
                    "</body></html>"
                ),
                encoding="utf-8",
            )
            outputs[html_path.name] = html_path
        return outputs

    def _build_chord_diagram(self, retained_edges: pd.DataFrame, tier_label: str = ""):
        from pycirclize import Circos

        if retained_edges.empty:
            placeholder = f"No {tier_label.lower()} data" if tier_label else "No data"
            matrix = pd.DataFrame([[1]], index=[placeholder], columns=[placeholder])
            color_map = {placeholder: "#94A3B8"}
            return Circos.chord_diagram(
                matrix,
                space=4,
                cmap=color_map,
                label_kws={"r": 112, "size": 13, "color": "#1E293B"},
                link_kws={"direction": 1, "alpha": 0.55, "lw": 0.6},
            )

        normalized = retained_edges.rename(
            columns={"positive_support_n": "resistant_support_n", "negative_support_n": "susceptible_support_n"}
        )

        # Build full-name matrix first
        full_matrix = normalized.pivot_table(
            index="upstream_antibiotic",
            columns="downstream_antibiotic",
            values="total_support_n",
            aggfunc="sum",
            fill_value=0,
        )
        if full_matrix.empty:
            placeholder = f"No {tier_label.lower()} data" if tier_label else "No data"
            full_matrix = pd.DataFrame([[1]], index=[placeholder], columns=[placeholder])

        # Map every raw label to its canonical abbreviation.
        # Synonymous drugs (same resolved abbreviation) collapse to the SAME node —
        # their matrix rows and columns are summed, eliminating collision suffixes.
        all_labels = sorted(set(full_matrix.index).union(set(full_matrix.columns)))
        abbrev_map: dict[str, str] = {}
        for label in all_labels:
            abbrev_map[label] = self._classification.get_abbreviation(label)

        # Rename and merge: group-sum rows and columns that share an abbreviation
        matrix = full_matrix.copy()
        matrix.index = pd.Index([abbrev_map[i] for i in full_matrix.index])
        matrix.columns = pd.Index([abbrev_map[c] for c in full_matrix.columns])
        matrix = matrix.groupby(matrix.index, observed=True).sum()
        matrix = matrix.T.groupby(matrix.T.index, observed=True).sum().T

        # Unique abbreviation labels after merging
        unique_abbrevs = sorted(matrix.index.union(matrix.columns))

        # Category and color maps keyed by abbreviation (use first matching raw label)
        abbrev_to_label: dict[str, str] = {}
        for label in all_labels:
            abbr = abbrev_map[label]
            if abbr not in abbrev_to_label:
                abbrev_to_label[abbr] = label

        categories: dict[str, str] = {
            abbr: self._classification.resolve(abbrev_to_label[abbr]).aware_category
            for abbr in unique_abbrevs
            if abbr in abbrev_to_label
        }
        order = sorted(
            unique_abbrevs,
            key=lambda abbr: (
                self._classification.category_sort_key(categories.get(abbr, "Unclassified")),
                abbr,
            ),
        )
        color_map: dict[str, str] = {
            abbr: self._classification.resolve(abbrev_to_label[abbr]).color
            for abbr in unique_abbrevs
            if abbr in abbrev_to_label
        }

        # Link color map keyed by (upstream_abbrev, downstream_abbrev)
        link_cmap = [
            (
                abbrev_map.get(row.upstream_antibiotic, row.upstream_antibiotic),
                abbrev_map.get(row.downstream_antibiotic, row.downstream_antibiotic),
                AntibioticClassificationResolver.with_alpha(
                    self._classification.resolve(row.upstream_antibiotic).color,
                    0.48,
                ),
            )
            for row in normalized.itertuples(index=False)
            if row.upstream_antibiotic in abbrev_map and row.downstream_antibiotic in abbrev_map
        ]

        return Circos.chord_diagram(
            matrix,
            space=4,
            cmap=color_map,
            link_cmap=link_cmap,
            ticks_interval=None,
            order=order,
            # Abbreviations are short — larger font is now readable
            label_kws={"r": 112, "size": 12, "color": "#1E293B"},
            link_kws={"direction": 1, "alpha": 0.55, "lw": 0.7, "ec": "#1E293B"},
            link_kws_handler=lambda source, target: {
                "ec": color_map.get(source, "#94A3B8"),
                "fc": AntibioticClassificationResolver.with_alpha(
                    color_map.get(source, "#94A3B8"), 0.48
                ),
            },
        )
