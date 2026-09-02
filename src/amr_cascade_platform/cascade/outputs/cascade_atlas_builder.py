"""Organism-aware atlas exports for descriptive and adjusted cascade results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class CascadeAtlasBuilder:
    """Export a compact atlas table that directly compares raw and adjusted cascade effects."""

    _ATLAS_COLUMNS = [
        "scope",
        "site",
        "organism_scope",
        "upstream_antibiotic",
        "downstream_antibiotic",
        "resistant_support_n",
        "susceptible_support_n",
        "resistant_tested_n",
        "susceptible_tested_n",
        "resistant_downstream_test_probability",
        "susceptible_downstream_test_probability",
        "escalation_ratio",
        "er_ci_lower",
        "er_ci_upper",
        "adjusted_odds_ratio",
        "adjusted_log_odds",
        "modeled_n",
        "passes_support_threshold",
        "supports_adjusted_model",
        "retained_edge",
        "raw_effect_direction",
        "adjusted_effect_direction",
        "raw_adjusted_direction_agreement",
    ]

    def export(
        self,
        edge_report: pd.DataFrame,
        output_dir: Path,
        *,
        scope: str,
        site: str | None,
        organism: str | None,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        atlas_path = output_dir / "cascade_atlas.parquet"
        summary_path = output_dir / "cascade_atlas_summary.json"

        atlas = edge_report.copy()
        if atlas.empty:
            atlas = pd.DataFrame(columns=self._ATLAS_COLUMNS)
        else:
            atlas["scope"] = scope
            atlas["site"] = site or "combined"
            atlas["organism_scope"] = organism or "all_organisms"
            atlas = atlas.reindex(columns=self._ATLAS_COLUMNS)
        atlas.to_parquet(atlas_path, index=False)

        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": scope,
            "site": site,
            "organism_scope": organism or "all_organisms",
            "row_count": int(len(atlas)),
            "retained_edges": int(pd.to_numeric(atlas.get("retained_edge", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int).sum()) if not atlas.empty else 0,
            "support_threshold_passes": int(pd.to_numeric(atlas.get("passes_support_threshold", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int).sum()) if not atlas.empty else 0,
            "direction_agreement_counts": atlas.get("raw_adjusted_direction_agreement", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "cascade_atlas_path": atlas_path,
            "cascade_atlas_summary_path": summary_path,
        }
