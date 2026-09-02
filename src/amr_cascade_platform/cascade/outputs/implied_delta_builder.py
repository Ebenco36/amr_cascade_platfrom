"""Compare primary observed cascade outputs with implied-sensitivity sensitivity outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class ImpliedDeltaBuilder:
    """Build delta summaries between primary and implied-sensitivity cascade outputs."""

    def export(
        self,
        primary_dir: Path,
        sensitivity_dir: Path,
    ) -> dict[str, Path]:
        primary_edge_path = primary_dir / "edge_report.parquet"
        sensitivity_edge_path = sensitivity_dir / "edge_report.parquet"
        if not primary_edge_path.exists() or not sensitivity_edge_path.exists():
            return {}

        primary = pd.read_parquet(primary_edge_path)
        sensitivity = pd.read_parquet(sensitivity_edge_path)
        delta = self._build_delta(primary, sensitivity)
        delta_path = sensitivity_dir / "implied_delta_edges.parquet"
        delta.to_parquet(delta_path, index=False)

        summary = self._build_summary(primary, sensitivity, delta)
        summary_path = sensitivity_dir / "implied_delta_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return {
            "implied_delta_edges_path": delta_path,
            "implied_delta_summary_path": summary_path,
        }

    def _build_delta(self, primary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
        primary_subset = primary.loc[
            :,
            [
                "upstream_antibiotic",
                "downstream_antibiotic",
                "escalation_ratio",
                "adjusted_odds_ratio",
                "rank_by_escalation_ratio",
            ],
        ].rename(
            columns={
                "escalation_ratio": "primary_escalation_ratio",
                "adjusted_odds_ratio": "primary_adjusted_odds_ratio",
                "rank_by_escalation_ratio": "primary_rank",
            }
        )
        sensitivity_subset = sensitivity.loc[
            :,
            [
                "upstream_antibiotic",
                "downstream_antibiotic",
                "escalation_ratio",
                "adjusted_odds_ratio",
                "rank_by_escalation_ratio",
            ],
        ].rename(
            columns={
                "escalation_ratio": "sensitivity_escalation_ratio",
                "adjusted_odds_ratio": "sensitivity_adjusted_odds_ratio",
                "rank_by_escalation_ratio": "sensitivity_rank",
            }
        )
        delta = primary_subset.merge(
            sensitivity_subset,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="outer",
        )
        delta["presence_change"] = delta.apply(self._presence_label, axis=1)
        delta["escalation_ratio_delta"] = (
            delta["sensitivity_escalation_ratio"] - delta["primary_escalation_ratio"]
        )
        delta["adjusted_odds_ratio_delta"] = (
            delta["sensitivity_adjusted_odds_ratio"] - delta["primary_adjusted_odds_ratio"]
        )
        return delta

    def _build_summary(
        self,
        primary: pd.DataFrame,
        sensitivity: pd.DataFrame,
        delta: pd.DataFrame,
    ) -> dict[str, object]:
        shared = delta[delta["presence_change"] == "shared"]
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "primary_edge_count": int(len(primary)),
            "sensitivity_edge_count": int(len(sensitivity)),
            "shared_edge_count": int((delta["presence_change"] == "shared").sum()),
            "primary_only_edge_count": int((delta["presence_change"] == "primary_only").sum()),
            "sensitivity_only_edge_count": int((delta["presence_change"] == "sensitivity_only").sum()),
            "mean_shared_escalation_ratio_delta": float(shared["escalation_ratio_delta"].mean()) if not shared.empty else 0.0,
            "mean_shared_adjusted_odds_ratio_delta": float(shared["adjusted_odds_ratio_delta"].mean()) if not shared.empty else 0.0,
        }

    @staticmethod
    def _presence_label(row: pd.Series) -> str:
        primary_present = pd.notna(row.get("primary_escalation_ratio"))
        sensitivity_present = pd.notna(row.get("sensitivity_escalation_ratio"))
        if primary_present and sensitivity_present:
            return "shared"
        if primary_present:
            return "primary_only"
        return "sensitivity_only"
