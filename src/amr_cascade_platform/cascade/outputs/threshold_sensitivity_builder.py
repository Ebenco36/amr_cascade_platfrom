"""Threshold sensitivity outputs for cascade retention."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
from amr_cascade_platform.core.config.config_models import Settings


class ThresholdSensitivityBuilder:
    """Evaluate cascade retention across threshold grids."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._retained_edge_analyzer = RetainedEdgeAnalyzer(settings)

    def export(
        self,
        escalation_results: pd.DataFrame,
        adjusted_results: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sensitivity_path = output_dir / "threshold_sensitivity.parquet"
        table = self._build_table(escalation_results, adjusted_results)
        table.to_parquet(sensitivity_path, index=False)
        return {"threshold_sensitivity_path": sensitivity_path}

    def _build_table(
        self,
        escalation_results: pd.DataFrame,
        adjusted_results: pd.DataFrame,
    ) -> pd.DataFrame:
        if escalation_results.empty:
            return pd.DataFrame()

        rows: list[dict[str, float | int]] = []
        for min_total_support in self._settings.cascade.sensitivity_min_total_supports:
            for min_result_support in self._settings.cascade.sensitivity_min_result_supports:
                support_mask = (
                    (escalation_results["total_support_n"] >= min_total_support)
                    & (escalation_results["positive_support_n"] >= min_result_support)
                    & (escalation_results["negative_support_n"] >= min_result_support)
                )
                support_passing = escalation_results[support_mask]
                for escalation_threshold in self._settings.cascade.sensitivity_retained_min_escalation_ratios:
                    retained = self._retained_edge_analyzer.analyze(
                        escalation_results=escalation_results,
                        adjusted_results=adjusted_results,
                        min_total_support=min_total_support,
                        min_result_support=min_result_support,
                        retained_min_escalation_ratio=escalation_threshold,
                    )
                    rows.append(
                        {
                            "min_total_support": min_total_support,
                            "min_result_support": min_result_support,
                            "retained_min_escalation_ratio": escalation_threshold,
                            "adjusted_or_positive_threshold": self._settings.cascade.adjusted_or_positive_threshold,
                            "support_passing_edges_n": int(len(support_passing)),
                            "retained_edges_n": int(len(retained)),
                            "mean_retained_escalation_ratio": float(retained["escalation_ratio"].mean()) if not retained.empty else 0.0,
                            "mean_retained_adjusted_odds_ratio": float(retained["adjusted_odds_ratio"].mean()) if not retained.empty else 0.0,
                        }
                    )
        return pd.DataFrame(rows)
