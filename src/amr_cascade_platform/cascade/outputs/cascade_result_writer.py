"""Write cascade analysis outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings


class CascadeResultWriter:
    """Persist cascade analysis results."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def write(
        self,
        conditional_probabilities: pd.DataFrame,
        escalation_results: pd.DataFrame,
        adjusted_results: pd.DataFrame,
        retained_edges: pd.DataFrame,
        validation_results: pd.DataFrame,
        dependence_results: pd.DataFrame,
        rule_concordance: pd.DataFrame,
        cotesting_pairs: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "conditional_probabilities": output_dir / "conditional_probabilities.parquet",
            "escalation_results": output_dir / "escalation_results.parquet",
            "adjusted_results": output_dir / "adjusted_results.parquet",
            "retained_edges": output_dir / "retained_edges.parquet",
            "validation_results": output_dir / "validation_results.parquet",
            "dependence_results": output_dir / "dependence_results.parquet",
            "rule_concordance": output_dir / "rule_concordance.parquet",
            "cotesting_pairs": output_dir / "cotesting_pairs.parquet",
            "summary": output_dir / self._settings.cascade.outputs.summary_filename,
        }
        conditional_probabilities.to_parquet(outputs["conditional_probabilities"], index=False)
        escalation_results.to_parquet(outputs["escalation_results"], index=False)
        adjusted_results.to_parquet(outputs["adjusted_results"], index=False)
        retained_edges.to_parquet(outputs["retained_edges"], index=False)
        validation_results.to_parquet(outputs["validation_results"], index=False)
        dependence_results.to_parquet(outputs["dependence_results"], index=False)
        rule_concordance.to_parquet(outputs["rule_concordance"], index=False)
        cotesting_pairs.to_parquet(outputs["cotesting_pairs"], index=False)

        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "positive_result": self._settings.cascade.upstream_result_positive,
            "negative_result": self._settings.cascade.upstream_result_negative,
            "min_total_support": self._settings.cascade.min_total_support,
            "min_result_support": self._settings.cascade.min_result_support,
            "rows": {
                "conditional_probabilities": len(conditional_probabilities),
                "escalation_results": len(escalation_results),
                "adjusted_results": len(adjusted_results),
                "retained_edges": len(retained_edges),
                "validation_results": len(validation_results),
                "dependence_results": len(dependence_results),
                "rule_concordance": len(rule_concordance),
                "cotesting_pairs": len(cotesting_pairs),
            },
        }
        with outputs["summary"].open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return outputs
