"""Write cascade analysis outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.utils.provenance import atomic_write_json, atomic_write_parquet


class CascadeResultWriter:
    """Persist cascade analysis results."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def write(
        self,
        conditional_probabilities: pd.DataFrame,
        escalation_results: pd.DataFrame,
        adjusted_results: pd.DataFrame,
        adjusted_model_shap_global: pd.DataFrame,
        adjusted_model_shap_values: pd.DataFrame,
        adjusted_model_shap_metadata: pd.DataFrame,
        adjusted_model_shap_domain: pd.DataFrame,
        retained_edges: pd.DataFrame,
        validation_results: pd.DataFrame,
        dependence_results: pd.DataFrame,
        rule_concordance: pd.DataFrame,
        cotesting_pairs: pd.DataFrame,
        cotesting_probabilities: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "conditional_probabilities": output_dir / "conditional_probabilities.parquet",
            "escalation_results": output_dir / "escalation_results.parquet",
            "adjusted_results": output_dir / "adjusted_results.parquet",
            "adjusted_model_shap_global": output_dir / "adjusted_model_shap_global.parquet",
            "adjusted_model_shap_values": output_dir / "adjusted_model_shap_values.parquet",
            "adjusted_model_shap_metadata": output_dir / "adjusted_model_shap_metadata.parquet",
            "adjusted_model_shap_domain": output_dir / "adjusted_model_shap_domain.parquet",
            "retained_edges": output_dir / "retained_edges.parquet",
            "validation_results": output_dir / "validation_results.parquet",
            "dependence_results": output_dir / "dependence_results.parquet",
            "rule_concordance": output_dir / "rule_concordance.parquet",
            "cotesting_pairs": output_dir / "cotesting_pairs.parquet",
            "cotesting_probabilities": output_dir / "cotesting_probabilities.parquet",
            "summary": output_dir / self._settings.cascade.outputs.summary_filename,
        }
        # Each file goes in under a temporary name and is renamed into place, so a
        # killed job never leaves a truncated parquet behind; the stage's
        # _SUCCESS.json (CascadeAnalysisWorkflow) is what declares the set complete.
        for name, frame in (
            ("conditional_probabilities", conditional_probabilities),
            ("escalation_results", escalation_results),
            ("adjusted_results", adjusted_results),
            ("adjusted_model_shap_global", adjusted_model_shap_global),
            ("adjusted_model_shap_values", adjusted_model_shap_values),
            ("adjusted_model_shap_metadata", adjusted_model_shap_metadata),
            ("adjusted_model_shap_domain", adjusted_model_shap_domain),
            ("retained_edges", retained_edges),
            ("validation_results", validation_results),
            ("dependence_results", dependence_results),
            ("rule_concordance", rule_concordance),
            ("cotesting_pairs", cotesting_pairs),
            ("cotesting_probabilities", cotesting_probabilities),
        ):
            atomic_write_parquet(frame, outputs[name])

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
                "adjusted_model_shap_global": len(adjusted_model_shap_global),
                "adjusted_model_shap_values": len(adjusted_model_shap_values),
                "adjusted_model_shap_metadata": len(adjusted_model_shap_metadata),
                "adjusted_model_shap_domain": len(adjusted_model_shap_domain),
                "retained_edges": len(retained_edges),
                "validation_results": len(validation_results),
                "dependence_results": len(dependence_results),
                "rule_concordance": len(rule_concordance),
                "cotesting_pairs": len(cotesting_pairs),
                "cotesting_probabilities": len(cotesting_probabilities),
            },
        }
        atomic_write_json(summary, outputs["summary"])
        return outputs
