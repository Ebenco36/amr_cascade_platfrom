"""Compare prespecified decision-threshold selection strategies."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError, ValidationError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore
from amr_cascade_platform.modeling.workflows.modeling_workflow import ModelingRequest, ModelingWorkflow


class ThresholdStrategyComparisonWorkflow:
    """Run a like-for-like threshold-strategy sensitivity analysis.

    The comparison deliberately uses ``clinical_no_cascade``. The full and
    cascade-aware feature sets contain target-derived ``train_*`` summaries
    computed before cross-validation; using those columns in grouped OOF would
    leak held-out outcomes unless the entire cascade feature pipeline were
    rebuilt independently inside every fold.
    """

    FEATURE_SET = "clinical_no_cascade"
    STRATEGIES = ("validation_holdout", "grouped_oof")
    OUTPUT_DIR = "threshold_strategy_comparison"

    def __init__(
        self,
        settings: Settings,
        path_manager: PathManager,
        modeling_workflow: ModelingWorkflow | None = None,
    ) -> None:
        self._settings = settings
        self._paths = path_manager
        self._estimator = settings.modeling.primary_estimator
        self._modeling = modeling_workflow or ModelingWorkflow(settings, path_manager)
        self._parquet_store = ParquetStore(settings)

    def run(
        self,
        *,
        scope: str = "combined",
        organism: str | None = None,
        split_strategy: str | None = None,
    ) -> dict[str, Path]:
        split_name = split_strategy or self._settings.modeling.default_split_strategy
        if split_name != "site":
            raise ValidationError(
                "Threshold-strategy comparison is defined for the prespecified site split: "
                "training site, development-validation site, and sealed test site."
            )

        strategy_outputs: dict[str, dict[str, Path]] = {}
        for strategy in self.STRATEGIES:
            strategy_outputs[strategy] = self._modeling.run(
                ModelingRequest(
                    scope=scope,
                    estimators=(self._estimator,),
                    feature_sets=(self.FEATURE_SET,),
                    organism=organism,
                    split_strategy=split_name,
                    threshold_selection_strategy=strategy,
                )
            )

        curves, selected, test_metrics = self._combine_strategy_outputs(strategy_outputs)
        output_dir = self._output_dir(scope=scope, organism=organism)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "threshold_strategy_curves": output_dir / "threshold_strategy_curves.parquet",
            "threshold_strategy_selected": output_dir / "threshold_strategy_selected.parquet",
            "threshold_strategy_test_metrics": output_dir / "threshold_strategy_test_metrics.parquet",
        }
        self._parquet_store.write_pandas(curves, outputs["threshold_strategy_curves"])
        self._parquet_store.write_pandas(selected, outputs["threshold_strategy_selected"])
        self._parquet_store.write_pandas(test_metrics, outputs["threshold_strategy_test_metrics"])

        summary_path = output_dir / "threshold_strategy_comparison_summary.json"
        summary = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": scope,
            "organism": organism,
            "split_strategy": split_name,
            "feature_set": self.FEATURE_SET,
            "estimator": self._estimator,
            "model_role": "prespecified_primary",
            "strategies": list(self.STRATEGIES),
            "selection_metric": self._settings.evaluation.threshold_selection_metric,
            "scientific_boundary": (
                "This sensitivity analysis compares threshold-selection procedures for the "
                "prespecified primary estimator using clinical_no_cascade features only. "
                "It is not cross-validation of precomputed cascade or network features. "
                "Sealed-test metrics compare the frozen strategies descriptively and must not "
                "be used to select between them."
            ),
            "test_sites": list(self._settings.modeling.test_sites),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs["summary"] = summary_path
        return outputs

    def _output_dir(self, *, scope: str, organism: str | None) -> Path:
        task_root = (
            self._paths.paths.artifacts
            / self._settings.modeling.output_dir
            / self._settings.modeling.task_name
        )
        if organism:
            task_root = scoped_output_dir(task_root, scope, organism=organism)
        return task_root / self.OUTPUT_DIR

    def _combine_strategy_outputs(
        self,
        strategy_outputs: dict[str, dict[str, Path]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        curve_frames: list[pd.DataFrame] = []
        selected_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []

        for strategy in self.STRATEGIES:
            outputs = strategy_outputs.get(strategy)
            if not outputs:
                raise DataDiscoveryError(f"Threshold comparison produced no outputs for {strategy}.")
            threshold_metrics = self._read_required(outputs, "threshold_metrics")
            selected = self._read_required(outputs, "selected_thresholds")
            metrics = self._read_required(outputs, "metrics")
            if len(selected) != 1:
                raise ValidationError(
                    f"Threshold comparison expected one selected-threshold row for {strategy}; "
                    f"found {len(selected)}."
                )

            selected_row = selected.iloc[0]
            recorded_strategy = str(selected_row["selection_strategy"])
            if recorded_strategy != strategy:
                raise ValidationError(
                    f"Threshold comparison requested {strategy} but artifact records "
                    f"{recorded_strategy}."
                )
            selection_split = str(selected_row["selection_split"])
            selection_curve = threshold_metrics.loc[
                threshold_metrics["split"].astype(str).eq(selection_split)
                & threshold_metrics["model_name"].astype(str).eq(self._estimator)
                & threshold_metrics["feature_set"].astype(str).eq(self.FEATURE_SET)
            ].copy()
            if selection_curve.empty:
                raise ValidationError(
                    f"Threshold comparison found no {selection_split} selection curve for {strategy}."
                )
            selection_curve["comparison_strategy"] = strategy
            selection_curve["comparison_label"] = self._strategy_label(strategy)
            selection_curve["selection_rows"] = int(selected_row["selection_rows"])
            selection_curve["selection_groups"] = selected_row.get("selection_groups")
            selection_curve["cv_folds"] = selected_row.get("cv_folds")
            curve_frames.append(selection_curve)

            selected_copy = selected.copy()
            selected_copy["comparison_strategy"] = strategy
            selected_copy["comparison_label"] = self._strategy_label(strategy)
            selected_frames.append(selected_copy)

            strategy_test = metrics.loc[
                metrics["split"].astype(str).eq("test")
                & metrics["model_name"].astype(str).eq(self._estimator)
                & metrics["feature_set"].astype(str).eq(self.FEATURE_SET)
            ].copy()
            if len(strategy_test) != 1:
                raise ValidationError(
                    f"Threshold comparison expected one sealed-test row for {strategy}; "
                    f"found {len(strategy_test)}."
                )
            strategy_test["comparison_strategy"] = strategy
            strategy_test["comparison_label"] = self._strategy_label(strategy)
            test_frames.append(strategy_test)

        curves = pd.concat(curve_frames, ignore_index=True)
        selected = pd.concat(selected_frames, ignore_index=True)
        test_metrics = pd.concat(test_frames, ignore_index=True)
        return curves, selected, test_metrics

    def _read_required(self, outputs: dict[str, Path], key: str) -> pd.DataFrame:
        path = outputs.get(key)
        if path is None or not path.exists():
            raise DataDiscoveryError(
                f"Threshold comparison requires the '{key}' artifact, but it was not generated."
            )
        frame = pd.read_parquet(path, engine=self._settings.parquet.engine)
        if frame.empty:
            raise ValidationError(f"Threshold comparison artifact is empty: {path}.")
        return frame

    @staticmethod
    def _strategy_label(strategy: str) -> str:
        labels = {
            "validation_holdout": "Validation-site holdout",
            "grouped_oof": "Patient-grouped OOF",
        }
        return labels.get(strategy, strategy.replace("_", " ").title())
