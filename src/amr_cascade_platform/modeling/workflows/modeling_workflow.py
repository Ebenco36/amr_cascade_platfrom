"""Train and evaluate downstream-testing prediction models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.parquet_store import ParquetStore
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.modeling.datasets.feature_set_selector import (
    ModelingFeatureSet,
    ModelingFeatureSetSelector,
)
from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import PairFeatureMatrixBuilder
from amr_cascade_platform.modeling.datasets.split_strategy import (
    DatasetSplit,
    SiteSplitStrategy,
    TemporalSplitStrategy,
)
from amr_cascade_platform.modeling.estimators.logistic_regression_estimator import LogisticRegressionEstimator
from amr_cascade_platform.modeling.estimators.neural_network_estimator import NeuralNetworkEstimator
from amr_cascade_platform.modeling.estimators.random_forest_estimator import RandomForestEstimator
from amr_cascade_platform.modeling.estimators.xgboost_estimator import XGBoostEstimator
from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics


@dataclass(frozen=True)
class ModelingRequest:
    scope: str = "combined"
    estimators: tuple[str, ...] | None = None
    feature_sets: tuple[str, ...] | None = None
    ablation: bool = False
    organism: str | None = None
    split_strategy: str | None = None


@dataclass(frozen=True)
class ModelingDatasetSelection:
    dataset: object
    base_dataset: object
    source_path: Path
    persisted: bool


class ModelingWorkflow:
    """Build features, split by site, train estimators, and export artifacts."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._feature_builder = PairFeatureMatrixBuilder(settings, path_manager)
        self._metrics = ClassificationMetrics()
        self._parquet_store = ParquetStore(settings)
        self._dataset_store = DatasetStore(settings)
        self._feature_set_selector = ModelingFeatureSetSelector(
            self._paths.paths.features / "combined" / self._settings.feature_build.metadata_filename
        )

    def run(self, request: ModelingRequest) -> dict[str, Path]:
        if request.scope != "combined":
            raise ValueError("Current modeling workflow supports scope='combined' only.")

        selection = self._load_modeling_dataset(request)
        dataset = selection.dataset
        split_strategy_name = request.split_strategy or self._settings.modeling.default_split_strategy
        feature_sets = self._feature_set_selector.select(
            base_dataset=selection.base_dataset,
            full_dataset=dataset,
            requested_sets=request.feature_sets,
            ablation=request.ablation,
        )
        run_label = self._run_label(request, feature_sets, split_strategy_name)
        use_parquet_site_loading = selection.persisted and split_strategy_name == "site"
        split = None
        if use_parquet_site_loading:
            # The parquet-filtered loading path below never materializes a full DatasetSplit,
            # so it can't run SiteSplitStrategy.split()'s episode-overlap check. It can and must
            # still run the cheap, data-free config check (train/validation/test sites are
            # disjoint) -- constructing the strategy triggers that in __init__. Skipping this
            # would let a misconfigured overlapping site list train and test on leaked data with
            # no error, unlike every other split path.
            SiteSplitStrategy(self._settings)
            row_counts = self._site_split_row_counts(selection.source_path)
        else:
            split = self._resolve_split_strategy(split_strategy_name).split(dataset.dataframe)
            row_counts = {
                "all": len(dataset.dataframe),
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            }

        task_root = (
            self._paths.paths.artifacts
            / self._settings.modeling.output_dir
            / self._settings.modeling.task_name
        )
        if request.organism:
            task_root = scoped_output_dir(task_root, request.scope, organism=request.organism)
        artifact_dir = task_root / run_label
        artifact_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {"feature_matrix": selection.source_path}

        insufficient_reason = self._insufficient_training_reason(row_counts)
        if insufficient_reason is not None:
            self._logger.warning("Skipping model training: %s", insufficient_reason)
            metrics_frame = pd.DataFrame()
            threshold_frame = pd.DataFrame()
            selected_thresholds = pd.DataFrame()
            threshold_recommendations = pd.DataFrame()

            metrics_path = artifact_dir / "metrics.parquet"
            threshold_path = artifact_dir / "threshold_metrics.parquet"
            selected_thresholds_path = artifact_dir / "selected_thresholds.parquet"
            threshold_recommendations_path = artifact_dir / "threshold_recommendations.parquet"
            self._parquet_store.write_pandas(metrics_frame, metrics_path)
            self._parquet_store.write_pandas(threshold_frame, threshold_path)
            self._parquet_store.write_pandas(selected_thresholds, selected_thresholds_path)
            self._parquet_store.write_pandas(threshold_recommendations, threshold_recommendations_path)
            outputs["metrics"] = metrics_path
            outputs["threshold_metrics"] = threshold_path
            outputs["selected_thresholds"] = selected_thresholds_path
            outputs["threshold_recommendations"] = threshold_recommendations_path

            skip_path = artifact_dir / "training_skipped.txt"
            skip_path.write_text(insufficient_reason + "\n", encoding="utf-8")
            outputs["training_skipped"] = skip_path

            summary_path = artifact_dir / "modeling_summary.json"
            self._write_summary(
                summary_path,
                outputs,
                dataset,
                row_counts,
                metrics_frame,
                feature_sets,
                skip_reason=insufficient_reason,
            )
            outputs["summary"] = summary_path

            if self._should_publish_to_task_root(request, feature_sets):
                self._publish_standard_outputs(
                    task_root,
                    outputs,
                    metrics_frame,
                    threshold_frame,
                    selected_thresholds,
                    threshold_recommendations,
                    summary_path,
                )
            return outputs

        metrics_rows: list[dict[str, object]] = []
        threshold_rows: list[dict[str, object]] = []
        selected_threshold_rows: list[dict[str, object]] = []
        threshold_recommendation_rows: list[dict[str, object]] = []
        for feature_set in feature_sets:
            estimators = self._build_estimators(
                feature_set.categorical_columns,
                feature_set.numeric_columns,
                request.estimators,
            )
            feature_dir = artifact_dir / feature_set.name
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_dataset = self._materialize_feature_set_dataset(
                selection=selection,
                dataset=dataset,
                feature_set=feature_set,
            )
            feature_dataset_path = feature_dir / "feature_dataset.parquet"
            self._parquet_store.write_pandas(feature_dataset, feature_dataset_path)
            outputs[f"{feature_set.name}__feature_dataset"] = feature_dataset_path
            for estimator in estimators:
                self._logger.info("Training %s on feature set %s", estimator.name, feature_set.name)
                if use_parquet_site_loading:
                    train_frame = self._load_site_partition(
                        selection.source_path,
                        self._settings.modeling.train_sites,
                        dataset,
                        feature_set,
                    )
                    if train_frame.empty:
                        self._logger.warning(
                            "Skipping %s on feature set %s: no rows available in the configured training sites.",
                            estimator.name,
                            feature_set.name,
                        )
                        continue
                    if not self._has_two_target_classes(train_frame["target"]):
                        self._logger.warning(
                            "Skipping %s on feature set %s: training sites contain fewer than two target classes.",
                            estimator.name,
                            feature_set.name,
                        )
                        del train_frame
                        continue
                    estimator.fit(
                        train_frame.loc[:, feature_set.feature_columns],
                        train_frame["target"],
                    )
                    del train_frame
                    # Post-hoc isotonic calibration for tree estimators.
                    # Fitted on validation data only — never on training data —
                    # so there is no leakage into test-set evaluation.
                    if hasattr(estimator, "calibrate"):
                        val_frame = self._load_site_partition(
                            selection.source_path,
                            self._settings.modeling.validation_sites,
                            dataset,
                            feature_set,
                        )
                        if not val_frame.empty:
                            estimator.calibrate(
                                val_frame.loc[:, feature_set.feature_columns],
                                val_frame["target"],
                            )
                            self._logger.info(
                                "Calibrated %s/%s using validation set (%d rows)",
                                estimator.name,
                                feature_set.name,
                                len(val_frame),
                            )
                        del val_frame
                    raw_predictions = self._predict_site_partitions(
                        source_path=selection.source_path,
                        model_name=estimator.name,
                        estimator=estimator,
                        dataset=dataset,
                        feature_set=feature_set,
                    )
                else:
                    if split.train.empty:
                        self._logger.warning(
                            "Skipping %s on feature set %s: no rows available in the configured training split.",
                            estimator.name,
                            feature_set.name,
                        )
                        continue
                    if not self._has_two_target_classes(split.train["target"]):
                        self._logger.warning(
                            "Skipping %s on feature set %s: training split contains fewer than two target classes.",
                            estimator.name,
                            feature_set.name,
                        )
                        continue
                    estimator.fit(
                        split.train.loc[:, feature_set.feature_columns],
                        split.train["target"],
                    )
                    # Post-hoc isotonic calibration for tree estimators.
                    if hasattr(estimator, "calibrate") and not split.validation.empty:
                        estimator.calibrate(
                            split.validation.loc[:, feature_set.feature_columns],
                            split.validation["target"],
                        )
                        self._logger.info(
                            "Calibrated %s/%s using validation set (%d rows)",
                            estimator.name,
                            feature_set.name,
                            len(split.validation),
                        )
                    raw_predictions = self._predict_all_splits(estimator.name, estimator, split, dataset, feature_set)
                selected_threshold, threshold_table = self._select_threshold(raw_predictions, estimator.name, feature_set.name)
                selected_threshold_rows.append(
                    {
                        "model_name": estimator.name,
                        "feature_set": feature_set.name,
                        "selection_split": "validation",
                        "selection_metric": self._settings.evaluation.threshold_selection_metric,
                        "selected_threshold": selected_threshold,
                        "default_threshold": self._settings.evaluation.probability_threshold,
                    }
                )
                threshold_recommendation_rows.extend(
                    self._build_threshold_recommendations(
                        model_name=estimator.name,
                        feature_set_name=feature_set.name,
                        threshold_table=threshold_table,
                        selected_threshold=selected_threshold,
                    )
                )
                predictions = self._apply_frozen_threshold(raw_predictions, selected_threshold)
                prediction_path = feature_dir / f"{estimator.name}_predictions.parquet"
                self._parquet_store.write_pandas(predictions, prediction_path)
                outputs[f"{feature_set.name}__{estimator.name}_predictions"] = prediction_path

                estimator_metrics, estimator_thresholds = self._evaluate_predictions(
                    estimator.name,
                    predictions,
                    feature_set.name,
                    selected_threshold,
                    threshold_table,
                )
                metrics_rows.extend(estimator_metrics)
                threshold_rows.extend(estimator_thresholds)

        metrics_frame = pd.DataFrame(metrics_rows)
        metrics_path = artifact_dir / "metrics.parquet"
        self._parquet_store.write_pandas(metrics_frame, metrics_path)
        outputs["metrics"] = metrics_path

        threshold_frame = pd.DataFrame(threshold_rows)
        threshold_path = artifact_dir / "threshold_metrics.parquet"
        self._parquet_store.write_pandas(threshold_frame, threshold_path)
        outputs["threshold_metrics"] = threshold_path

        selected_thresholds = pd.DataFrame(selected_threshold_rows)
        selected_thresholds_path = artifact_dir / "selected_thresholds.parquet"
        self._parquet_store.write_pandas(selected_thresholds, selected_thresholds_path)
        outputs["selected_thresholds"] = selected_thresholds_path

        threshold_recommendations = pd.DataFrame(threshold_recommendation_rows)
        threshold_recommendations_path = artifact_dir / "threshold_recommendations.parquet"
        self._parquet_store.write_pandas(threshold_recommendations, threshold_recommendations_path)
        outputs["threshold_recommendations"] = threshold_recommendations_path

        if len(feature_sets) > 1:
            ablation_summary = self._build_ablation_summary(metrics_frame)
            ablation_path = artifact_dir / "ablation_summary.parquet"
            self._parquet_store.write_pandas(ablation_summary, ablation_path)
            outputs["ablation_summary"] = ablation_path

        summary_path = artifact_dir / "modeling_summary.json"
        self._write_summary(summary_path, outputs, dataset, row_counts, metrics_frame, feature_sets)
        outputs["summary"] = summary_path

        if self._should_publish_to_task_root(request, feature_sets):
            self._publish_standard_outputs(
                task_root,
                outputs,
                metrics_frame,
                threshold_frame,
                selected_thresholds,
                threshold_recommendations,
                summary_path,
            )
        return outputs

    @staticmethod
    def _insufficient_training_reason(row_counts: dict[str, int]) -> str | None:
        if row_counts["all"] <= 0:
            return "No model-ready rows were available for this request."
        if row_counts["train"] <= 0:
            return "No model-ready rows were available in the configured training split."
        return None

    @staticmethod
    def _has_two_target_classes(target: pd.Series) -> bool:
        return pd.Series(target).dropna().nunique() >= 2

    @staticmethod
    def _run_label(
        request: ModelingRequest,
        feature_sets: tuple[ModelingFeatureSet, ...],
        split_strategy_name: str,
    ) -> str:
        feature_label = "__".join(feature_set.name for feature_set in feature_sets)
        if request.ablation or len(feature_sets) > 1:
            return f"{split_strategy_name}__ablation__{feature_label}"
        if request.estimators:
            return f"{split_strategy_name}__{feature_label}__{'__'.join(request.estimators)}"
        return f"{split_strategy_name}__all_models"

    def _resolve_split_strategy(self, split_strategy_name: str):
        if split_strategy_name == "site":
            return SiteSplitStrategy(self._settings)
        if split_strategy_name == "temporal":
            return TemporalSplitStrategy(self._settings)
        raise ValueError(f"Unsupported modeling split strategy: {split_strategy_name}")

    def _load_modeling_dataset(self, request: ModelingRequest) -> ModelingDatasetSelection:
        features_dir = scoped_output_dir(
            root=self._paths.paths.features,
            scope=request.scope,
            organism=request.organism,
        )
        enriched_path = features_dir / self._settings.feature_build.combined_output_filename
        if not enriched_path.exists():
            base_path = features_dir / "pair_testing_feature_matrix.parquet"
            bundle = self._feature_builder.build(scope=request.scope, organism=request.organism)
            if not base_path.exists():
                base_path.parent.mkdir(parents=True, exist_ok=True)
                self._parquet_store.write_pandas(bundle.dataframe, base_path)
            return ModelingDatasetSelection(dataset=bundle, base_dataset=bundle, source_path=base_path, persisted=True)

        layout = self._feature_builder.feature_layout()
        schema_columns = tuple(pq.ParquetFile(enriched_path).schema_arrow.names)
        base_numeric = tuple(layout["numeric_columns"])
        metadata_path = features_dir / self._settings.feature_build.metadata_filename
        clinical_numeric = self._feature_set_selector._clinical_numeric_columns(  # noqa: SLF001
            type(
                "SchemaOnlyBundle",
                (),
                {
                    "dataframe": pd.DataFrame(columns=list(schema_columns)),
                    "numeric_columns": base_numeric,
                },
            )()
        )
        base_bundle = type("ModelingBundle", (), {})()
        base_bundle.dataframe = pd.DataFrame(columns=list(schema_columns))
        base_bundle.feature_columns = layout["feature_columns"]
        base_bundle.categorical_columns = layout["categorical_columns"]
        base_bundle.numeric_columns = base_numeric
        base_bundle.id_columns = layout["id_columns"]

        full_bundle = type("ModelingBundle", (), {})()
        full_bundle.dataframe = pd.DataFrame(columns=list(schema_columns))
        full_bundle.feature_columns = layout["feature_columns"] + tuple(
            column for column in clinical_numeric if column not in layout["feature_columns"]
        )
        full_bundle.categorical_columns = layout["categorical_columns"]
        full_bundle.numeric_columns = base_numeric + tuple(
            column for column in clinical_numeric if column not in base_numeric
        )
        full_bundle.id_columns = layout["id_columns"]
        return ModelingDatasetSelection(dataset=full_bundle, base_dataset=base_bundle, source_path=enriched_path, persisted=True)

    def _build_estimators(
        self,
        categorical_columns: tuple[str, ...],
        numeric_columns: tuple[str, ...],
        requested_estimators: tuple[str, ...] | None,
    ) -> list[object]:
        estimators: list[object] = []
        estimator_names = requested_estimators or self._settings.modeling.estimators
        for estimator_name in estimator_names:
            if estimator_name == "logistic_regression":
                estimators.append(
                    LogisticRegressionEstimator(self._settings, categorical_columns, numeric_columns)
                )
            elif estimator_name == "random_forest":
                estimators.append(
                    RandomForestEstimator(self._settings, categorical_columns, numeric_columns)
                )
            elif estimator_name == "xgboost":
                estimators.append(
                    XGBoostEstimator(self._settings, categorical_columns, numeric_columns)
                )
            elif estimator_name == "neural_network":
                estimators.append(
                    NeuralNetworkEstimator(self._settings, categorical_columns, numeric_columns)
                )
            else:
                raise ValueError(f"Unsupported estimator configured: {estimator_name}")
        return estimators

    def _site_split_row_counts(self, source_path: Path) -> dict[str, int]:
        split_column = self._settings.modeling.split_column
        counts = {"all": 0, "train": 0, "validation": 0, "test": 0}
        for label, sites in (
            ("train", self._settings.modeling.train_sites),
            ("validation", self._settings.modeling.validation_sites),
            ("test", self._settings.modeling.test_sites),
        ):
            frame = self._dataset_store.read_pandas(
                source_path,
                columns=[split_column],
                filters=[(split_column, "in", list(sites))],
            )
            counts[label] = len(frame)
            counts["all"] += len(frame)
        return counts

    def _required_columns_for_feature_set(
        self,
        dataset,
        feature_set: ModelingFeatureSet,
    ) -> list[str]:
        columns = list(dict.fromkeys(dataset.id_columns + feature_set.feature_columns + (self._settings.modeling.split_column,)))
        return columns

    def _materialize_feature_set_dataset(
        self,
        selection: ModelingDatasetSelection,
        dataset,
        feature_set: ModelingFeatureSet,
    ) -> pd.DataFrame:
        required_columns = self._required_columns_for_feature_set(dataset, feature_set)
        if selection.persisted:
            return self._dataset_store.read_pandas(
                selection.source_path,
                columns=required_columns,
            )
        return dataset.dataframe.loc[:, required_columns].copy()

    def _load_site_partition(
        self,
        source_path: Path,
        sites: tuple[str, ...],
        dataset,
        feature_set: ModelingFeatureSet,
    ) -> pd.DataFrame:
        return self._dataset_store.read_pandas(
            source_path,
            columns=self._required_columns_for_feature_set(dataset, feature_set),
            filters=[(self._settings.modeling.split_column, "in", list(sites))],
        )

    def _predict_site_partitions(
        self,
        source_path: Path,
        model_name: str,
        estimator,
        dataset,
        feature_set: ModelingFeatureSet,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split_name, sites in (
            ("train", self._settings.modeling.train_sites),
            ("validation", self._settings.modeling.validation_sites),
            ("test", self._settings.modeling.test_sites),
        ):
            frame = self._load_site_partition(source_path, sites, dataset, feature_set)
            if frame.empty:
                continue
            probabilities = estimator.predict_proba(frame.loc[:, feature_set.feature_columns])
            result = frame.loc[:, dataset.id_columns].copy()
            result["model_name"] = model_name
            result["feature_set"] = feature_set.name
            result["split"] = split_name
            result["predicted_probability"] = probabilities
            frames.append(result)
            del frame
        if not frames:
            return pd.DataFrame(
                columns=list(dataset.id_columns)
                + ["model_name", "feature_set", "split", "predicted_probability"]
            )
        return pd.concat(frames, ignore_index=True)

    def _predict_all_splits(
        self,
        model_name: str,
        estimator,
        split: DatasetSplit,
        dataset,
        feature_set: ModelingFeatureSet,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split_name, frame in {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }.items():
            if frame.empty:
                continue
            probabilities = estimator.predict_proba(frame.loc[:, feature_set.feature_columns])
            result = frame.loc[:, dataset.id_columns].copy()
            result["model_name"] = model_name
            result["feature_set"] = feature_set.name
            result["split"] = split_name
            result["predicted_probability"] = probabilities
            frames.append(result)
        if not frames:
            return pd.DataFrame(
                columns=list(dataset.id_columns)
                + ["model_name", "feature_set", "split", "predicted_probability"]
            )
        return pd.concat(frames, ignore_index=True)

    def _select_threshold(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        feature_set_name: str,
    ) -> tuple[float, pd.DataFrame]:
        validation_predictions = predictions.loc[predictions["split"] == "validation"].copy()
        if validation_predictions.empty:
            return self._settings.evaluation.probability_threshold, pd.DataFrame()
        selected_threshold, threshold_table = self._metrics.select_threshold(
            y_true=validation_predictions["target"],
            y_probability=validation_predictions["predicted_probability"],
            objective=self._settings.evaluation.threshold_selection_metric,
            thresholds=self._settings.evaluation.threshold_candidates,
            default_threshold=self._settings.evaluation.probability_threshold,
        )
        self._logger.info(
            "Selected threshold %.3f for %s/%s using validation %s",
            selected_threshold,
            feature_set_name,
            model_name,
            self._settings.evaluation.threshold_selection_metric,
        )
        return selected_threshold, threshold_table

    @staticmethod
    def _apply_frozen_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
        result = predictions.copy()
        result["decision_threshold"] = float(threshold)
        result["predicted_label"] = (result["predicted_probability"] >= threshold).astype(int)
        return result

    def _evaluate_predictions(
        self,
        model_name: str,
        predictions: pd.DataFrame,
        feature_set_name: str,
        selected_threshold: float,
        validation_threshold_table: pd.DataFrame,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        metrics_rows: list[dict[str, object]] = []
        threshold_rows: list[dict[str, object]] = []
        for split_name, frame in predictions.groupby("split"):
            metrics = self._metrics.evaluate(
                y_true=frame["target"],
                y_probability=frame["predicted_probability"],
                threshold=selected_threshold,
            )
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "split": split_name,
                    "decision_threshold": selected_threshold,
                    "threshold_selection_metric": self._settings.evaluation.threshold_selection_metric,
                    **metrics,
                }
            )
            # The validation threshold table is the only one used to choose the
            # operating point. We still export per-split sweeps for diagnostic
            # review, but the selected threshold is frozen before test scoring.
            threshold_table = (
                validation_threshold_table.copy()
                if split_name == "validation" and not validation_threshold_table.empty
                else self._metrics.threshold_sweep(
                    y_true=frame["target"],
                    y_probability=frame["predicted_probability"],
                    thresholds=self._settings.evaluation.threshold_candidates,
                )
            )
            threshold_table["model_name"] = model_name
            threshold_table["feature_set"] = feature_set_name
            threshold_table["split"] = split_name
            threshold_table["selected_threshold"] = selected_threshold
            threshold_table["threshold_selection_metric"] = self._settings.evaluation.threshold_selection_metric
            threshold_table["selected_for_decision"] = threshold_table["threshold"].eq(selected_threshold)
            threshold_rows.extend(threshold_table.to_dict(orient="records"))
        return metrics_rows, threshold_rows

    def _build_threshold_recommendations(
        self,
        model_name: str,
        feature_set_name: str,
        threshold_table: pd.DataFrame,
        selected_threshold: float,
    ) -> list[dict[str, object]]:
        if threshold_table.empty:
            return []
        rows: list[dict[str, object]] = []
        for metric_name in self._settings.evaluation.reporting_threshold_metrics:
            if metric_name not in threshold_table.columns:
                continue
            ranked = threshold_table.sort_values(
                by=[metric_name, "recall", "precision", "threshold"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            best = ranked.iloc[0]
            rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "metric_name": metric_name,
                    "recommended_threshold": float(best["threshold"]),
                    "metric_value": float(best[metric_name]),
                    "is_primary_selection_metric": metric_name == self._settings.evaluation.threshold_selection_metric,
                    "matches_selected_threshold": float(best["threshold"]) == float(selected_threshold),
                }
            )
        return rows

    def _write_summary(
        self,
        path: Path,
        outputs: dict[str, Path],
        dataset,
        row_counts: dict[str, int],
        metrics_frame: pd.DataFrame,
        feature_sets: tuple[ModelingFeatureSet, ...],
        skip_reason: str | None = None,
    ) -> None:
        payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "task_name": self._settings.modeling.task_name,
            "feature_columns": list(dataset.feature_columns),
            "categorical_columns": list(dataset.categorical_columns),
            "numeric_columns": list(dataset.numeric_columns),
            "feature_sets": {
                feature_set.name: {
                    "description": feature_set.description,
                    "feature_columns": list(feature_set.feature_columns),
                    "categorical_columns": list(feature_set.categorical_columns),
                    "numeric_columns": list(feature_set.numeric_columns),
                }
                for feature_set in feature_sets
            },
            "row_counts": {
                "all": int(row_counts["all"]),
                "train": int(row_counts["train"]),
                "validation": int(row_counts["validation"]),
                "test": int(row_counts["test"]),
            },
            "best_test_model": self._best_model(metrics_frame),
            "outputs": {name: str(path_value) for name, path_value in outputs.items()},
        }
        if skip_reason is not None:
            payload["training_status"] = "skipped"
            payload["skip_reason"] = skip_reason
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _best_model(metrics_frame: pd.DataFrame) -> dict[str, object]:
        if metrics_frame.empty:
            return {}
        test_rows = metrics_frame[metrics_frame["split"] == "test"].copy()
        if test_rows.empty:
            return {}
        ranked = test_rows.sort_values(by=["pr_auc", "roc_auc", "f1"], ascending=[False, False, False])
        best = ranked.iloc[0]
        return {
            "model_name": best["model_name"],
            "feature_set": best.get("feature_set"),
            "pr_auc": float(best["pr_auc"]),
            "roc_auc": float(best["roc_auc"]) if pd.notna(best["roc_auc"]) else None,
            "f1": float(best["f1"]),
            "mcc": float(best["mcc"]) if "mcc" in best and pd.notna(best["mcc"]) else None,
        }

    @staticmethod
    def _should_publish_to_task_root(
        request: ModelingRequest,
        feature_sets: tuple[ModelingFeatureSet, ...],
    ) -> bool:
        return not request.ablation and len(feature_sets) == 1 and feature_sets[0].name == "full"

    def _publish_standard_outputs(
        self,
        task_root: Path,
        outputs: dict[str, Path],
        metrics_frame: pd.DataFrame,
        threshold_frame: pd.DataFrame,
        selected_thresholds: pd.DataFrame,
        threshold_recommendations: pd.DataFrame,
        summary_path: Path,
    ) -> None:
        task_root.mkdir(parents=True, exist_ok=True)
        metrics_path = task_root / "metrics.parquet"
        threshold_path = task_root / "threshold_metrics.parquet"
        selected_thresholds_path = task_root / "selected_thresholds.parquet"
        threshold_recommendations_path = task_root / "threshold_recommendations.parquet"
        self._parquet_store.write_pandas(metrics_frame, metrics_path)
        self._parquet_store.write_pandas(threshold_frame, threshold_path)
        self._parquet_store.write_pandas(selected_thresholds, selected_thresholds_path)
        self._parquet_store.write_pandas(threshold_recommendations, threshold_recommendations_path)
        for name, path in outputs.items():
            if not (name.endswith("_predictions") or name.endswith("__feature_dataset")):
                continue
            target_path = task_root / path.name
            frame = pd.read_parquet(path, engine=self._settings.parquet.engine)
            self._parquet_store.write_pandas(frame, target_path)
        summary_target = task_root / "modeling_summary.json"
        summary_target.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    @staticmethod
    def _build_ablation_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
        if metrics_frame.empty:
            return pd.DataFrame()
        test_rows = metrics_frame.loc[metrics_frame["split"] == "test"].copy()
        if test_rows.empty:
            return pd.DataFrame()
        observed = (
            test_rows.loc[test_rows["feature_set"] == "observed_only", ["model_name", "pr_auc", "roc_auc", "f1", "mcc"]]
            .rename(
                columns={
                    "pr_auc": "observed_only_pr_auc",
                    "roc_auc": "observed_only_roc_auc",
                    "f1": "observed_only_f1",
                    "mcc": "observed_only_mcc",
                }
            )
        )
        full = (
            test_rows.loc[test_rows["feature_set"] == "full", ["model_name", "pr_auc", "roc_auc", "f1", "mcc"]]
            .rename(
                columns={
                    "pr_auc": "full_pr_auc",
                    "roc_auc": "full_roc_auc",
                    "f1": "full_f1",
                    "mcc": "full_mcc",
                }
            )
        )
        cascade = (
            test_rows.loc[test_rows["feature_set"] == "cascade_aware", ["model_name", "pr_auc", "roc_auc", "f1", "mcc"]]
            .rename(
                columns={
                    "pr_auc": "cascade_aware_pr_auc",
                    "roc_auc": "cascade_aware_roc_auc",
                    "f1": "cascade_aware_f1",
                    "mcc": "cascade_aware_mcc",
                }
            )
        )
        clinical = (
            test_rows.loc[test_rows["feature_set"] == "clinical_no_cascade", ["model_name", "pr_auc", "roc_auc", "f1", "mcc"]]
            .rename(
                columns={
                    "pr_auc": "clinical_no_cascade_pr_auc",
                    "roc_auc": "clinical_no_cascade_roc_auc",
                    "f1": "clinical_no_cascade_f1",
                    "mcc": "clinical_no_cascade_mcc",
                }
            )
        )
        summary = observed.merge(cascade, on="model_name", how="outer").merge(clinical, on="model_name", how="outer").merge(full, on="model_name", how="outer")
        if "observed_only_pr_auc" in summary.columns and "cascade_aware_pr_auc" in summary.columns:
            summary["delta_pr_auc_cascade_vs_observed"] = summary["cascade_aware_pr_auc"] - summary["observed_only_pr_auc"]
        if "clinical_no_cascade_pr_auc" in summary.columns and "full_pr_auc" in summary.columns:
            summary["delta_pr_auc_full_vs_clinical"] = summary["full_pr_auc"] - summary["clinical_no_cascade_pr_auc"]
        return summary.sort_values("model_name").reset_index(drop=True)
