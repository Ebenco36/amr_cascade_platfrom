"""Patient-grouped out-of-fold decision-threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.modeling.evaluation.classification_metrics import ClassificationMetrics
from amr_cascade_platform.modeling.thresholding.base import (
    EstimatorFactory,
    ThresholdSelectionResult,
)


class GroupedOOFThresholdSelector:
    """Tune the threshold on leak-safe out-of-fold training predictions.

    All rows from one patient remain in one fold. This is essential because a
    culture episode contributes several ordered drug-pair rows and a patient
    may contribute multiple culture episodes.
    """

    name = "grouped_oof"
    uses_external_validation = False
    _TARGET_DERIVED_FEATURE_PREFIXES = (
        "train_pair_",
        "train_upstream_",
        "train_downstream_",
    )

    def __init__(self, settings: Settings, metrics: ClassificationMetrics) -> None:
        self._settings = settings
        self._metrics = metrics

    def validate_feature_columns(self, feature_columns: tuple[str, ...]) -> None:
        self._reject_precomputed_target_features(feature_columns)

    def select(
        self,
        *,
        predictions: pd.DataFrame,
        train_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        id_columns: tuple[str, ...],
        estimator_factory: EstimatorFactory,
        model_name: str,
        feature_set_name: str,
    ) -> ThresholdSelectionResult:
        del predictions
        frame = train_frame.reset_index(drop=True).copy()
        group_column = self._settings.evaluation.threshold_cv_group_column
        n_splits = int(self._settings.evaluation.threshold_cv_folds)
        self.validate_feature_columns(feature_columns)
        self._validate_training_frame(frame, group_column, n_splits)

        target = pd.to_numeric(frame["target"], errors="raise").astype(int)
        groups = frame[group_column]
        folds, fold_assignment = self._patient_folds(target, groups, n_splits)
        selection_frames: list[pd.DataFrame] = []
        assigned = np.zeros(len(frame), dtype=np.int8)

        for fold_number, (fit_positions, holdout_positions) in enumerate(folds, start=1):
            fit_groups = set(groups.iloc[fit_positions].tolist())
            holdout_groups = set(groups.iloc[holdout_positions].tolist())
            if fit_groups & holdout_groups:
                raise ValidationError(
                    "Grouped threshold cross-validation leaked patient groups across folds."
                )
            fit_target = target.iloc[fit_positions]
            holdout_target = target.iloc[holdout_positions]
            if fit_target.nunique() < 2 or holdout_target.nunique() < 2:
                raise ValidationError(
                    "Grouped threshold cross-validation produced a fold without both target "
                    f"classes (fold={fold_number}). Reduce threshold_cv_folds or increase "
                    "training support."
                )

            fold_estimator = estimator_factory()
            fold_estimator.fit(
                frame.iloc[fit_positions].loc[:, feature_columns],
                fit_target,
            )
            probabilities = fold_estimator.predict_proba(
                frame.iloc[holdout_positions].loc[:, feature_columns]
            )
            output_columns = list(dict.fromkeys(id_columns + (group_column, "target")))
            fold_output = frame.iloc[holdout_positions].loc[:, output_columns].copy()
            fold_output["model_name"] = model_name
            fold_output["feature_set"] = feature_set_name
            fold_output["split"] = "training_oof"
            fold_output["selection_source"] = "training_oof"
            fold_output["selection_fold"] = fold_number
            fold_output["selection_row_position"] = holdout_positions
            fold_output["predicted_probability"] = np.asarray(probabilities, dtype=float)
            selection_frames.append(fold_output)
            assigned[holdout_positions] += 1

        if not np.all(assigned == 1):
            raise ValidationError(
                "Grouped threshold cross-validation did not predict every training row exactly once."
            )
        selection_predictions = (
            pd.concat(selection_frames, ignore_index=True)
            .sort_values("selection_row_position")
            .reset_index(drop=True)
        )
        selected_threshold, threshold_table = self._metrics.select_threshold(
            y_true=selection_predictions["target"],
            y_probability=selection_predictions["predicted_probability"],
            objective=self._settings.evaluation.threshold_selection_metric,
            thresholds=self._settings.evaluation.threshold_candidates,
            default_threshold=self._settings.evaluation.probability_threshold,
        )
        return ThresholdSelectionResult(
            selected_threshold=selected_threshold,
            threshold_table=threshold_table,
            selection_predictions=selection_predictions,
            strategy=self.name,
            selection_source="training_oof",
            group_column=group_column,
            cv_folds=n_splits,
            selection_rows=len(selection_predictions),
            selection_groups=int(groups.nunique()),
            fold_assignment=fold_assignment,
        )

    def _patient_folds(
        self, target: pd.Series, groups: pd.Series, n_splits: int
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
        """Patient-grouped, outcome-stratified folds in which every fold holds both classes.

        StratifiedGroupKFold without shuffling assigns patients greedily to keep
        each fold's outcome distribution closest to the whole; it is
        deterministic, so no seed is involved. (With shuffle=True, scikit-learn
        1.6 can put every patient of one class into a single fold even when a
        balanced split exists.) If a fold still lacks a class, patients are
        ranked by their positive rate and dealt round-robin, which puts at
        least one patient of each class in every fold whenever each class
        spans at least ``n_splits`` patients -- the condition
        _validate_training_frame already enforces.
        """
        positions = np.arange(len(target))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
        folds = [
            (fit, holdout)
            for fit, holdout in splitter.split(positions.reshape(-1, 1), target.to_numpy(), groups.to_numpy())
        ]
        if all(target.iloc[holdout].nunique() == 2 for _, holdout in folds):
            return folds, "stratified_group_kfold"
        codes, uniques = pd.factorize(groups, sort=True)
        rate = np.bincount(codes, weights=target.to_numpy(dtype=float)) / np.bincount(codes)
        tie_break = np.random.default_rng(self._settings.modeling.random_state).random(len(uniques))
        order = np.lexsort((tie_break, -rate))
        fold_of_group = np.empty(len(uniques), dtype=int)
        fold_of_group[order] = np.arange(len(uniques)) % n_splits
        fold_of_row = fold_of_group[codes]
        folds = [(positions[fold_of_row != fold], positions[fold_of_row == fold]) for fold in range(n_splits)]
        return folds, "rate_ranked_round_robin"

    @classmethod
    def _reject_precomputed_target_features(cls, feature_columns: tuple[str, ...]) -> None:
        contaminated = sorted(
            column
            for column in feature_columns
            if column.startswith(cls._TARGET_DERIVED_FEATURE_PREFIXES)
        )
        if contaminated:
            raise ValidationError(
                "grouped_oof cannot use cascade features precomputed from all training-site "
                "outcomes because the held-out fold influenced those features. Use "
                "validation_holdout for cascade_aware/full models, or regenerate cascade "
                "features independently inside every fold. Contaminated columns: "
                f"{contaminated}"
            )

    @staticmethod
    def _validate_training_frame(
        frame: pd.DataFrame,
        group_column: str,
        n_splits: int,
    ) -> None:
        if n_splits < 2:
            raise ValidationError("threshold_cv_folds must be at least 2.")
        required = {group_column, "target"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValidationError(
                "Grouped threshold cross-validation is missing required columns: "
                f"{missing}."
            )
        if frame.empty:
            raise ValidationError("Grouped threshold cross-validation received no training rows.")
        if frame[group_column].isna().any():
            raise ValidationError(
                f"Grouped threshold cross-validation requires complete '{group_column}' values."
            )
        if frame[group_column].nunique() < n_splits:
            raise ValidationError(
                f"Grouped threshold cross-validation requires at least {n_splits} distinct "
                f"'{group_column}' groups."
            )
        target = pd.to_numeric(frame["target"], errors="raise").astype(int)
        if set(target.unique()) != {0, 1}:
            raise ValidationError(
                "Grouped threshold cross-validation requires a binary target containing 0 and 1."
            )
        groups_per_class = frame.assign(__target=target).groupby("__target")[group_column].nunique()
        if int(groups_per_class.min()) < n_splits:
            raise ValidationError(
                "Grouped threshold cross-validation has fewer patient groups than folds in "
                "at least one target class. Reduce threshold_cv_folds or increase support."
            )
