"""Feature-set selection for modeling ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from amr_cascade_platform.modeling.datasets.pair_feature_matrix_builder import ModelingDatasetBundle


@dataclass(frozen=True)
class ModelingFeatureSet:
    """One named feature-set view over a common modeling dataframe."""

    name: str
    description: str
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


class ModelingFeatureSetSelector:
    """Construct publication-facing feature-set ablations."""

    CASCADE_PREFIXES = ("train_pair_", "train_upstream_", "train_downstream_")

    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path

    def select(
        self,
        base_dataset: ModelingDatasetBundle,
        full_dataset: ModelingDatasetBundle,
        requested_sets: tuple[str, ...] | None = None,
        ablation: bool = False,
    ) -> tuple[ModelingFeatureSet, ...]:
        clinical_numeric = self._clinical_numeric_columns(full_dataset)
        cascade_numeric = tuple(
            column
            for column in base_dataset.numeric_columns
            if column.startswith(self.CASCADE_PREFIXES)
        )
        core_numeric = tuple(
            column for column in base_dataset.numeric_columns if column not in cascade_numeric
        )
        categorical = base_dataset.categorical_columns

        feature_sets = {
            "observed_only": ModelingFeatureSet(
                name="observed_only",
                description="Observed pair/context features without cascade or clinical enrichment.",
                feature_columns=categorical + core_numeric,
                categorical_columns=categorical,
                numeric_columns=core_numeric,
            ),
            "cascade_aware": ModelingFeatureSet(
                name="cascade_aware",
                description="Observed pair/context features plus cascade-derived pair and network summaries.",
                feature_columns=categorical + core_numeric + cascade_numeric,
                categorical_columns=categorical,
                numeric_columns=core_numeric + cascade_numeric,
            ),
            "clinical_no_cascade": ModelingFeatureSet(
                name="clinical_no_cascade",
                description="Observed pair/context features plus baseline, acute, history, and comorbidity enrichment, excluding cascade features.",
                feature_columns=categorical + core_numeric + clinical_numeric,
                categorical_columns=categorical,
                numeric_columns=core_numeric + clinical_numeric,
            ),
            "full": ModelingFeatureSet(
                name="full",
                description="Observed pair/context, cascade-aware, and clinical enrichment combined.",
                feature_columns=categorical + core_numeric + cascade_numeric + clinical_numeric,
                categorical_columns=categorical,
                numeric_columns=core_numeric + cascade_numeric + clinical_numeric,
            ),
        }
        for key, feature_set in tuple(feature_sets.items()):
            feature_sets[key] = ModelingFeatureSet(
                name=feature_set.name,
                description=feature_set.description,
                feature_columns=self._dedupe_existing(feature_set.feature_columns, full_dataset),
                categorical_columns=self._dedupe_existing(feature_set.categorical_columns, full_dataset),
                numeric_columns=self._dedupe_existing(feature_set.numeric_columns, full_dataset),
            )

        if requested_sets:
            return tuple(feature_sets[name] for name in requested_sets)
        if ablation:
            return tuple(feature_sets[name] for name in ("observed_only", "cascade_aware", "clinical_no_cascade", "full"))
        return (feature_sets["full"],)

    def _clinical_numeric_columns(self, dataset: ModelingDatasetBundle) -> tuple[str, ...]:
        if not self._metadata_path.exists():
            return tuple(
                column
                for column in dataset.numeric_columns
                if not column.startswith(self.CASCADE_PREFIXES)
                and column not in ("upstream_positive", "was_positive_numeric", "downstream_intrinsic_resistance")
            )
        payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        clinical_columns: list[str] = []
        for key in (
            "baseline_feature_columns",
            "acute_feature_columns",
            "history_feature_columns",
            "comorbidity_feature_columns",
        ):
            clinical_columns.extend(payload.get(key, []))
        return self._dedupe_existing(tuple(clinical_columns), dataset)

    @staticmethod
    def _dedupe_existing(columns: tuple[str, ...], dataset: ModelingDatasetBundle) -> tuple[str, ...]:
        seen: set[str] = set()
        available = set(dataset.dataframe.columns)
        ordered: list[str] = []
        for column in columns:
            if column not in available or column in seen:
                continue
            seen.add(column)
            ordered.append(column)
        return tuple(ordered)
