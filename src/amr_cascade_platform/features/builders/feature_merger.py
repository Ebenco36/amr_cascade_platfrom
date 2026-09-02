"""Lossless feature merging utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError


@dataclass(frozen=True)
class MergeResult:
    dataframe: pd.DataFrame
    added_feature_columns: tuple[str, ...]
    matched_rows: int
    unmatched_rows: int


class FeatureMerger:
    """Merge episode-level features onto base modeling rows without row loss."""

    def merge_left(
        self,
        base: pd.DataFrame,
        features: pd.DataFrame,
        join_keys: tuple[str, ...],
    ) -> MergeResult:
        if base.empty:
            return MergeResult(
                dataframe=base.copy(),
                added_feature_columns=tuple(),
                matched_rows=0,
                unmatched_rows=0,
            )
        if features.empty:
            return MergeResult(
                dataframe=base.copy(),
                added_feature_columns=tuple(),
                matched_rows=0,
                unmatched_rows=len(base),
            )

        missing_keys = [key for key in join_keys if key not in base.columns or key not in features.columns]
        if missing_keys:
            raise ValidationError(f"Missing merge keys for feature join: {missing_keys}")

        duplicate_mask = features.duplicated(list(join_keys), keep=False)
        if duplicate_mask.any():
            raise ValidationError("Feature source contains duplicate rows for merge keys; refusing lossy join.")

        pre_merge_rows = len(base)
        feature_columns = [column for column in features.columns if column not in join_keys]
        merged = base.merge(
            features,
            on=list(join_keys),
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if len(merged) != pre_merge_rows:
            raise ValidationError(
                f"Feature merge changed row count from {pre_merge_rows} to {len(merged)}."
            )

        matched_rows = int((merged["_merge"] == "both").sum())
        unmatched_rows = int((merged["_merge"] == "left_only").sum())
        merged = merged.drop(columns="_merge")

        added_columns: list[str] = []
        for column in feature_columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
            if pd.api.types.is_float_dtype(merged[column]) and (merged[column] % 1 == 0).all():
                merged[column] = merged[column].astype("int64")
            added_columns.append(column)
        return MergeResult(
            dataframe=merged,
            added_feature_columns=tuple(sorted(added_columns)),
            matched_rows=matched_rows,
            unmatched_rows=unmatched_rows,
        )
