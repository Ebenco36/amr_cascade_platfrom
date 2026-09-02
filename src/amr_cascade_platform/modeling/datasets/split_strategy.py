"""Site-based split strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class SiteSplitStrategy:
    """Split modeling datasets by predefined site membership."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._validate_configured_sites()

    def split(self, dataframe: pd.DataFrame) -> DatasetSplit:
        split_column = self._settings.modeling.split_column
        if split_column not in dataframe.columns:
            raise ValidationError(f"Split column '{split_column}' not found in modeling dataset.")

        train = dataframe[dataframe[split_column].isin(self._settings.modeling.train_sites)].copy()
        validation = dataframe[dataframe[split_column].isin(self._settings.modeling.validation_sites)].copy()
        test = dataframe[dataframe[split_column].isin(self._settings.modeling.test_sites)].copy()

        if train.empty or validation.empty or test.empty:
            raise ValidationError(
                "Site split produced an empty partition. Confirm combined-site inputs are available for "
                f"train={self._settings.modeling.train_sites}, "
                f"validation={self._settings.modeling.validation_sites}, "
                f"test={self._settings.modeling.test_sites}."
            )
        self._validate_partition_overlap(train, validation, test)
        return DatasetSplit(train=train, validation=validation, test=test)

    def _validate_configured_sites(self) -> None:
        configured = {
            "train": set(self._settings.modeling.train_sites),
            "validation": set(self._settings.modeling.validation_sites),
            "test": set(self._settings.modeling.test_sites),
        }
        for left_name, left_sites in configured.items():
            for right_name, right_sites in configured.items():
                if left_name >= right_name:
                    continue
                overlap = left_sites & right_sites
                if overlap:
                    raise ValidationError(
                        f"Modeling split configuration is not disjoint: {left_name} and {right_name} "
                        f"share sites {sorted(overlap)}."
                    )

    def _validate_partition_overlap(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        episode_keys = list(self._settings.gold.episode_key_columns)
        if not all(column in train.columns for column in episode_keys):
            return

        # Site-based external validation should keep each culture episode in exactly one split.
        # This explicit check catches accidental leakage if the split column or source data are
        # ever changed in a way that breaks that assumption.
        train_keys = self._as_key_set(train, episode_keys)
        validation_keys = self._as_key_set(validation, episode_keys)
        test_keys = self._as_key_set(test, episode_keys)
        if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
            raise ValidationError("Modeling split produced overlapping episode keys across partitions.")

    @staticmethod
    def _as_key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[object, ...]]:
        return set(map(tuple, frame.loc[:, columns].drop_duplicates().itertuples(index=False, name=None)))


class TemporalSplitStrategy:
    """Split modeling datasets into train/validation/test partitions by culture year."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.modeling.temporal_train_end_year >= settings.modeling.temporal_validation_end_year:
            raise ValidationError("Temporal split configuration must satisfy train_end_year < validation_end_year.")
        if settings.modeling.temporal_validation_end_year >= settings.modeling.temporal_test_end_year:
            raise ValidationError("Temporal split configuration must satisfy validation_end_year < test_end_year.")

    def split(self, dataframe: pd.DataFrame) -> DatasetSplit:
        time_column = "order_time_jittered"
        if time_column not in dataframe.columns:
            raise ValidationError(f"Temporal split requires '{time_column}' in modeling dataset.")

        frame = dataframe.copy()
        # The harmonized pipeline may preserve equivalent timestamps in multiple valid string
        # layouts (for example ISO-8601 with `T` or a space separator). `format='mixed'` keeps
        # temporal validation robust to that harmless formatting variation.
        timestamps = pd.to_datetime(frame[time_column], errors="coerce", utc=True, format="mixed")
        frame["__culture_year"] = timestamps.dt.year
        if frame["__culture_year"].isna().any():
            raise ValidationError("Temporal split requires parseable culture timestamps for all rows.")

        train = frame[frame["__culture_year"] <= self._settings.modeling.temporal_train_end_year].copy()
        validation = frame[
            (frame["__culture_year"] > self._settings.modeling.temporal_train_end_year)
            & (frame["__culture_year"] <= self._settings.modeling.temporal_validation_end_year)
        ].copy()
        test = frame[
            (frame["__culture_year"] > self._settings.modeling.temporal_validation_end_year)
            & (frame["__culture_year"] <= self._settings.modeling.temporal_test_end_year)
        ].copy()
        if train.empty or validation.empty or test.empty:
            raise ValidationError(
                "Temporal split produced an empty partition. Confirm culture years span "
                f"train<= {self._settings.modeling.temporal_train_end_year}, "
                f"validation<= {self._settings.modeling.temporal_validation_end_year}, "
                f"test<= {self._settings.modeling.temporal_test_end_year}."
            )
        episode_keys = list(self._settings.gold.episode_key_columns)
        if all(column in frame.columns for column in episode_keys):
            train_keys = SiteSplitStrategy._as_key_set(train, episode_keys)
            validation_keys = SiteSplitStrategy._as_key_set(validation, episode_keys)
            test_keys = SiteSplitStrategy._as_key_set(test, episode_keys)
            if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
                raise ValidationError("Temporal split produced overlapping episode keys across partitions.")
        return DatasetSplit(
            train=train.drop(columns="__culture_year"),
            validation=validation.drop(columns="__culture_year"),
            test=test.drop(columns="__culture_year"),
        )
