"""Silver-stage table cleaning orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.data.cleaning.key_normalizer import KeyNormalizer
from amr_cascade_platform.data.cleaning.null_handler import NullHandler


@dataclass(frozen=True)
class CleaningResult:
    dataframe: pd.DataFrame
    rows_before: int
    rows_after: int
    duplicates_removed: int
    discordant_susceptibility_group_n: int = 0
    discordant_susceptibility_row_n: int = 0
    shard_coalesced_group_n: int = 0
    shard_conflict_group_n: int = 0
    shard_conflict_cell_n: int = 0


class TableCleaner:
    """Apply deterministic cleaning rules to bronze tables."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._null_handler = NullHandler(settings.platform.missing_tokens)
        self._key_normalizer = KeyNormalizer(
            id_columns=settings.platform.id_columns,
            trim_strings=settings.cleaning.trim_strings,
            uppercase_standard_columns=settings.cleaning.uppercase_standard_columns,
        )

    def clean(self, dataframe: pd.DataFrame, canonical_table: str) -> CleaningResult:
        return self._clean_dataframe(dataframe, canonical_table, log_result=True)

    def clean_partition(self, dataframe: pd.DataFrame, canonical_table: str) -> pd.DataFrame:
        return self._clean_dataframe(dataframe, canonical_table, log_result=False).dataframe

    def _clean_dataframe(
        self,
        dataframe: pd.DataFrame,
        canonical_table: str,
        *,
        log_result: bool,
    ) -> CleaningResult:
        rows_before = len(dataframe)
        cleaned = self._null_handler.normalize(dataframe)
        cleaned = self._key_normalizer.normalize(cleaned)
        cleaned = self._harmonize_time_columns(cleaned)
        cleaned, discordant_group_n, discordant_row_n = self._exclude_discordant_susceptibility(
            cleaned, canonical_table
        )
        (
            cleaned,
            duplicates_removed,
            shard_coalesced_group_n,
            shard_conflict_group_n,
            shard_conflict_cell_n,
        ) = self._drop_duplicates(cleaned, canonical_table)
        rows_after = len(cleaned)
        if log_result:
            self._logger.info(
                "Cleaned %s | rows_before=%s rows_after=%s duplicates_removed=%s discordant_susceptibility_groups=%s discordant_susceptibility_rows=%s shard_coalesced_groups=%s shard_conflict_groups=%s shard_conflict_cells=%s",
                canonical_table,
                rows_before,
                rows_after,
                duplicates_removed,
                discordant_group_n,
                discordant_row_n,
                shard_coalesced_group_n,
                shard_conflict_group_n,
                shard_conflict_cell_n,
            )
        return CleaningResult(
            dataframe=cleaned,
            rows_before=rows_before,
            rows_after=rows_after,
            duplicates_removed=duplicates_removed,
            discordant_susceptibility_group_n=discordant_group_n,
            discordant_susceptibility_row_n=discordant_row_n,
            shard_coalesced_group_n=shard_coalesced_group_n,
            shard_conflict_group_n=shard_conflict_group_n,
            shard_conflict_cell_n=shard_conflict_cell_n,
        )

    def _harmonize_time_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = dataframe.copy()
        if "order_time_jittered_utc" in normalized.columns and "order_time_jittered" not in normalized.columns:
            normalized["order_time_jittered"] = normalized["order_time_jittered_utc"]
        return normalized

    def _drop_duplicates(self, dataframe: pd.DataFrame, canonical_table: str) -> tuple[pd.DataFrame, int, int, int, int]:
        rule = self._settings.cleaning.duplicate_resolution.per_table.get(canonical_table)
        before = len(dataframe)
        shard_coalesced_group_n = 0
        shard_conflict_group_n = 0
        shard_conflict_cell_n = 0
        if rule and rule.strategy == "shard_coalesce":
            key = [c for c in rule.subset if c in dataframe.columns]
            if key:
                duplicate_mask = dataframe.duplicated(subset=key, keep=False)
                shard_coalesced_group_n = int(dataframe.loc[duplicate_mask, key].drop_duplicates().shape[0])
                shard_conflict_group_n, shard_conflict_cell_n = self._audit_shard_conflicts(dataframe, key)
                deduplicated = self._coalesce_shards(dataframe, key)
            else:
                deduplicated = dataframe.drop_duplicates()
        else:
            subset = None
            if rule:
                available_subset = [column for column in rule.subset if column in dataframe.columns]
                subset = available_subset or None
            deduplicated = dataframe.drop_duplicates(subset=subset)
        removed = before - len(deduplicated)
        return deduplicated, removed, shard_coalesced_group_n, shard_conflict_group_n, shard_conflict_cell_n

    @staticmethod
    def _audit_shard_conflicts(dataframe: pd.DataFrame, key: list[str]) -> tuple[int, int]:
        """Count rare non-null value disagreements within shard groups.

        A conflict is column-specific: the same shard key has more than one
        distinct non-null value for the same measurement column. Complementary
        sparse rows that fill different null slots are not conflicts.
        """
        value_cols = [c for c in dataframe.columns if c not in key]
        if not value_cols:
            return 0, 0
        conflict_keys: set[tuple[object, ...]] = set()
        conflict_cell_n = 0
        for column in value_cols:
            non_null = dataframe.loc[dataframe[column].notna(), key + [column]]
            if non_null.empty:
                continue
            nunique = (
                non_null.groupby(key, dropna=False)[column]
                .nunique(dropna=True)
                .reset_index(name="_nunique")
            )
            conflicted = nunique.loc[nunique["_nunique"] > 1, key]
            if conflicted.empty:
                continue
            conflict_cell_n += int(len(conflicted))
            conflict_keys.update(map(tuple, conflicted.itertuples(index=False, name=None)))
        return len(conflict_keys), conflict_cell_n

    @staticmethod
    def _coalesce_shards(dataframe: pd.DataFrame, key: list[str]) -> pd.DataFrame:
        """Merge measurement-shard rows into one record per key.

        Each unique key may have N rows where each row is a sparse "shard"
        carrying values for a different measurement component (e.g. one lab
        panel per row). Aggregation is column-prefix-aware:

        - ``median_*``, ``Q25_*``, ``Q75_*``  → numeric median of non-null values
          (one shard: passes through; rare conflicts: robust pooled estimate)
        - ``first_*``  → first non-null across shards (earliest value retained)
        - ``last_*``   → last non-null across shards (latest value retained)
        - everything else → first non-null (covers Period_Day and similar metadata)
        """
        value_cols = [c for c in dataframe.columns if c not in key]
        if not value_cols:
            return dataframe.drop_duplicates(subset=key)

        stat_prefixes = ("median_", "Q25_", "Q75_")
        stat_cols = [c for c in value_cols if c.startswith(stat_prefixes)]
        first_cols = [c for c in value_cols if c.startswith("first_")]
        last_cols = [c for c in value_cols if c.startswith("last_")]
        other_cols = [c for c in value_cols if c not in stat_cols + first_cols + last_cols]

        def _to_numeric_frame(cols: list[str]) -> pd.DataFrame:
            sub = dataframe[key + cols].copy()
            for c in cols:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            return sub

        parts: list[pd.DataFrame] = []
        if stat_cols:
            parts.append(_to_numeric_frame(stat_cols).groupby(key, dropna=False)[stat_cols].median())
        if first_cols:
            parts.append(_to_numeric_frame(first_cols).groupby(key, dropna=False)[first_cols].first())
        if last_cols:
            parts.append(_to_numeric_frame(last_cols).groupby(key, dropna=False)[last_cols].last())
        if other_cols:
            parts.append(dataframe[key + other_cols].groupby(key, dropna=False)[other_cols].first())

        if not parts:
            return dataframe.drop_duplicates(subset=key)

        combined = pd.concat(parts, axis=1).reset_index()
        # Restore original column order
        original_order = [c for c in dataframe.columns if c in combined.columns]
        return combined[original_order].reset_index(drop=True)

    def _exclude_discordant_susceptibility(
        self,
        dataframe: pd.DataFrame,
        canonical_table: str,
    ) -> tuple[pd.DataFrame, int, int]:
        """Drop entire groups whose dedup key maps to more than one distinct
        susceptibility value, rather than letting the later same-key dedup step
        silently keep whichever row happens to come first.

        The configured dedup subset for microbial_resistance does not include
        susceptibility, so two rows for the same episode/drug/time with
        RESISTANT vs SUSCEPTIBLE are "duplicates" by that key -- an unresolvable
        conflict, not noise. Mirrors PairGenerator's discordant-duplicate
        handling in the gold layer: exclude the whole group, never guess a
        branch, and log the exclusion for audit.
        """
        if canonical_table != "microbial_resistance" or "susceptibility" not in dataframe.columns:
            return dataframe, 0, 0

        rule = self._settings.cleaning.duplicate_resolution.per_table.get(canonical_table)
        if not rule:
            return dataframe, 0, 0

        subset = [column for column in rule.subset if column in dataframe.columns]
        if not subset:
            return dataframe, 0, 0

        audit_frame = dataframe.loc[:, subset + ["susceptibility"]].copy()
        audit_frame["susceptibility"] = audit_frame["susceptibility"].astype("string").str.strip()
        valid = audit_frame["susceptibility"].notna() & audit_frame["susceptibility"].ne("")
        audit_frame = audit_frame.loc[valid]
        if audit_frame.empty:
            return dataframe, 0, 0

        susceptibility_counts = (
            audit_frame.groupby(subset, dropna=False)["susceptibility"]
            .nunique(dropna=True)
            .reset_index(name="susceptibility_nunique")
        )
        discordant_keys = susceptibility_counts.loc[
            susceptibility_counts["susceptibility_nunique"] > 1,
            subset,
        ]
        if discordant_keys.empty:
            return dataframe, 0, 0

        marker = discordant_keys.copy()
        marker["_discordant"] = True
        merged = dataframe.merge(marker, on=subset, how="left", validate="many_to_one")
        discordant_mask = merged["_discordant"].to_numpy(dtype=bool, na_value=False)
        excluded = dataframe.loc[discordant_mask]
        retained = dataframe.loc[~discordant_mask]
        return retained.reset_index(drop=True), int(len(discordant_keys)), int(len(excluded))
