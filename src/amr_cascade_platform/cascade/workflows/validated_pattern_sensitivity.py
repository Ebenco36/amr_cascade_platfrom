"""Sensitivity analyses applied to every pattern in the final validated set, run in shards."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer

PAIR_KEY = ["upstream_antibiotic", "downstream_antibiotic"]


@dataclass(frozen=True)
class ValidatedPatternAnalysis:
    output_name: str
    shard_dir_name: str
    statuses: frozenset[str]
    run: Callable[[CascadeValidationAnalyzer, pd.DataFrame, pd.DataFrame], pd.DataFrame]


ANALYSES: dict[str, ValidatedPatternAnalysis] = {
    "era-stratified": ValidatedPatternAnalysis(
        output_name="era_stratified_permutation_sensitivity.parquet",
        shard_dir_name="era_stratified_shards",
        statuses=CascadeValidationAnalyzer.VALIDATED_STATUSES,
        run=lambda analyzer, pairs, edges: analyzer.era_stratified_sensitivity_summary(pairs, edges),
    ),
    "patient-cluster": ValidatedPatternAnalysis(
        output_name="patient_cluster_bootstrap_sensitivity.parquet",
        shard_dir_name="patient_cluster_shards",
        statuses=frozenset({"robust"}),
        run=lambda analyzer, pairs, edges: analyzer.patient_cluster_sensitivity_summary(pairs, edges),
    ),
}


def select_edges(validation_results: pd.DataFrame, statuses: frozenset[str]) -> pd.DataFrame:
    selected = validation_results.loc[
        validation_results["validation_status"].isin(statuses), PAIR_KEY + ["observed_escalation_ratio"]
    ].rename(columns={"observed_escalation_ratio": "escalation_ratio"})
    return selected.sort_values(PAIR_KEY, kind="mergesort").reset_index(drop=True)


def shard_slice(edges: pd.DataFrame, shard_index: int, shard_total: int) -> pd.DataFrame:
    ordered = edges.sort_values(PAIR_KEY, kind="mergesort").reset_index(drop=True)
    per_shard = math.ceil(len(ordered) / shard_total)
    return ordered.iloc[shard_index * per_shard:(shard_index + 1) * per_shard].reset_index(drop=True)


def merge_shard_frames(frames: list[pd.DataFrame], expected_keys: set[tuple[str, str]]) -> pd.DataFrame:
    merged = pd.concat([frame for frame in frames if not frame.empty] or frames[:1], ignore_index=True)
    keys = list(zip(merged["upstream_antibiotic"].astype(str), merged["downstream_antibiotic"].astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError("Shards overlap: a pattern appears in more than one shard.")
    missing, unexpected = expected_keys - set(keys), set(keys) - expected_keys
    if missing or unexpected:
        raise ValueError(
            f"Shards do not match the selected pattern set: {len(missing)} missing, {len(unexpected)} unexpected."
        )
    return merged


def expected_keys(edges: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(edges["upstream_antibiotic"].astype(str), edges["downstream_antibiotic"].astype(str)))


def shard_path(shard_dir: Path, shard_index: int, shard_total: int) -> Path:
    return shard_dir / f"shard_{shard_index:04d}_of_{shard_total:04d}.parquet"
