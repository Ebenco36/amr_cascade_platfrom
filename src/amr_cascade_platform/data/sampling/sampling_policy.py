"""Sampling strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplingDecision:
    use_dask: bool


class FileSizeSamplingPolicy:
    """Choose a sampling backend from file size."""

    def __init__(self, dask_threshold_mb: float) -> None:
        self._dask_threshold_mb = dask_threshold_mb

    def decide(self, file_path: Path, dask_available: bool) -> SamplingDecision:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return SamplingDecision(use_dask=dask_available and size_mb > self._dask_threshold_mb)
