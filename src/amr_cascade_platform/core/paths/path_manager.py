"""Project path management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings


@dataclass(frozen=True)
class DatasetPaths:
    raw: Path
    sample: Path
    bronze: Path
    silver: Path
    harmonized: Path
    gold: Path
    interim: Path
    features: Path
    artifacts: Path
    metadata: Path
    reference: Path
    outputs: Path


class PathManager:
    """Resolve canonical repository paths from settings."""

    def __init__(self, project_root: Path, settings: Settings) -> None:
        self._project_root = project_root
        data_root = project_root / settings.environment.data_root
        self._paths = DatasetPaths(
            raw=data_root / "raw",
            sample=data_root / "sample",
            bronze=data_root / "bronze",
            silver=data_root / "silver",
            harmonized=data_root / "harmonized",
            gold=data_root / "gold",
            interim=data_root / "interim",
            features=data_root / "features",
            artifacts=data_root / "artifacts",
            metadata=data_root / "metadata",
            reference=data_root / "reference",
            outputs=project_root / "outputs",
        )

    @property
    def paths(self) -> DatasetPaths:
        return self._paths

    @property
    def project_root(self) -> Path:
        return self._project_root

    def ensure_standard_directories(self) -> None:
        for path in self._paths.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)

    def site_dir(self, layer: str, site: str) -> Path:
        root = getattr(self._paths, layer)
        return root / site
