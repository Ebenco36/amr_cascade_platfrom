"""Canonical table discovery."""

from __future__ import annotations

from pathlib import Path

import yaml

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager


class CanonicalTableMapper:
    """Map site-specific source file names onto canonical domain names."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._path_manager = path_manager
        mapping_path = (
            path_manager.paths.reference
            / settings.platform.reference_files["canonical_table_map"]
        )
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        self._mapping = payload.get("sites", {})

    def site_tables(self, site: str) -> dict[str, Path]:
        if site not in self._mapping:
            raise DataDiscoveryError(f"No canonical mapping configured for site '{site}'")

        site_root = self._path_manager.site_dir("raw", site)
        mapped_paths: dict[str, Path] = {}
        missing_required: list[str] = []

        for canonical_name, filename in self._mapping[site].items():
            path = site_root / filename
            if path.exists():
                mapped_paths[canonical_name] = path
            elif canonical_name not in self._settings.platform.optional_tables:
                missing_required.append(canonical_name)

        if missing_required:
            raise DataDiscoveryError(
                f"Missing required mapped tables for site '{site}': {', '.join(sorted(missing_required))}"
            )
        return mapped_paths
