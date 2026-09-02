"""Local filesystem repository implementation."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.core.interfaces.repository import FileRepository


class LocalFileRepository(FileRepository):
    """Filesystem-backed repository."""

    def exists(self, path: Path) -> bool:
        return path.exists()
