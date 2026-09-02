"""Repository interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class FileRepository(ABC):
    """Abstract file repository contract."""

    @abstractmethod
    def exists(self, path: Path) -> bool:
        """Return whether a path exists."""
