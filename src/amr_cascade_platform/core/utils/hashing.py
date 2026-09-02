"""Hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return an MD5 checksum for a file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
