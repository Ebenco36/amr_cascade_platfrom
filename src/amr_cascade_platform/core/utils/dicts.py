"""Dictionary helpers."""

from __future__ import annotations

from collections.abc import Mapping


def deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge dictionaries without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged
