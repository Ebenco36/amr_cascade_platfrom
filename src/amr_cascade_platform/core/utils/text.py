"""Text normalization helpers."""

from __future__ import annotations


def normalize_label(value: str | None) -> str:
    """Normalize organism or antibiotic labels for joins and comparisons."""
    if value is None:
        return ""
    return " ".join(str(value).strip().upper().split())


def safe_feature_name(value: str) -> str:
    """Convert a text label into a stable feature suffix."""
    normalized = normalize_label(value).lower()
    for token in (" ", "/", "-", "(", ")", ",", "."):
        normalized = normalized.replace(token, "_")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
