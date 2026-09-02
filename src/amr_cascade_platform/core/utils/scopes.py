"""Helpers for scoped output directories and organism labels."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.core.utils.text import normalize_label, safe_feature_name


def organism_slug(organism: str | None) -> str | None:
    """Return a stable directory-friendly slug for an organism label."""
    if not organism:
        return None
    normalized = normalize_label(organism)
    return safe_feature_name(normalized) or None


def scoped_output_dir(
    root: Path,
    scope: str,
    site: str | None = None,
    organism: str | None = None,
) -> Path:
    """Build a scope-aware directory path, optionally nested by organism."""
    if scope == "site" and site:
        base = root / site
    else:
        base = root / "combined"
    slug = organism_slug(organism)
    if slug:
        return base / "organisms" / slug
    return base
