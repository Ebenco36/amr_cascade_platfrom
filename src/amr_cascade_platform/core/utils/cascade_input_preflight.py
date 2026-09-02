"""Preflight validation helpers for cascade-ready organism inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.core.utils.eskape import EskapeTarget
from amr_cascade_platform.core.utils.text import normalize_label


def available_labels(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique non-null labels from a dataframe column."""
    if column not in frame.columns:
        return []
    labels = frame[column].dropna().astype("string").tolist()
    return sorted({str(label) for label in labels if str(label).strip()})


def require_organism_present(frame: pd.DataFrame, organism: str, dataset_label: str, column: str = "organism") -> None:
    """Raise a clear error if the requested organism label is not present."""
    if column not in frame.columns:
        raise ValidationError(
            f"{dataset_label} does not contain the required '{column}' column for organism filtering."
        )
    normalized = frame[column].map(normalize_label)
    organism_key = normalize_label(organism)
    match_count = int((normalized == organism_key).sum())
    if match_count > 0:
        return
    present = available_labels(frame, column)
    preview = ", ".join(present[:20]) if present else "none"
    raise ValidationError(
        f"{dataset_label} has no usable rows for organism '{organism}' in column '{column}'. "
        f"Available labels: {preview}"
    )


def require_armd_eskape_targets(
    cohort: pd.DataFrame,
    targets: list[EskapeTarget],
    *,
    allow_missing: bool = False,
) -> list[EskapeTarget]:
    """Validate ARMD cohort availability for requested ESKAPE targets."""
    normalized = cohort["organism"].map(normalize_label) if "organism" in cohort.columns else pd.Series(dtype="string")
    missing: list[EskapeTarget] = []
    for target in targets:
        match_count = int(normalized.str.contains(target.armd_match_pattern, regex=True, na=False).sum())
        if match_count == 0:
            missing.append(target)
    if missing and not allow_missing:
        missing_text = ", ".join(f"{target.display_name} ({target.scientific_focus})" for target in missing)
        present_preview = ", ".join(available_labels(cohort, "organism")[:20])
        raise ValidationError(
            "ARMD ESKAPE preflight failed because some requested targets have no usable rows in the "
            f"current harmonized cohort: {missing_text}. Example available labels: {present_preview}. "
            "Use --allow-missing-targets if you want a partial comparison summary instead."
        )
    return missing


def require_path_exists(path: str | Path, label: str) -> Path:
    """Raise a clear validation error if a required file path is missing."""
    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"{label} not found: {resolved}")
    return resolved
