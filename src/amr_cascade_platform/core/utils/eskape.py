"""Definitions and helpers for ARMD ESKAPE-family analyses."""

from __future__ import annotations

from dataclasses import dataclass
from amr_cascade_platform.core.utils.text import normalize_label


@dataclass(frozen=True)
class EskapeTarget:
    """Definition for one ESKAPE pathogen group in ARMD-style organism labels."""

    key: str
    display_name: str
    scientific_focus: str
    armd_match_pattern: str


ESKAPE_TARGETS: tuple[EskapeTarget, ...] = (
    EskapeTarget(
        key="enterococcus",
        display_name="Enterococcus",
        scientific_focus="Enterococcus faecium",
        armd_match_pattern=r"ENTEROCOCCUS FAECIUM",
    ),
    EskapeTarget(
        key="staphylococcus",
        display_name="Staphylococcus",
        scientific_focus="Staphylococcus aureus",
        armd_match_pattern=r"STAPH(?:YLOCOCCUS)? AUREUS",
    ),
    EskapeTarget(
        key="klebsiella",
        display_name="Klebsiella",
        scientific_focus="Klebsiella pneumoniae",
        armd_match_pattern=r"KLEBSIELLA PNEUMONIAE",
    ),
    EskapeTarget(
        key="acinetobacter",
        display_name="Acinetobacter",
        scientific_focus="Acinetobacter baumannii",
        armd_match_pattern=r"ACINETOBACTER BAUMANNII",
    ),
    EskapeTarget(
        key="pseudomonas",
        display_name="Pseudomonas",
        scientific_focus="Pseudomonas aeruginosa",
        armd_match_pattern=r"PSEUDOMONAS AERUGINOSA",
    ),
    EskapeTarget(
        key="enterobacter",
        display_name="Enterobacter",
        scientific_focus="Enterobacter spp.",
        armd_match_pattern=r"ENTEROBACTER",
    ),
)


def get_eskape_target(name: str) -> EskapeTarget:
    """Resolve an ESKAPE target by key, display name, or scientific focus."""
    normalized = normalize_label(name)
    for target in ESKAPE_TARGETS:
        candidates = {
            normalize_label(target.key),
            normalize_label(target.display_name),
            normalize_label(target.scientific_focus),
        }
        if normalized in candidates:
            return target
    available = ", ".join(target.display_name for target in ESKAPE_TARGETS)
    raise ValueError(f"Unknown ESKAPE target '{name}'. Available targets: {available}")


def resolve_eskape_targets(names: list[str] | tuple[str, ...] | None = None) -> list[EskapeTarget]:
    """Return all ESKAPE targets, or the requested subset in the requested order."""
    if not names:
        return list(ESKAPE_TARGETS)
    return [get_eskape_target(name) for name in names]
