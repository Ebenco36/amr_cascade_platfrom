"""Shared helper for turning a raw organism slug into a display label."""

from __future__ import annotations

_ACRONYM_OVERRIDES = {
    "mrsa": "MRSA",
    "mssa": "MSSA",
    "vre": "VRE",
    "esbl": "ESBL",
}


def format_organism_label(organism: str) -> str:
    """Convert a slug like "staph_aureus_mrsa" into "Staph Aureus MRSA"."""
    words = organism.replace("_", " ").split()
    return " ".join(_ACRONYM_OVERRIDES.get(word.lower(), word.title()) for word in words)
