"""Antimicrobial names recognised by the project's antibiotic classification reference.

The gold layer keeps every label that ``classify_ast_entity`` does not type as a
special entity, as an ordinary antibacterial (``DEFAULT_AST_ENTITY_TYPE``). A new
screening or confirmatory assay in a future extract would therefore enter the
observed layer silently. The gold build compares the kept labels with this
vocabulary and records any it does not recognise (see ObservationSpaceBuilder).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label

# Name columns of data/antibiotic_classification_complete.csv. The German and full
# names carry an abbreviation prefix ("AMC - Amoxicillin/Clavulansaeure").
REFERENCE_NAME_COLUMNS = ("Antibiotic Name", "WHO_Match", "Antibiotic Name German", "Full Name")
_ABBREVIATION_PREFIX = re.compile(r"^[A-Z0-9]{2,5}\s+-\s+")


def classification_reference_path(project_root: Path) -> Path:
    """The classification reference shared with the reporting stage."""
    return Path(project_root) / "data" / "antibiotic_classification_complete.csv"


@lru_cache(maxsize=4)
def reference_antibiotic_labels(path: Path) -> frozenset[str]:
    """Canonical labels (normalize_antibiotic_label) of every agent in the reference."""
    table = pd.read_csv(path)
    labels: set[str] = set()
    for column in REFERENCE_NAME_COLUMNS:
        if column not in table.columns:
            continue
        for value in table[column].dropna().astype(str):
            label = normalize_antibiotic_label(_ABBREVIATION_PREFIX.sub("", value.strip()))
            if label:
                labels.add(label)
    return frozenset(labels)
