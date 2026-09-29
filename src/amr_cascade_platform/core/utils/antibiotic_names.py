"""Canonical antimicrobial-name normalization shared across ARMD analyses.

Site-specific microbiology systems record the same antimicrobial or AST
concept under different raw labels, including:

- abbreviations,
- brand names,
- alternate spellings,
- reference-table spellings,
- separator differences,
- source-system typographical errors,
- laboratory-method suffixes.

All code that groups, joins, classifies, or compares observations by
antimicrobial name should normalize through :func:`normalize_antibiotic_label`
before operating on the antibiotic-name field.

Important
---------
This module performs antimicrobial-identity normalization. It does NOT merge
biologically distinct screening, confirmatory, or resistance-mechanism tests
into therapeutic antimicrobial AST measurements.

For example:

    CEFOXITIN SCREEN != CEFOXITIN
    GENTAMICIN SCREEN != GENTAMICIN
    INDUCIBLE CLINDAMYCIN != CLINDAMYCIN
    ECIM / MCIM are resistance-mechanism tests

By contrast, site-specific ``RM`` suffixes in the UTSW data denote
"Reference Method". They describe how susceptibility was measured rather
than a different antimicrobial identity:

    TOBRAMYCIN RM -> TOBRAMYCIN
    VANCOMYCIN RM -> VANCOMYCIN

The measurement-method information can be retained separately through
:func:`infer_ast_method`.

Which entity types enter the primary observed-AST layer is decided by
``gold.primary_ast_entity_types`` (see ObservationSpaceBuilder); this module
only classifies.

Canonical drug spellings intentionally follow the project's reference
crosswalk where applicable, e.g. CEFOTAXIM, CEFEPIM, CEFTAZIDIM,
CEFTRIAXON, CEFUROXIM, and CO-TRIMOXAZOL.
"""

from __future__ import annotations

import re
from typing import Final

from amr_cascade_platform.core.utils.text import normalize_label


# =============================================================================
# Missing labels
# =============================================================================

MISSING_ANTIBIOTIC_LABELS: Final[frozenset[str]] = frozenset(
    {
        "",
        "NULL",
        "NONE",
        "NAN",
        "NA",
        "N/A",
        # pandas missing values as normalize_label renders them: pd.NA -> "<NA>",
        # pd.NaT -> "NAT" (float NaN -> "NAN" is covered above). Without these a
        # string-dtype column's missing entries became an "antibiotic" called <NA>.
        "<NA>",
        "NAT",
    }
)


# =============================================================================
# Site-specific AST method labels
# =============================================================================

REFERENCE_METHOD_LABELS: Final[frozenset[str]] = frozenset(
    {
        "TOBRAMYCIN RM",
        "VANCOMYCIN RM",
    }
)


# =============================================================================
# Canonical antimicrobial aliases
# =============================================================================

ANTIBIOTIC_NAME_ALIASES: Final[dict[str, str]] = {
    # -------------------------------------------------------------------------
    # Aminoglycosides
    # -------------------------------------------------------------------------
    "AMIKACIN IV": "AMIKACIN",

    # Site abbreviation for a screening test.
    # This is intentionally NOT mapped to ordinary GENTAMICIN.
    "GENT SCREEN": "GENTAMICIN SCREEN",

    # UTSW "RM" = Reference Method.
    # Method differs, antimicrobial identity does not.
    "TOBRAMYCIN RM": "TOBRAMYCIN",

    # -------------------------------------------------------------------------
    # Glycopeptides
    # -------------------------------------------------------------------------
    # UTSW "RM" = Reference Method.
    "VANCOMYCIN RM": "VANCOMYCIN",

    # -------------------------------------------------------------------------
    # Aminopenicillin / beta-lactamase inhibitor combinations
    # -------------------------------------------------------------------------
    "AMOX/CLAVULANATE": "AMOXICILLIN/CLAVULANIC ACID",
    "AMOXICILLIN/K CLAVULANATE AUG": "AMOXICILLIN/CLAVULANIC ACID",
    "AUGMENTIN": "AMOXICILLIN/CLAVULANIC ACID",

    "AMP/SULBACTAM": "AMPICILLIN/SULBACTAM",

    # -------------------------------------------------------------------------
    # Penicillins
    # -------------------------------------------------------------------------
    "BENZYLPENICILLIN": "PENICILLIN G",
    "PENICILLIN": "PENICILLIN G",

    # -------------------------------------------------------------------------
    # Piperacillin / beta-lactamase inhibitor combinations
    # -------------------------------------------------------------------------
    "PIP/TAZOBACTAM": "PIPERACILLIN/TAZOBACTAM",

    # -------------------------------------------------------------------------
    # Ticarcillin / clavulanate variants
    # -------------------------------------------------------------------------
    "TICACARCILLIN/CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",
    "TICARILLIN/CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",
    "TICARCIL/CLAVULANATE": "TICARCILLIN/CLAVULANIC ACID",
    "TICARCILLIN-CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",

    # -------------------------------------------------------------------------
    # Cephalosporins
    #
    # Canonical spellings intentionally follow the project's reference
    # antimicrobial crosswalk.
    # -------------------------------------------------------------------------
    "CEFEPIME": "CEFEPIM",
    "CEFOTAXIME": "CEFOTAXIM",
    "CEFTAZIDIME": "CEFTAZIDIM",
    "CEFTRIAXONE": "CEFTRIAXON",
    "CEFUROXIME": "CEFUROXIM",

    # Reference crosswalk uses "Cefpodoxim".
    "CEFPODOXIME": "CEFPODOXIM",

    # Reference crosswalk uses "Cefalotin".
    "CEPHALOTHIN": "CEFALOTIN",

    # -------------------------------------------------------------------------
    # Phenotypic / beta-lactamase confirmatory tests
    #
    # normalize_antibiotic_label() converts '+' delimiters to '/' before
    # alias lookup, so keys must use '/'.
    #
    # These remain distinct from ordinary cefotaxime/ceftazidime/cefepime AST.
    # -------------------------------------------------------------------------
    "CEFOTAXIME/CLAVULANIC ACID": "CEFOTAXIM/CLAVULANIC ACID",
    "CEFTAZIDIME/CLAVULANIC ACID": "CEFTAZIDIM/CLAVULANIC ACID",
    "CEFEPIME/CLAVULANIC ACID": "CEFEPIM/CLAVULANIC ACID",

    # -------------------------------------------------------------------------
    # Newer cephalosporin / beta-lactamase inhibitor combinations
    # -------------------------------------------------------------------------
    "CEFTAZIDIME/AVIBACTAM": "CEFTAZIDIM/AVIBACTAM",

    # -------------------------------------------------------------------------
    # Carbapenem / beta-lactamase inhibitor combinations
    # -------------------------------------------------------------------------
    # ARMD contains "IMIPENEM/EBACTAM", while UTSW explicitly contains
    # IMIPENEM/RELEBACTAM. This is treated as a source-string error.
    "IMIPENEM/EBACTAM": "IMIPENEM/RELEBACTAM",

    # -------------------------------------------------------------------------
    # Tetracycline-class agents
    #
    # Canonical spellings intentionally follow the reference crosswalk.
    # -------------------------------------------------------------------------
    "DOXYCYCLINE": "DOXYCYCLIN",
    "MINOCYCLINE": "MINOCYCLIN",
    "TETRACYCLINE": "TETRACYCLIN",
    "TIGECYCLINE": "TIGECYCLIN",

    # -------------------------------------------------------------------------
    # Nitroimidazoles
    # -------------------------------------------------------------------------
    "METRONIDAZOLE": "METRONIDAZOL",

    # -------------------------------------------------------------------------
    # Folate-pathway inhibitors
    # -------------------------------------------------------------------------
    "TRIMETH-SULFAMETHOXAZOLE": "CO-TRIMOXAZOL",
    "TRIMETH/SULF": "CO-TRIMOXAZOL",
    "TRIMETHOPRIM/SULFAMETHOXAZOLE": "CO-TRIMOXAZOL",

    # Defensive normalization if alternate spelling occurs later.
    "CO-TRIMOXAZOLE": "CO-TRIMOXAZOL",

    # -------------------------------------------------------------------------
    # Polymyxins
    # -------------------------------------------------------------------------
    "POLYMIXIN E": "COLISTIN",
    "POLYMYXIN E": "COLISTIN",

    # -------------------------------------------------------------------------
    # Rifamycins
    # -------------------------------------------------------------------------
    "RIFAMPIN": "RIFAMPICIN",

    # -------------------------------------------------------------------------
    # Streptogramins
    # -------------------------------------------------------------------------
    "SYNERCID": "QUINUPRISTIN/DALFOPRISTIN",

    # -------------------------------------------------------------------------
    # Antifungals
    #
    # These are present in microbiology source data. They are normalized here
    # but excluded from antibacterial-only analyses through entity
    # classification.
    # -------------------------------------------------------------------------
    "FLUCONIZOLE": "FLUCONAZOLE",
    "CASPOFUNGIN ACETATE": "CASPOFUNGIN",
    "5 FLUOROCYTOSINE": "FLUCYTOSINE",

    # -------------------------------------------------------------------------
    # Antimycobacterial agents
    # -------------------------------------------------------------------------
    "BEDAQUILLINE": "BEDAQUILINE",

    # -------------------------------------------------------------------------
    # Deliberately NOT aliased
    # -------------------------------------------------------------------------
    #
    # CEFOXITIN SCREEN
    #     Screening test; must remain distinct from CEFOXITIN.
    #
    # INDUCIBLE CLINDAMYCIN
    #     Phenotypic inducible-resistance test; must remain distinct from
    #     ordinary CLINDAMYCIN AST.
    #
    # ECIM / MCIM
    #     Carbapenemase resistance-mechanism tests; not therapeutic agents.
    #
    # CEPHALEXIN/CEPHALOTHIN
    #     Composite/surrogate-labelled source entity. Do not arbitrarily
    #     collapse to either component without source-system evidence.
}


# =============================================================================
# AST entity types
# =============================================================================

AST_ENTITY_TYPE: Final[dict[str, str]] = {
    # -------------------------------------------------------------------------
    # Screening tests
    # -------------------------------------------------------------------------
    "CEFOXITIN SCREEN": "SCREEN",
    "GENTAMICIN SCREEN": "SCREEN",

    # -------------------------------------------------------------------------
    # Phenotypic resistance tests
    # -------------------------------------------------------------------------
    "INDUCIBLE CLINDAMYCIN": "PHENOTYPIC_TEST",
    "CEFOTAXIM/CLAVULANIC ACID": "PHENOTYPIC_TEST",
    "CEFTAZIDIM/CLAVULANIC ACID": "PHENOTYPIC_TEST",
    "CEFEPIM/CLAVULANIC ACID": "PHENOTYPIC_TEST",

    # -------------------------------------------------------------------------
    # Resistance-mechanism tests
    # -------------------------------------------------------------------------
    "ECIM": "RESISTANCE_MECHANISM_TEST",
    "MCIM": "RESISTANCE_MECHANISM_TEST",

    # -------------------------------------------------------------------------
    # Composite / surrogate-labelled AST entities
    # -------------------------------------------------------------------------
    "CEPHALEXIN/CEPHALOTHIN": "COMPOSITE_AST",

    # -------------------------------------------------------------------------
    # Antifungals
    # -------------------------------------------------------------------------
    "AMPHOTERICIN B": "ANTIFUNGAL",
    "CASPOFUNGIN": "ANTIFUNGAL",
    "FLUCONAZOLE": "ANTIFUNGAL",
    "FLUCYTOSINE": "ANTIFUNGAL",
    "ITRACONAZOLE": "ANTIFUNGAL",
    "MICAFUNGIN": "ANTIFUNGAL",
    "MICONAZOLE": "ANTIFUNGAL",
    "VORICONAZOLE": "ANTIFUNGAL",

    # -------------------------------------------------------------------------
    # Antimycobacterial agents
    # -------------------------------------------------------------------------
    "BEDAQUILINE": "ANTIMYCOBACTERIAL",
    "ETHAMBUTOL": "ANTIMYCOBACTERIAL",
    "ISONIAZID": "ANTIMYCOBACTERIAL",
    "PYRAZINAMIDE": "ANTIMYCOBACTERIAL",

    # Concentration-specific susceptibility tests should not be silently merged
    # into generic isoniazid.
    "ISONIAZID, LOW": "ANTIMYCOBACTERIAL_TEST",
    "ISONIAZID, HIGH": "ANTIMYCOBACTERIAL_TEST",
}


DEFAULT_AST_ENTITY_TYPE: Final[str] = "ANTIBACTERIAL"


# =============================================================================
# AST method types
# =============================================================================

AST_METHOD_REFERENCE: Final[str] = "REFERENCE_METHOD"
AST_METHOD_UNKNOWN: Final[str] = "UNKNOWN"


# =============================================================================
# Internal helpers
# =============================================================================

def _normalize_raw_antibiotic_text(value: str | None) -> str | None:
    """Perform lexical normalization before biological alias resolution.

    This helper only standardizes text representation. It does not resolve
    aliases or change biological identity.
    """
    if value is None:
        return None

    normalized = normalize_label(value)

    if normalized is None:
        return None

    normalized = normalized.strip()

    if not normalized:
        return None

    # Collapse repeated whitespace.
    normalized = " ".join(normalized.split())

    # Normalize '+' combination separators.
    #
    # Examples:
    #   CEFTAZIDIME+CLAVULANIC ACID
    #   CEFTAZIDIME + CLAVULANIC ACID
    #
    # become:
    #   CEFTAZIDIME/CLAVULANIC ACID
    normalized = re.sub(r"\s*\+\s*", "/", normalized)

    # Remove inconsistent whitespace around '/':
    #
    #   AMPICILLIN / SULBACTAM
    #
    # becomes:
    #   AMPICILLIN/SULBACTAM
    normalized = re.sub(r"\s*/\s*", "/", normalized)

    # Normalize spaced separators while preserving genuine hyphenated names.
    normalized = normalized.replace(" - ", " ")

    # Collapse whitespace again after separator transformations.
    normalized = " ".join(normalized.split())

    if normalized in MISSING_ANTIBIOTIC_LABELS:
        return None

    return normalized


# =============================================================================
# Public normalization API
# =============================================================================

def normalize_antibiotic_label(value: str | None) -> str | None:
    """Normalize a raw antimicrobial/AST label to canonical identity.

    Parameters
    ----------
    value
        Raw antimicrobial or AST test label.

    Returns
    -------
    str | None
        Canonical antimicrobial/test identity, or ``None`` for missing
        placeholders.

    Processing
    ----------
    1. Generic case/whitespace normalization.
    2. Combination-separator normalization.
    3. Missing-value recognition.
    4. Site-specific and spelling alias resolution.

    Important
    ---------
    Biological distinctions are preserved. Screening and resistance-mechanism
    tests do not collapse into ordinary therapeutic antimicrobial AST.

    Laboratory-method suffixes such as UTSW ``RM`` are different: ``RM`` means
    Reference Method and therefore collapses to the underlying drug identity.

    Examples
    --------
    >>> normalize_antibiotic_label("Amp/sulbactam")
    'AMPICILLIN/SULBACTAM'

    >>> normalize_antibiotic_label("Pip/tazobactam")
    'PIPERACILLIN/TAZOBACTAM'

    >>> normalize_antibiotic_label("Ceftazidime+clavulanic Acid")
    'CEFTAZIDIM/CLAVULANIC ACID'

    >>> normalize_antibiotic_label("Tobramycin rm")
    'TOBRAMYCIN'

    >>> normalize_antibiotic_label("Vancomycin rm")
    'VANCOMYCIN'

    >>> normalize_antibiotic_label("Cefoxitin Screen")
    'CEFOXITIN SCREEN'

    >>> normalize_antibiotic_label("Inducible Clindamycin")
    'INDUCIBLE CLINDAMYCIN'

    >>> normalize_antibiotic_label("Null") is None
    True
    """
    normalized = _normalize_raw_antibiotic_text(value)

    if normalized is None:
        return None

    return ANTIBIOTIC_NAME_ALIASES.get(
        normalized,
        normalized,
    )


def classify_ast_entity(value: str | None) -> str | None:
    """Classify an antimicrobial/AST label by analytical entity type.

    Ordinary therapeutic antibacterial agents default to ``ANTIBACTERIAL``.
    Special entities such as screening tests, antifungals, resistance-
    mechanism tests, and antimycobacterial agents are explicitly classified.

    Parameters
    ----------
    value
        Raw or canonical antimicrobial/AST label.

    Returns
    -------
    str | None
        Entity type, or ``None`` for missing labels.
    """
    normalized = normalize_antibiotic_label(value)

    if normalized is None:
        return None

    return AST_ENTITY_TYPE.get(
        normalized,
        DEFAULT_AST_ENTITY_TYPE,
    )


def infer_ast_method(value: str | None) -> str | None:
    """Infer AST measurement method encoded in a source antimicrobial label.

    Currently UTSW labels ending in the confirmed ``RM`` convention denote
    susceptibility measurements generated using a reference method.

    The method is inferred from the raw label BEFORE the ``RM`` suffix is
    collapsed by :func:`normalize_antibiotic_label`.

    Parameters
    ----------
    value
        Raw source antimicrobial label.

    Returns
    -------
    str | None
        ``REFERENCE_METHOD`` for known RM labels,
        ``UNKNOWN`` for other non-missing labels,
        or ``None`` for missing labels.

    Examples
    --------
    >>> infer_ast_method("Tobramycin rm")
    'REFERENCE_METHOD'

    >>> infer_ast_method("Vancomycin rm")
    'REFERENCE_METHOD'

    >>> infer_ast_method("Vancomycin")
    'UNKNOWN'

    >>> infer_ast_method("Null") is None
    True
    """
    normalized = _normalize_raw_antibiotic_text(value)

    if normalized is None:
        return None

    if normalized in REFERENCE_METHOD_LABELS:
        return AST_METHOD_REFERENCE

    return AST_METHOD_UNKNOWN


def is_antibacterial_ast(value: str | None) -> bool:
    """Return True only for ordinary therapeutic antibacterial AST."""
    return classify_ast_entity(value) == DEFAULT_AST_ENTITY_TYPE


def is_special_ast_entity(value: str | None) -> bool:
    """Return True for non-standard therapeutic-antibacterial AST entities."""
    entity_type = classify_ast_entity(value)

    return (
        entity_type is not None
        and entity_type != DEFAULT_AST_ENTITY_TYPE
    )


def is_reference_method_ast(value: str | None) -> bool:
    """Return True when the raw source label encodes Reference Method AST."""
    return infer_ast_method(value) == AST_METHOD_REFERENCE
