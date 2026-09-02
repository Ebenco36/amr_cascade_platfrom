"""Canonical antibiotic-name normalization, shared across analysis and reporting.

Site-specific source data records the same antibiotic under different raw
spellings (abbreviated order-set names, alternate salts, trailing suffixes).
Any code that groups or joins rows by antibiotic name must normalize through
:func:`normalize_antibiotic_label` first, or rows for the same drug will
silently split into separate groups.
"""

from __future__ import annotations

from amr_cascade_platform.core.utils.text import normalize_label

ANTIBIOTIC_NAME_ALIASES: dict[str, str] = {
    "AMIKACIN IV": "AMIKACIN",
    "AMOX/CLAVULANATE": "AMOXICILLIN/CLAVULANIC ACID",
    "AMOXICILLIN/K CLAVULANATE AUG": "AMOXICILLIN/CLAVULANIC ACID",
    "AMP/SULBACTAM": "AMPICILLIN/SULBACTAM",
    "AUGMENTIN": "AMOXICILLIN/CLAVULANIC ACID",
    "CEFEPIME": "CEFEPIM",
    "CEFOTAXIME": "CEFOTAXIM",
    # NOTE: normalize_antibiotic_label() converts '+' -> '/' before alias lookup,
    # so these keys must use '/' not '+'.
    "CEFOTAXIME/CLAVULANIC ACID": "CEFOTAXIM/CLAVULANIC ACID",
    # Reference crosswalk's own name/full-name fields are "Cefpodoxim"
    # (German-style spelling) and "Cefpodoxim-Proxetil" (the oral prodrug) --
    # neither matches the plain English "Cefpodoxime" spelling used in the
    # raw AST data, so it fell through to Unclassified without this alias.
    "CEFPODOXIME": "CEFPODOXIM",
    "CEFOXITIN SCREEN": "CEFOXITIN",
    "CEFTAZIDIME": "CEFTAZIDIM",
    "CEFTAZIDIME/CLAVULANIC ACID": "CEFTAZIDIM/CLAVULANIC ACID",
    "CEFTAZIDIME/AVIBACTAM": "CEFTAZIDIM/AVIBACTAM",
    "CEFTRIAXONE": "CEFTRIAXON",
    "CEFUROXIME": "CEFUROXIM",
    "CEPHALOTHIN": "CEFALOTIN",
    "DOXYCYCLINE": "DOXYCYCLIN",
    "GENT SCREEN": "GENTAMICIN SCREEN",
    "INDUCIBLE CLINDAMYCIN": "CLINDAMYCIN",
    "IMIPENEM/EBACTAM": "IMIPENEM/RELEBACTAM",
    "METRONIDAZOLE": "METRONIDAZOL",
    "MINOCYCLINE": "MINOCYCLIN",
    "PENICILLIN": "PENICILLIN G",
    "PIP/TAZOBACTAM": "PIPERACILLIN/TAZOBACTAM",
    "POLYMIXIN E": "COLISTIN",
    "RIFAMPIN": "RIFAMPICIN",
    "SYNERCID": "QUINUPRISTIN/DALFOPRISTIN",
    "TETRACYCLINE": "TETRACYCLIN",
    "TICACARCILLIN/CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",
    "TICARILLIN/CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",
    "TICARCIL/CLAVULANATE": "TICARCILLIN/CLAVULANIC ACID",
    "TICARCILLIN-CLAVULANIC ACID": "TICARCILLIN/CLAVULANIC ACID",
    "TIGECYCLINE": "TIGECYCLIN",
    "TOBRAMYCIN RM": "TOBRAMYCIN",
    "TRIMETH-SULFAMETHOXAZOLE": "CO-TRIMOXAZOL",
    "TRIMETH/SULF": "CO-TRIMOXAZOL",
    "TRIMETHOPRIM/SULFAMETHOXAZOLE": "CO-TRIMOXAZOL",
    "VANCOMYCIN RM": "VANCOMYCIN",
}


def normalize_antibiotic_label(value: str | None) -> str:
    """Normalize a raw antibiotic label to its canonical alias-resolved form.

    Applies the generic label normalizer (case/whitespace), collapses '+'
    delimiters to '/', collapses ' - ' to a single space, then resolves
    known site-specific spelling variants via ``ANTIBIOTIC_NAME_ALIASES``.
    Two raw labels that denote the same drug always normalize to the same
    string; use this before any groupby or join keyed on antibiotic name.
    """
    normalized = normalize_label(value)
    normalized = normalized.replace("+", "/")
    normalized = normalized.replace(" - ", " ")
    normalized = " ".join(normalized.split())
    return ANTIBIOTIC_NAME_ALIASES.get(normalized, normalized)
