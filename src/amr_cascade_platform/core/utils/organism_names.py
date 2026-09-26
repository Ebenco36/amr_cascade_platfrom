"""Canonical organism-name normalization, shared across eligibility and reporting.

Raw EHR organism labels append phenotype/resistance annotations (MRSA, ESBL,
mucoid, biotype numbers, coagulase status), legacy record-system prefixes, or
abbreviated genus spellings onto an otherwise-recognized species name. The
intrinsic-resistance reference (data/reference/intrinsic_resistance.csv) is
keyed on plain binomial/complex names, so an unmapped raw label joins to
nothing and silently defaults to "not intrinsically resistant" rather than
inheriting its species' real profile -- any code that joins organism labels
against that reference, or pools testing history across spelling variants of
the same organism, must normalize through :func:`normalize_organism_label`
first.

``ORGANISM_NAME_ALIASES`` was built empirically: every entry corresponds to a
raw organism string observed in real ARMD/ESKAPE data (see
data/artifacts/eskape_validation/.../gold/eligible_pairs.parquet) that did not
exact-match the reference after plain :func:`normalize_label`, mapped to the
verified reference entry that represents the same organism (or, for genuinely
indistinguishable species pairs like "cloacae/asburiae", the matching
"... COMPLEX" reference entry). If a future data refresh introduces new
unmatched organism strings, find them the same way: normalize
eligible_pairs.parquet's ``organism`` column and diff against the reference's
normalized ``microorganism`` column.
"""

from __future__ import annotations

import re

from amr_cascade_platform.core.utils.text import normalize_label

ORGANISM_NAME_ALIASES: dict[str, str] = {
    # Abbreviated genus spelling.
    "STAPH AUREUS": "STAPHYLOCOCCUS AUREUS",
    # Resistance/susceptibility phenotype prefix (methicillin status does not
    # change the underlying species' intrinsic-resistance profile).
    "METHICILLIN RESISTANT STAPHYLOCOCCUS AUREUS": "STAPHYLOCOCCUS AUREUS",
    "METHICILLIN SENSITIVE STAPHYLOCOCCUS AUREUS": "STAPHYLOCOCCUS AUREUS",
    "METHICILLINSENSITIVE STAPHYLOCOCCUS AUREUS": "STAPHYLOCOCCUS AUREUS",
    "STAPH AUREUS {MRSA}": "STAPHYLOCOCCUS AUREUS",
    "STAPH AUREUS(COLONY VARIANT - SMALL COLONY OR OTHER MORPHOTYPE)": "STAPHYLOCOCCUS AUREUS",
    "STAPHYLOCOCCUS AUREUS BIOTYPE 1": "STAPHYLOCOCCUS AUREUS",
    "STAPHYLOCOCCUS AUREUS BIOTYPE 2": "STAPHYLOCOCCUS AUREUS",
    "ENTEROCOCCUS FAECIUM - VANCO RESISTANT": "ENTEROCOCCUS FAECIUM",
    # Escherichia coli written with a resistance phenotype or a biotype (UTSW
    # records ESBL producers under their own label; Stanford marks carbapenem
    # resistance in the label). These are E. coli isolates: the annotation is a
    # result, not a different organism, and leaving them out would drop the most
    # resistant episodes from the E. coli cohort.
    "ESBL ESCHERICHIA COLI": "ESCHERICHIA COLI",
    "ESCHERICHIA COLI (CARBAPENEM RESISTANT)": "ESCHERICHIA COLI",
    "ESCHERICHIA COLI BIOTYPE 1": "ESCHERICHIA COLI",
    "ESCHERICHIA COLI BIOTYPE 2": "ESCHERICHIA COLI",
    "ESCHERICHIA COLI BIO TYPE 1": "ESCHERICHIA COLI",
    "ESCHERICHIA COLI BIO TYPE 2": "ESCHERICHIA COLI",
    "ESBL KLEBSIELLA PNEUMONIAE": "KLEBSIELLA PNEUMONIAE",
    "KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)": "KLEBSIELLA PNEUMONIAE",
    "KLEBSIELLA PNEUMONIAE CARBAPENEMASE PRODUCER": "KLEBSIELLA PNEUMONIAE",
    "KLEBSIELLA PNEUMONIAE BIOTYPE 2": "KLEBSIELLA PNEUMONIAE",
    # Klebsiella pneumoniae subspecies: reference carries this as its own
    # (space-separated, not "SSP."/"SS"-joined) trinomial entry.
    "KLEBSIELLA PNEUMONIAE SSP. OZAENAE": "KLEBSIELLA PNEUMONIAE OZAENAE",
    "KLEBSIELLA PNEUMONIAE SS OZAENAE": "KLEBSIELLA PNEUMONIAE OZAENAE",
    "ENTEROBACTER CLOACAE CARBAPENEMASE PRODUCER": "ENTEROBACTER CLOACAE",
    "ENTEROBACTER CLOACAE COMPLEX (CARBAPENEM RESISTANT)": "ENTEROBACTER CLOACAE COMPLEX",
    # Mucoid/non-mucoid is a P. aeruginosa colony phenotype (common in cystic
    # fibrosis sputum cultures), not a different organism.
    "MUCOID PSEUDOMONAS AERUGINOSA": "PSEUDOMONAS AERUGINOSA",
    "NONMUCOID PSEUDOMONAS AERUGINOSA": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA NONMUCOID": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA (NON-MUCOID CF)": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA BIOTYPE": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA BIOTYPE 1": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA BIOTYPE 2": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA BIOTYPE 3": "PSEUDOMONAS AERUGINOSA",
    "PSEUDOMONAS AERUGINOSA BIOTYPE 4": "PSEUDOMONAS AERUGINOSA",
    # Legacy/deprecated Epic-style record prefix ("ZZZ..."); the organism
    # itself is unambiguous once the prefix is stripped.
    "ZZZENTEROBACTER AEROGENES": "ENTEROBACTER AEROGENES",
    # Trailing biotype/group number; reference carries the bare species.
    "ENTEROBACTER AMNIGENUS 1": "ENTEROBACTER AMNIGENUS",
    # Genus reported with no species resolved -- the genus-level reference
    # entry is the correct (conservative) fallback, not a specific species.
    "ENTEROBACTER SPECIES": "ENTEROBACTER",
    # Two species that are difficult to distinguish by routine phenotypic
    # methods and are conventionally reported jointly as a "complex"; the
    # reference carries the recognized complex-level entry for each.
    "ENTEROBACTER CLOACAE/ASBURIAE": "ENTEROBACTER CLOACAE COMPLEX",
    "ENTEROBACTER CLOACAEASBURIAE": "ENTEROBACTER CLOACAE COMPLEX",
    "ACINETOBACTER BAUMANNII/HAEMOLYTICUS": "ACINETOBACTER BAUMANNII COMPLEX",
    "ACINETOBACTER BAUMANNII NOSOCOMIALIS GROUP": "ACINETOBACTER BAUMANNII COMPLEX",
}


def normalize_organism_label(value: str | None) -> str:
    """Normalize a raw organism label to its canonical alias-resolved form.

    Applies the generic label normalizer (case/whitespace) then resolves known
    site-specific phenotype/resistance-annotated, abbreviated, or legacy-coded
    spelling variants via ``ORGANISM_NAME_ALIASES``. Two raw labels that denote
    the same organism (for the purpose of intrinsic-resistance lookup and
    operational-availability pooling) always normalize to the same string; use
    this -- not the bare :func:`normalize_label` -- before any groupby or join
    keyed on organism identity.
    """
    normalized = normalize_label(value)
    return ORGANISM_NAME_ALIASES.get(normalized, normalized)


def matches_requested_organism(label: object, requested: str) -> bool:
    """Whether a raw organism label belongs to the organism an analysis requests.

    A request for a species (a name that is already canonical, e.g.
    "ESCHERICHIA COLI") includes every raw label that resolves to it through
    ``ORGANISM_NAME_ALIASES`` -- resistance-phenotype ("ESBL ...",
    "... (CARBAPENEM RESISTANT)"), biotype and colony-variant spellings -- since
    those annotations describe the isolate, not a different organism. A request
    for an annotated label itself (e.g. "STAPH AUREUS {MRSA}") keeps exactly
    that label, so a phenotype-specific analysis stays phenotype-specific.
    """
    if not isinstance(label, str):
        return False
    requested_label = normalize_label(requested)
    canonical = normalize_organism_label(requested)
    if requested_label != canonical:
        return normalize_label(label) == requested_label
    return normalize_organism_label(label) == canonical


# Words some raw labels put before the genus (phenotype, test result or grouping).
_NON_GENUS_WORDS = frozenset(
    {
        "ALPHA", "ANAEROBIC", "BETA", "COAG", "COAGULASE", "ESBL", "GRAM", "GROUP",
        "HAEMOLYTIC", "HEMOLYTIC", "METHICILLIN", "METHICILLINSENSITIVE", "MUCOID",
        "NEGATIVE", "NONMUCOID", "POSITIVE", "RESISTANT", "SENSITIVE", "VIRIDANS",
    }
)
# Genus abbreviations in the extracts; the prior-infection extract writes
# coagulase-negative staphylococci as "CONS".
_GENUS_ABBREVIATIONS = {"CONS": "STAPHYLOCOCCUS", "STAPH": "STAPHYLOCOCCUS", "STREP": "STREPTOCOCCUS"}


def organism_genus(label: object) -> str:
    """Genus of an organism label, for comparisons at genus level ("" if none).

    The prior-infection extracts record prior organisms at genus level
    ("Escherichia", "CONS") while culture episodes carry species labels
    ("ESCHERICHIA COLI", "COAG NEGATIVE STAPHYLOCOCCUS"), so "same organism"
    history can only be established at genus level. The label is first
    alias-resolved, then the first word that is not a phenotype or grouping
    qualifier is taken, with legacy "ZZZ" prefixes and abbreviations resolved.
    """
    if not isinstance(label, str):
        return ""
    for word in re.split(r"[^A-Z]+", normalize_organism_label(label)):
        word = _GENUS_ABBREVIATIONS.get(word.removeprefix("ZZZ"), word.removeprefix("ZZZ"))
        if word and word not in _NON_GENUS_WORDS:
            return word
    return ""
