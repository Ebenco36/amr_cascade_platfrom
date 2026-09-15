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
