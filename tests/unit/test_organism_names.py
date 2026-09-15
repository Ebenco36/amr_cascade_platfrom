from amr_cascade_platform.core.utils.organism_names import (
    ORGANISM_NAME_ALIASES,
    normalize_organism_label,
)
from amr_cascade_platform.core.utils.text import normalize_label


def test_normalize_organism_label_resolves_known_aliases() -> None:
    assert normalize_organism_label("Staph Aureus {MRSA}") == "STAPHYLOCOCCUS AUREUS"
    assert normalize_organism_label("Mucoid Pseudomonas Aeruginosa") == "PSEUDOMONAS AERUGINOSA"
    assert normalize_organism_label("ZZZEnterobacter Aerogenes") == "ENTEROBACTER AEROGENES"
    assert normalize_organism_label("Enterobacter Cloacae/Asburiae") == "ENTEROBACTER CLOACAE COMPLEX"
    assert normalize_organism_label("Acinetobacter Baumannii/Haemolyticus") == "ACINETOBACTER BAUMANNII COMPLEX"
    assert normalize_organism_label("Klebsiella Pneumoniae SSP. Ozaenae") == "KLEBSIELLA PNEUMONIAE OZAENAE"
    assert normalize_organism_label("Enterobacter Species") == "ENTEROBACTER"


def test_normalize_organism_label_passes_through_unknown_organisms_unchanged() -> None:
    assert normalize_organism_label("Escherichia coli") == "ESCHERICHIA COLI"
    assert normalize_organism_label("Staphylococcus aureus") == "STAPHYLOCOCCUS AUREUS"
    assert normalize_organism_label(None) == ""


def test_every_alias_key_is_the_normalize_label_form_of_itself() -> None:
    """Alias keys must already be in normalize_label's canonical form (uppercase,

    single-spaced), or a raw label that should match one would silently fall
    through unresolved -- this guards against a typo'd key that looks right but
    never actually matches after normalize_label runs first.
    """
    for raw_key in ORGANISM_NAME_ALIASES:
        assert normalize_label(raw_key) == raw_key, f"key {raw_key!r} is not its own normalize_label form"
