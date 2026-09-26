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


def test_phenotype_annotated_escherichia_coli_labels_resolve_to_the_species() -> None:
    for raw in (
        "ESBL Escherichia coli",
        "Escherichia coli (Carbapenem Resistant)",
        "Escherichia coli biotype 2",
        "Escherichia coli bio type 1",
    ):
        assert normalize_organism_label(raw) == "ESCHERICHIA COLI"
    # Other Escherichia species and unresolved genus labels stay distinct.
    assert normalize_organism_label("Escherichia fergusonii") == "ESCHERICHIA FERGUSONII"
    assert normalize_organism_label("Escherichia species") == "ESCHERICHIA SPECIES"


def test_species_requests_pool_annotated_labels_but_annotated_requests_stay_exact() -> None:
    from amr_cascade_platform.core.utils.organism_names import matches_requested_organism

    ecoli = ["ESCHERICHIA COLI", "ESBL ESCHERICHIA COLI", "ESCHERICHIA COLI (CARBAPENEM RESISTANT)"]
    assert all(matches_requested_organism(label, "Escherichia coli") for label in ecoli)
    assert not matches_requested_organism("ESCHERICHIA FERGUSONII", "ESCHERICHIA COLI")
    assert not matches_requested_organism(None, "ESCHERICHIA COLI")
    # A phenotype-specific request selects only that label.
    assert matches_requested_organism("STAPH AUREUS {MRSA}", "STAPH AUREUS {MRSA}")
    assert not matches_requested_organism("STAPHYLOCOCCUS AUREUS", "STAPH AUREUS {MRSA}")
    assert matches_requested_organism("STAPH AUREUS {MRSA}", "STAPHYLOCOCCUS AUREUS")


def test_organism_genus_links_the_genus_level_prior_extract_to_species_labels() -> None:
    from amr_cascade_platform.core.utils.organism_names import organism_genus

    # Every prior_organism value in the three raw extracts.
    prior_vocabulary = {
        "Acinetobacter": "ACINETOBACTER", "CONS": "STAPHYLOCOCCUS", "Candida": "CANDIDA",
        "Citrobacter": "CITROBACTER", "Enterobacter": "ENTEROBACTER", "Enterococcus": "ENTEROCOCCUS",
        "Escherichia": "ESCHERICHIA", "Klebsiella": "KLEBSIELLA", "Morganella": "MORGANELLA",
        "Proteus": "PROTEUS", "Providencia": "PROVIDENCIA", "Pseudomonas": "PSEUDOMONAS",
        "Serratia": "SERRATIA", "Staphylococcus": "STAPHYLOCOCCUS",
        "Stenotrophomonas": "STENOTROPHOMONAS", "Streptococcus": "STREPTOCOCCUS",
    }
    for raw, genus in prior_vocabulary.items():
        assert organism_genus(raw) == genus
    episode_labels = {
        "ESCHERICHIA COLI": "ESCHERICHIA",
        "ESBL ESCHERICHIA COLI": "ESCHERICHIA",
        "COAG NEGATIVE STAPHYLOCOCCUS": "STAPHYLOCOCCUS",
        "STAPH AUREUS {MRSA}": "STAPHYLOCOCCUS",
        "METHICILLIN RESISTANT STAPHYLOCOCCUS AUREUS": "STAPHYLOCOCCUS",
        "MUCOID PSEUDOMONAS AERUGINOSA": "PSEUDOMONAS",
        "STREPTOCOCCUS AGALACTIAE (GROUP B)": "STREPTOCOCCUS",
        "ZZZCITROBACTER AMALONATICUS": "CITROBACTER",
        "KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)": "KLEBSIELLA",
    }
    for raw, genus in episode_labels.items():
        assert organism_genus(raw) == genus
    assert organism_genus(None) == ""
    assert organism_genus(float("nan")) == ""
