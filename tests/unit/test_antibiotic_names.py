import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.core.utils.antibiotic_names import (
    ANTIBIOTIC_NAME_ALIASES,
    classify_ast_entity,
    infer_ast_method,
    is_antibacterial_ast,
    normalize_antibiotic_label,
)


@pytest.mark.parametrize(
    ("raw", "canonical", "entity"),
    [
        ("Cefoxitin Screen", "CEFOXITIN SCREEN", "SCREEN"),
        ("GENT SCREEN", "GENTAMICIN SCREEN", "SCREEN"),
        ("Inducible Clindamycin", "INDUCIBLE CLINDAMYCIN", "PHENOTYPIC_TEST"),
        ("ECIM", "ECIM", "RESISTANCE_MECHANISM_TEST"),
        ("Ceftazidime + Clavulanic Acid", "CEFTAZIDIM/CLAVULANIC ACID", "PHENOTYPIC_TEST"),
        ("CEFEPIME/CLAVULANIC ACID", "CEFEPIM/CLAVULANIC ACID", "PHENOTYPIC_TEST"),
        ("Caspofungin acetate", "CASPOFUNGIN", "ANTIFUNGAL"),
    ],
)
def test_screening_confirmatory_and_mechanism_assays_keep_their_own_identity(raw, canonical, entity):
    assert normalize_antibiotic_label(raw) == canonical
    assert classify_ast_entity(raw) == entity
    assert not is_antibacterial_ast(raw)


def test_reference_method_suffix_is_a_result_for_the_same_drug():
    assert normalize_antibiotic_label("Tobramycin rm") == "TOBRAMYCIN"
    assert infer_ast_method("Tobramycin rm") == "REFERENCE_METHOD"
    assert infer_ast_method("TOBRAMYCIN") == "UNKNOWN"
    assert is_antibacterial_ast("Tobramycin rm")


@pytest.mark.parametrize(
    "missing",
    [None, "", "  ", "Null", "n/a", "NaN", "NONE", pd.NA, np.nan, pd.NaT, "<NA>"],
)
def test_missing_placeholders_are_not_antibiotics(missing):
    assert normalize_antibiotic_label(missing) is None
    assert classify_ast_entity(missing) is None


def test_missing_entries_of_a_string_column_map_to_none():
    labels = pd.Series(["Tobramycin rm", pd.NA, None], dtype="string")
    assert [normalize_antibiotic_label(value) for value in labels] == ["TOBRAMYCIN", None, None]


def test_separator_variants_share_one_identity():
    assert normalize_antibiotic_label("Ampicillin / Sulbactam") == normalize_antibiotic_label("AMPICILLIN/SULBACTAM")
    assert normalize_antibiotic_label("Amp/sulbactam") == "AMPICILLIN/SULBACTAM"


def test_normalization_is_idempotent_on_every_canonical_name():
    for canonical in set(ANTIBIOTIC_NAME_ALIASES.values()):
        assert normalize_antibiotic_label(canonical) == canonical
