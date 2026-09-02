from pathlib import Path

from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver

_CLASSIFICATION_PATH = Path(__file__).resolve().parents[2] / "data" / "antibiotic_classification_complete.csv"


def test_abbreviation_table_preserves_source_and_canonical_labels() -> None:
    resolver = AntibioticClassificationResolver(_CLASSIFICATION_PATH)

    table = resolver.abbreviation_table(["CO-TRIMOXAZOL", "SULFAMETHOXAZOLE/TRIMETHOPRIM"])
    row = table.loc[table["abbreviation"] == "SXT"].iloc[0]

    assert row["full_name"] == "SULFAMETHOXAZOLE/TRIMETHOPRIM"
    assert "CO-TRIMOXAZOL" in row["source_labels"]
    assert "SULFAMETHOXAZOLE/TRIMETHOPRIM" in row["source_labels"]
    assert row["canonical_labels"] == "SULFAMETHOXAZOLE/TRIMETHOPRIM"
    assert "SXT - CO-TRIMOXAZOL" in row["reference_labels"]
    assert row["abbreviation_collision"] == False


def test_axis_labels_collapse_synonyms_but_disambiguate_true_collisions() -> None:
    resolver = AntibioticClassificationResolver(_CLASSIFICATION_PATH)

    synonym_map = resolver.abbreviation_axis_label_map(["CO-TRIMOXAZOL", "SULFAMETHOXAZOLE/TRIMETHOPRIM"])
    assert synonym_map["CO-TRIMOXAZOL"] == "SXT"
    assert synonym_map["SULFAMETHOXAZOLE/TRIMETHOPRIM"] == "SXT"

    collision_map = resolver.abbreviation_axis_label_map(["TESTDRUGA", "TESTDRUGB"])
    assert collision_map["TESTDRUGA"] != collision_map["TESTDRUGB"]
    assert collision_map["TESTDRUGA"].startswith("TEST (")
    assert collision_map["TESTDRUGB"].startswith("TEST (")


def test_known_unclassified_screen_aliases_do_not_create_false_collisions() -> None:
    resolver = AntibioticClassificationResolver(_CLASSIFICATION_PATH)

    table = resolver.abbreviation_table(["GENT SCREEN", "GENTAMICIN SCREEN", "POLYMIXIN E"])

    gent = table.loc[table["full_name"] == "GENTAMICIN SCREEN"].iloc[0]
    assert gent["abbreviation"] == "GS"
    assert gent["abbreviation_collision"] == False
    assert "GENT SCREEN" in gent["source_labels"]
    assert "GENTAMICIN SCREEN" in gent["source_labels"]

    colistin = table.loc[table["abbreviation"] == "COL"].iloc[0]
    assert colistin["full_name"] == "COLISTIN"
    assert colistin["aware_category"] == "Reserve"
    assert "POLYMIXIN E" in colistin["source_labels"]


def test_known_source_spelling_defect_resolves_to_reference_drug() -> None:
    resolver = AntibioticClassificationResolver(_CLASSIFICATION_PATH)

    table = resolver.abbreviation_table(["IMIPENEM/EBACTAM"])
    row = table.loc[table["abbreviation"] == "IMR"].iloc[0]

    assert row["full_name"] == "IMIPENEM/RELEBACTAM"
    assert row["aware_category"] == "Reserve"
    assert "IMIPENEM/EBACTAM" in row["source_labels"]
    assert "IMIPENEM/RELEBACTAM" in row["reference_labels"]
