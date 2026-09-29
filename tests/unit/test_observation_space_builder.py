from dataclasses import replace
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.utils.antibiotic_names import classify_ast_entity, normalize_antibiotic_label
from amr_cascade_platform.core.utils.antibiotic_reference import (
    classification_reference_path,
    reference_antibiotic_labels,
)
from amr_cascade_platform.data.transformations.observation_space_builder import ObservationSpaceBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _row(antibiotic: str, susceptibility: object) -> dict:
    return {
        "anon_id": "1",
        "pat_enc_csn_id_coded": "2",
        "order_proc_id_coded": "3",
        "order_time_jittered": "2024-01-01",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd_ecuh",
        "antibiotic": antibiotic,
        "susceptibility": susceptibility,
    }


def test_only_interpreted_results_of_therapeutic_drugs_count_as_observed() -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    cohort = pd.DataFrame(
        [
            _row("CEFTRIAXONE", "RESISTANT"),
            _row("CIPROFLOXACIN", "SUSCEPTIBLE"),
            _row("CEFAZOLIN", "INTERMEDIATE"),
            _row("NITROFURANTOIN", None),
            _row("COLISTIN", "INCONCLUSIVE"),
            _row("CEFTAZIDIME+CLAVULANIC ACID", "SUSCEPTIBLE"),
            _row("CEFOTAXIME+CLAVULANIC ACID", None),
        ]
    )

    observed = ObservationSpaceBuilder(settings).build(cohort)

    assert sorted(observed["antibiotic"]) == ["CEFAZOLIN", "CEFTRIAXONE", "CIPROFLOXACIN"]


def test_screening_confirmation_and_non_antibacterial_entries_leave_the_observed_layer() -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    cohort = pd.DataFrame(
        [
            _row("CEFOXITIN", "SUSCEPTIBLE"),
            _row("CEFOXITIN SCREEN", "RESISTANT"),
            _row("INDUCIBLE CLINDAMYCIN", "RESISTANT"),
            _row("ECIM", "RESISTANT"),
            _row("FLUCONAZOLE", "SUSCEPTIBLE"),
            _row("CEPHALEXIN/CEPHALOTHIN", "SUSCEPTIBLE"),
            _row("TOBRAMYCIN RM", "SUSCEPTIBLE"),
            _row(None, "RESISTANT"),
        ]
    )

    builder = ObservationSpaceBuilder(settings)
    observed, ledger = builder.build_with_ledger(cohort)

    # The screen is not merged into cefoxitin, and the composite label (one result
    # for two agents) is not a drug node; the reference-method tobramycin result
    # stays (names are canonicalised in gold).
    assert sorted(observed["antibiotic"]) == ["CEFOXITIN", "TOBRAMYCIN RM"]
    assert int(ledger.loc["armd_ecuh", "excluded_assay_rows"]) == 6
    assert int(ledger.loc["armd_ecuh", "composite_label_rows"]) == 1
    assert int(ledger.loc["armd_ecuh", "interpretable_rows"]) == 2
    assert set(builder.excluded_entities["entity_type"]) == {
        "SCREEN", "PHENOTYPIC_TEST", "RESISTANCE_MECHANISM_TEST", "COMPOSITE_AST", "ANTIFUNGAL", "MISSING_LABEL",
    }
    assert int(builder.excluded_entities["rows"].sum()) == 6


def test_composite_labels_stay_only_when_the_sensitivity_setting_keeps_them() -> None:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    assert settings.gold.primary_ast_entity_types == ("ANTIBACTERIAL",)
    keep = replace(settings, gold=replace(settings.gold, primary_ast_entity_types=("ANTIBACTERIAL", "COMPOSITE_AST")))
    cohort = pd.DataFrame([_row("CEFAZOLIN", "SUSCEPTIBLE"), _row("Cephalexin/Cephalothin", "RESISTANT")])

    observed, ledger = ObservationSpaceBuilder(keep).build_with_ledger(cohort)

    assert sorted(observed["antibiotic"]) == ["CEFAZOLIN", "Cephalexin/Cephalothin"]
    assert int(ledger.loc["armd_ecuh", "composite_label_rows"]) == 0


def test_labels_outside_the_reference_are_kept_but_recorded() -> None:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    reference = reference_antibiotic_labels(classification_reference_path(PROJECT_ROOT))
    cohort = pd.DataFrame(
        [
            _row("Ceftriaxone", "RESISTANT"),
            _row("CARBA NP", "RESISTANT"),
            _row("CARBA NP", "SUSCEPTIBLE"),
            _row("Cefoxitin Screen", "RESISTANT"),
        ]
    )

    builder = ObservationSpaceBuilder(settings, reference_labels=reference)
    observed = builder.build(cohort)

    # Audit, not exclusion: an unknown label stays in the observed layer, and the
    # gold metadata names it with its row count.
    assert sorted(observed["antibiotic"]) == ["CARBA NP", "CARBA NP", "Ceftriaxone"]
    assert builder.unrecognised_labels.to_dict(orient="records") == [
        {"source_site": "armd_ecuh", "antibiotic": "CARBA NP", "canonical_label": "CARBA NP", "rows": 2}
    ]
    unaudited = ObservationSpaceBuilder(settings)
    unaudited.build(cohort)
    assert unaudited.unrecognised_labels is None


# Every distinct AST label in the three ARMD source extracts (the antibiotic column
# of the cohort tables, 2026-09-29), apart from "Null".
ARMD_AST_LABELS = (
    '5 fluorocytosine', 'Amikacin', 'Amikacin iv', 'Amox/clavulanate', 'Amoxicillin/Clavulanic Acid',
    'Amoxicillin/clavulanic Acid', 'Amoxicillin/clavulanic acid', 'Amoxicillin/k clavulanate aug', 'Amp/sulbactam',
    'Amphotericin B', 'Ampicillin', 'Ampicillin/Sulbactam', 'Ampicillin/sulbactam', 'Augmentin', 'Azithromycin',
    'Aztreonam', 'Bedaquilline', 'Caspofungin', 'Caspofungin acetate', 'Cefaclor', 'Cefazolin', 'Cefepime',
    'Cefiderocol', 'Cefotaxime', 'Cefotaxime+clavulanic Acid', 'Cefotetan', 'Cefoxitin', 'Cefoxitin Screen',
    'Cefpodoxime', 'Ceftaroline', 'Ceftazidime', 'Ceftazidime+clavulanic Acid', 'Ceftazidime/Avibactam',
    'Ceftazidime/avibactam', 'Ceftazidime/avibactam ', 'Ceftolozane/Tazobactam', 'Ceftolozane/tazobactam',
    'Ceftriaxone', 'Cefuroxime', 'Cephalexin/Cephalothin', 'Cephalothin', 'Chloramphenicol', 'Ciprofloxacin',
    'Clarithromycin', 'Clindamycin', 'Colistin', 'Daptomycin', 'Doripenem', 'Doxycycline', 'Ecim', 'Eravacycline',
    'Ertapenem', 'Erythromycin', 'Fluconazole', 'Fluconizole', 'Fosfomycin', 'Gatifloxacin', 'Gent screen',
    'Gentamicin', 'Gentamicin screen', 'Imipenem', 'Imipenem/Ebactam', 'Imipenem/relebactam',
    'Inducible Clindamycin', 'Isoniazid', 'Isoniazid, high', 'Isoniazid, low', 'Itraconazole', 'Kanamycin',
    'Levofloxacin', 'Linezolid', 'Mcim', 'Meropenem', 'Meropenem/Vaborbactam', 'Meropenem/vaborbactam',
    'Metronidazole', 'Micafungin', 'Miconazole', 'Minocycline', 'Moxifloxacin', 'Nitrofurantoin', 'Norfloxacin',
    'Ofloxacin', 'Oxacillin', 'Penicillin', 'Pip/tazobactam', 'Piperacillin', 'Piperacillin/Tazobactam',
    'Piperacillin/tazobactam', 'Polymixin e', 'Quinupristin/Dalfopristin', 'Rifampin', 'Streptomycin', 'Synercid',
    'Tetracycline', 'Ticacarcillin/Clavulanic Acid', 'Ticarcil/clavulanate', 'Ticarcillin',
    'Ticarcillin-clavulanic acid', 'Ticarillin/clavulanic Acid', 'Tigecycline', 'Tobramycin', 'Tobramycin rm',
    'Trimeth-sulfamethoxazole', 'Trimeth/sulf', 'Trimethoprim/Sulfamethoxazole', 'Trimethoprim/sulfamethoxazole',
    'Vancomycin', 'Vancomycin rm', 'Voriconazole',
)


def test_every_current_armd_label_is_a_typed_entity_or_in_the_reference() -> None:
    reference = reference_antibiotic_labels(classification_reference_path(PROJECT_ROOT))
    unrecognised = [
        label
        for label in ARMD_AST_LABELS
        if classify_ast_entity(label) == "ANTIBACTERIAL" and normalize_antibiotic_label(label) not in reference
    ]
    assert unrecognised == []
    assert classify_ast_entity("Cephalexin/Cephalothin") == "COMPOSITE_AST"
