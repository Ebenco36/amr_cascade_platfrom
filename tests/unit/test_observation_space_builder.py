from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.transformations.observation_space_builder import ObservationSpaceBuilder


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

    # The screen is not merged into cefoxitin; the composite surrogate report and
    # the reference-method tobramycin result stay (names are canonicalised in gold).
    assert sorted(observed["antibiotic"]) == ["CEFOXITIN", "CEPHALEXIN/CEPHALOTHIN", "TOBRAMYCIN RM"]
    assert int(ledger.loc["armd_ecuh", "excluded_assay_rows"]) == 5
    assert int(ledger.loc["armd_ecuh", "interpretable_rows"]) == 3
    assert set(builder.excluded_entities["entity_type"]) == {
        "SCREEN", "PHENOTYPIC_TEST", "RESISTANCE_MECHANISM_TEST", "ANTIFUNGAL", "MISSING_LABEL",
    }
    assert int(builder.excluded_entities["rows"].sum()) == 5
