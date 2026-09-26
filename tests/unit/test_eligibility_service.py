from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.domain.services.eligibility_service import EligibilityService


def test_eligibility_service_marks_intrinsic_pairs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2024-01-01",
                "organism": "(unknown Gram-positives)".upper(),
                "source_site": "armd",
            }
        ]
    )
    culture_drug_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2024-01-01",
                "organism": "(unknown Gram-positives)".upper(),
                "source_site": "armd",
                "antibiotic": "AZTREONAM",
                "susceptibility": "RESISTANT",
                "was_tested": 1,
            }
        ]
    )

    eligible = service.build_episode_eligibility(culture_episodes, culture_drug_episodes)
    assert int(eligible.iloc[0]["is_intrinsic_resistance"]) == 1
    assert int(eligible.iloc[0]["is_eligible"]) == 0


def test_operational_eligibility_excludes_site_era_unavailable_drugs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_a",
            },
            {
                "anon_id": "B1",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_b",
            },
        ]
    )
    culture_drug_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_a",
                "antibiotic": "AMPICILLIN",
                "susceptibility": "SUSCEPTIBLE",
                "was_tested": 1,
            },
            {
                "anon_id": "B1",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_b",
                "antibiotic": "MEROPENEM",
                "susceptibility": "SUSCEPTIBLE",
                "was_tested": 1,
            },
        ]
    )

    eligible = service.build_episode_eligibility(culture_episodes, culture_drug_episodes)
    site_a_meropenem = eligible[
        (eligible["source_site"] == "site_a") & (eligible["antibiotic"] == "MEROPENEM")
    ].iloc[0]
    site_b_meropenem = eligible[
        (eligible["source_site"] == "site_b") & (eligible["antibiotic"] == "MEROPENEM")
    ].iloc[0]

    assert int(site_a_meropenem["is_biologically_eligible"]) == 1
    assert int(site_a_meropenem["is_operationally_available"]) == 0
    assert int(site_a_meropenem["is_eligible"]) == 0
    assert site_a_meropenem["observation_status"] == "operationally_unavailable"
    assert int(site_b_meropenem["is_operationally_available"]) == 1
    assert int(site_b_meropenem["is_eligible"]) == 1


def test_biological_denominator_keeps_non_intrinsic_unavailable_drugs_eligible() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        gold=replace(
            settings.gold,
            eligibility=replace(settings.gold.eligibility, denominator="biological"),
        ),
    )
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_a",
            }
        ]
    )
    culture_drug_episodes = pd.DataFrame(
        [
            {
                "anon_id": "B1",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2020-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_b",
                "antibiotic": "MEROPENEM",
                "susceptibility": "SUSCEPTIBLE",
                "was_tested": 1,
            }
        ]
    )

    eligible = service.build_episode_eligibility(culture_episodes, culture_drug_episodes)
    row = eligible.iloc[0]
    assert int(row["is_biologically_eligible"]) == 1
    assert int(row["is_operationally_available"]) == 0
    assert int(row["is_eligible"]) == 1


def test_organism_alias_resolves_intrinsic_resistance_for_annotated_spelling() -> None:
    """A phenotype-annotated raw organism string must inherit its species' real

    intrinsic-resistance profile, not silently default to "not resistant".
    Regression test for the organism-alias gap: before normalize_organism_label,
    "STAPH AUREUS {MRSA}" had zero match against the reference (exact-string
    join on the bare, un-aliased label), so is_intrinsic_resistance defaulted
    to 0 for every MRSA-labeled row -- verified against real data to affect
    12.05% of the audited ESKAPE cohort (577,805 of 4,795,529 rows).
    Staphylococcus aureus is a real, verified intrinsic-resistance entry in
    data/reference/intrinsic_resistance.csv for aztreonam (a monobactam with
    no gram-positive activity).
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2024-01-01",
                "organism": "STAPH AUREUS {MRSA}",
                "source_site": "armd",
            }
        ]
    )
    culture_drug_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2024-01-01",
                "organism": "STAPH AUREUS {MRSA}",
                "source_site": "armd",
                "antibiotic": "AZTREONAM",
                "susceptibility": "RESISTANT",
                "was_tested": 1,
            }
        ]
    )

    eligible = service.build_episode_eligibility(culture_episodes, culture_drug_episodes)
    row = eligible.iloc[0]
    assert int(row["is_intrinsic_resistance"]) == 1
    assert int(row["is_biologically_eligible"]) == 0
    assert int(row["is_eligible"]) == 0


def test_organism_alias_pools_operational_availability_across_spelling_variants() -> None:
    """Testing history for one spelling variant must count toward another

    variant's availability -- otherwise a drug well-established under
    "STAPHYLOCOCCUS AUREUS" would wrongly read as operationally unavailable
    for "METHICILLIN RESISTANT STAPHYLOCOCCUS AUREUS"-labeled episodes at the
    same site/era, even though they are the same organism and the same testing
    program. This is the availability-pooling half of the organism-alias fix,
    separate from the intrinsic-resistance join covered above.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "2020-01-01",
                "organism": "STAPHYLOCOCCUS AUREUS",
                "source_site": "site_a",
            },
            {
                "anon_id": "B1",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "2",
                "order_time_jittered": "2020-01-01",
                "organism": "METHICILLIN RESISTANT STAPHYLOCOCCUS AUREUS",
                "source_site": "site_a",
            },
        ]
    )
    culture_drug_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "2020-01-01",
                "organism": "STAPHYLOCOCCUS AUREUS",
                "source_site": "site_a",
                "antibiotic": "VANCOMYCIN",
                "susceptibility": "SUSCEPTIBLE",
                "was_tested": 1,
            },
        ]
    )

    eligible = service.build_episode_eligibility(culture_episodes, culture_drug_episodes)
    mrsa_vancomycin = eligible[
        (eligible["organism"] == "METHICILLIN RESISTANT STAPHYLOCOCCUS AUREUS")
        & (eligible["antibiotic"] == "VANCOMYCIN")
    ].iloc[0]
    assert int(mrsa_vancomycin["availability_support_n"]) == 1
    assert int(mrsa_vancomycin["is_operationally_available"]) == 1


def test_operational_eligibility_requires_parseable_time_column() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    service = EligibilityService(settings, path_manager.paths.reference)

    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "1",
                "order_time_jittered": "not-a-date",
                "organism": "ESCHERICHIA COLI",
                "source_site": "site_a",
            }
        ]
    )
    culture_drug_episodes = culture_episodes.assign(
        antibiotic="AMPICILLIN",
        susceptibility="SUSCEPTIBLE",
        was_tested=1,
    )

    with pytest.raises(ValueError, match="could not parse any years"):
        service.build_episode_eligibility(culture_episodes, culture_drug_episodes)


def test_benzylpenicillin_reference_entry_marks_penicillin_g_intrinsic_for_e_coli() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    service = EligibilityService(settings, PathManager(project_root, settings).paths.reference)
    episode = {
        "anon_id": "A1",
        "pat_enc_csn_id_coded": "1",
        "order_proc_id_coded": "2",
        "order_time_jittered": "2024-01-01",
        "organism": "ESCHERICHIA COLI",
        "source_site": "armd",
    }
    culture_drug_episodes = pd.DataFrame(
        [
            {**episode, "antibiotic": "PENICILLIN G", "susceptibility": "RESISTANT"},
            {**episode, "antibiotic": "CEFTRIAXON", "susceptibility": "SUSCEPTIBLE"},
        ]
    )

    eligibility = service.build_episode_eligibility(pd.DataFrame([episode]), culture_drug_episodes)
    by_drug = eligibility.set_index("antibiotic")

    assert by_drug.loc["PENICILLIN G", "is_intrinsic_resistance"] == 1
    assert by_drug.loc["PENICILLIN G", "is_eligible"] == 0
    assert by_drug.loc["CEFTRIAXON", "is_eligible"] == 1
