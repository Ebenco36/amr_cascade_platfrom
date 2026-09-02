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
