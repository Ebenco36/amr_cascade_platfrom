from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.features.builders.history_feature_builder import HistoryFeatureBuilder


def _settings():
    project_root = Path(__file__).resolve().parents[2]
    return ConfigLoader(project_root).load("mac")


def _culture_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "organism": "ESCHERICHIA COLI",
                "order_time_jittered": "2024-06-01T00:00:00Z",
            },
            {
                "anon_id": "A2",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "20",
                "source_site": "armd",
                "organism": "ESCHERICHIA COLI",
                "order_time_jittered": "2024-06-01T00:00:00Z",
            },
        ]
    )


def _loader(tables: dict[str, pd.DataFrame]):
    def load_site_table(site: str, canonical_table: str) -> pd.DataFrame:
        if site != "armd":
            return pd.DataFrame()
        return tables.get(canonical_table, pd.DataFrame())

    return load_site_table


def test_prior_procedures_history_counts_pre_culture_events_within_window() -> None:
    builder = HistoryFeatureBuilder(_settings())
    procedures = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "procedure_description": "central line placement",
                "procedure_time_to_culturetime": 5,
            },
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "procedure_description": "post-culture wound care",
                "procedure_time_to_culturetime": -3,
            },
        ]
    )
    result = builder.build(_culture_episodes(), _loader({"prior_procedures": procedures}))
    a1 = result.loc[result["anon_id"] == "A1"].iloc[0]
    a2 = result.loc[result["anon_id"] == "A2"].iloc[0]

    assert a1["history_procedure_available"] == 1
    assert a1["history_procedure_count"] == 1
    assert a1["history_procedure_any_30d"] == 1
    assert a2["history_procedure_available"] == 0
    assert a2["history_procedure_count"] == 0


def test_antibiotic_subtype_history_counts_pre_culture_exposures() -> None:
    builder = HistoryFeatureBuilder(_settings())
    exposures = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "antibiotic_subtype": "ceftriaxone",
                "medication_time_to_culturetime": 10,
            }
        ]
    )
    result = builder.build(_culture_episodes(), _loader({"antibiotic_subtype_exposure": exposures}))
    a1 = result.loc[result["anon_id"] == "A1"].iloc[0]
    a2 = result.loc[result["anon_id"] == "A2"].iloc[0]

    assert a1["history_abx_subtype_available"] == 1
    assert a1["history_abx_subtype_exposure_count"] == 1
    assert a1["history_abx_subtype_any_30d"] == 1
    assert a2["history_abx_subtype_available"] == 0


def test_nursing_home_visits_history_treats_column_as_day_offset_not_boolean_flag() -> None:
    # `nursing_home_visit_culture` carries the day-offset to the culture (0 = same day),
    # not a boolean presence flag -- the most common real value is 0, which a naive ">0"
    # filter would incorrectly drop.
    builder = HistoryFeatureBuilder(_settings())
    visits = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "source_site": "armd",
                "nursing_home_visit_culture": 0,
            },
            {
                "anon_id": "A2",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "20",
                "source_site": "armd",
                "nursing_home_visit_culture": "Null",
            },
        ]
    )
    result = builder.build(_culture_episodes(), _loader({"nursing_home_visits": visits}))
    a1 = result.loc[result["anon_id"] == "A1"].iloc[0]
    a2 = result.loc[result["anon_id"] == "A2"].iloc[0]

    assert a1["history_nursing_home_available"] == 1
    assert a1["history_nursing_home_visit_count"] == 1
    assert a1["history_nursing_home_any_30d"] == 1
    assert a2["history_nursing_home_available"] == 0
