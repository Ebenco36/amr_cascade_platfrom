import pandas as pd
from pathlib import Path

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.cleaning.table_cleaner import TableCleaner


def test_table_cleaner_trims_and_deduplicates() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    cleaner = TableCleaner(settings)

    dataframe = pd.DataFrame(
        [
            {
                "anon_id": " A1 ",
                "pat_enc_csn_id_coded": " 10 ",
                "order_proc_id_coded": " 20 ",
                "organism": "klebsiella pneumoniae ",
                "antibiotic": " ertapenem ",
                "susceptibility": " susceptible ",
                "source_site": "armd",
            },
            {
                "anon_id": " A1 ",
                "pat_enc_csn_id_coded": " 10 ",
                "order_proc_id_coded": " 20 ",
                "organism": "klebsiella pneumoniae ",
                "antibiotic": " ertapenem ",
                "susceptibility": " susceptible ",
                "source_site": "armd",
            },
        ]
    )

    result = cleaner.clean(dataframe, "cohort")
    assert result.rows_before == 2
    assert result.rows_after == 1
    assert result.duplicates_removed == 1
    assert result.dataframe.iloc[0]["organism"] == "KLEBSIELLA PNEUMONIAE"


def test_table_cleaner_audits_discordant_microbial_resistance_duplicates() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    cleaner = TableCleaner(settings)

    dataframe = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "10",
                "order_proc_id_coded": "20",
                "organism": "escherichia coli",
                "antibiotic": "ceftriaxone",
                "resistant_time_to_culturetime": 0.0,
                "susceptibility": "susceptible",
                "source_site": "armd",
            },
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "10",
                "order_proc_id_coded": "20",
                "organism": "escherichia coli",
                "antibiotic": "ceftriaxone",
                "resistant_time_to_culturetime": 0.0,
                "susceptibility": "resistant",
                "source_site": "armd",
            },
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "10",
                "order_proc_id_coded": "20",
                "organism": "escherichia coli",
                "antibiotic": "ceftriaxone",
                "resistant_time_to_culturetime": 24.0,
                "susceptibility": "resistant",
                "source_site": "armd",
            },
        ]
    )

    result = cleaner.clean(dataframe, "microbial_resistance")
    assert result.rows_before == 3
    # The first two rows share the same key (time=0.0) but conflict on
    # susceptibility -- both must be excluded, not silently collapsed to one.
    # Only the third (time=24.0, its own key, no conflict) survives.
    assert result.rows_after == 1
    assert result.duplicates_removed == 0
    assert result.discordant_susceptibility_group_n == 1
    assert result.discordant_susceptibility_row_n == 2
    assert result.dataframe["resistant_time_to_culturetime"].tolist() == [24.0]
    assert result.discordant_susceptibility_row_n == 2


def test_table_cleaner_coalesces_measurement_shards_for_labs() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    cleaner = TableCleaner(settings)

    dataframe = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "E1",
                "order_proc_id_coded": "LAB1",
                "Period_Day": 14,
                "median_wbc": 11.0,
                "median_cr": None,
                "first_wbc": 10.0,
                "last_wbc": None,
            },
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "E1",
                "order_proc_id_coded": "LAB1",
                "Period_Day": 14,
                "median_wbc": 13.0,
                "median_cr": 1.8,
                "first_wbc": None,
                "last_wbc": 12.0,
            },
            {
                "anon_id": "A2",
                "pat_enc_csn_id_coded": "E2",
                "order_proc_id_coded": "LAB2",
                "Period_Day": 14,
                "median_wbc": 7.0,
                "median_cr": 0.9,
                "first_wbc": 7.0,
                "last_wbc": 7.2,
            },
        ]
    )

    result = cleaner.clean(dataframe, "labs")

    assert result.rows_before == 3
    assert result.rows_after == 2
    assert result.duplicates_removed == 1
    assert result.shard_coalesced_group_n == 1
    assert result.shard_conflict_group_n == 1
    assert result.shard_conflict_cell_n == 1
    collapsed = result.dataframe.loc[result.dataframe["anon_id"].eq("A1")].iloc[0]
    assert float(collapsed["median_wbc"]) == 12.0
    assert float(collapsed["median_cr"]) == 1.8
    assert float(collapsed["first_wbc"]) == 10.0
    assert float(collapsed["last_wbc"]) == 12.0
    assert int(collapsed["Period_Day"]) == 14
