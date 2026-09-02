from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.transformations.culture_episode_builder import CultureEpisodeBuilder


def test_culture_episode_builder_filters_to_requested_organism() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    builder = CultureEpisodeBuilder(settings)
    cohort = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "10",
                "order_time_jittered": "2024-01-01",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "ordering_mode": "lab",
                "culture_description": "urine",
                "was_positive": 1,
            },
            {
                "anon_id": "A2",
                "pat_enc_csn_id_coded": "2",
                "order_proc_id_coded": "20",
                "order_time_jittered": "2024-01-02",
                "organism": "KLEBSIELLA PNEUMONIAE",
                "source_site": "armd",
                "ordering_mode": "lab",
                "culture_description": "urine",
                "was_positive": 1,
            },
        ]
    )
    result = builder.build(cohort, organism_list=("ESCHERICHIA COLI",))
    assert len(result) == 1
    assert result.iloc[0]["organism"] == "ESCHERICHIA COLI"
