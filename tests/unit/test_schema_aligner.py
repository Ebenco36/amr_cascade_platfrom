from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.harmonization.schema_aligner import SchemaAligner


def test_schema_aligner_renames_and_orders_columns() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    aligner = SchemaAligner(settings)

    dataframe = pd.DataFrame(
        [
            {
                "anon_id": "A1",
                "pat_enc_csn_id_coded": "1",
                "order_proc_id_coded": "2",
                "order_time_jittered_utc": "2024-01-01 00:00:00+00:00",
                "ordering_mode": "INPATIENT",
                "culture_description": "URINE",
                "was_positive": 1,
                "organism": "E. COLI",
                "antibiotic": "ERTAPENEM",
                "susceptibility": "SUSCEPTIBLE",
                "source_site": "armd",
            }
        ]
    )

    aligned = aligner.align(dataframe, "cohort")
    assert "order_time_jittered" in aligned.columns
    assert "order_time_jittered_utc" not in aligned.columns
    assert list(aligned.columns) == list(settings.harmonization.contracts["cohort"].columns)
