"""join_colon in the HPC job-graph script: empty job lists must not look like a dependency."""
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "submit_pipeline_dag_hpc.sh"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('"" ""', ""),  # harmonization skipped, no --after-job: the case that requeued every finished stage
        ("", ""),
        ('"" 12 "" 34', "12:34"),
        ("12", "12"),
        ('"12:34" 56', "12:34:56"),
    ],
)
def test_join_colon_drops_empty_items(arguments, expected):
    function = re.search(r"^join_colon\(\) \{.*?^\}", SCRIPT.read_text(), flags=re.S | re.M).group(0)
    result = subprocess.run(["bash", "-c", f"{function}\njoin_colon {arguments}"], capture_output=True, text=True, check=True)
    assert result.stdout == expected
