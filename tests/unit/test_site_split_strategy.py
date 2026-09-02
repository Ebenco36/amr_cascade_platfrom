from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.modeling.datasets.split_strategy import SiteSplitStrategy, TemporalSplitStrategy


def test_site_split_strategy_uses_configured_sites() -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    dataframe = pd.DataFrame(
        {
            "source_site": ["armd", "armd_ecuh", "armd_utsw"],
            "target": [1, 0, 1],
        }
    )
    split = SiteSplitStrategy(settings).split(dataframe)
    assert split.train["source_site"].tolist() == ["armd"]
    assert split.validation["source_site"].tolist() == ["armd_ecuh"]
    assert split.test["source_site"].tolist() == ["armd_utsw"]


def test_site_split_strategy_rejects_overlapping_site_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")

    class OverlappingModeling:
        split_column = settings.modeling.split_column
        train_sites = ("armd",)
        validation_sites = ("armd",)
        test_sites = settings.modeling.test_sites

    overlapping_settings = type("OverlappingSettings", (), {"modeling": OverlappingModeling(), "gold": settings.gold})()

    with pytest.raises(ValidationError, match="not disjoint"):
        SiteSplitStrategy(overlapping_settings)


def test_temporal_split_strategy_uses_year_windows() -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    dataframe = pd.DataFrame(
        {
            "source_site": ["armd", "armd", "armd"],
            "order_time_jittered": [
                "2018-01-01T00:00:00Z",
                "2020-01-01 00:00:00+00:00",
                "2023-01-01T00:00:00.000Z",
            ],
            "anon_id": ["p1", "p2", "p3"],
            "pat_enc_csn_id_coded": ["e1", "e2", "e3"],
            "order_proc_id_coded": ["o1", "o2", "o3"],
            "organism": ["ESCHERICHIA COLI"] * 3,
            "target": [1, 0, 1],
        }
    )
    split = TemporalSplitStrategy(settings).split(dataframe)
    assert split.train["anon_id"].tolist() == ["p1"]
    assert split.validation["anon_id"].tolist() == ["p2"]
    assert split.test["anon_id"].tolist() == ["p3"]
