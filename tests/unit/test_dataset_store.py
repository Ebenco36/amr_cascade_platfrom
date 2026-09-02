from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


def _settings():
    project_root = Path(__file__).resolve().parents[2]
    return ConfigLoader(project_root).load("mac")


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anon_id": [f"p{i}" for i in range(20)],
            "upstream_antibiotic": (["CIPROFLOXACIN", "AMPICILLIN"] * 10),
            "downstream_tested": ([1, 0] * 10),
        }
    )


def test_read_pandas_without_categorical_columns_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    _sample_frame().to_parquet(path, index=False)

    store = DatasetStore(_settings())
    result = store.read_pandas(path)

    assert result["upstream_antibiotic"].dtype == object
    assert len(result) == 20


def test_read_pandas_with_categorical_columns_preserves_values_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    original = _sample_frame()
    original.to_parquet(path, index=False)

    store = DatasetStore(_settings())
    result = store.read_pandas(path, categorical_columns=["upstream_antibiotic"])

    assert isinstance(result["upstream_antibiotic"].dtype, pd.CategoricalDtype)
    # Non-requested columns are unaffected.
    assert result["anon_id"].dtype == object
    assert result["downstream_tested"].dtype in (int, "int64")
    # Values round-trip exactly; only the dtype changes.
    pd.testing.assert_series_equal(
        result["upstream_antibiotic"].astype(object),
        original["upstream_antibiotic"],
        check_names=False,
    )


def test_read_pandas_categorical_column_groupby_observed_true_matches_object_dtype(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    _sample_frame().to_parquet(path, index=False)
    store = DatasetStore(_settings())

    object_dtype = store.read_pandas(path)
    categorical_dtype = store.read_pandas(path, categorical_columns=["upstream_antibiotic"])

    object_counts = object_dtype.groupby("upstream_antibiotic", observed=True)["downstream_tested"].sum()
    categorical_counts = categorical_dtype.groupby("upstream_antibiotic", observed=True)["downstream_tested"].sum()

    # observed=True must produce the same real groups regardless of dtype -- no
    # phantom cartesian-product categories from the categorical conversion. Index
    # to plain strings before comparing: CategoricalIndex sorts by dictionary
    # order, not alphabetically, so the raw index objects aren't comparable.
    object_counts.index = object_counts.index.astype(str)
    categorical_counts.index = categorical_counts.index.astype(str)
    pd.testing.assert_series_equal(
        categorical_counts.astype(object_counts.dtype).sort_index(),
        object_counts.sort_index(),
        check_names=False,
    )


def test_read_pandas_categorical_columns_respects_column_and_filter_selection(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    frame = _sample_frame()
    frame["source_site"] = (["armd", "armd_ecuh"] * 10)
    frame.to_parquet(path, index=False)

    store = DatasetStore(_settings())
    result = store.read_pandas(
        path,
        columns=["upstream_antibiotic", "source_site"],
        filters=[("source_site", "==", "armd")],
        categorical_columns=["upstream_antibiotic", "source_site"],
    )

    assert list(result.columns) == ["upstream_antibiotic", "source_site"]
    assert (result["source_site"] == "armd").all()
    assert len(result) == 10
