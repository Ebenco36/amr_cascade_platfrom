"""Tests for ESKAPE-aware sample dataset creation."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from amr_cascade_platform.data.sampling.sample_dataset_creator import (
    PandasCsvSampler,
    SampleDatasetCreator,
    SamplingRequest,
)


class _NoopLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


class _FakeMapper:
    def __init__(self, table_map):
        self._table_map = table_map

    def site_tables(self, site: str):
        return self._table_map


class _FakePaths:
    def __init__(self, sample_root):
        self._sample_root = sample_root

    def site_dir(self, layer: str, site: str):
        return self._sample_root / site


class _NoDaskPolicy:
    def decide(self, input_path, dask_available: bool):
        return SimpleNamespace(use_dask=False)


def test_sample_cohort_dataframe_with_eskape_floor_preserves_available_targets() -> None:
    dataframe = pd.DataFrame(
        {
            "anon_id": [f"a{i}" for i in range(12)],
            "pat_enc_csn_id_coded": [f"e{i}" for i in range(12)],
            "order_proc_id_coded": [f"o{i}" for i in range(12)],
            "organism": [
                "KLEBSIELLA PNEUMONIAE",
                "PSEUDOMONAS AERUGINOSA",
                "STAPHYLOCOCCUS AUREUS",
                "ENTEROCOCCUS FAECIUM",
                "ACINETOBACTER BAUMANNII",
                "ENTEROBACTER CLOACAE COMPLEX",
                "ESCHERICHIA COLI",
                "ESCHERICHIA COLI",
                "PROTEUS MIRABILIS",
                "SERRATIA MARCESCENS",
                "ENTEROBACTER AEROGENES",
                "CITROBACTER FREUNDII",
            ],
        }
    )

    sampled, coverage = SampleDatasetCreator._sample_cohort_dataframe_with_eskape_floor(
        dataframe=dataframe,
        fraction=0.10,
        seed=42,
        min_rows_per_target=1,
    )

    assert len(sampled) >= 6
    present_targets = coverage.loc[coverage["present_in_raw"], "present_in_sample"]
    assert present_targets.all()


def test_sample_site_filters_all_keyed_tables_to_sampled_cohort(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    sample_dir = tmp_path / "sample"
    raw_dir.mkdir()
    cohort_path = raw_dir / "microbiology_cultures_cohort.csv"
    comorbidity_path = raw_dir / "microbiology_cultures_comorbidity.csv"
    cohort = pd.DataFrame(
        {
            "anon_id": [f"a{i}" for i in range(10)],
            "pat_enc_csn_id_coded": [f"e{i}" for i in range(10)],
            "order_proc_id_coded": [f"o{i}" for i in range(10)],
            "organism": ["ESCHERICHIA COLI"] * 10,
            "antibiotic": ["CEFTRIAXONE"] * 10,
            "susceptibility": ["Susceptible"] * 10,
        }
    )
    comorbidity = pd.DataFrame(
        {
            "anon_id": [f"a{i}" for i in range(10)] + ["unlinked"],
            "pat_enc_csn_id_coded": [f"e{i}" for i in range(10)] + ["unlinked"],
            "order_proc_id_coded": [f"o{i}" for i in range(10)] + ["unlinked"],
            "comorbidity_component": ["diabetes"] * 11,
        }
    )
    cohort.to_csv(cohort_path, index=False)
    comorbidity.to_csv(comorbidity_path, index=False)

    creator = SampleDatasetCreator.__new__(SampleDatasetCreator)
    creator._settings = SimpleNamespace(
        platform=SimpleNamespace(
            sites=("armd",),
            id_columns=("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"),
            missing_tokens=("",),
        ),
        sampling=SimpleNamespace(dask_blocksize="1MB"),
    )
    creator._paths = _FakePaths(sample_dir)
    creator._mapper = _FakeMapper({"cohort": cohort_path, "comorbidity": comorbidity_path})
    creator._logger = _NoopLogger()
    creator._policy = _NoDaskPolicy()
    creator._pandas_sampler = PandasCsvSampler(large_file_threshold_mb=10_000)

    creator._sample_site(
        site="armd",
        request=SamplingRequest(
            fraction=0.2,
            seed=17,
            ensure_eskape_coverage=True,
            eskape_min_rows_per_target=2,
        ),
    )

    sampled_cohort = pd.read_csv(sample_dir / "armd" / cohort_path.name, dtype=str)
    sampled_comorbidity = pd.read_csv(sample_dir / "armd" / comorbidity_path.name, dtype=str)
    sampled_order_ids = set(sampled_cohort["order_proc_id_coded"])

    assert not sampled_cohort.empty
    assert set(sampled_comorbidity["order_proc_id_coded"]).issubset(sampled_order_ids)
    assert "unlinked" not in set(sampled_comorbidity["order_proc_id_coded"])


def test_sample_cohort_dataframe_with_eskape_floor_applies_minimum_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "anon_id": [f"a{i}" for i in range(20)],
            "pat_enc_csn_id_coded": [f"e{i}" for i in range(20)],
            "order_proc_id_coded": [f"o{i}" for i in range(20)],
            "organism": (
                ["ACINETOBACTER BAUMANNII"] * 4
                + ["KLEBSIELLA PNEUMONIAE"] * 8
                + ["PSEUDOMONAS AERUGINOSA"] * 8
            ),
        }
    )

    sampled, coverage = SampleDatasetCreator._sample_cohort_dataframe_with_eskape_floor(
        dataframe=dataframe,
        fraction=0.10,
        seed=7,
        min_rows_per_target=3,
    )

    assert len(sampled) >= 9
    acinetobacter_row = coverage.loc[coverage["eskape_target"].eq("Acinetobacter")].iloc[0]
    klebsiella_row = coverage.loc[coverage["eskape_target"].eq("Klebsiella")].iloc[0]
    pseudomonas_row = coverage.loc[coverage["eskape_target"].eq("Pseudomonas")].iloc[0]
    assert acinetobacter_row["sampled_count"] == 3
    assert klebsiella_row["sampled_count"] == 3
    assert pseudomonas_row["sampled_count"] == 3
