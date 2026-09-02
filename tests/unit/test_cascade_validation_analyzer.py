from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import (
    CascadeValidationAnalyzer,
)
from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_cascade_validation_analyzer_flags_strong_replicated_edge() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(
                settings.cascade.validation,
                permutation_iterations=40,
                bootstrap_iterations=40,
                random_seed=7,
            ),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    rows: list[dict[str, object]] = []
    for site in ("armd", "armd_ecuh"):
        for year in (2020, 2024):
            for idx in range(10):
                resistant_order_id = f"{site}_{year}_resistant_{idx}"
                susceptible_order_id = f"{site}_{year}_susceptible_{idx}"
                rows.append(
                    {
                        "anon_id": f"{site}_anon_{idx}",
                        "pat_enc_csn_id_coded": f"{site}_enc_{year}_{idx}",
                        "order_proc_id_coded": resistant_order_id,
                        "upstream_antibiotic": "CEFTRIAXONE",
                        "downstream_antibiotic": "MEROPENEM",
                        "upstream_susceptibility": "RESISTANT",
                        "downstream_tested": 1 if idx < 8 else 0,
                        "downstream_eligible": 1,
                        "source_site": site,
                        "order_time_jittered": f"{year}-03-01T00:00:00Z",
                    }
                )
                rows.append(
                    {
                        "anon_id": f"{site}_anon_{idx}",
                        "pat_enc_csn_id_coded": f"{site}_enc_{year}_{idx}",
                        "order_proc_id_coded": susceptible_order_id,
                        "upstream_antibiotic": "CEFTRIAXONE",
                        "downstream_antibiotic": "MEROPENEM",
                        "upstream_susceptibility": "SUSCEPTIBLE",
                        "downstream_tested": 1 if idx < 2 else 0,
                        "downstream_eligible": 1,
                        "source_site": site,
                        "order_time_jittered": f"{year}-03-15T00:00:00Z",
                    }
                )

    drug_pairs = pd.DataFrame(rows)
    retained_edges = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "MEROPENEM",
                "escalation_ratio": 4.0,
            }
        ]
    )

    result = analyzer.analyze(drug_pairs, retained_edges)
    assert len(result) == 1
    row = result.iloc[0]
    assert float(row["permutation_p_value"]) < 0.05
    assert float(row["permutation_fdr_q_value"]) == float(row["permutation_p_value"])
    assert bool(row["permutation_fdr_supported"]) is True
    assert float(row["bootstrap_sign_stability"]) >= 0.8
    assert int(row["site_replication_n"]) == 2
    assert row["validation_status"] == "robust"


def test_benjamini_hochberg_q_values_are_monotone_and_index_aligned() -> None:
    q_values = CascadeValidationAnalyzer._benjamini_hochberg(
        pd.Series([0.03, 0.01, 0.20], index=["edge_b", "edge_a", "edge_c"])
    )

    assert q_values.index.tolist() == ["edge_b", "edge_a", "edge_c"]
    assert abs(float(q_values.loc["edge_a"]) - 0.03) < 1e-12
    assert abs(float(q_values.loc["edge_b"]) - 0.045) < 1e-12
    assert abs(float(q_values.loc["edge_c"]) - 0.20) < 1e-12


def test_i_squared_threshold_uses_percent_scale() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")

    assert settings.cascade.validation.i_squared_threshold == 50.0


def test_between_site_permutation_is_invariant_to_input_row_order() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(
                settings.cascade.validation,
                permutation_iterations=25,
                random_seed=11,
                between_site_permutation_enabled=True,
            ),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    rows: list[dict[str, object]] = []
    for site in ("site_a", "site_b"):
        for idx in range(12):
            rows.append(
                {
                    "source_site": site,
                    "upstream_result_group": "positive" if idx < 6 else "negative",
                    "downstream_tested": 1 if idx < 5 else 0,
                }
            )
    ordered = pd.DataFrame(rows)
    shuffled = ordered.sample(frac=1.0, random_state=3).reset_index(drop=True)

    ordered_result = analyzer._between_site_permutation_summary(
        ordered,
        observed_er=5.0,
        upstream_antibiotic="CEFTRIAXONE",
        downstream_antibiotic="MEROPENEM",
    )
    shuffled_result = analyzer._between_site_permutation_summary(
        shuffled,
        observed_er=5.0,
        upstream_antibiotic="CEFTRIAXONE",
        downstream_antibiotic="MEROPENEM",
    )

    assert ordered_result == shuffled_result


def test_shuffle_within_strata_keeps_episode_clusters_together() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadeValidationAnalyzer(settings)
    frame = pd.DataFrame(
        [
            {"source_site": "armd", "anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "t1", "organism": "E", "upstream_result_group": "positive"},
            {"source_site": "armd", "anon_id": "p1", "pat_enc_csn_id_coded": "e1", "order_proc_id_coded": "o1", "order_time_jittered": "t1", "organism": "E", "upstream_result_group": "positive"},
            {"source_site": "armd", "anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "t2", "organism": "E", "upstream_result_group": "negative"},
            {"source_site": "armd", "anon_id": "p2", "pat_enc_csn_id_coded": "e2", "order_proc_id_coded": "o2", "order_time_jittered": "t2", "organism": "E", "upstream_result_group": "negative"},
        ]
    )

    shuffled = analyzer._shuffle_within_strata(
        values=frame["upstream_result_group"],
        strata=frame["source_site"],
        rng=np.random.default_rng(1),
        episode_keys=frame,
    )

    grouped_unique_counts = shuffled.groupby(
        [frame["anon_id"], frame["pat_enc_csn_id_coded"], frame["order_proc_id_coded"], frame["order_time_jittered"], frame["organism"], frame["source_site"]]
    ).nunique()
    assert grouped_unique_counts.max() == 1
