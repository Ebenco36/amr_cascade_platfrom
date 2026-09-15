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


def test_between_site_permutation_two_sided_matches_direct_formula() -> None:
    """between_site_permutation_p_value_two_sided must use the two-sided formula,

    not silently alias the one-sided value. Regression test for the gap where this
    statistic was left on the pre-fix one-sided calculation after the primary
    permutation was already corrected -- same anti-conservative-direction issue
    _empirical_p_value_two_sided's docstring describes, just never propagated here.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(
                settings.cascade.validation,
                permutation_iterations=30,
                random_seed=11,
                between_site_permutation_enabled=True,
            ),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    # Each site needs >= min_total_support (25) rows with >= min_result_support (5)
    # in each result-group for _dl_pooled_log_er_stat to not skip it -- the earlier
    # row-order-invariance test only needs *some* dict back (even an all-NaN one
    # compares equal to itself), but this test needs a genuine finite statistic,
    # so it must actually clear those per-site gates.
    rows: list[dict[str, object]] = []
    for site in ("site_a", "site_b"):
        for idx in range(30):
            is_positive = idx < 15
            if is_positive:
                tested = 1 if idx < 12 else 0  # 12/15 tested when positive
            else:
                tested = 1 if idx < 18 else 0  # 3/15 tested when negative
            rows.append(
                {
                    "source_site": site,
                    "upstream_result_group": "positive" if is_positive else "negative",
                    "downstream_tested": tested,
                }
            )
    subset = pd.DataFrame(rows)

    result = analyzer._between_site_permutation_summary(
        subset,
        observed_er=4.0,
        upstream_antibiotic="CEFTRIAXONE",
        downstream_antibiotic="MEROPENEM",
    )

    assert "between_site_permutation_p_value_two_sided" in result
    one_sided = result["between_site_permutation_p_value"]
    two_sided = result["between_site_permutation_p_value_two_sided"]
    assert pd.notna(two_sided), "test data must clear min_total_support/min_result_support to be meaningful"
    # Two-sided counts exceedances in both tails, so for the same null distribution
    # it can never be strictly smaller than the direction-matched one-sided value.
    assert two_sided >= one_sided - 1e-12


def test_merge_shards_applies_fdr_to_between_site_two_sided_column() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadeValidationAnalyzer(settings)

    shard = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "cascade_direction": "escalation",
                "observed_escalation_ratio": 4.0,
                "permutation_p_value": 0.02,
                "permutation_p_value_two_sided": 0.04,
                "permutation_null_median_er": 1.0,
                "permutation_null_q05_er": 0.5,
                "permutation_null_q95_er": 2.0,
                "bootstrap_median_er": 4.0,
                "bootstrap_er_ci_lower": 2.0,
                "bootstrap_er_ci_upper": 8.0,
                "bootstrap_sign_stability": 0.9,
                "bootstrap_retention_rate": 0.95,
                "site_replication_n": 2,
                "site_direction_agreement_rate": 1.0,
                "site_median_er": 4.0,
                "site_pooled_log_er": 1.38,
                "site_pooled_se": 0.2,
                "site_i_squared": 0.0,
                "site_cochran_q": 0.1,
                "site_heterogeneity_p": 0.9,
                "between_site_permutation_p_value": 0.03,
                "between_site_permutation_p_value_two_sided": 0.06,
                "temporal_early_er": 3.5,
                "temporal_late_er": 4.5,
                "temporal_n_early": 10,
                "temporal_n_late": 10,
                "temporal_direction_agreement": True,
                "temporal_log_ratio_delta": 0.1,
            }
        ]
    )

    merged = analyzer.merge_shards([shard])

    assert "between_site_permutation_fdr_q_value_two_sided" in merged.columns
    assert "between_site_permutation_supported_two_sided" in merged.columns
    row = merged.iloc[0]
    assert float(row["between_site_permutation_fdr_q_value_two_sided"]) == 0.06
    # threshold is 0.05 (configs/base/cascade.yaml); 0.06 > 0.05 -> not supported.
    assert bool(row["between_site_permutation_supported_two_sided"]) is False


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


def test_permutation_summary_era_stratified_separates_site_and_era() -> None:
    """Two episodes at the same site but different five-year eras must not be
    poolable in the sensitivity permutation's null -- only the primary,
    site-only permutation is allowed to mix them.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(settings.cascade.validation, permutation_iterations=5),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    subset = pd.DataFrame(
        {
            "source_site": ["armd", "armd", "armd", "armd"],
            "upstream_result_group": ["positive", "positive", "negative", "negative"],
            "downstream_tested": [1, 0, 0, 1],
            "_event_time": pd.to_datetime(
                ["2005-03-01", "2005-06-01", "2022-01-01", "2022-05-01"], utc=True
            ),
        }
    )

    captured_strata: list[pd.Series] = []
    real_shuffle = analyzer._shuffle_within_strata

    def _spy(*, values, strata, rng, episode_keys=None):
        captured_strata.append(strata.reset_index(drop=True))
        return real_shuffle(values=values, strata=strata, rng=rng, episode_keys=episode_keys)

    analyzer._shuffle_within_strata = _spy  # type: ignore[method-assign]

    result = analyzer.permutation_summary_era_stratified(
        subset=subset,
        observed_er=2.0,
        upstream_antibiotic="DRUG_A",
        downstream_antibiotic="DRUG_B",
    )

    assert "permutation_p_value_two_sided_era_stratified" in result
    p_value = result["permutation_p_value_two_sided_era_stratified"]
    assert p_value == p_value and 0.0 <= p_value <= 1.0  # not NaN, valid range

    assert len(captured_strata) == 5
    stratum = captured_strata[0]
    # Same site, but 2005 and 2022 fall in different five-year eras: rows 0-1
    # must carry a different stratum label than rows 2-3.
    assert stratum.iloc[0] == stratum.iloc[1]
    assert stratum.iloc[2] == stratum.iloc[3]
    assert stratum.iloc[0] != stratum.iloc[2]


def test_era_stratified_degeneracy_diagnostic_counts_frozen_strata() -> None:
    """Reviewer-requested diagnostic: report how much of the data is actually

    permutable under the site x era stratification, not just the p-value.
    One stratum here has both labels (non-degenerate, contributes to the
    null); the other has only one label present (degenerate, a no-op under
    shuffling) -- the diagnostic must correctly separate these.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(settings.cascade.validation, permutation_iterations=5),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    subset = pd.DataFrame(
        {
            "source_site": ["armd"] * 6,
            # Rows 0-3: era 2020, both labels present (2 positive, 2 negative) -- permutable.
            # Rows 4-5: era 2005, both positive -- degenerate, frozen under shuffling.
            "upstream_result_group": ["positive", "positive", "negative", "negative", "positive", "positive"],
            "downstream_tested": [1, 0, 0, 1, 1, 0],
            "_event_time": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2005-01-01", "2005-02-01"],
                utc=True,
            ),
        }
    )

    result = analyzer.permutation_summary_era_stratified(
        subset=subset,
        observed_er=2.0,
        upstream_antibiotic="DRUG_A",
        downstream_antibiotic="DRUG_B",
    )

    assert result["era_stratified_n_strata"] == 2.0
    assert result["era_stratified_n_strata_with_both_labels"] == 1.0
    assert result["era_stratified_n_permutable_rows"] == 4.0
    assert abs(result["era_stratified_fraction_fixed_rows"] - (1.0 - 4.0 / 6.0)) < 1e-9


def test_era_stratified_degeneracy_diagnostic_present_even_when_permutation_skipped() -> None:
    """The degeneracy diagnostic is a property of the data, not of a specific

    permutation draw -- it must still be reported when iterations<=0 or
    observed_er is NaN short-circuits the actual shuffling loop, not just
    silently omitted alongside the resulting NaN p-value.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadeValidationAnalyzer(settings)

    subset = pd.DataFrame(
        {
            "source_site": ["armd", "armd"],
            "upstream_result_group": ["positive", "negative"],
            "downstream_tested": [1, 0],
            "_event_time": pd.to_datetime(["2020-01-01", "2020-02-01"], utc=True),
        }
    )

    result = analyzer.permutation_summary_era_stratified(
        subset=subset,
        observed_er=float("nan"),
        upstream_antibiotic="DRUG_A",
        downstream_antibiotic="DRUG_B",
    )

    assert pd.isna(result["permutation_p_value_two_sided_era_stratified"])
    assert result["era_stratified_n_strata"] == 1.0
    assert result["era_stratified_n_strata_with_both_labels"] == 1.0


def test_era_stratified_sensitivity_summary_runs_over_validated_edges() -> None:
    """Regression test for the wiring gap: permutation_summary_era_stratified

    existed and was unit-tested, but nothing in the production pipeline ever
    called it -- so even after a full HPC regeneration, this sensitivity check
    would never actually run. This orchestration method (and the script that
    calls it) closes that gap: given a raw drug_pairs frame and a validated
    edge list, it prepares the pairs itself and returns one row per edge.
    """
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    settings = replace(
        settings,
        cascade=replace(
            settings.cascade,
            validation=replace(settings.cascade.validation, permutation_iterations=5),
        ),
    )
    analyzer = CascadeValidationAnalyzer(settings)

    rows: list[dict[str, object]] = []
    for site in ("armd", "armd_ecuh"):
        for idx in range(6):
            rows.append(
                {
                    "anon_id": f"{site}_p{idx}",
                    "pat_enc_csn_id_coded": f"{site}_e{idx}",
                    "order_proc_id_coded": f"{site}_r{idx}",
                    "upstream_antibiotic": "CEFTRIAXONE",
                    "downstream_antibiotic": "MEROPENEM",
                    "upstream_susceptibility": "RESISTANT" if idx < 3 else "SUSCEPTIBLE",
                    "downstream_tested": 1 if idx % 2 == 0 else 0,
                    "downstream_eligible": 1,
                    "source_site": site,
                    "order_time_jittered": "2020-03-01T00:00:00Z",
                }
            )
    drug_pairs = pd.DataFrame(rows)
    edges = pd.DataFrame(
        [
            {"upstream_antibiotic": "CEFTRIAXONE", "downstream_antibiotic": "MEROPENEM", "escalation_ratio": 2.0},
            # An edge with no matching data must be silently skipped, not raise.
            {"upstream_antibiotic": "VANCOMYCIN", "downstream_antibiotic": "DAPTOMYCIN", "escalation_ratio": 3.0},
        ]
    )

    result = analyzer.era_stratified_sensitivity_summary(drug_pairs, edges)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["upstream_antibiotic"] == "CEFTRIAXONE"
    assert row["downstream_antibiotic"] == "MEROPENEM"
    assert "permutation_p_value_two_sided_era_stratified" in result.columns
    assert 0.0 <= row["permutation_p_value_two_sided_era_stratified"] <= 1.0


def test_era_stratified_sensitivity_summary_empty_inputs_keep_schema() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    analyzer = CascadeValidationAnalyzer(settings)

    result = analyzer.era_stratified_sensitivity_summary(pd.DataFrame(), pd.DataFrame())

    assert result.empty
    assert "permutation_p_value_two_sided_era_stratified" in result.columns
