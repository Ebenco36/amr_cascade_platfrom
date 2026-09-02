import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.reporting.builders.manuscript_table_builder import ManuscriptTableBuilder


def test_edge_presence_label() -> None:
    row = pd.Series({"site_escalation_ratio": 2.0, "combined_escalation_ratio": None})
    assert ManuscriptTableBuilder._edge_presence_label(row) == "site_only"


def test_resolve_sites_for_combined() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    builder = ManuscriptTableBuilder(settings, PathManager(project_root, settings))
    assert builder._resolve_sites("combined", None) == list(settings.platform.sites)


def test_normalize_edge_report_collapses_antibiotic_aliases() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    builder = ManuscriptTableBuilder(settings, PathManager(project_root, settings))

    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "AUGMENTIN",
                "downstream_antibiotic": "PIP/TAZOBACTAM",
                "positive_support_n": 10,
                "negative_support_n": 50,
                "positive_probability": 0.1,
                "negative_probability": 0.01,
                "escalation_ratio": 10.0,
                "adjusted_odds_ratio": 2.0,
                "total_support_n": 60,
            },
            {
                "upstream_antibiotic": "AMOX/CLAVULANATE",
                "downstream_antibiotic": "PIPERACILLIN/TAZOBACTAM",
                "positive_support_n": 8,
                "negative_support_n": 40,
                "positive_probability": 0.08,
                "negative_probability": 0.008,
                "escalation_ratio": 10.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 48,
            },
        ]
    )

    normalized = builder._normalize_edge_report(edge_report)

    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["upstream_antibiotic"] == "AMOXICILLIN/CLAVULANIC ACID"
    assert row["downstream_antibiotic"] == "PIPERACILLIN/TAZOBACTAM"


def test_classification_display_prefers_who_match_for_publication_labels() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    builder = ManuscriptTableBuilder(settings, PathManager(project_root, settings))

    assert builder._display_antibiotic_label("CEFEPIM") == "CEFEPIME"


def test_validated_primary_cascade_table_filters_to_supported_statuses(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    artifact_dir = (
        path_manager.paths.artifacts
        / settings.cascade.outputs.result_dir
        / "combined"
        / "organisms"
        / "unit_test_bug"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "positive_support_n": 10,
                "negative_support_n": 20,
                "positive_probability": 0.4,
                "negative_probability": 0.1,
                "escalation_ratio": 4.0,
                "adjusted_odds_ratio": 2.0,
                "total_support_n": 30,
                "validation_status": "robust",
                "permutation_p_value": 0.01,
                "bootstrap_sign_stability": 0.9,
                "site_replication_n": 2,
                "site_direction_agreement_rate": 1.0,
            },
            {
                "upstream_antibiotic": "C",
                "downstream_antibiotic": "D",
                "positive_support_n": 10,
                "negative_support_n": 20,
                "positive_probability": 0.3,
                "negative_probability": 0.1,
                "escalation_ratio": 3.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 30,
                "validation_status": "mixed",
                "permutation_p_value": 0.2,
                "bootstrap_sign_stability": 0.8,
                "site_replication_n": 1,
                "site_direction_agreement_rate": 1.0,
            },
        ]
    )
    edge_report.to_parquet(artifact_dir / "edge_report.parquet", index=False)

    table = builder.build_validated_primary_cascade_table("combined", None, "unit_test_bug")
    assert len(table) == 1
    assert table.iloc[0]["upstream_drug"] == "A"
    assert table.iloc[0]["validation_status"] == "robust"
    assert "aware_transition" in table.columns
    assert "aware_step" in table.columns


def test_site_vs_combined_summary_builds_from_organism_scoped_edge_reports(tmp_path: Path) -> None:
    cascade_root = tmp_path / "artifacts" / "cascade"
    organism_suffix = Path("organisms") / "escherichia_coli"
    combined_dir = cascade_root / "combined" / organism_suffix
    site_dir = cascade_root / "armd" / organism_suffix
    combined_dir.mkdir(parents=True)
    site_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
            "escalation_ratio": [2.0],
            "adjusted_odds_ratio": [1.2],
            "total_support_n": [30],
        }
    ).to_parquet(combined_dir / "edge_report.parquet", index=False)
    pd.DataFrame(
        {
            "upstream_antibiotic": ["A"],
            "downstream_antibiotic": ["B"],
            "escalation_ratio": [2.4],
            "adjusted_odds_ratio": [1.4],
            "total_support_n": [28],
        }
    ).to_parquet(site_dir / "edge_report.parquet", index=False)

    builder = ManuscriptTableBuilder.__new__(ManuscriptTableBuilder)
    builder._settings = SimpleNamespace(
        cascade=SimpleNamespace(outputs=SimpleNamespace(result_dir="cascade"))
    )
    builder._paths = SimpleNamespace(paths=SimpleNamespace(artifacts=tmp_path / "artifacts"))

    def safe_read(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    builder._safe_read = safe_read

    summary = builder.build_site_vs_combined_summary("combined", None, "ESCHERICHIA COLI")

    assert not summary.empty
    assert (combined_dir / "site_vs_combined_summary.parquet").exists()
    assert summary.iloc[0]["site"] == "armd"


def test_downstream_trigger_availability_uses_only_validated_edges(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    artifact_dir = (
        path_manager.paths.artifacts
        / settings.cascade.outputs.result_dir
        / "combined"
        / "organisms"
        / "unit_test_validated_trigger"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "total_support_n": 30,
                "validation_status": "robust",
            },
            {
                "upstream_antibiotic": "C",
                "downstream_antibiotic": "B",
                "escalation_ratio": 0.5,
                "total_support_n": 40,
                "validation_status": "supported",
            },
            {
                "upstream_antibiotic": "D",
                "downstream_antibiotic": "B",
                "escalation_ratio": 5.0,
                "total_support_n": 50,
                "validation_status": "mixed",
            },
        ]
    ).to_parquet(artifact_dir / "edge_report.parquet", index=False)

    table = builder.build_downstream_trigger_availability_table("combined", None, "unit_test_validated_trigger")

    assert len(table) == 1
    row = table.iloc[0]
    assert row["validated_upstream_n"] == 2
    assert row["validated_escalation_upstream_n"] == 1
    assert row["validated_suppression_upstream_n"] == 1
    assert bool(row["has_validated_trigger_forest"])


def test_aware_transition_summary_table_aggregates_validated_edges(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    artifact_dir = (
        path_manager.paths.artifacts
        / settings.cascade.outputs.result_dir
        / "combined"
        / "organisms"
        / "unit_test_aware"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "AMOX/CLAVULANATE",
                "downstream_antibiotic": "CEFTRIAXONE",
                "positive_support_n": 10,
                "negative_support_n": 20,
                "positive_probability": 0.4,
                "negative_probability": 0.2,
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.8,
                "total_support_n": 30,
                "validation_status": "robust",
            },
            {
                "upstream_antibiotic": "AMPICILLIN/SULBACTAM",
                "downstream_antibiotic": "CEFEPIME",
                "positive_support_n": 8,
                "negative_support_n": 16,
                "positive_probability": 0.5,
                "negative_probability": 0.25,
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.6,
                "total_support_n": 24,
                "validation_status": "supported",
            },
            {
                "upstream_antibiotic": "VANCOMYCIN",
                "downstream_antibiotic": "LINEZOLID",
                "positive_support_n": 6,
                "negative_support_n": 12,
                "positive_probability": 0.5,
                "negative_probability": 0.4,
                "escalation_ratio": 1.25,
                "adjusted_odds_ratio": 1.2,
                "total_support_n": 18,
                "validation_status": "mixed",
            },
        ]
    )
    edge_report.to_parquet(artifact_dir / "edge_report.parquet", index=False)

    table = builder.build_aware_transition_summary_table("combined", None, "unit_test_aware")
    assert not table.empty
    assert "aware_transition" in table.columns
    assert "support_weighted_mean_escalation_ratio" in table.columns
    assert set(table["aware_direction"]) <= {"upward", "downward", "lateral", "unclassified"}
    assert int(table["validated_edge_n"].sum()) == 2


def test_aware_direction_hypothesis_table_rolls_up_by_direction(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    artifact_dir = (
        path_manager.paths.artifacts
        / settings.cascade.outputs.result_dir
        / "combined"
        / "organisms"
        / "unit_test_aware_direction"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "AMOX/CLAVULANATE",
                "downstream_antibiotic": "CEFTRIAXONE",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.8,
                "total_support_n": 30,
                "validation_status": "robust",
            },
            {
                "upstream_antibiotic": "CEFTRIAXONE",
                "downstream_antibiotic": "AMOX/CLAVULANATE",
                "escalation_ratio": 1.8,
                "adjusted_odds_ratio": 1.6,
                "total_support_n": 20,
                "validation_status": "supported",
            },
        ]
    )
    edge_report.to_parquet(artifact_dir / "edge_report.parquet", index=False)

    table = builder.build_aware_direction_hypothesis_table("combined", None, "unit_test_aware_direction")
    assert not table.empty
    assert {"aware_direction", "validated_edge_n", "validated_edge_share", "supports_upward_hypothesis"}.issubset(table.columns)


def test_covariate_correlation_table_flags_high_dependence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    gold_dir = (
        path_manager.paths.gold
        / "combined"
        / "organisms"
        / "unit_test_corr"
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2020-01-01T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "ordering_mode": "Inpatient",
                "culture_description": "URINE",
                "cov_ordering_mode": "inpatient",
                "cov_specimen_type": "urine",
                "comorb_asthma": 1,
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e2",
                "order_proc_id_coded": "o2",
                "order_time_jittered": "2020-01-02T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "ordering_mode": "Inpatient",
                "culture_description": "URINE",
                "cov_ordering_mode": "inpatient",
                "cov_specimen_type": "urine",
                "comorb_asthma": 1,
            },
            {
                "anon_id": "p3",
                "pat_enc_csn_id_coded": "e3",
                "order_proc_id_coded": "o3",
                "order_time_jittered": "2020-01-03T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "ordering_mode": "Outpatient",
                "culture_description": "URINE",
                "cov_ordering_mode": "outpatient",
                "cov_specimen_type": "urine",
                "comorb_asthma": 0,
            },
            {
                "anon_id": "p4",
                "pat_enc_csn_id_coded": "e4",
                "order_proc_id_coded": "o4",
                "order_time_jittered": "2020-01-04T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "ordering_mode": "Outpatient",
                "culture_description": "URINE",
                "cov_ordering_mode": "outpatient",
                "cov_specimen_type": "urine",
                "comorb_asthma": 0,
            },
        ]
    )
    culture_episodes.to_parquet(gold_dir / "culture_episodes.parquet", index=False)

    table = builder.build_covariate_correlation_table("combined", None, "unit_test_corr")

    assert not table.empty
    high_row = table.loc[
        (
            (table["feature_a"] == "cov_ordering_mode=inpatient")
            & (table["feature_b"] == "comorb_asthma")
        )
        | (
            (table["feature_b"] == "cov_ordering_mode=inpatient")
            & (table["feature_a"] == "comorb_asthma")
        )
    ].iloc[0]
    assert float(high_row["absolute_correlation"]) == 1.0
    assert high_row["correlation_flag"] == "very_high"
    assert "comorb_asthma" in set(table["feature_a"]) | set(table["feature_b"])


def test_asymmetry_sensitivity_table_summarizes_bins(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    artifact_dir = (
        path_manager.paths.artifacts
        / settings.cascade.outputs.result_dir
        / "combined"
        / "organisms"
        / "unit_test_asymmetry"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    edge_report = pd.DataFrame(
        [
            {
                "upstream_antibiotic": "A",
                "downstream_antibiotic": "B",
                "escalation_ratio": 2.0,
                "adjusted_odds_ratio": 1.5,
                "total_support_n": 30,
                "validation_status": "robust",
                "testing_asymmetry_score": 0.1,
                "testing_asymmetry_bin": "low_asymmetry",
                "bundled_panel_fraction": 0.8,
            },
            {
                "upstream_antibiotic": "C",
                "downstream_antibiotic": "D",
                "escalation_ratio": 4.0,
                "adjusted_odds_ratio": 2.5,
                "total_support_n": 20,
                "validation_status": "supported",
                "testing_asymmetry_score": 1.2,
                "testing_asymmetry_bin": "high_asymmetry",
                "bundled_panel_fraction": 0.3,
            },
        ]
    )
    edge_report.to_parquet(artifact_dir / "edge_report.parquet", index=False)

    table = builder.build_asymmetry_sensitivity_table("combined", None, "unit_test_asymmetry")
    assert not table.empty
    assert {"testing_asymmetry_bin", "retained_edge_n", "validated_edge_rate", "mean_testing_asymmetry_score"}.issubset(table.columns)


def test_prevalence_shift_table_exports_main_summary(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    prevalence_dir = (
        path_manager.paths.artifacts
        / settings.prevalence.output_dir
        / "combined"
        / "organisms"
        / "unit_test_prevalence_table"
    )
    prevalence_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "organism": "ESCHERICHIA COLI",
                "drug": "CEFEPIME",
                "eligible_n": 100,
                "tested_n": 20,
                "unknown_binary_outcome_n": 80,
                "naive_prevalence_pct": 25.123,
                "mnar_lambda0_prevalence_pct": 10.456,
                "mnar_lambda0_shift_from_naive_pct": 14.667,
                "mnar_lambda0_absolute_shift_pct": 14.667,
                "mnar_status_lambda0": "estimated",
                "prevalence_lower_bound_pct": 5.0,
                "prevalence_upper_bound_pct": 85.0,
                "standardised_prevalence_pct": 12.345,
                "rho_independent_vs_cascade": 0.25,
                "cascade_trigger_fraction": 0.4,
            }
        ]
    ).to_parquet(prevalence_dir / "prevalence_shift.parquet", index=False)

    table = builder.build_prevalence_shift_table("combined", None, "unit_test_prevalence_table")

    assert len(table) == 1
    assert "naive_prevalence_pct" in table.columns
    assert "mnar_lambda0_prevalence_pct" in table.columns
    assert "standardised_prevalence_pct" in table.columns
    assert float(table.iloc[0]["naive_prevalence_pct"]) == 25.12


def test_provenance_table_includes_discordant_susceptibility_audit() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    metadata_dir = path_manager.paths.metadata / "datasets" / "silver" / "armd"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "microbial_resistance.json").write_text(
        json.dumps(
            {
                "canonical_table": "microbial_resistance",
                "duplicates_removed": 17,
                "discordant_susceptibility_group_n": 3,
                "discordant_susceptibility_row_n": 7,
            }
        ),
        encoding="utf-8",
    )

    gold_dir = path_manager.paths.gold / "armd" / "organisms" / "unit_test_provenance"
    gold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_id": [1, 2]}).to_parquet(gold_dir / "culture_episodes.parquet", index=False)
    pd.DataFrame({"pair_id": [1, 2, 3]}).to_parquet(gold_dir / "eligible_pairs.parquet", index=False)
    pd.DataFrame({"pair_id": [1, 2, 3, 4]}).to_parquet(gold_dir / "drug_pair_episodes.parquet", index=False)

    table = builder.build_provenance_table("site", "armd", "unit_test_provenance")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["microbial_resistance_duplicates_removed"] == 17
    assert row["discordant_susceptibility_group_n"] == 3
    assert row["discordant_susceptibility_row_n"] == 7


def test_upstream_selection_balance_table_summarizes_tested_vs_untested_groups(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    path_manager = PathManager(project_root, settings)
    builder = ManuscriptTableBuilder(settings, path_manager)

    gold_dir = path_manager.paths.gold / "combined" / "organisms" / "unit_test_selection_balance"
    gold_dir.mkdir(parents=True, exist_ok=True)
    culture_episodes = pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2024-01-01T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e2",
                "order_proc_id_coded": "o2",
                "order_time_jittered": "2024-01-02T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
            },
        ]
    )
    eligible_pairs = pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2024-01-01T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "antibiotic": "CIPROFLOXACIN",
                "is_observed_tested": 1,
                "is_eligible": 1,
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e2",
                "order_proc_id_coded": "o2",
                "order_time_jittered": "2024-01-02T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "antibiotic": "CIPROFLOXACIN",
                "is_observed_tested": 0,
                "is_eligible": 1,
            },
        ]
    )
    culture_episodes.to_parquet(gold_dir / "culture_episodes.parquet", index=False)
    eligible_pairs.to_parquet(gold_dir / "eligible_pairs.parquet", index=False)

    builder._cascade_covariates.build = lambda frame: pd.DataFrame(
        [
            {
                "anon_id": "p1",
                "pat_enc_csn_id_coded": "e1",
                "order_proc_id_coded": "o1",
                "order_time_jittered": "2024-01-01T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "cov_icu_status": 1,
                "cov_prior_abx_any_90d": 1,
                "cov_prior_same_organism_any_90d": 0,
                "cov_icu_available": 1,
                "cov_prior_abx_available": 1,
                "cov_prior_organism_available": 1,
                "cov_comorbidity_available": 1,
                "cov_comorbidity_count": 4,
                "cov_age_bin": "adult",
                "cov_sex": "female",
            },
            {
                "anon_id": "p2",
                "pat_enc_csn_id_coded": "e2",
                "order_proc_id_coded": "o2",
                "order_time_jittered": "2024-01-02T00:00:00Z",
                "organism": "ESCHERICHIA COLI",
                "source_site": "armd",
                "cov_icu_status": 0,
                "cov_prior_abx_any_90d": 0,
                "cov_prior_same_organism_any_90d": 1,
                "cov_icu_available": 1,
                "cov_prior_abx_available": 1,
                "cov_prior_organism_available": 1,
                "cov_comorbidity_available": 1,
                "cov_comorbidity_count": 1,
                "cov_age_bin": "older_adult",
                "cov_sex": "male",
            },
        ]
    )

    table = builder.build_upstream_selection_balance_table("combined", None, "unit_test_selection_balance")

    assert not table.empty
    assert {
        "site",
        "organism",
        "upstream_antibiotic",
        "covariate",
        "upstream_tested_n",
        "upstream_untested_n",
        "standardised_mean_difference",
        "balance_flag",
    }.issubset(table.columns)
    icu_row = table.loc[table["covariate"] == "ICU status"].iloc[0]
    assert icu_row["upstream_tested_n"] == 1
    assert icu_row["upstream_untested_n"] == 1
    assert icu_row["upstream_tested_value"] == 1.0
    assert icu_row["upstream_untested_value"] == 0.0
    assert icu_row["balance_flag"] == "high_imbalance"
