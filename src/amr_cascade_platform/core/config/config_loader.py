"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from amr_cascade_platform.core.config.config_models import (
    AppConfig,
    BronzeConfig,
    CascadeConfig,
    CascadeOutputConfig,
    CascadeValidationConfig,
    CleaningConfig,
    CsvReadConfig,
    ComorbidityConfig,
    DuplicateResolutionConfig,
    DuplicateResolutionRule,
    EnvironmentConfig,
    FeatureBuildConfig,
    GoldConfig,
    HarmonizationConfig,
    HarmonizationTableContract,
    IngestionConfig,
    LogisticRegressionModelConfig,
    ModelingConfig,
    ParquetConfig,
    PlatformConfig,
    RandomForestModelConfig,
    XGBoostModelConfig,
    NeuralNetworkModelConfig,
    OrganismConfig,
    ReportingConfig,
    SamplingConfig,
    SchemaConfig,
    EvaluationConfig,
    EligibilityConfig,
    PrevalenceConfig,
    Settings,
)
from amr_cascade_platform.core.exceptions.custom_exceptions import ConfigurationError
from amr_cascade_platform.core.utils.dicts import deep_merge


class ConfigLoader:
    """Load base and environment-specific YAML settings."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._config_root = project_root / "configs"

    def load(self, environment_name: str) -> Settings:
        base_dir = self._config_root / "base"
        env_path = self._config_root / "environments" / f"{environment_name}.yaml"
        if not base_dir.exists():
            raise ConfigurationError(f"Base config directory not found: {base_dir}")
        if not env_path.exists():
            raise ConfigurationError(f"Environment config not found: {env_path}")

        merged: dict = {}
        for path in sorted(base_dir.glob("*.yaml")):
            merged = deep_merge(merged, self._read_yaml(path))
        merged = deep_merge(merged, self._read_yaml(env_path))
        return self._to_settings(merged)

    def _read_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"Expected YAML mapping in {path}")
        return data

    def _to_settings(self, raw: dict) -> Settings:
        try:
            return Settings(
                app=AppConfig(**raw["app"]),
                platform=PlatformConfig(
                    sites=tuple(raw["platform"]["sites"]),
                    id_columns=tuple(raw["platform"]["id_columns"]),
                    missing_tokens=tuple(raw["platform"]["missing_tokens"]),
                    time_columns=tuple(raw["platform"]["time_columns"]),
                    reference_files=dict(raw["platform"]["reference_files"]),
                    optional_tables=tuple(raw["platform"]["optional_tables"]),
                ),
                organisms=OrganismConfig(
                    priority_organisms=tuple(raw["organisms"]["priority_organisms"]),
                    min_episode_threshold=raw["organisms"]["min_episode_threshold"],
                ),
                sampling=SamplingConfig(**raw["preprocessing"]["sampling"]),
                comorbidity=ComorbidityConfig(
                    output_filename=raw["preprocessing"]["comorbidity"]["output_filename"],
                    dask_blocksize=raw["preprocessing"]["comorbidity"]["dask_blocksize"],
                    default_top_k=raw["preprocessing"]["comorbidity"]["default_top_k"],
                    rarest_count=raw["preprocessing"]["comorbidity"]["rarest_count"],
                    coverage_report_k_values=tuple(
                        raw["preprocessing"]["comorbidity"]["coverage_report_k_values"]
                    ),
                ),
                environment=EnvironmentConfig(**raw["environment"]),
                ingestion=IngestionConfig(
                    csv_read=CsvReadConfig(**raw["ingestion"]["csv_read"]),
                    bronze=BronzeConfig(**raw["ingestion"]["bronze"]),
                ),
                parquet=ParquetConfig(**raw["parquet"]),
                cleaning=CleaningConfig(
                    trim_strings=raw["cleaning"]["trim_strings"],
                    uppercase_standard_columns=tuple(raw["cleaning"]["uppercase_standard_columns"]),
                    duplicate_resolution=DuplicateResolutionConfig(
                        default_strategy=raw["cleaning"]["duplicate_resolution"]["default_strategy"],
                        per_table={
                            name: DuplicateResolutionRule(
                                subset=tuple(rule["subset"]),
                                strategy=rule.get("strategy", "drop_exact"),
                            )
                            for name, rule in raw["cleaning"]["duplicate_resolution"]["per_table"].items()
                        },
                    ),
                ),
                schema=SchemaConfig(
                    default_exempt_tables=tuple(raw["schema"]["default_exempt_tables"]),
                    required_columns={
                        name: tuple(columns)
                        for name, columns in raw["schema"]["required_columns"].items()
                    },
                    required_non_null_columns={
                        name: tuple(columns)
                        for name, columns in raw["schema"]["required_non_null_columns"].items()
                    },
                    allowed_time_columns=tuple(raw["schema"]["allowed_time_columns"]),
                ),
                harmonization=HarmonizationConfig(
                    shared_tables=tuple(raw["harmonization"]["shared_tables"]),
                    optional_shared_tables=tuple(raw["harmonization"]["optional_shared_tables"]),
                    rename_columns=dict(raw["harmonization"]["rename_columns"]),
                    contracts={
                        name: HarmonizationTableContract(columns=tuple(contract["columns"]))
                        for name, contract in raw["harmonization"]["contracts"].items()
                    },
                ),
                gold=GoldConfig(
                    shared_tables=tuple(raw["gold"]["shared_tables"]),
                    episode_key_columns=tuple(raw["gold"]["episode_key_columns"]),
                    culture_episode_columns=tuple(raw["gold"]["culture_episode_columns"]),
                    culture_drug_columns=tuple(raw["gold"]["culture_drug_columns"]),
                    testing_matrix_prefix=raw["gold"]["testing_matrix_prefix"],
                    eligibility=EligibilityConfig(**raw["gold"]["eligibility"]),
                ),
                cascade=CascadeConfig(
                    primary_observation_layer=raw["cascade"]["primary_observation_layer"],
                    upstream_result_positive=raw["cascade"]["upstream_result_positive"],
                    upstream_result_negative=raw["cascade"]["upstream_result_negative"],
                    continuity_correction=raw["cascade"]["continuity_correction"],
                    min_total_support=raw["cascade"]["min_total_support"],
                    min_result_support=raw["cascade"]["min_result_support"],
                    min_total_tested_events_for_retention=raw["cascade"]["min_total_tested_events_for_retention"],
                    sensitivity_min_total_supports=tuple(raw["cascade"]["sensitivity_min_total_supports"]),
                    sensitivity_min_result_supports=tuple(raw["cascade"]["sensitivity_min_result_supports"]),
                    sensitivity_retained_min_escalation_ratios=tuple(raw["cascade"]["sensitivity_retained_min_escalation_ratios"]),
                    sensitivity_continuity_corrections=tuple(raw["cascade"]["sensitivity_continuity_corrections"]),
                    validate_suppression_cascades=raw["cascade"]["validate_suppression_cascades"],
                    include_acuity_covariates=raw["cascade"].get("include_acuity_covariates", False),
                    pathway_top_n=raw["cascade"]["pathway_top_n"],
                    require_downstream_eligible=raw["cascade"]["require_downstream_eligible"],
                    filter_symmetric_cotesting=raw["cascade"]["filter_symmetric_cotesting"],
                    cotesting_probability_threshold=raw["cascade"]["cotesting_probability_threshold"],
                    retained_min_escalation_ratio=raw["cascade"]["retained_min_escalation_ratio"],
                    adjusted_or_positive_threshold=raw["cascade"].get(
                        "adjusted_or_positive_threshold",
                        raw["cascade"].get("retained_min_adjusted_odds_ratio", 1.0),
                    ),
                    validation=CascadeValidationConfig(**raw["cascade"]["validation"]),
                    outputs=CascadeOutputConfig(**raw["cascade"]["outputs"]),
                ),
                reporting=ReportingConfig(
                    tables_dir=raw["reporting"]["tables_dir"],
                    figures_dir=raw["reporting"]["figures_dir"],
                    reports_dir=raw["reporting"]["reports_dir"],
                    forest_top_n=raw["reporting"]["forest_top_n"],
                    sankey_top_n=raw["reporting"]["sankey_top_n"],
                    default_figure_formats=tuple(raw["reporting"]["default_figure_formats"]),
                    default_figures=tuple(raw["reporting"]["default_figures"]),
                    plotly_template=raw["reporting"]["plotly_template"],
                    image_width=raw["reporting"]["image_width"],
                    image_height=raw["reporting"]["image_height"],
                ),
                feature_build=FeatureBuildConfig(
                    combined_output_filename=raw["features"]["combined_output_filename"],
                    metadata_filename=raw["features"]["metadata_filename"],
                    auto_build_comorbidity=raw["features"]["auto_build_comorbidity"],
                    auto_prepare_silver_feature_sources=raw["features"]["auto_prepare_silver_feature_sources"],
                    comorbidity_output_filename=raw["features"]["comorbidity_output_filename"],
                    comorbidity_feature_prefix=raw["features"]["comorbidity_feature_prefix"],
                    baseline_output_filename=raw["features"]["baseline_output_filename"],
                    acute_output_filename=raw["features"]["acute_output_filename"],
                    history_output_filename=raw["features"]["history_output_filename"],
                    history_windows_days=tuple(raw["features"]["history_windows_days"]),
                ),
                modeling=ModelingConfig(
                    task_name=raw["modeling"]["task_name"],
                    target_column=raw["modeling"]["target_column"],
                    split_column=raw["modeling"]["split_column"],
                    default_split_strategy=raw["modeling"]["default_split_strategy"],
                    train_sites=tuple(raw["modeling"]["train_sites"]),
                    validation_sites=tuple(raw["modeling"]["validation_sites"]),
                    test_sites=tuple(raw["modeling"]["test_sites"]),
                    temporal_train_end_year=raw["modeling"]["temporal_train_end_year"],
                    temporal_validation_end_year=raw["modeling"]["temporal_validation_end_year"],
                    temporal_test_end_year=raw["modeling"]["temporal_test_end_year"],
                    estimators=tuple(raw["modeling"]["estimators"]),
                    include_cascade_features=raw["modeling"]["include_cascade_features"],
                    output_dir=raw["modeling"]["output_dir"],
                    random_state=raw["modeling"]["random_state"],
                    logistic_regression=LogisticRegressionModelConfig(
                        **raw["modeling"]["logistic_regression"]
                    ),
                    random_forest=RandomForestModelConfig(
                        **raw["modeling"]["random_forest"]
                    ),
                    xgboost=XGBoostModelConfig(**raw["modeling"]["xgboost"]),
                    neural_network=(
                        NeuralNetworkModelConfig(
                            hidden_layer_sizes=tuple(
                                raw["modeling"]["neural_network"]["hidden_layer_sizes"]
                            ),
                            max_iter=raw["modeling"]["neural_network"]["max_iter"],
                            alpha=raw["modeling"]["neural_network"]["alpha"],
                            learning_rate_init=raw["modeling"]["neural_network"][
                                "learning_rate_init"
                            ],
                            early_stopping=raw["modeling"]["neural_network"][
                                "early_stopping"
                            ],
                            batch_size=raw["modeling"]["neural_network"]["batch_size"],
                        )
                        if "neural_network" in raw.get("modeling", {})
                        else None
                    ),
                ),
                evaluation=EvaluationConfig(
                    probability_threshold=raw["evaluation"]["probability_threshold"],
                    threshold_selection_metric=raw["evaluation"]["threshold_selection_metric"],
                    threshold_candidates=tuple(raw["evaluation"]["threshold_candidates"]),
                    reporting_threshold_metrics=tuple(raw["evaluation"]["reporting_threshold_metrics"]),
                ),
                prevalence=PrevalenceConfig(
                    output_dir=raw["prevalence"]["output_dir"],
                    min_eligible_support=raw["prevalence"]["min_eligible_support"],
                    min_tested_support=raw["prevalence"]["min_tested_support"],
                    min_resistant_support=raw["prevalence"]["min_resistant_support"],
                    default_top_n=raw["prevalence"]["default_top_n"],
                    delta_curve_points=raw["prevalence"]["delta_curve_points"],
                    mnar_lambda_grid=tuple(raw["prevalence"].get("mnar_lambda_grid", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])),
                    decision_thresholds=tuple(raw["prevalence"].get("decision_thresholds", [0.10, 0.20, 0.30])),
                ),
            )
        except KeyError as exc:
            raise ConfigurationError(f"Missing configuration section: {exc}") from exc
