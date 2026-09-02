"""Typed configuration models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    name: str
    default_environment: str


@dataclass(frozen=True)
class PlatformConfig:
    sites: tuple[str, ...]
    id_columns: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    time_columns: tuple[str, ...]
    reference_files: dict[str, str]
    optional_tables: tuple[str, ...]


@dataclass(frozen=True)
class OrganismConfig:
    priority_organisms: tuple[str, ...]
    min_episode_threshold: int


@dataclass(frozen=True)
class SamplingConfig:
    default_fraction: float
    random_seed: int
    use_dask_threshold_mb: float
    dask_blocksize: str
    copy_reference_files_to_output: bool


@dataclass(frozen=True)
class ComorbidityConfig:
    output_filename: str
    dask_blocksize: str
    default_top_k: int
    rarest_count: int
    coverage_report_k_values: tuple[int, ...]


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    data_root: str
    execution_mode: str
    default_data_layer: str
    allow_raw_processing: bool


@dataclass(frozen=True)
class CsvReadConfig:
    dask_blocksize: str
    use_dask_threshold_mb: float
    encoding: str


@dataclass(frozen=True)
class BronzeConfig:
    write_index: bool
    compression: str
    metadata_subdir: str


@dataclass(frozen=True)
class IngestionConfig:
    csv_read: CsvReadConfig
    bronze: BronzeConfig


@dataclass(frozen=True)
class ParquetConfig:
    compression: str
    engine: str


@dataclass(frozen=True)
class DuplicateResolutionRule:
    subset: tuple[str, ...]
    strategy: str = "drop_exact"  # "drop_exact" | "shard_coalesce"


@dataclass(frozen=True)
class DuplicateResolutionConfig:
    default_strategy: str
    per_table: dict[str, DuplicateResolutionRule]


@dataclass(frozen=True)
class CleaningConfig:
    trim_strings: bool
    uppercase_standard_columns: tuple[str, ...]
    duplicate_resolution: DuplicateResolutionConfig


@dataclass(frozen=True)
class SchemaConfig:
    default_exempt_tables: tuple[str, ...]
    required_columns: dict[str, tuple[str, ...]]
    required_non_null_columns: dict[str, tuple[str, ...]]
    allowed_time_columns: tuple[str, ...]


@dataclass(frozen=True)
class HarmonizationTableContract:
    columns: tuple[str, ...]


@dataclass(frozen=True)
class HarmonizationConfig:
    shared_tables: tuple[str, ...]
    optional_shared_tables: tuple[str, ...]
    rename_columns: dict[str, str]
    contracts: dict[str, HarmonizationTableContract]


@dataclass(frozen=True)
class EligibilityConfig:
    denominator: str
    availability_era_years: int
    availability_min_observed: int
    availability_time_column: str


@dataclass(frozen=True)
class GoldConfig:
    shared_tables: tuple[str, ...]
    episode_key_columns: tuple[str, ...]
    culture_episode_columns: tuple[str, ...]
    culture_drug_columns: tuple[str, ...]
    testing_matrix_prefix: str
    eligibility: EligibilityConfig


@dataclass(frozen=True)
class CascadeOutputConfig:
    result_dir: str
    summary_filename: str


@dataclass(frozen=True)
class CascadeValidationConfig:
    enabled: bool
    permutation_iterations: int
    bootstrap_iterations: int
    random_seed: int
    permutation_p_value_threshold: float
    bootstrap_sign_stability_threshold: float
    minimum_replicated_sites: int
    minimum_site_direction_agreement_rate: float
    i_squared_threshold: float
    between_site_permutation_enabled: bool


@dataclass(frozen=True)
class CascadeConfig:
    primary_observation_layer: str
    upstream_result_positive: str
    upstream_result_negative: str
    continuity_correction: float
    min_total_support: int
    min_result_support: int
    min_total_tested_events_for_retention: int
    sensitivity_min_total_supports: tuple[int, ...]
    sensitivity_min_result_supports: tuple[int, ...]
    sensitivity_retained_min_escalation_ratios: tuple[float, ...]
    sensitivity_continuity_corrections: tuple[float, ...]
    pathway_top_n: int
    require_downstream_eligible: bool
    filter_symmetric_cotesting: bool
    cotesting_probability_threshold: float
    retained_min_escalation_ratio: float
    adjusted_or_positive_threshold: float
    validate_suppression_cascades: bool
    include_acuity_covariates: bool
    validation: CascadeValidationConfig
    outputs: CascadeOutputConfig


@dataclass(frozen=True)
class ReportingConfig:
    tables_dir: str
    figures_dir: str
    reports_dir: str
    forest_top_n: int
    sankey_top_n: int
    default_figure_formats: tuple[str, ...]
    default_figures: tuple[str, ...]
    plotly_template: str
    image_width: int
    image_height: int


@dataclass(frozen=True)
class FeatureBuildConfig:
    combined_output_filename: str
    metadata_filename: str
    auto_build_comorbidity: bool
    auto_prepare_silver_feature_sources: bool
    comorbidity_output_filename: str
    comorbidity_feature_prefix: str
    baseline_output_filename: str
    acute_output_filename: str
    history_output_filename: str
    history_windows_days: tuple[int, ...]


@dataclass(frozen=True)
class LogisticRegressionModelConfig:
    max_iter: int
    solver: str
    class_weight: str | None


@dataclass(frozen=True)
class RandomForestModelConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None


@dataclass(frozen=True)
class XGBoostModelConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


@dataclass(frozen=True)
class NeuralNetworkModelConfig:
    hidden_layer_sizes: tuple[int, ...]
    max_iter: int
    alpha: float
    learning_rate_init: float
    early_stopping: bool
    batch_size: int


@dataclass(frozen=True)
class ModelingConfig:
    task_name: str
    target_column: str
    split_column: str
    default_split_strategy: str
    train_sites: tuple[str, ...]
    validation_sites: tuple[str, ...]
    test_sites: tuple[str, ...]
    temporal_train_end_year: int
    temporal_validation_end_year: int
    temporal_test_end_year: int
    estimators: tuple[str, ...]
    include_cascade_features: bool
    output_dir: str
    random_state: int
    logistic_regression: LogisticRegressionModelConfig
    random_forest: RandomForestModelConfig
    xgboost: XGBoostModelConfig
    neural_network: NeuralNetworkModelConfig | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    probability_threshold: float
    threshold_selection_metric: str
    threshold_candidates: tuple[float, ...]
    reporting_threshold_metrics: tuple[str, ...]


@dataclass(frozen=True)
class PrevalenceConfig:
    output_dir: str
    min_eligible_support: int
    min_tested_support: int
    min_resistant_support: int
    default_top_n: int
    delta_curve_points: int
    mnar_lambda_grid: tuple[float, ...]
    decision_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    platform: PlatformConfig
    organisms: OrganismConfig
    sampling: SamplingConfig
    comorbidity: ComorbidityConfig
    environment: EnvironmentConfig
    ingestion: IngestionConfig
    parquet: ParquetConfig
    cleaning: CleaningConfig
    schema: SchemaConfig
    harmonization: HarmonizationConfig
    gold: GoldConfig
    cascade: CascadeConfig
    reporting: ReportingConfig
    feature_build: FeatureBuildConfig
    modeling: ModelingConfig
    evaluation: EvaluationConfig
    prevalence: PrevalenceConfig
