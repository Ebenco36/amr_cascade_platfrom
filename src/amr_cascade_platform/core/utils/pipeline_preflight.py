"""Pipeline preflight checks for operator scripts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import ValidationError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir


@dataclass(frozen=True)
class PipelinePreflightRequest:
    environment: str
    sites: tuple[str, ...]
    organisms: tuple[str, ...] = ()
    estimators: tuple[str, ...] = ()
    report_formats: tuple[str, ...] = ()
    run_sampling: bool = False
    run_organism_primary: bool = True
    run_pooled_sensitivity: bool = True
    run_ingestion: bool = True
    run_preprocessing: bool = True
    run_harmonization: bool = True
    run_gold: bool = True
    run_cascade: bool = True
    run_site_cascade: bool = True
    run_sensitivity: bool = True
    run_features: bool = True
    run_training: bool = True
    run_ablation: bool = False
    run_reporting: bool = True


@dataclass
class PipelinePreflightResult:
    environment: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class PipelinePreflightChecker:
    """Validate runner arguments, dependencies, and stage prerequisites."""

    _REPORT_FORMATS = {"html", "png", "svg", "pdf"}
    _BASE_DEPENDENCIES = ("pandas", "pyarrow", "plotly", "sklearn", "dask")
    _REPORT_DEPENDENCIES = ("matplotlib", "pycirclize")

    def __init__(self, project_root: Path, environment: str) -> None:
        self._project_root = project_root
        self._settings = ConfigLoader(project_root).load(environment)
        self._paths = PathManager(project_root, self._settings)

    @property
    def settings(self) -> Settings:
        return self._settings

    def run(self, request: PipelinePreflightRequest) -> PipelinePreflightResult:
        result = PipelinePreflightResult(environment=request.environment)
        self._validate_selection_arguments(request, result)
        self._validate_dependencies(request, result)
        self._validate_reference_files(result)
        self._validate_stage_prerequisites(request, result)
        return result

    def raise_if_failed(self, result: PipelinePreflightResult) -> None:
        if result.errors:
            raise ValidationError("\n".join(result.errors))

    def _validate_selection_arguments(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        known_sites = set(self._settings.platform.sites)
        unknown_sites = [site for site in request.sites if site not in known_sites]
        if unknown_sites:
            result.errors.append(
                f"Unknown site selection: {', '.join(sorted(unknown_sites))}. Known sites: {', '.join(self._settings.platform.sites)}"
            )

        if request.run_organism_primary and not request.organisms:
            result.errors.append("Organism-primary execution requested but no organisms were provided.")

        supported_estimators = set(self._settings.modeling.estimators)
        unknown_estimators = [name for name in request.estimators if name not in supported_estimators]
        if unknown_estimators:
            result.errors.append(
                f"Unsupported estimators: {', '.join(sorted(unknown_estimators))}. Supported: {', '.join(self._settings.modeling.estimators)}"
            )

        unknown_formats = [fmt for fmt in request.report_formats if fmt not in self._REPORT_FORMATS]
        if unknown_formats:
            result.errors.append(
                f"Unsupported report formats: {', '.join(sorted(unknown_formats))}. Supported: {', '.join(sorted(self._REPORT_FORMATS))}"
            )

        if request.run_sensitivity and "armd" not in request.sites:
            result.warnings.append(
                "Sensitivity run requested but 'armd' is not in selected sites; the runner will skip implied-susceptibility sensitivity."
            )

    def _validate_dependencies(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        required = list(self._BASE_DEPENDENCIES)
        if request.run_training or request.run_ablation or "xgboost" in request.estimators:
            required.append("xgboost")
        if request.run_reporting:
            required.extend(self._REPORT_DEPENDENCIES)

        for module_name in dict.fromkeys(required):
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - import failure details vary by env
                result.errors.append(f"Missing or broken Python dependency '{module_name}': {exc}")

        if request.run_reporting and any(fmt in {"png", "svg", "pdf"} for fmt in request.report_formats):
            try:
                importlib.import_module("kaleido")
            except Exception:
                result.warnings.append(
                    "kaleido is unavailable; static Plotly export will rely on the project's fallback rendering path."
                )

    def _validate_reference_files(self, result: PipelinePreflightResult) -> None:
        for reference_name, filename in self._settings.platform.reference_files.items():
            path = self._paths.paths.reference / filename
            if not path.exists():
                result.errors.append(f"Required reference file missing: {path} ({reference_name})")

    def _validate_stage_prerequisites(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        if request.environment == "hpc":
            if self._settings.feature_build.auto_prepare_silver_feature_sources:
                result.warnings.append(
                    "HPC feature build is configured to auto-prepare missing silver feature sources; this can mask upstream incompleteness on full runs."
                )
            if self._settings.feature_build.auto_build_comorbidity:
                result.warnings.append(
                    "HPC feature build is configured to auto-build missing comorbidity features; consider materializing them explicitly before full production runs."
                )

        if request.run_features and not self._settings.feature_build.auto_build_comorbidity:
            comorbidity_filename = self._settings.feature_build.comorbidity_output_filename
            for site in request.sites:
                comorb_path = (
                    self._paths.site_dir("interim", site)
                    / "feature_matrices"
                    / comorbidity_filename
                )
                if not comorb_path.exists():
                    result.warnings.append(
                        f"Comorbidity feature matrix not yet present for site '{site}': "
                        f"{comorb_path}. The pipeline script will pre-aggregate it before the feature build step; "
                        f"if running manually, execute scripts/pre_aggregate_comorbidities.py --env {request.environment} "
                        f"--site {site} --min-coverage 0.8 first."
                    )

        if request.run_sampling:
            for site in request.sites:
                self._require_dir(
                    self._paths.paths.raw / site,
                    result,
                    f"Raw input directory for sampling is missing: {self._paths.paths.raw / site}",
                )

        if request.run_ingestion:
            for site in request.sites:
                input_root = self._paths.paths.raw if request.environment == "hpc" else self._paths.paths.sample
                if request.environment != "hpc" and request.run_sampling:
                    # In sample-first local runs, sampling is executed after preflight and
                    # before ingestion. Requiring the sample directory here prevents a
                    # clean run from recreating deleted sample artefacts.
                    continue
                self._require_dir(
                    input_root / site,
                    result,
                    f"Ingestion input directory is missing for site '{site}': {input_root / site}",
                )
        elif request.run_preprocessing:
            for site in request.sites:
                self._require_site_stage_dir("bronze", site, result)

        if not request.run_preprocessing and request.run_harmonization:
            for site in request.sites:
                self._require_file(
                    self._paths.paths.silver / site / "cohort.parquet",
                    result,
                    f"Silver cohort dataset is required before harmonization for site '{site}'.",
                )

        if not request.run_harmonization and (request.run_gold or request.run_cascade or request.run_features or request.run_reporting):
            if request.run_site_cascade:
                for site in request.sites:
                    self._require_file(
                        self._paths.paths.harmonized / "site_aligned" / site / "cohort.parquet",
                        result,
                        f"Harmonized site cohort is required for site '{site}'.",
                    )
            self._require_file(
                self._paths.paths.harmonized / "combined" / "cohort.parquet",
                result,
                "Harmonized combined cohort is required.",
            )

        if not request.run_gold and request.run_cascade:
            self._validate_gold_outputs_exist(request, result)

        if not request.run_cascade and request.run_features:
            self._validate_gold_outputs_exist(request, result)
            self._validate_training_cascade_artifacts_exist(request, result)

        if not request.run_features and (request.run_training or request.run_ablation):
            self._validate_feature_outputs_exist(request, result)
            if not request.run_cascade:
                self._validate_training_cascade_artifacts_exist(request, result)

        if request.run_reporting and not request.run_cascade:
            self._validate_reporting_artifacts_exist(request, result)

    def _validate_gold_outputs_exist(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        scopes = self._iter_requested_scopes(request)
        for scope, site, organism in scopes:
            root = scoped_output_dir(self._paths.paths.gold, scope, site=site, organism=organism)
            self._require_file(root / "culture_episodes.parquet", result, f"Gold culture episodes missing: {root / 'culture_episodes.parquet'}")
            self._require_file(root / "drug_pair_episodes.parquet", result, f"Gold pair dataset missing: {root / 'drug_pair_episodes.parquet'}")

    def _validate_training_cascade_artifacts_exist(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        train_site = self._settings.modeling.train_sites[0]
        organisms = request.organisms if request.run_organism_primary else (None,)
        for organism in organisms:
            root = scoped_output_dir(
                self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
                "site",
                site=train_site,
                organism=organism,
            )
            self._require_file(root / "edge_report.parquet", result, f"Training-site cascade edge report missing: {root / 'edge_report.parquet'}")
            self._require_file(root / "network_nodes.parquet", result, f"Training-site cascade network nodes missing: {root / 'network_nodes.parquet'}")

    def _validate_feature_outputs_exist(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        organisms = request.organisms if request.run_organism_primary else (None,)
        if request.run_pooled_sensitivity:
            organisms = tuple({*organisms, None})
        for organism in organisms:
            root = scoped_output_dir(self._paths.paths.features, "combined", organism=organism)
            self._require_file(
                root / self._settings.feature_build.combined_output_filename,
                result,
                f"Model-ready feature matrix missing: {root / self._settings.feature_build.combined_output_filename}",
            )

    def _validate_reporting_artifacts_exist(
        self,
        request: PipelinePreflightRequest,
        result: PipelinePreflightResult,
    ) -> None:
        scopes = self._iter_requested_scopes(request)
        for scope, site, organism in scopes:
            root = scoped_output_dir(
                self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
                scope,
                site=site,
                organism=organism,
            )
            for filename in (
                "edge_report.parquet",
                "escalation_results.parquet",
                "threshold_sensitivity.parquet",
                "pathway_flows.parquet",
                "retained_edges.parquet",
                "network_nodes.parquet",
                "network_edges.parquet",
            ):
                self._require_file(root / filename, result, f"Reporting prerequisite missing: {root / filename}")

    def _iter_requested_scopes(
        self,
        request: PipelinePreflightRequest,
    ) -> list[tuple[str, str | None, str | None]]:
        scopes: list[tuple[str, str | None, str | None]] = []
        if request.run_organism_primary:
            for organism in request.organisms:
                if request.run_site_cascade:
                    for site in request.sites:
                        scopes.append(("site", site, organism))
                scopes.append(("combined", None, organism))
        if request.run_pooled_sensitivity:
            if request.run_site_cascade:
                for site in request.sites:
                    scopes.append(("site", site, None))
            scopes.append(("combined", None, None))
        # de-duplicate while preserving order
        seen: set[tuple[str, str | None, str | None]] = set()
        ordered: list[tuple[str, str | None, str | None]] = []
        for item in scopes:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _require_site_stage_dir(self, stage: str, site: str, result: PipelinePreflightResult) -> None:
        path = getattr(self._paths.paths, stage) / site
        self._require_dir(path, result, f"Required {stage} directory missing for site '{site}': {path}")

    @staticmethod
    def _require_dir(path: Path, result: PipelinePreflightResult, message: str) -> None:
        if not path.exists() or not path.is_dir():
            result.errors.append(message)

    @staticmethod
    def _require_file(path: Path, result: PipelinePreflightResult, message: str) -> None:
        if not path.exists():
            result.errors.append(message)
