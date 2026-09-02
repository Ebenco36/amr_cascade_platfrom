"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amr_cascade_platform.cli.commands.build_features import register_feature_build_command
from amr_cascade_platform.cli.commands.audit import register_audit_command
from amr_cascade_platform.cli.commands.cascade import register_cascade_command
from amr_cascade_platform.cli.commands.cascade_sensitivity import register_cascade_sensitivity_command
from amr_cascade_platform.cli.commands.features import register_comorbidity_command
from amr_cascade_platform.cli.commands.gold import register_gold_command
from amr_cascade_platform.cli.commands.harmonize import register_harmonization_command
from amr_cascade_platform.cli.commands.ingest import register_ingestion_command
from amr_cascade_platform.cli.commands.preprocess import register_preprocessing_command
from amr_cascade_platform.cli.commands.prevalence import register_prevalence_command
from amr_cascade_platform.cli.commands.report import register_reporting_command
from amr_cascade_platform.cli.commands.sample import register_sampling_command
from amr_cascade_platform.cli.commands.train import register_training_command
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.config.environment_resolver import EnvironmentResolver
from amr_cascade_platform.core.exceptions.custom_exceptions import AmrCascadeError
from amr_cascade_platform.core.logging.logger_factory import LoggerFactory
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.data.ingestion.canonical_table_mapper import CanonicalTableMapper


def build_context(project_root: Path, environment_name: str | None):
    loader = ConfigLoader(project_root)
    app_config = loader.load(environment_name or "mac").app
    resolved_environment = EnvironmentResolver().resolve(
        environment_name or app_config.default_environment
    )
    settings = loader.load(resolved_environment)
    path_manager = PathManager(project_root, settings)
    path_manager.ensure_standard_directories()
    canonical_mapper = CanonicalTableMapper(settings, path_manager)
    return settings, path_manager, canonical_mapper


def main() -> None:
    LoggerFactory.configure()
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(prog="amr-cascade")
    parser.add_argument("--env", default=None, help="Runtime environment name, e.g. mac or hpc.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_ingestion_command(subparsers, project_root, build_context)
    register_preprocessing_command(subparsers, project_root, build_context)
    register_harmonization_command(subparsers, project_root, build_context)
    register_gold_command(subparsers, project_root, build_context)
    register_cascade_command(subparsers, project_root, build_context)
    register_cascade_sensitivity_command(subparsers, project_root, build_context)
    register_reporting_command(subparsers, project_root, build_context)
    register_prevalence_command(subparsers, project_root, build_context)
    register_audit_command(subparsers, project_root, build_context)
    register_feature_build_command(subparsers, project_root, build_context)
    register_training_command(subparsers, project_root, build_context)
    register_sampling_command(subparsers, project_root, build_context)
    register_comorbidity_command(subparsers, project_root, build_context)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (AmrCascadeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
