"""Run specification traceability and scientific QA audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.reporting.builders.scientific_audit_builder import ScientificAuditBuilder


@dataclass(frozen=True)
class ScientificAuditRequest:
    scope: str = "combined"
    site: str | None = None
    organism: str | None = None


class ScientificAuditWorkflow:
    """Materialize traceability, contract, and QA audit artifacts."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._builder = ScientificAuditBuilder(settings, path_manager)

    def run(self, request: ScientificAuditRequest) -> dict[str, Path]:
        if request.scope == "site" and request.site:
            scope_dir = self._paths.project_root / self._settings.reporting.reports_dir / request.site
        else:
            scope_dir = self._paths.project_root / self._settings.reporting.reports_dir / "combined"
        if request.organism:
            from amr_cascade_platform.core.utils.scopes import organism_slug

            scope_dir = scope_dir / "organisms" / str(organism_slug(request.organism))
        reports_dir = scope_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        traceability = self._builder.build_traceability_table()
        contracts = self._builder.build_contract_check_table(request.scope, request.site, request.organism)
        scientific_qa = self._builder.build_scientific_qa_payload(request.scope, request.site, request.organism)

        outputs = {
            "cascade_traceability_audit.csv": reports_dir / "cascade_traceability_audit.csv",
            "cascade_traceability_audit.md": reports_dir / "cascade_traceability_audit.md",
            "pipeline_contract_audit.csv": reports_dir / "pipeline_contract_audit.csv",
            "pipeline_contract_audit.md": reports_dir / "pipeline_contract_audit.md",
            "scientific_qa_summary.json": reports_dir / "scientific_qa_summary.json",
            "scientific_qa_summary.md": reports_dir / "scientific_qa_summary.md",
        }

        traceability.to_csv(outputs["cascade_traceability_audit.csv"], index=False)
        outputs["cascade_traceability_audit.md"].write_text(
            self._builder.render_traceability_markdown(traceability),
            encoding="utf-8",
        )
        contracts.to_csv(outputs["pipeline_contract_audit.csv"], index=False)
        outputs["pipeline_contract_audit.md"].write_text(
            self._builder.render_contract_markdown(contracts),
            encoding="utf-8",
        )
        outputs["scientific_qa_summary.json"].write_text(
            json.dumps(scientific_qa, indent=2),
            encoding="utf-8",
        )
        outputs["scientific_qa_summary.md"].write_text(
            self._builder.render_scientific_qa_markdown(scientific_qa),
            encoding="utf-8",
        )
        return outputs
