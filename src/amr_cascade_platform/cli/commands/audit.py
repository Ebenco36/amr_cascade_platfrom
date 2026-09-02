"""Scientific audit command."""

from __future__ import annotations

from pathlib import Path

from amr_cascade_platform.reporting.workflows.scientific_audit_workflow import (
    ScientificAuditRequest,
    ScientificAuditWorkflow,
)


def register_audit_command(subparsers, project_root: Path, context_builder) -> None:
    parser = subparsers.add_parser(
        "run-audit",
        help="Export traceability, contract, and scientific QA audit artifacts.",
    )
    parser.add_argument(
        "--scope",
        choices=("combined", "site"),
        default="combined",
        help="Audit combined outputs or a single site scope.",
    )
    parser.add_argument("--site", default=None, help="Site name when using --scope site.")
    parser.add_argument("--organism", default=None, help="Optional organism filter for organism-stratified audit outputs.")
    parser.set_defaults(handler=lambda args: _run(args, project_root, context_builder))


def _run(args, project_root: Path, context_builder) -> None:
    settings, path_manager, _ = context_builder(project_root, args.env)
    workflow = ScientificAuditWorkflow(settings, path_manager)
    workflow.run(ScientificAuditRequest(scope=args.scope, site=args.site, organism=args.organism))
