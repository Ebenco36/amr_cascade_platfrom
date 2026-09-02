"""Traceability, contract, and scientific QA reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


class ScientificAuditBuilder:
    """Build auditable specification, contract, and result summaries."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)

    def build_traceability_table(self) -> pd.DataFrame:
        rows = [
            {
                "requirement_id": "1.1",
                "section": "Observed data are selected data",
                "requirement": "Model AST as an observation process rather than a neutral label table.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/data/transformations/observation_space_builder.py",
                        "src/amr_cascade_platform/data/gold/gold_build_manager.py",
                        "src/amr_cascade_platform/cascade/workflows/cascade_analysis_workflow.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Gold and cascade stages operate on observed testing availability and eligible denominators.",
            },
            {
                "requirement_id": "4.1",
                "section": "Bronze ingestion",
                "requirement": "Convert site-specific raw CSV tables into stable Bronze parquet with source_site and canonical mapping.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/data/ingestion/raw_ingestion_manager.py",
                        "src/amr_cascade_platform/data/ingestion/table_ingestor.py",
                        "src/amr_cascade_platform/data/ingestion/canonical_table_mapper.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Local verification runs use sampled inputs; production configuration still needs full-environment validation.",
            },
            {
                "requirement_id": "4.2",
                "section": "Silver cleaning",
                "requirement": "Normalize nulls, keys, duplicates, and schema before downstream use.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/data/cleaning/table_cleaner.py",
                        "src/amr_cascade_platform/data/cleaning/null_handler.py",
                        "src/amr_cascade_platform/data/validators/schema_validator.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Reference-style implied susceptibility rule tables are exempted from episode-key validation, and same-event discordant susceptibility labels in microbial_resistance are audited explicitly before de-duplication.",
            },
            {
                "requirement_id": "4.3",
                "section": "Cross-site harmonization",
                "requirement": "Align schemas semantically across sites while preserving source_site and site-level availability differences.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/data/harmonization/site_harmonizer.py",
                        "src/amr_cascade_platform/data/harmonization/combined_dataset_builder.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Optional domains such as vitals remain site-aware rather than silently row-missing.",
            },
            {
                "requirement_id": "4.4",
                "section": "Gold analytical datasets",
                "requirement": "Materialize culture episodes, culture-drug, drug-pair, testing matrix, and eligibility datasets as explicit contracts.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/data/transformations/culture_episode_builder.py",
                        "src/amr_cascade_platform/data/transformations/testing_matrix_builder.py",
                        "src/amr_cascade_platform/data/transformations/pair_generator.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Gold outputs are explicit parquet products and used as contracts by modeling and reporting layers.",
            },
            {
                "requirement_id": "8.1",
                "section": "Intrinsic resistance and structural omission",
                "requirement": "Use intrinsic resistance to separate structural omission from selective non-testing.",
                "implementation_paths": "src/amr_cascade_platform/domain/services/eligibility_service.py",
                "status": "implemented",
                "notes": "Eligibility joins normalized organism-antibiotic pairs to the intrinsic resistance reference table.",
            },
            {
                "requirement_id": "9.1",
                "section": "Conditional testing",
                "requirement": "Estimate downstream testing probabilities conditional on observed upstream resistance state.",
                "implementation_paths": "src/amr_cascade_platform/cascade/analyzers/conditional_probability_analyzer.py",
                "status": "implemented",
                "notes": "Conditional probabilities are built from eligible directed pair episodes only.",
            },
            {
                "requirement_id": "9.2",
                "section": "Escalation ratio",
                "requirement": "Compute ER with support thresholds and interpret ER > 1 as stronger downstream testing after upstream resistance.",
                "implementation_paths": "src/amr_cascade_platform/cascade/analyzers/escalation_ratio_analyzer.py",
                "status": "implemented",
                "notes": "Reporting now treats downstream trigger forests as trigger-only when ER > 1.",
            },
            {
                "requirement_id": "9.3",
                "section": "Adjusted logistic regression",
                "requirement": "Model downstream testing as an adjusted robustness analysis, not a causal claim.",
                "implementation_paths": "src/amr_cascade_platform/cascade/statistics/downstream_testing_regression.py",
                "status": "implemented",
                "notes": "Adjusted ORs are exported and clearly framed as selection-conditioned descriptive robustness estimates rather than causal effects.",
            },
            {
                "requirement_id": "9.3a",
                "section": "Bootstrap stability",
                "requirement": "Preserve within-episode dependence in cascade stability resampling.",
                "implementation_paths": "src/amr_cascade_platform/cascade/analyzers/cascade_validation_analyzer.py",
                "status": "implemented",
                "notes": "Within-site bootstrap resampling now samples culture episodes and carries all associated pair rows together.",
            },
            {
                "requirement_id": "9.3b",
                "section": "Selection diagnostics",
                "requirement": "Characterize the upstream-observed subset rather than assuming it is representative of all eligible episodes.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/reporting/builders/manuscript_table_builder.py",
                        "src/amr_cascade_platform/reporting/workflows/report_export_workflow.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Reporting exports upstream-tested versus upstream-untested balance diagnostics on the main episode-level covariates used in the adjusted cascade models.",
            },
            {
                "requirement_id": "9.3c",
                "section": "Multiplicity-aware validation diagnostics",
                "requirement": "Report false-discovery-rate-adjusted permutation diagnostics across retained edges without collapsing the multi-criterion validation framework into a single p-value rule.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/cascade/analyzers/cascade_validation_analyzer.py",
                        "src/amr_cascade_platform/cascade/outputs/cascade_report_builder.py",
                        "src/amr_cascade_platform/reporting/builders/manuscript_table_builder.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Validation outputs now include Benjamini-Hochberg-adjusted permutation q-values and an accompanying support flag as multiplicity-aware supplementary diagnostics.",
            },
            {
                "requirement_id": "9.3d",
                "section": "Covariate dependence diagnostics",
                "requirement": "Expose pairwise dependence among the model-ready episode-level covariates so collinearity risks are inspectable rather than implicit.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/reporting/builders/covariate_correlation_builder.py",
                        "src/amr_cascade_platform/reporting/builders/manuscript_table_builder.py",
                        "src/amr_cascade_platform/reporting/workflows/report_export_workflow.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Reporting exports pairwise correlation diagnostics for numeric and one-hot-encoded categorical design columns, excluding trivial within-factor dummy contrasts and flagging high absolute correlations for review.",
            },
            {
                "requirement_id": "9.4",
                "section": "Cascade network",
                "requirement": "Construct directed antibiotic network from retained empirical cascade edges.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/cascade/outputs/network_exporter.py",
                        "src/amr_cascade_platform/visualization/report/plotly_cascade_plotters.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Network exports and figures use retained directed edges and node-level centrality summaries.",
            },
            {
                "requirement_id": "9.5",
                "section": "Rule concordance",
                "requirement": "Compare empirical edges with intrinsic/rule-informed expectations without claiming intent.",
                "implementation_paths": "src/amr_cascade_platform/cascade/analyzers/guideline_concordance_analyzer.py",
                "status": "implemented",
                "notes": "Concordance outputs are descriptive only.",
            },
            {
                "requirement_id": "9.6",
                "section": "Implied susceptibility boundary",
                "requirement": "Keep implied susceptibility out of primary cross-site cascade analyses and use it only as armd sensitivity layer.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/cascade/workflows/implied_sensitivity_workflow.py",
                        "src/amr_cascade_platform/cli/commands/cascade_sensitivity.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Primary outputs are observed-AST only; sensitivity outputs are written to a separate artifact namespace.",
            },
            {
                "requirement_id": "10.1",
                "section": "Prevalence-shift robustness",
                "requirement": "Estimate denominator-aware prevalence shift using model-free bounds, cascade-trigger diagnostics, observed enrichment summaries, and MNAR odds-tilt sensitivity curves.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/surveillance/prevalence_shift_analyzer.py",
                        "src/amr_cascade_platform/surveillance/workflows/prevalence_shift_workflow.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Primary prevalence outputs now report naive tested-row prevalence, eligible-denominator bounds, MNAR sensitivity summaries, cascade-trigger concentration, and observed enrichment diagnostics; IPW is not used for prevalence recovery.",
            },
            {
                "requirement_id": "12.1-12.6",
                "section": "Feature layer",
                "requirement": "Build baseline, acute, history, microbiology, and cascade-derived features without leakage.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/features/workflows/feature_build_workflow.py",
                        "src/amr_cascade_platform/features/builders/feature_merger.py",
                        "src/amr_cascade_platform/modeling/datasets/pair_feature_matrix_builder.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Model-ready features preserve row counts and keep training-site cascade features isolated from held-out sites.",
            },
            {
                "requirement_id": "13-14",
                "section": "Predictive modeling and split strategy",
                "requirement": "Evaluate learnability and cascade-aware improvement on fixed site-based splits with leakage guards.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/modeling/workflows/modeling_workflow.py",
                        "src/amr_cascade_platform/modeling/datasets/split_strategy.py",
                        "src/amr_cascade_platform/modeling/datasets/feature_set_selector.py",
                    ]
                ),
                "status": "implemented",
                "notes": "Ablation now explicitly compares observed-only, cascade-aware, clinical-no-cascade, and full feature sets.",
            },
            {
                "requirement_id": "16-17",
                "section": "Outputs and visualization",
                "requirement": "Export inspectable tables and publication-facing figures with honest interpretation boundaries.",
                "implementation_paths": "; ".join(
                    [
                        "src/amr_cascade_platform/reporting/workflows/report_export_workflow.py",
                        "src/amr_cascade_platform/visualization/report/figure_manager.py",
                    ]
                ),
                "status": "implemented_with_sampling_limit",
                "notes": "Local report outputs are valid for sampled runs, but sparsity remains data-dependent until full production execution.",
            },
        ]
        return pd.DataFrame.from_records(rows)

    def build_contract_check_table(self, scope: str, site: str | None, organism: str | None = None) -> pd.DataFrame:
        checks: list[dict[str, object]] = []
        gold_root = scoped_output_dir(self._paths.paths.gold, scope, site=site, organism=organism)
        culture = self._safe_read(gold_root / "culture_episodes.parquet")
        drug = self._safe_read(gold_root / "culture_drug_episodes.parquet")
        pair = self._safe_read(gold_root / "drug_pair_episodes.parquet")
        eligible = self._safe_read(gold_root / "eligible_pairs.parquet")
        cascade_root = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope,
            site=site,
            organism=organism,
        )
        retained = self._safe_read(cascade_root / "retained_edges.parquet")
        edge_report = self._safe_read(cascade_root / "edge_report.parquet")
        threshold = self._safe_read(cascade_root / "threshold_sensitivity.parquet")
        trigger_availability = self._safe_read(
            scoped_output_dir(
                self._paths.project_root / self._settings.reporting.tables_dir,
                scope,
                site=site,
                organism=organism,
            )
            / "table_j_downstream_trigger_availability.csv"
        )
        modeling_root = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.modeling.output_dir / self._settings.modeling.task_name,
            scope,
            site=site,
            organism=organism,
        )
        metrics = self._safe_read(modeling_root / "metrics.parquet")

        episode_keys = list(self._settings.gold.episode_key_columns)
        checks.append(
            self._check(
                "gold_culture_episode_uniqueness",
                bool(not culture.empty and not culture.duplicated(subset=episode_keys).any()),
                "Culture episodes are unique on the configured episode key.",
                {"rows": len(culture)},
            )
        )
        checks.append(
            self._check(
                "gold_drug_episode_observed_layer",
                bool(not drug.empty and set(drug["observation_layer"].dropna().unique()) <= {"observed_ast"}),
                "Culture-drug episodes use the observed AST layer in primary outputs.",
                {"rows": len(drug)},
            )
        )
        checks.append(
            self._check(
                "gold_pair_direction_consistency",
                bool(
                    not pair.empty
                    and pair["pair_direction"].fillna("").str.contains("->", regex=False).all()
                    and pair["downstream_eligible"].isin([0, 1]).all()
                ),
                "Directed pair episodes encode explicit direction and binary downstream eligibility.",
                {"rows": len(pair)},
            )
        )
        checks.append(
            self._check(
                "eligibility_structural_omission_rule",
                bool(not eligible.empty and not ((eligible["is_intrinsic_resistance"] == 1) & (eligible["is_eligible"] == 1)).any()),
                "Intrinsic resistance never remains marked as eligible.",
                {"rows": len(eligible)},
            )
        )
        checks.append(
            self._check(
                "retained_edges_are_directional_candidates",
                bool(
                    not retained.empty
                    and retained["is_retained_edge"].astype(bool).all()
                    and retained["passes_support_threshold"].astype(bool).all()
                    and retained["escalation_ratio"].notna().all()
                ),
                "Retained cascade edges satisfy support thresholds and have a finite directional ER.",
                {"rows": len(retained)},
            )
        )
        checks.append(
            self._check(
                "edge_report_matches_retained_count",
                bool(len(edge_report) == len(retained)),
                "Edge report row count matches retained edge row count.",
                {"edge_report_rows": len(edge_report), "retained_rows": len(retained)},
            )
        )
        if not trigger_availability.empty and "has_validated_trigger_forest" in trigger_availability.columns:
            estimable_downstreams = int(trigger_availability["has_validated_trigger_forest"].sum())
        elif not trigger_availability.empty and "has_estimable_trigger_forest" in trigger_availability.columns:
            estimable_downstreams = int(trigger_availability["has_estimable_trigger_forest"].sum())
        else:
            estimable_downstreams = 0
        screened_downstreams = int(len(trigger_availability)) if not trigger_availability.empty else 0
        checks.append(
            self._check(
                "downstream_trigger_availability_table_present",
                bool(not trigger_availability.empty and "has_estimable_trigger_forest" in trigger_availability.columns),
                "Report exports a downstream trigger availability table for screened downstream antibiotics.",
                {
                    "screened_downstreams": screened_downstreams,
                    "estimable_downstreams": estimable_downstreams,
                },
            )
        )
        checks.append(
            self._check(
                "threshold_sensitivity_nonempty",
                bool(not threshold.empty and {"retained_edges_n", "support_passing_edges_n"}.issubset(threshold.columns)),
                "Threshold sensitivity output is present and structurally complete.",
                {"rows": len(threshold)},
            )
        )
        expected_splits = {"train", "validation", "test"}
        observed_feature_sets = set(metrics["feature_set"].unique()) if not metrics.empty and "feature_set" in metrics.columns else set()
        observed_models = set(metrics["model_name"].unique()) if not metrics.empty else set()
        observed_splits = set(metrics["split"].unique()) if not metrics.empty else set()
        checks.append(
            self._check(
                "model_metrics_cover_evaluation_splits",
                bool(
                    observed_models
                    and expected_splits.issubset(observed_splits)
                ),
                "Model metrics include at least one estimator and all evaluation splits.",
                {
                    "feature_sets": sorted(observed_feature_sets),
                    "models": sorted(observed_models),
                    "splits": sorted(observed_splits),
                },
            )
        )
        checks.append(
            self._check(
                "model_feature_dataset_materialized",
                bool((modeling_root / "feature_dataset.parquet").exists()),
                "The model exports an inspectable feature dataset artifact.",
                {
                    "feature_dataset": str(modeling_root / "feature_dataset.parquet"),
                },
            )
        )
        return pd.DataFrame.from_records(checks)

    def build_scientific_qa_payload(self, scope: str, site: str | None, organism: str | None = None) -> dict[str, object]:
        scope_dir = site if scope == "site" and site else "combined"
        cascade_root = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope,
            site=site,
            organism=organism,
        )
        cascade_summary = self._read_json(cascade_root / "cascade_summary.json")
        report_summary = self._read_json(cascade_root / "report_summary.json")
        threshold = self._safe_read(cascade_root / "threshold_sensitivity.parquet")
        pathways = self._safe_read(cascade_root / "pathway_flows.parquet")
        availability = self._safe_read(
            scoped_output_dir(
                self._paths.project_root / self._settings.reporting.tables_dir,
                scope,
                site=site,
                organism=organism,
            )
            / "table_j_downstream_trigger_availability.csv"
        )
        modeling_root = scoped_output_dir(
            self._paths.paths.artifacts / self._settings.modeling.output_dir / self._settings.modeling.task_name,
            scope,
            site=site,
            organism=organism,
        )
        metrics = self._safe_read(modeling_root / "metrics.parquet")
        modeling_summary = self._read_json(modeling_root / "modeling_summary.json")
        rows = cascade_summary.get("rows", {})
        retained_edges_n = int(rows.get("retained_edges", 0))
        if not availability.empty and "has_validated_trigger_forest" in availability.columns:
            estimable_downstreams = int(availability["has_validated_trigger_forest"].sum())
        elif not availability.empty and "has_estimable_trigger_forest" in availability.columns:
            estimable_downstreams = int(availability["has_estimable_trigger_forest"].sum())
        else:
            estimable_downstreams = 0
        total_downstreams = int(len(availability)) if not availability.empty else 0
        threshold_invariant = bool(not threshold.empty and threshold["retained_edges_n"].nunique() == 1)
        sparse_pathways = bool(pathways.empty or len(pathways) <= 2)
        test_metrics = metrics.loc[metrics["split"] == "test"].copy() if not metrics.empty else pd.DataFrame()

        return {
            "scope": scope_dir,
            "organism": organism,
            "sampling_context": {
                "environment": "mac_sampled" if self._settings.environment.name == "mac" else self._settings.environment.name,
                "sample_fraction": self._settings.sampling.default_fraction,
                "interpretation_note": (
                    "These summaries are computed from the current local sampled pipeline outputs. "
                    "They are dynamic and will change when rerun on larger samples or full production data."
                ),
            },
            "cascade_counts": {
                "conditional_probabilities": int(rows.get("conditional_probabilities", 0)),
                "escalation_results": int(rows.get("escalation_results", 0)),
                "adjusted_results": int(rows.get("adjusted_results", 0)),
                "retained_edges": retained_edges_n,
                "two_step_paths": int(report_summary.get("row_counts", {}).get("top_paths", 0)),
            },
            "downstream_trigger_availability": {
                "estimable_downstreams": estimable_downstreams,
                "total_downstreams_screened": total_downstreams,
                "non_estimable_downstreams": total_downstreams - estimable_downstreams,
            },
            "sparsity_flags": {
                "threshold_retained_edge_count_invariant": threshold_invariant,
                "pathway_flow_sparse": sparse_pathways,
            },
            "modeling": {
                "best_test_model": modeling_summary.get("best_test_model", {}),
                "test_metrics": test_metrics.to_dict(orient="records"),
            },
            "interpretation_boundary": {
                "primary_analysis": "directly observed AST only",
                "sensitivity_analysis": "armd implied susceptibility remains separate from primary cross-site cascade outputs",
                "causal_boundary": "adjusted models are robustness analyses of association, not causal claims",
            },
        }

    def render_traceability_markdown(self, traceability: pd.DataFrame) -> str:
        lines = [
            "# Cascade Traceability Audit",
            "",
            "This audit maps major methodological requirements from `cascade.txt` to their current implementation paths.",
            "",
        ]
        for row in traceability.itertuples(index=False):
            lines.extend(
                [
                    f"## {row.requirement_id} {row.section}",
                    f"- Requirement: {row.requirement}",
                    f"- Status: {row.status}",
                    f"- Implementation: {row.implementation_paths}",
                    f"- Notes: {row.notes}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def render_contract_markdown(self, checks: pd.DataFrame) -> str:
        lines = [
            "# Pipeline Contract Audit",
            "",
            "These checks are evaluated against the currently materialized local pipeline outputs.",
            "",
        ]
        for row in checks.itertuples(index=False):
            lines.extend(
                [
                    f"## {row.check_id}",
                    f"- Status: {row.status}",
                    f"- Description: {row.description}",
                    f"- Evidence: {row.evidence_json}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def render_scientific_qa_markdown(self, payload: dict[str, object]) -> str:
        lines = [
            f"# Scientific QA Summary: {payload['scope']}",
            "",
            "## Context",
            payload["sampling_context"]["interpretation_note"],
            "",
            "## Cascade Counts",
        ]
        for key, value in payload["cascade_counts"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                "## Downstream Trigger Availability",
                f"- estimable_downstreams: {payload['downstream_trigger_availability']['estimable_downstreams']}",
                f"- total_downstreams_screened: {payload['downstream_trigger_availability']['total_downstreams_screened']}",
                f"- non_estimable_downstreams: {payload['downstream_trigger_availability']['non_estimable_downstreams']}",
                "",
                "## Sparsity Flags",
                f"- threshold_retained_edge_count_invariant: {payload['sparsity_flags']['threshold_retained_edge_count_invariant']}",
                f"- pathway_flow_sparse: {payload['sparsity_flags']['pathway_flow_sparse']}",
                "",
                "## Modeling",
                f"- best_test_model: {json.dumps(payload['modeling']['best_test_model'])}",
                "- test_metrics:",
            ]
        )
        for item in payload["modeling"].get("test_metrics", []):
            lines.append(
                f"  - {item.get('model_name')}: feature_set={item.get('feature_set')}; "
                f"roc_auc={item.get('roc_auc')}; pr_auc={item.get('pr_auc')}"
            )
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                f"- primary_analysis: {payload['interpretation_boundary']['primary_analysis']}",
                f"- sensitivity_analysis: {payload['interpretation_boundary']['sensitivity_analysis']}",
                f"- causal_boundary: {payload['interpretation_boundary']['causal_boundary']}",
                "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _check(check_id: str, passed: bool, description: str, evidence: dict[str, object]) -> dict[str, object]:
        return {
            "check_id": check_id,
            "status": "pass" if passed else "fail",
            "description": description,
            "evidence_json": json.dumps(evidence, sort_keys=True),
        }

    def _safe_read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return self._dataset_store.read_pandas(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
