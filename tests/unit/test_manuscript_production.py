import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.cascade.workflows.sensitivity_cache import SensitivityCache
from amr_cascade_platform.reporting.builders.manuscript_package import CSV_SOURCES, scan_citations
from amr_cascade_platform.reporting.manuscript_profile import (
    MANUSCRIPT_FIGURE_GROUPS, MANUSCRIPT_REPORT_FIGURES, PAPER2_REPORT_FIGURES,
    required_exports,
)
from amr_cascade_platform.reporting.paper1_profile import PAPER1_PREDICTION_FIGURES, report_ready

ROOT = Path(__file__).resolve().parents[2]


def test_both_manuscripts_match_the_named_output_contract():
    folders = [ROOT / "paper1_detection_validation", ROOT / "paper2_prevalence_surveillance"]
    if not all(path.is_dir() for path in folders):
        pytest.skip("Paper folders are not distributed in git")
    citations = [scan_citations(path) for path in folders]
    assert {Path(name).stem for name in citations[1].figures} == set(PAPER2_REPORT_FIGURES)
    assert set.union(*(c.figures for c in citations)) == {
        f"{stem}.pdf" for stem in MANUSCRIPT_REPORT_FIGURES + PAPER1_PREDICTION_FIGURES
    } | {"figure_pipeline_overview.pdf"}
    assert set.union(*(c.csv_cited | c.csvreader for c in citations)) == set(CSV_SOURCES)


def test_contract_excludes_exploratory_suites_and_impossible_html():
    assert not {"adjusted_model_shap", "dataset_characterization", "mnar_tipping_point",
                "threshold_strategy_comparison"}.intersection(MANUSCRIPT_FIGURE_GROUPS)
    exports = required_exports("manuscripts", ("pdf", "png", "svg", "html"))
    assert len(exports) == 105  # 27 report families, three static-only.
    assert "figure_prevalence_mnar_sensitivity_curves_stable.pdf" in exports
    assert "figure_prevalence_mnar_sensitivity_curves_stable.html" not in exports


def test_sensitivity_cache_rejects_changed_inputs_parameters_and_output(tmp_path):
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame({"edge": ["a"]})
    frame.to_parquet(source)
    cache = SensitivityCache(tmp_path, {"iterations": 1000}, {"validation": source}, analysis="era")
    target = tmp_path / "result.parquet"
    frame.to_parquet(target)
    assert not cache.ready(target)  # Existence without provenance is not completion.
    cache.save(frame, target, shard=(0, 2))
    assert cache.ready(target, shard=(0, 2))
    assert not cache.ready(target, shard=(0, 3))
    assert not cache.ready(target)
    cache.save(frame, target)
    assert cache.ready(target)
    changed = SensitivityCache(tmp_path, {"iterations": 50}, {"validation": source}, analysis="era")
    assert not changed.ready(target)
    frame.assign(new=1).to_parquet(source)
    assert not SensitivityCache(tmp_path, {"iterations": 1000}, {"validation": source}, analysis="era").ready(target)
    target.write_bytes(b"interrupted write")
    assert not cache.ready(target)


def test_manuscript_report_reuse_requires_paper2_exports_and_current_prevalence(tmp_path):
    edges = tmp_path / "edge_report.parquet"
    prevalence = tmp_path / "prevalence_shift.parquet"
    for path in (edges, prevalence):
        path.write_bytes(b"input")
        os.utime(path, (1, 1))
    exports = {}
    for name in required_exports("manuscripts", ("pdf", "png", "svg", "html")):
        path = tmp_path / name
        path.write_bytes(b"export")
        exports[name] = str(path)
    table = tmp_path / "table.csv"
    table.write_text("a\n1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"publication": "manuscripts", "figure_exports": exports,
                                    "missing_figure_groups": {}, "tables": {"table": str(table)}}))
    def ready():
        return report_ready(manifest, edges, publication="manuscripts", additional_inputs=(prevalence,))
    assert ready()
    assert not report_ready(manifest, edges, publication="paper1")
    from amr_cascade_platform.core.utils.provenance import file_fingerprint, git_state
    assert not report_ready(manifest, edges, publication="manuscripts", project_root=tmp_path)
    recorded = json.loads(manifest.read_text())
    recorded["provenance"] = {"git": git_state(tmp_path), "inputs": {"prevalence": file_fingerprint(prevalence)}}
    manifest.write_text(json.dumps(recorded))
    assert report_ready(manifest, edges, publication="manuscripts", project_root=tmp_path)
    recorded["provenance"]["git"] = {"commit": "stale-code"}
    manifest.write_text(json.dumps(recorded))
    assert not report_ready(manifest, edges, publication="manuscripts", project_root=tmp_path)
    os.utime(prevalence, (manifest.stat().st_mtime + 1,) * 2)
    assert not ready()
    os.utime(prevalence, (1, 1))
    (tmp_path / "figure_antibiogram.pdf").unlink()
    assert not ready()


def test_combined_supplement_requires_prevalence_inputs(tmp_path):
    from amr_cascade_platform.core.config.config_loader import ConfigLoader
    from amr_cascade_platform.core.paths.path_manager import PathManager
    from amr_cascade_platform.reporting.builders.supplementary_table_builder import SupplementaryTableBuilder
    settings = ConfigLoader(ROOT).load("test_dataset")
    builder = SupplementaryTableBuilder(settings, PathManager(tmp_path, settings),
                                       classification_path=ROOT / "data/antibiotic_classification_complete.csv")
    result = builder.build("ESCHERICHIA COLI", publication="manuscripts")
    assert any("table_k_prevalence_shift.csv" in name for name in result.missing_inputs)
    assert any("table_k_prevalence_shift_diagnostics.csv" in name for name in result.missing_inputs)


def test_both_papers_asset_audit_rejects_missing_csv(tmp_path):
    spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_paper1_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(tmp_path, tmp_path, tmp_path, publication="manuscripts")
    assert not result["passed"]
    assert any("table_s3_smd_balance_summary.csv" in error for error in result["errors"])
    assert any("figure_prevalence_mnar_sensitivity_curves_stable.pdf" in error for error in result["errors"])
    assert not any("figure_prevalence_mnar_sensitivity_curves_stable.html" in error for error in result["errors"])


def test_manuscript_wrapper_dry_run_covers_both_and_orders_variants():
    result = subprocess.run([
        "bash", "scripts/submit_everything_hpc.sh", "--dry-run", "--skip-install",
        "--allow-dirty-checkout", "--era-stratified-shards", "2", "--patient-cluster-shards", "2",
    ], cwd=ROOT, env={**os.environ, "CASCADE_SHARD_NUM": "2", "CASCADE_SITE_SHARD_NUM": "2"},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.splitlines() + result.stderr.splitlines()
    jobs = [line for line in lines if "--job-name=" in line]
    for name in ("_prev", "_training", "_threshold_compare", "_report", "amr_era_stratified_merge",
                 "amr_patient_cluster_merge", "amr_ridge_penalty_sensitivity", "amr_supplementary_prediction",
                 "amr_supplementary_tables", "amr_manuscripts_artifact_audit"):
        assert any(f"{name} " in line for line in jobs), name
    assert not any("--job-name=amr_eskape" in line or "_dataset_summary " in line or "_readiness_audit " in line for line in jobs)
    variant_gold = [line for line in jobs if "--job-name=amr_escherichia_coli_gold " in line
                    and "availability" in line]
    assert len(variant_gold) == 3
    assert all("--dependency=afterok:DRYRUN_amr_harmonize" in line for line in variant_gold)
    cascades = [line for line in jobs if "scripts/run_cascade_analysis.py" in line]
    assert len(cascades) == 4  # Primary combined plus the three sites; no variant refits.
    assert all("--skip-shap" in line for line in cascades)
    reports = [line for line in jobs if "scripts/run_reporting.py" in line]
    assert len(reports) == 1
    assert "_armd_cascade" in reports[0] and "_prev" in reports[0]
