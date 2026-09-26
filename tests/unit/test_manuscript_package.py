import json
import os
import time
from pathlib import Path

from amr_cascade_platform.reporting.builders import manuscript_package as mp

PAPER1, PAPER2 = mp.PAPERS


def _project(tmp_path: Path) -> Path:
    (tmp_path / PAPER1).mkdir()
    (tmp_path / PAPER2).mkdir()
    (tmp_path / PAPER1 / "results_only.tex").write_text(
        "\\includegraphics[width=\\linewidth]{../submission_figures/paper1_detection_validation/figure_escalation_forest_validated.pdf}\n"
        "% \\includegraphics{figure_commented_out.pdf}\n"
        "\\includegraphics{figure_pipeline_overview.pdf}\n"
    )
    (tmp_path / PAPER1 / "supplementary.tex").write_text(
        "\\graphicspath{{../submission_figures/paper1_detection_validation/}}\n"
        "\\csvreader[]{table_s7_patient_cluster_bootstrap_sensitivity.csv}{a=\\a}{\\a}\n"
        "(\\texttt{table\\_s9\\_pbi\\_sensitivity.csv}) and \\texttt{data/antibiotic\\_classification\\_complete.csv}\n"
    )
    (tmp_path / PAPER2 / "supplementary.tex").write_text(
        "\\InputIfFileExists{generated_numbers.tex}{}{}\n(\\texttt{table\\_s3\\_smd\\_balance\\_summary.csv})\n"
    )
    figures = tmp_path / "outputs" / "figures"
    tables = tmp_path / "outputs" / "tables"
    (tables / "supplementary").mkdir(parents=True)
    figures.mkdir(parents=True)
    (figures / "figure_escalation_forest_validated.pdf").write_bytes(b"new forest")
    for name in ("table_s7_patient_cluster_bootstrap_sensitivity.csv", "table_s9_pbi_sensitivity.csv", "table_s3_smd_balance_summary.csv"):
        (tables / "supplementary" / name).write_text(f"{name},new\n")
    (tables / "supplementary" / "latex").mkdir()
    (tables / "supplementary" / "latex" / "generated_numbers.tex").write_text("% numbers\n")
    submission = tmp_path / "submission_figures" / PAPER1
    submission.mkdir(parents=True)
    (submission / "figure_pipeline_overview.pdf").write_bytes(b"hand drawn")
    (submission / "figure_escalation_forest_validated.pdf").write_bytes(b"old forest")
    (submission / "table_s16_ridge_penalty_sensitivity_2site.csv").write_text("stale\n")
    (tmp_path / PAPER1 / "table_s7_patient_cluster_bootstrap_sensitivity.csv").write_text("old s7\n")
    return tmp_path


def test_scan_citations_reads_figures_csvreader_and_cited_csv_names(tmp_path):
    root = _project(tmp_path)
    citations = mp.scan_citations(root / PAPER1)
    assert citations.figures == {"figure_escalation_forest_validated.pdf", "figure_pipeline_overview.pdf"}
    assert citations.csvreader == {"table_s7_patient_cluster_bootstrap_sensitivity.csv"}
    assert citations.csv_cited == {"table_s9_pbi_sensitivity.csv"}
    assert mp.scan_citations(root / PAPER2).generated_tex == {"generated_numbers.tex"}


def test_plan_package_refuses_missing_stale_and_unregistered_files(tmp_path):
    root = _project(tmp_path)
    figures, tables = root / "outputs" / "figures", root / "outputs" / "tables"
    old = time.time() - 3600
    os.utime(tables / "supplementary" / "table_s9_pbi_sensitivity.csv", (old, old))
    (root / PAPER2 / "supplementary.tex").write_text(
        "(\\texttt{table\\_s3\\_smd\\_balance\\_summary.csv}) (\\texttt{table\\_s99\\_unknown.csv}) "
        "\\includegraphics{figure_not_generated.pdf}\n"
    )
    plans = mp.plan_package(root, figures, tables, run_started=time.time() - 60)
    errors = plans[PAPER1].errors + plans[PAPER2].errors
    assert any("table_s9_pbi_sensitivity.csv" in e and "older than this run" in e for e in errors)
    assert any("table_s99_unknown.csv" in e and "no generator" in e for e in errors)
    assert any("figure_not_generated.pdf" in e and "does not exist" in e for e in errors)


def test_execute_backs_up_what_it_replaces_and_moves_uncited_files(tmp_path):
    root = _project(tmp_path)
    plans = mp.plan_package(root, root / "outputs" / "figures", root / "outputs" / "tables", run_started=time.time() - 60)
    assert not plans[PAPER1].errors and not plans[PAPER2].errors

    mp.execute(plans, root, prune=True, dry_run=False, context={"organism": "E. coli"})

    submission = root / "submission_figures" / PAPER1
    assert (submission / "figure_escalation_forest_validated.pdf").read_bytes() == b"new forest"
    assert (submission / "figure_pipeline_overview.pdf").read_bytes() == b"hand drawn"
    assert not (submission / "table_s16_ridge_penalty_sensitivity_2site.csv").exists()
    superseded = list((submission / mp.SUPERSEDED_DIR).rglob("*"))
    assert {p.name for p in superseded if p.is_file()} == {"figure_escalation_forest_validated.pdf", "table_s16_ridge_penalty_sensitivity_2site.csv"}
    assert (root / PAPER1 / "table_s7_patient_cluster_bootstrap_sensitivity.csv").read_text().endswith("new\n")
    backups = list((root / "manuscript_edit_log").rglob("table_s7_patient_cluster_bootstrap_sensitivity.csv"))
    assert len(backups) == 1 and backups[0].read_text() == "old s7\n"
    manifest = json.loads((submission / mp.MANIFEST_NAME).read_text())
    assert {entry["name"] for entry in manifest["packaged"]} == {
        "figure_escalation_forest_validated.pdf",
        "table_s7_patient_cluster_bootstrap_sensitivity.csv",
        "table_s9_pbi_sensitivity.csv",
    }
    assert manifest["uncited_moved_aside"] == ["table_s16_ridge_penalty_sensitivity_2site.csv"]
    assert (root / "submission_figures" / PAPER2 / "table_s3_smd_balance_summary.csv").exists()
    assert (root / PAPER2 / "generated_numbers.tex").read_text() == "% numbers\n"
    assert not (root / "submission_figures" / PAPER2 / "generated_numbers.tex").exists()


def test_dry_run_changes_nothing(tmp_path):
    root = _project(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    plans = mp.plan_package(root, root / "outputs" / "figures", root / "outputs" / "tables", run_started=time.time() - 60)
    mp.execute(plans, root, prune=True, dry_run=True, context={})
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
