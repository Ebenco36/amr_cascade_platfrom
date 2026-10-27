import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.modeling.estimators.logistic_regression_estimator import LogisticRegressionEstimator
from amr_cascade_platform.modeling.workflows.modeling_workflow import ModelingWorkflow
from amr_cascade_platform.reporting.paper1_profile import PAPER1_REPORT_FIGURES, PAPER1_PREDICTION_FIGURES

ROOT = Path(__file__).resolve().parents[2]


def test_saved_lr_coefficients_reproduce_the_fitted_probability():
    settings = ConfigLoader(ROOT).load("test_dataset")
    estimator = LogisticRegressionEstimator(settings, ("category",), ("age", "missing"))
    frame = pd.DataFrame({"category": ["a", "b", "a", "b"] * 8,
                          "age": np.arange(32) * 2.0, "missing": np.nan})
    estimator.fit(frame, pd.Series([0, 1, 1, 0] * 8))
    coefficients = estimator.coefficients()
    design = estimator._pipeline.named_steps["preprocess"].transform(frame)
    reconstructed = expit(np.asarray(design @ coefficients.raw_unit_coefficient) + coefficients.intercept.iloc[0])
    np.testing.assert_allclose(reconstructed, estimator.predict_proba(frame), atol=1e-12)
    assert {"category_a", "category_b", "age"}.issubset(coefficients.feature_raw)
    # sklearn versions differ on retaining constant-imputed all-missing fields.
    assert len(coefficients) == design.shape[1]
    assert coefficients.coefficient_scale.unique().tolist() == ["MaxAbs-scaled fitted design"]


def test_feature_export_streams_without_loading_all_rows(tmp_path, monkeypatch):
    settings = ConfigLoader(ROOT).load("test_dataset")
    workflow = ModelingWorkflow(settings, PathManager(tmp_path, settings))
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame({"source_site": ["armd", "armd_ecuh", "armd_utsw"] * 25000,
                          "target": [0, 1, 0] * 25000, "age": np.arange(75000), "unused": "ignore"})
    frame.to_parquet(source, index=False, row_group_size=20000)
    def forbidden(*args, **kwargs):
        raise AssertionError("Full pandas read must not be used to export or count sites")
    monkeypatch.setattr(workflow._dataset_store, "read_pandas", forbidden)
    selection = SimpleNamespace(persisted=True, source_path=source)
    dataset = SimpleNamespace(id_columns=("target",))
    feature_set = SimpleNamespace(feature_columns=("age",))
    destination = tmp_path / "export.parquet"
    workflow._export_feature_set_dataset(selection, dataset, feature_set, destination)
    pd.testing.assert_frame_equal(pd.read_parquet(destination), frame[["target", "age", "source_site"]])
    assert workflow._site_split_row_counts(source) == {"all": 75000, "train": 25000, "validation": 25000, "test": 25000}
    assert not list(tmp_path.glob("*.tmp"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper1_asset_audit_fails_on_missing_figures(tmp_path):
    module = _script("audit_paper1_outputs")
    result = module.audit(tmp_path, tmp_path, tmp_path)
    assert not result["passed"]
    assert any("figure_suppression_matrix_validated.pdf" in error for error in result["errors"])
    assert len(PAPER1_REPORT_FIGURES + PAPER1_PREDICTION_FIGURES) == 23


def test_paper1_supplement_does_not_require_prevalence_inputs(tmp_path):
    from amr_cascade_platform.reporting.builders.supplementary_table_builder import SupplementaryTableBuilder
    settings = ConfigLoader(ROOT).load("test_dataset")
    builder = SupplementaryTableBuilder(
        settings, PathManager(tmp_path, settings),
        classification_path=ROOT / "data/antibiotic_classification_complete.csv",
    )
    result = builder.build("ESCHERICHIA COLI", publication="paper1")
    assert result.missing_inputs
    assert any("validation_results.parquet" in name for name in result.missing_inputs)
    assert not any("prevalence_shift" in name for name in result.missing_inputs)


def test_paper1_contract_matches_manuscript_if_available():
    import re
    folder = ROOT / "paper1_detection_validation"
    if not folder.is_dir():
        pytest.skip("Manuscript is not distributed in the git repository")
    content = "\n".join((folder / name).read_text() for name in ("results_only.tex", "supplementary.tex"))
    stems = set(re.findall(r"\\includegraphics[^\n]*\{\./([^}]+)\.pdf\}", content))
    assert stems == set(PAPER1_REPORT_FIGURES + PAPER1_PREDICTION_FIGURES) | {"Cascade_publication.drawio"}  # Fig. 1: the author's drawio figure (2026-10-27)


def test_report_reuse_requires_all_saved_exports(tmp_path):
    import json
    import os
    from amr_cascade_platform.reporting.paper1_profile import report_ready
    edges = tmp_path / "edge_report.parquet"
    edges.write_bytes(b"edge data")
    os.utime(edges, (1, 1))
    exports = {}
    for stem in PAPER1_REPORT_FIGURES:
        for fmt in ("pdf", "png", "svg", "html"):
            path = tmp_path / f"{stem}.{fmt}"
            path.write_bytes(b"test export")
            exports[path.name] = str(path)
    table = tmp_path / "table.csv"
    table.write_text("column\n1\n")
    manifest = tmp_path / "report_manifest.json"
    manifest.write_text(json.dumps({"publication": "paper1", "missing_figure_groups": {},
                                    "figure_exports": exports, "tables": {table.name: str(table)}}))
    assert report_ready(manifest, edges)
    assert not report_ready(manifest, edges, hide_titles=True)
    content = json.loads(manifest.read_text())
    content["hide_figure_titles"] = True
    manifest.write_text(json.dumps(content))
    assert report_ready(manifest, edges, hide_titles=True)
    assert not report_ready(manifest, edges)
    content["hide_figure_titles"] = False
    manifest.write_text(json.dumps(content))
    figure = tmp_path / f"{PAPER1_REPORT_FIGURES[0]}.pdf"
    figure.unlink()
    assert not report_ready(manifest, edges)
    figure.write_bytes(b"test export")
    table.write_text("")
    assert not report_ready(manifest, edges)


def test_paper1_disables_generic_renderer_fallback(tmp_path, monkeypatch):
    import plotly.graph_objects as go
    from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
    def fail(*args, **kwargs):
        raise RuntimeError("renderer unavailable")
    monkeypatch.setattr(go.Figure, "write_image", fail)
    exporter = PlotlyFigureExporter(1200, 800, allow_generic_fallback=False)
    with pytest.raises(RuntimeError, match="Generic approximation is disabled"):
        exporter.write(go.Figure(go.Bar(x=[1], y=[2])), tmp_path / "figure", ("png",))
    assert not (tmp_path / "figure.png").exists()


@pytest.mark.parametrize("title,has_math", [("Observed AST", False), (r"$P(O=1)$", True)])
def test_native_export_uses_mathjax_only_when_needed(tmp_path, monkeypatch, title, has_math):
    import plotly.graph_objects as go
    import plotly.io as pio
    from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
    scope = SimpleNamespace(mathjax="configured-mathjax")
    monkeypatch.setattr(pio.kaleido, "scope", scope)
    def fail(self, *args, **kwargs):
        assert scope.mathjax == ("configured-mathjax" if has_math else None)
        raise RuntimeError("expected test error")
    monkeypatch.setattr(go.Figure, "write_image", fail)
    figure = go.Figure(layout={"title": title, "width": 800, "height": 600})
    with pytest.raises(RuntimeError, match="expected test error"):
        PlotlyFigureExporter._write_native(figure, tmp_path / "figure.pdf", "pdf")
    assert scope.mathjax == "configured-mathjax"


def test_paper1_can_skip_shap_without_changing_validation_settings(tmp_path):
    from amr_cascade_platform.cascade.workflows.cascade_analysis_workflow import CascadeAnalysisWorkflow
    settings = ConfigLoader(ROOT).load("test_dataset")
    workflow = CascadeAnalysisWorkflow(settings, PathManager(tmp_path, settings), explain=False)
    assert not workflow._regression_analyzer._explain
    assert workflow._settings is settings
    assert workflow._validation_analyzer is not None


def test_validation_reuse_refuses_incompatible_sidecar(tmp_path, monkeypatch):
    import json
    module = _script("check_validation_reuse")
    settings = ConfigLoader(ROOT).load("test_dataset")
    paths = PathManager(tmp_path, settings)
    directory = paths.paths.artifacts / settings.cascade.outputs.result_dir / "combined/organisms/escherichia_coli"
    directory.mkdir(parents=True)
    result = directory / "validation_results.parquet"
    pd.DataFrame({"edge": [1]}).to_parquet(result)
    module.sidecar_path(result).write_text(json.dumps({"fingerprint": "old", "git": {"commit": "old-commit"}}))
    monkeypatch.setattr(module, "validation_fingerprint", lambda *args: ("new", {}))
    errors = module.check(tmp_path, settings, paths, ["armd"], ["ESCHERICHIA COLI"], include_sites=True)
    assert len(errors) == 1
    assert "old-commit" in errors[0]
    assert json.loads(module.sidecar_path(result).read_text())["fingerprint"] == "old"
