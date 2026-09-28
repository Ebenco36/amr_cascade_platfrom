from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from amr_cascade_platform.cascade.statistics.downstream_testing_regression import (
    DownstreamTestingRegression,
)
from amr_cascade_platform.cascade.statistics.downstream_testing_shap import (
    DownstreamTestingShapExplainer,
)
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.config.config_models import CascadeExplainabilityConfig
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_shap_plotter import PlotlyShapPlotter


def _config(**overrides) -> CascadeExplainabilityConfig:
    values = {
        "enabled": True,
        "background_size": 4,
        "explanation_sample_size": 5,
        "random_seed": 17,
        "additivity_tolerance": 1e-10,
    }
    values.update(overrides)
    return CascadeExplainabilityConfig(**values)


def test_linear_shap_explains_exact_fitted_log_odds() -> None:
    design = pd.DataFrame(
        {
            "_intercept": 1.0,
            "upstream_positive": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "cov_icu_status": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "comorb_diabetes": [-0.5, -0.5, 2.0, -0.5, 2.0, -0.5],
        }
    )
    coefficients = {
        "_intercept": -0.7,
        "upstream_positive": 1.4,
        "cov_icu_status": 0.5,
        "comorb_diabetes": -0.25,
    }

    artifacts = DownstreamTestingShapExplainer(_config()).explain(
        design_matrix=design,
        coefficients=coefficients,
        upstream_antibiotic="CEFTRIAXONE",
        downstream_antibiotic="MEROPENEM",
    )

    assert len(artifacts.global_importance) == 3
    assert len(artifacts.sampled_values) == 5 * 3
    assert artifacts.metadata.loc[0, "explainer"] == "shap.LinearExplainer"
    assert artifacts.metadata.loc[0, "output_scale"] == "log_odds"
    assert bool(artifacts.metadata.loc[0, "additivity_passed"])
    assert artifacts.metadata.loc[0, "maximum_additivity_error"] <= 1e-10
    assert set(artifacts.global_importance["importance_rank"]) == {1, 2, 3}
    assert set(artifacts.sampled_values["feature_domain"]) == {
        "upstream_result",
        "care_setting",
        "comorbidity",
    }


def test_shap_sampling_is_deterministic_for_an_edge() -> None:
    rng = np.random.default_rng(5)
    design = pd.DataFrame(rng.normal(size=(30, 4)), columns=["x1", "x2", "x3", "x4"])
    design.insert(0, "_intercept", 1.0)
    coefficients = {column: float(index) / 10 for index, column in enumerate(design.columns)}
    explainer = DownstreamTestingShapExplainer(_config(background_size=7, explanation_sample_size=9))

    first = explainer.explain(
        design_matrix=design,
        coefficients=coefficients,
        upstream_antibiotic="A",
        downstream_antibiotic="B",
    )
    second = explainer.explain(
        design_matrix=design,
        coefficients=coefficients,
        upstream_antibiotic="A",
        downstream_antibiotic="B",
    )

    pd.testing.assert_frame_equal(first.global_importance, second.global_importance)
    pd.testing.assert_frame_equal(first.sampled_values, second.sampled_values)
    pd.testing.assert_frame_equal(first.metadata, second.metadata)


def test_disabled_shap_returns_schema_stable_empty_artifacts() -> None:
    artifacts = DownstreamTestingShapExplainer(_config(enabled=False)).explain(
        design_matrix=pd.DataFrame({"_intercept": [1.0], "x": [0.0]}),
        coefficients={"_intercept": 0.0, "x": 1.0},
        upstream_antibiotic="A",
        downstream_antibiotic="B",
    )

    assert artifacts.global_importance.empty
    assert artifacts.sampled_values.empty
    assert artifacts.metadata.empty
    assert "mean_absolute_shap" in artifacts.global_importance.columns
    assert "shap_value" in artifacts.sampled_values.columns
    assert "additivity_passed" in artifacts.metadata.columns


def test_adjusted_regression_exposes_shap_for_an_estimable_pair(tmp_path: Path) -> None:
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    regression = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))
    rng = np.random.default_rng(23)
    n = 600
    upstream = rng.integers(0, 2, n)
    icu = rng.integers(0, 2, n)
    tested = (rng.random(n) < expit(-1.2 + 1.1 * upstream + 0.35 * icu)).astype(int)
    group = pd.DataFrame(
        {
            "anon_id": [f"p{index // 2}" for index in range(n)],
            "source_site": np.where(np.arange(n) % 3 == 0, "armd", "armd_ecuh"),
            "organism": "ESCHERICHIA COLI",
            "upstream_positive": upstream,
            "downstream_tested": tested,
            "cov_icu_status": icu,
        }
    )

    result = regression._fit_pair_model(group, "CEFTRIAXONE", "MEROPENEM")
    outputs = regression.explainability_outputs()

    assert bool(result["supports_adjusted_model"])
    assert not outputs["adjusted_model_shap_global"].empty
    assert not outputs["adjusted_model_shap_values"].empty
    assert len(outputs["adjusted_model_shap_metadata"]) == 1
    assert bool(outputs["adjusted_model_shap_metadata"].loc[0, "additivity_passed"])
    assert set(outputs["adjusted_model_shap_domain"]["feature_domain"]) >= {"upstream_result", "care_setting"}


def test_regression_without_explanations_fits_the_pair_but_computes_no_shap(tmp_path: Path) -> None:
    """The ridge-penalty refits pass explain=False: same adjusted fit, no SHAP work."""
    settings = ConfigLoader(Path(__file__).resolve().parents[2]).load("mac")
    rng = np.random.default_rng(23)
    n = 600
    upstream = rng.integers(0, 2, n)
    icu = rng.integers(0, 2, n)
    tested = (rng.random(n) < expit(-1.2 + 1.1 * upstream + 0.35 * icu)).astype(int)
    group = pd.DataFrame(
        {
            "anon_id": [f"p{index // 2}" for index in range(n)],
            "source_site": np.where(np.arange(n) % 3 == 0, "armd", "armd_ecuh"),
            "organism": "ESCHERICHIA COLI",
            "upstream_positive": upstream,
            "downstream_tested": tested,
            "cov_icu_status": icu,
        }
    )
    explained = DownstreamTestingRegression(settings, PathManager(tmp_path, settings))
    silent = DownstreamTestingRegression(settings, PathManager(tmp_path, settings), explain=False)

    with_shap = explained._fit_pair_model(group, "CEFTRIAXONE", "MEROPENEM")
    without_shap = silent._fit_pair_model(group, "CEFTRIAXONE", "MEROPENEM")

    assert without_shap["adjusted_odds_ratio"] == with_shap["adjusted_odds_ratio"]
    assert all(frame.empty for frame in silent.explainability_outputs().values())


def test_shap_plotter_exports_interactive_and_static_pair_figures(tmp_path: Path) -> None:
    design = pd.DataFrame(
        {
            "_intercept": 1.0,
            "upstream_positive": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "cov_icu_status": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        }
    )
    artifacts = DownstreamTestingShapExplainer(_config()).explain(
        design_matrix=design,
        coefficients={"_intercept": -0.5, "upstream_positive": 1.2, "cov_icu_status": 0.4},
        upstream_antibiotic="CEFTRIAXONE",
        downstream_antibiotic="MEROPENEM",
    )
    plotter = PlotlyShapPlotter(
        PlotlyFigureExporter(width=1000, height=700),
        "plotly_white",
        width=1000,
        height=700,
    )

    outputs = plotter.export_pair_suite(
        artifacts.global_importance,
        artifacts.sampled_values,
        tmp_path,
        ("html", "png", "pdf", "svg"),
    )

    assert len(outputs) == 8
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())


def test_domain_attribution_sums_a_categorys_dummies_within_each_row() -> None:
    """A one-hot category's dummies offset each other within a row; summing their
    per-feature mean |SHAP| overstated such domains, the grouped value does not."""
    design = pd.DataFrame(
        {
            "_intercept": 1.0,
            "upstream_positive": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            # Two dummies of one category (calendar year): exactly one is 1 per row.
            "cov_calendar_year_2020": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "cov_calendar_year_2021": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    coefficients = {"_intercept": 0.1, "upstream_positive": 1.0, "cov_calendar_year_2020": 0.8, "cov_calendar_year_2021": -0.8}
    artifacts = DownstreamTestingShapExplainer(_config(explanation_sample_size=6, background_size=6)).explain(
        design_matrix=design, coefficients=coefficients, upstream_antibiotic="A", downstream_antibiotic="B",
    )
    domains = artifacts.domain_importance.set_index("feature_domain")
    assert set(domains.index) == {"upstream_result", "calendar_time"}
    assert domains.loc["calendar_time", "feature_n"] == 2

    values = artifacts.sampled_values
    per_row = values.loc[values["feature_domain"].eq("calendar_time")].groupby("sample_row")["shap_value"].sum()
    assert np.isclose(domains.loc["calendar_time", "mean_absolute_shap"], per_row.abs().mean())
    per_feature_sum = artifacts.global_importance.loc[
        artifacts.global_importance["feature_domain"].eq("calendar_time"), "mean_absolute_shap"
    ].sum()
    assert domains.loc["calendar_time", "mean_absolute_shap"] <= per_feature_sum + 1e-12


def test_pair_ranking_and_domain_summary_for_investigation() -> None:
    global_importance = pd.DataFrame(
        {
            "upstream_antibiotic": ["A", "A", "C", "C"],
            "downstream_antibiotic": ["B", "B", "D", "D"],
            "feature": ["upstream_positive", "cov_icu_status", "upstream_positive", "cov_icu_status"],
            "feature_domain": ["upstream_result", "care_setting", "upstream_result", "care_setting"],
            "mean_absolute_shap": [0.2, 0.6, 0.9, 0.1],
        }
    )
    domain = global_importance.rename(columns={"feature": "unused"})
    by_pair, overall = PlotlyShapPlotter.domain_summary(domain)
    assert np.allclose(by_pair.groupby(["upstream_antibiotic", "downstream_antibiotic"])["pair_share"].sum(), 1.0)
    ranking = PlotlyShapPlotter.rank_pairs(global_importance, by_pair)
    # C -> D has the larger upstream-result attribution, so it is drawn first.
    assert ranking[["upstream_antibiotic", "downstream_antibiotic"]].iloc[0].tolist() == ["C", "D"]
    assert np.isclose(ranking.set_index("upstream_antibiotic").loc["C", "upstream_result_share"], 0.9)
    assert set(overall["feature_domain"]) == {"upstream_result", "care_setting"}
    assert overall["pairs"].eq(2).all()
    assert overall["pairs_where_largest"].sum() == 2


def test_pair_suite_draws_only_the_selected_pairs(tmp_path: Path) -> None:
    rows = []
    for up, down in (("A", "B"), ("C", "D")):
        for feature, domain in (("upstream_positive", "upstream_result"), ("cov_icu_status", "care_setting")):
            rows.append({"upstream_antibiotic": up, "downstream_antibiotic": down, "feature": feature,
                         "feature_domain": domain, "mean_absolute_shap": 0.5, "mean_signed_shap": 0.1,
                         "positive_attribution_fraction": 0.5, "importance_rank": 1, "explained_n": 3,
                         "output_scale": "log_odds"})
    global_importance = pd.DataFrame(rows)
    values = pd.DataFrame(
        [
            {"upstream_antibiotic": up, "downstream_antibiotic": down, "sample_row": r, "model_row_position": r,
             "feature": feature, "feature_domain": domain, "feature_value": float(r), "shap_value": 0.1 * r,
             "absolute_shap_value": 0.1 * r, "output_scale": "log_odds"}
            for up, down in (("A", "B"), ("C", "D"))
            for r in range(3)
            for feature, domain in (("upstream_positive", "upstream_result"), ("cov_icu_status", "care_setting"))
        ]
    )
    plotter = PlotlyShapPlotter(PlotlyFigureExporter(width=900, height=600), "plotly_white", 900, 600)
    outputs = plotter.export_pair_suite(global_importance, values, tmp_path, ("png",), pairs=[("C", "D")])
    assert {path.name for path in outputs.values()} == {
        "figure_adjusted_model_shap_importance__c_to_d.png",
        "figure_adjusted_model_shap_beeswarm__c_to_d.png",
    }
    _, overall = PlotlyShapPlotter.domain_summary(global_importance)
    summary = plotter.export_domain_summary(overall, tmp_path, ("png",))
    assert all(path.exists() and path.stat().st_size > 0 for path in summary.values())
