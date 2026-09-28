from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.modeling.workflows.threshold_strategy_comparison_workflow import (
    ThresholdStrategyComparisonWorkflow,
)
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter
from amr_cascade_platform.visualization.report.plotly_threshold_strategy_comparison_plotter import (
    PlotlyThresholdStrategyComparisonPlotter,
)


class _FakeModelingWorkflow:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        strategy = request.threshold_selection_strategy
        output_dir = self.root / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        selection_split = "validation" if strategy == "validation_holdout" else "training_oof"
        selected = pd.DataFrame(
            [
                {
                    "model_name": "logistic_regression",
                    "feature_set": "clinical_no_cascade",
                    "selection_strategy": strategy,
                    "selection_split": selection_split,
                    "selection_metric": "f1",
                    "selected_threshold": 0.3,
                    "default_threshold": 0.5,
                    "selection_rows": 100,
                    "selection_groups": 20 if strategy == "grouped_oof" else None,
                    "group_column": "anon_id" if strategy == "grouped_oof" else None,
                    "cv_folds": 5 if strategy == "grouped_oof" else None,
                }
            ]
        )
        curve = pd.DataFrame(
            [
                {
                    "model_name": "logistic_regression",
                    "feature_set": "clinical_no_cascade",
                    "split": selection_split,
                    "threshold": threshold,
                    "precision": 0.65,
                    "recall": 0.85,
                    "f1": f1,
                    "mcc": 0.45,
                    "selected_for_decision": threshold == 0.3,
                }
                for threshold, f1 in ((0.2, 0.72), (0.3, 0.75), (0.4, 0.74))
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "model_name": "logistic_regression",
                    "feature_set": "clinical_no_cascade",
                    "split": "test",
                    "decision_threshold": 0.3,
                    "balanced_accuracy": 0.70,
                    "precision": 0.58,
                    "recall": 0.87,
                    "f1": 0.70,
                    "mcc": 0.41,
                }
            ]
        )
        outputs = {}
        for key, frame in (
            ("selected_thresholds", selected),
            ("threshold_metrics", curve),
            ("metrics", metrics),
        ):
            path = output_dir / f"{key}.parquet"
            frame.to_parquet(path, index=False)
            outputs[key] = path
        return outputs


def test_threshold_strategy_comparison_runs_same_model_with_both_strategies(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    paths = PathManager(tmp_path, settings)
    fake_modeling = _FakeModelingWorkflow(tmp_path / "model_runs")
    workflow = ThresholdStrategyComparisonWorkflow(settings, paths, modeling_workflow=fake_modeling)

    outputs = workflow.run(scope="combined", organism="ESCHERICHIA COLI")

    assert [request.threshold_selection_strategy for request in fake_modeling.requests] == [
        "validation_holdout",
        "grouped_oof",
    ]
    assert all(request.feature_sets == ("clinical_no_cascade",) for request in fake_modeling.requests)
    assert all(request.estimators == ("logistic_regression",) for request in fake_modeling.requests)
    curves = pd.read_parquet(outputs["threshold_strategy_curves"])
    selected = pd.read_parquet(outputs["threshold_strategy_selected"])
    test_metrics = pd.read_parquet(outputs["threshold_strategy_test_metrics"])
    assert set(curves["comparison_strategy"]) == {"validation_holdout", "grouped_oof"}
    assert selected.groupby("comparison_strategy").size().eq(1).all()
    assert test_metrics.groupby("comparison_strategy").size().eq(1).all()


def test_threshold_strategy_plotter_exports_interactive_and_static_formats(tmp_path: Path) -> None:
    curves = pd.DataFrame(
        [
            {
                "comparison_strategy": strategy,
                "threshold": threshold,
                "precision": 0.65,
                "recall": 0.85,
                "f1": f1,
                "mcc": 0.45,
            }
            for strategy in ("validation_holdout", "grouped_oof")
            for threshold, f1 in ((0.2, 0.72), (0.3, 0.75), (0.4, 0.74))
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "comparison_strategy": "validation_holdout",
                "selected_threshold": 0.3,
                "selection_rows": 200,
                "selection_groups": None,
                "cv_folds": None,
            },
            {
                "comparison_strategy": "grouped_oof",
                "selected_threshold": 0.3,
                "selection_rows": 180,
                "selection_groups": 30,
                "cv_folds": 5,
            },
        ]
    )
    test_metrics = pd.DataFrame(
        [
            {
                "comparison_strategy": strategy,
                "balanced_accuracy": 0.70,
                "precision": 0.58,
                "recall": 0.87,
                "f1": 0.70,
                "mcc": 0.41,
            }
            for strategy in ("validation_holdout", "grouped_oof")
        ]
    )
    plotter = PlotlyThresholdStrategyComparisonPlotter(
        PlotlyFigureExporter(width=1200, height=628),
        template="plotly_white",
        width=1200,
        height=628,
    )

    outputs = plotter.export(
        curves,
        selected,
        test_metrics,
        tmp_path / "figure_threshold_strategy_comparison",
        ("html", "png", "svg", "pdf"),
    )

    assert set(outputs) == {
        "figure_threshold_strategy_comparison.html",
        "figure_threshold_strategy_comparison.png",
        "figure_threshold_strategy_comparison.svg",
        "figure_threshold_strategy_comparison.pdf",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
