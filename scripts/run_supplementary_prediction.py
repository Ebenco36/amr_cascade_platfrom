"""
Supplementary predictive-modelling comparison — publication figures and tables.

Loads saved prediction artefacts and fitted LR coefficients (no retraining),
then writes six publication-quality figures (each as .html, .png, and .pdf,
matching the house style in
src/amr_cascade_platform/visualization/report/plotly_model_plotters.py) and
three CSV tables to:

    outputs/figures/combined/organisms/escherichia_coli/
    outputs/tables/combined/organisms/escherichia_coli/

All filenames carry the prefix ``supp_pred_`` so they do not collide with
primary pipeline outputs.

Usage
-----
    python scripts/run_supplementary_prediction.py [--env hpc] [--organism ESCHERICHIA_COLI]

Input and output locations follow the environment config (--env, default hpc):
modelling artefacts and features under its data_root, figures and tables under
its reporting.figures_dir and reporting.tables_dir. The organism string is
normalised to lower-snake for path construction; the default is
ESCHERICHIA_COLI.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Plotly house style ───────────────────────────────────────────────────────
# Matches src/amr_cascade_platform/visualization/report/plotly_model_plotters.py
# (_apply_standard_layout) exactly, so these supplementary figures are visually
# consistent with the primary pipeline's figure_model_* outputs.
PLOTLY_TEMPLATE = "plotly_white"
FIG_WIDTH = 1600
FIG_HEIGHT = 900
GRID_X = "#E8EEF5"
GRID_Y = "#F2F4F7"
REFERENCE_COLOR = "#7F8C8D"
EXPORT_FORMATS = ("html", "png", "svg", "pdf")

# ── Model display properties ─────────────────────────────────────────────────
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLOR = {
    "logistic_regression": "#1F77B4",
    "random_forest": "#2E8B57",
    "xgboost": "#E69F00",
}
SPLIT_COLOR = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
SPLIT_LABEL = {"train": "Train (armd)", "validation": "Validation (armd_ecuh)", "test": "Test (armd_utsw)"}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from amr_cascade_platform.visualization.report.figure_presentation import figure_titles_hidden, save_matplotlib, title_policy
from amr_cascade_platform.visualization.report.plotly_model_plotters import (  # noqa: E402
    CURVE_DISPLAY_POINTS as _CURVE_DISPLAY_POINTS,
    thin_curve as _thin_curve,
)
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver  # noqa: E402


@dataclass(frozen=True)
class _Layout:
    """Where one environment keeps its modelling inputs and report outputs."""

    modeling_root: Path
    features_root: Path
    figures_root: Path
    tables_root: Path


_LAYOUT: _Layout | None = None


def _configure_layout(environment: str) -> _Layout:
    global _LAYOUT
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from amr_cascade_platform.core.config.config_loader import ConfigLoader

    settings = ConfigLoader(PROJECT_ROOT).load(environment)
    data_root = PROJECT_ROOT / settings.environment.data_root
    _LAYOUT = _Layout(
        modeling_root=data_root / "artifacts" / settings.modeling.output_dir / settings.modeling.task_name,
        features_root=data_root / "features",
        figures_root=PROJECT_ROOT / settings.reporting.figures_dir,
        tables_root=PROJECT_ROOT / settings.reporting.tables_dir,
    )
    return _LAYOUT


def _layout() -> _Layout:
    return _LAYOUT if _LAYOUT is not None else _configure_layout("hpc")


def _org_slug(organism: str) -> str:
    return organism.lower().replace(" ", "_")


def _model_dir(organism: str) -> Path:
    return _layout().modeling_root / "combined/organisms" / _org_slug(organism) / "site__all_models"


def _artifact_dir(organism: str) -> Path:
    return _model_dir(organism) / "full"


def _feat_path(organism: str) -> Path:
    return (
        _layout().features_root
        / "combined/organisms"
        / _org_slug(organism)
        / "model_ready_pair_features.parquet"
    )


def _fig_dir(organism: str) -> Path:
    d = _layout().figures_root / "combined/organisms" / _org_slug(organism)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tab_dir(organism: str) -> Path:
    d = _layout().tables_root / "combined/organisms" / _org_slug(organism)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Plotly layout / export helpers ───────────────────────────────────────────

def _apply_standard_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    margin: dict[str, int] | None = None,
    legend_title: str = "Model",
) -> None:
    """Mirrors PlotlyModelEvaluationPlotter._apply_standard_layout exactly."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=width or FIG_WIDTH,
        height=height or FIG_HEIGHT,
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend={
            "title": {"text": legend_title},
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13},
        },
        margin=margin or {"l": 80, "r": 40, "t": 90, "b": 90},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"size": 16, "family": "Arial"},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_X, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False)


def _model_static_plotter():
    """Reuse the production model plotter's dedicated static renderers."""
    from amr_cascade_platform.visualization.report.plotly_exporter import (
        PlotlyFigureExporter,
    )
    from amr_cascade_platform.visualization.report.plotly_model_plotters import (
        PlotlyModelEvaluationPlotter,
    )

    exporter = PlotlyFigureExporter(width=FIG_WIDTH, height=FIG_HEIGHT)
    return PlotlyModelEvaluationPlotter(
        exporter=exporter,
        template=PLOTLY_TEMPLATE,
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
    )


def _write_all_formats(
    fig: go.Figure,
    out_stem: Path,
    static_fallback: Callable[[str, Path], None] | None = None,
) -> dict[str, Path]:
    """Write interactive and publication-static outputs with a real fallback."""
    from amr_cascade_platform.visualization.report.plotly_exporter import (
        PlotlyFigureExporter,
    )

    width = int(fig.layout.width or FIG_WIDTH)
    height = int(fig.layout.height or FIG_HEIGHT)
    PlotlyFigureExporter(width=width, height=height, allow_generic_fallback=False, hide_titles=figure_titles_hidden()).write(
        fig,
        out_stem,
        EXPORT_FORMATS,
        static_fallback=static_fallback,
    )
    return {fmt: out_stem.with_suffix(f".{fmt}") for fmt in EXPORT_FORMATS}


def _write_cross_split_static(metrics: pd.DataFrame, fmt: str, path: Path) -> None:
    """Publication-static grouped bars for train/validation/test ROC-AUC."""
    models = [model for model in MODEL_ORDER if model in set(metrics["model_name"])]
    splits = ["train", "validation", "test"]
    positions = np.arange(len(models), dtype=float)
    bar_width = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    for index, split in enumerate(splits):
        split_rows = metrics.loc[metrics["split"].eq(split)].set_index("model_name")
        values = [
            float(split_rows.loc[model, "roc_auc"]) if model in split_rows.index else np.nan
            for model in models
        ]
        offset = (index - 1) * bar_width
        bars = ax.bar(
            positions + offset,
            values,
            width=bar_width,
            color=SPLIT_COLOR[split],
            label=SPLIT_LABEL[split],
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" if np.isfinite(value) else "" for value in values],
            padding=3,
            fontsize=10,
        )

    ax.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.set_xticks(positions, [MODEL_LABEL[model] for model in models])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("ROC-AUC Across Train, Development-Validation, and Test Sites", fontsize=15, fontweight="bold")
    ax.grid(axis="y", color=GRID_Y, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_matplotlib(fig, path, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_coefficients_static(plot_df: pd.DataFrame, fmt: str, path: Path) -> None:
    """Publication-static horizontal coefficient plot with readable labels."""
    frame = plot_df.iloc[::-1].copy()
    labels = frame["feature"].map(_axis_label).str.replace("<br>", "\n", regex=False)
    values = frame["coefficient"].astype(float)
    colors = ["#1F77B4" if value > 0 else "#D62728" for value in values]
    height = max(8.0, 0.43 * len(frame) + 2.5)
    fig, ax = plt.subplots(figsize=(13.5, height))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Coefficient per unit of MaxAbs-scaled predictor (log-OR)")
    ax.set_title(
        "Largest Positive and Negative Logistic-Regression Coefficients\n"
        "Stanford training site; full feature set",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID_X, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}" for value in values],
        padding=4,
        fontsize=9,
    )
    x_limit = max(abs(float(values.min())), abs(float(values.max())), 0.01)
    ax.set_xlim(-1.30 * x_limit, 1.30 * x_limit)
    fig.tight_layout()
    save_matplotlib(fig, path, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(organism: str) -> pd.DataFrame:
    """Concatenate saved per-model prediction parquets."""
    art = _artifact_dir(organism)
    dfs = []
    for model in MODEL_ORDER:
        path = art / f"{model}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"The three-model comparison requires {path}")
        df = pd.read_parquet(path, columns=["target", "split", "predicted_probability"])
        df["model_name"] = model
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No prediction parquets found in {art}")
    return pd.concat(dfs, ignore_index=True)


def load_metrics(organism: str) -> pd.DataFrame:
    """Load the pre-computed per-split metrics parquet."""
    path = _model_dir(organism) / "metrics.parquet"
    return pd.read_parquet(path)


def load_threshold_metrics(organism: str) -> pd.DataFrame:
    """Load threshold-sweep metrics parquet."""
    path = _model_dir(organism) / "threshold_metrics.parquet"
    return pd.read_parquet(path)


# ── LR coefficient extraction ─────────────────────────────────────────────────

_FEATURE_RENAME = {
    "upstream_susceptibility_RESISTANT": "Upstream result: Resistant",
    "upstream_susceptibility_SUSCEPTIBLE": "Upstream result: Susceptible",
    "baseline_demographics_available": "Demographics data available",
    "demo_age": "Patient age",
    "demo_gender_female": "Sex: Female",
    "demo_gender_male": "Sex: Male",
    "demo_gender_unknown": "Sex: Unknown",
    "acute_labs_available": "Acute labs available",
    "acute_vitals_available": "Acute vitals available",
    "ward_hosp_ward_ip": "Ward: Inpatient",
    "ward_hosp_ward_op": "Ward: Outpatient",
    "ward_hosp_ward_er": "Ward: Emergency",
    "ward_hosp_ward_icu": "Ward: ICU",
    "acute_ward_available": "Acute ward data available",
    "history_abx_available": "Prior ABX data available",
    "history_abx_exposure_count": "Prior ABX count (90d)",
    "history_abx_unique_class_count": "Unique ABX classes (90d)",
    "history_abx_min_days": "Days since last ABX",
    "history_abx_any_30d": "Any prior ABX (30d)",
    "history_abx_any_90d": "Any prior ABX (90d)",
    "history_abx_any_365d": "Any prior ABX (365d)",
    "history_prior_organism_available": "Prior organism data available",
    "history_prior_organism_count": "Prior organism count",
    "history_prior_organism_min_days": "Days since prior organism",
    "history_prior_same_organism_any_30d": "Prior same-genus infection (30d)",
    "history_prior_same_organism_any_90d": "Prior same-genus infection (90d)",
    "history_prior_same_organism_any_365d": "Prior same-genus infection (365d)",
    "comorbidity_count": "Total comorbidity count",
}


# One-hot antibiotic features carry the source label (e.g. MINOCYCLIN, CO-TRIMOXAZOL); show the
# display name the manuscripts use, as the Supplementary tables of the companion builder do.
_ANTIBIOTIC_FEATURE_PREFIXES = ("upstream_antibiotic_", "downstream_antibiotic_")
_CLASSIFICATION_PATH = PROJECT_ROOT / "data" / "antibiotic_classification_complete.csv"
_RESOLVER: AntibioticClassificationResolver | None = None


def _antibiotic_display(label: str) -> str:
    global _RESOLVER
    if not _CLASSIFICATION_PATH.exists():
        return label
    if _RESOLVER is None:
        _RESOLVER = AntibioticClassificationResolver(_CLASSIFICATION_PATH)
    return _RESOLVER.canonical_display_label(label)


def _clean_feature_name(raw: str) -> str:
    if raw in _FEATURE_RENAME:
        return _FEATURE_RENAME[raw]
    for prefix in _ANTIBIOTIC_FEATURE_PREFIXES:
        if raw.startswith(prefix):
            return f"{prefix.replace('_', ' ').title()}{_antibiotic_display(raw[len(prefix):]).title()}"
    if raw.startswith("comorb_"):
        name = raw[len("comorb_"):]
        name = name.replace("_", " ").title()
        return f"Comorbidity: {name}"
    return raw.replace("_", " ").title()


def _axis_label(name: str, limit: int = 58) -> str:
    """Shorten a feature name for a figure axis only; tables keep the full name."""
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


def load_lr_coefficients(organism: str) -> pd.DataFrame:
    """Read coefficients of the model that generated the saved predictions."""
    path = _artifact_dir(organism) / "logistic_regression_coefficients.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fitted-model coefficients: {path}. Run the updated training "
            "workflow; refusing to substitute a sampled, differently specified refit."
        )
    frame = pd.read_parquet(path)
    required = {"feature_raw", "coefficient", "odds_ratio", "coefficient_scale"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Incomplete fitted-model coefficient artifact: {path}")
    frame["feature"] = frame["feature_raw"].map(_clean_feature_name)
    return frame.sort_values("coefficient", ascending=False).reset_index(drop=True)

# ── Figure 1 — ROC curves (test split) ───────────────────────────────────────

def fig_roc_curves(preds: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = preds[preds["split"] == "test"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"dash": "dash", "color": REFERENCE_COLOR},
        name="Chance (AUC = 0.50)",
    ))

    for model in MODEL_ORDER:
        sub = test[test["model_name"] == model]
        if sub.empty:
            continue
        fpr, tpr, _ = roc_curve(sub["target"], sub["predicted_probability"])
        fpr, tpr = _thin_curve(fpr, tpr)
        auc = roc_auc_score(sub["target"], sub["predicted_probability"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line={"width": 3, "color": MODEL_COLOR[model]},
            name=f"{MODEL_LABEL[model]} (AUC = {auc:.3f})",
            hovertemplate="FPR=%{x:.4f}<br>TPR=%{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    # Titles stay on one line at print scale (the exporter enlarges text 2.6x);
    # what they used to explain is in the Supplement caption.
    _apply_standard_layout(
        fig,
        title="ROC curves, test site (armd_utsw)",
        xaxis_title="False positive rate (1 − specificity)",
        yaxis_title="True positive rate",
    )
    fig.update_xaxes(range=[-0.01, 1.01], dtick=0.2)
    fig.update_yaxes(range=[-0.01, 1.01], dtick=0.2)

    static_plotter = _model_static_plotter()
    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig1_roc_curves",
        static_fallback=lambda fmt, path: static_plotter._write_curve_static(
            test, fmt, path, curve_type="roc"
        ),
    )


# ── Figure 2 — PR curves (test split) ────────────────────────────────────────

def fig_pr_curves(preds: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = preds[preds["split"] == "test"]
    prevalence = float(test["target"].mean())

    fig = go.Figure()
    # Below the line at the right: every curve ends on the baseline from above, so the
    # label is clear of them (at the left it ran into the y tick labels).
    fig.add_hline(
        y=prevalence, line_dash="dash", line_color=REFERENCE_COLOR,
        annotation_text=f"No skill (prevalence {prevalence:.2f})",
        annotation_position="bottom right",
    )

    for model in MODEL_ORDER:
        sub = test[test["model_name"] == model]
        if sub.empty:
            continue
        prec, rec, _ = precision_recall_curve(sub["target"], sub["predicted_probability"])
        rec, prec = _thin_curve(rec, prec)
        pr_auc = average_precision_score(sub["target"], sub["predicted_probability"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            line={"width": 3, "color": MODEL_COLOR[model]},
            name=f"{MODEL_LABEL[model]} (PR-AUC = {pr_auc:.3f})",
            hovertemplate="Recall=%{x:.4f}<br>Precision=%{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    _apply_standard_layout(
        fig,
        title="Precision–recall curves, test site (armd_utsw)",
        xaxis_title="Recall (sensitivity)",
        yaxis_title="Precision (PPV)",
    )
    fig.update_xaxes(range=[-0.01, 1.01], dtick=0.2)
    # Precision is always in [0, 1]. Only zoom toward the baseline for a
    # genuinely rare positive class (prevalence <= 5%, so prevalence*20 <= 1.0
    # stays a valid upper bound) — otherwise use the full range, since a
    # near-balanced positive class (as in this run) would otherwise have its
    # whole curve flattened into a sliver at the bottom of the plot.
    if prevalence <= 0.05:
        fig.update_yaxes(range=[-0.001, max(0.025, prevalence * 20)])
    else:
        fig.update_yaxes(range=[-0.01, 1.01])

    static_plotter = _model_static_plotter()
    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig2_pr_curves",
        static_fallback=lambda fmt, path: static_plotter._write_curve_static(
            test, fmt, path, curve_type="pr"
        ),
    )


# ── Figure 3 — Calibration plots (test split, 2 × 2 grid) ────────────────────

def fig_calibration(preds: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = preds[preds["split"] == "test"]
    models = [m for m in MODEL_ORDER if not test[test["model_name"] == m].empty]

    # One row of panels with a shared y-axis: each axis title appears once, so
    # the print-scaled titles cannot run into a neighbouring panel.
    fig = make_subplots(
        rows=1, cols=max(len(models), 1), subplot_titles=[MODEL_LABEL[m] for m in models],
        shared_yaxes=True, horizontal_spacing=0.05,
    )

    for idx, model in enumerate(models):
        row, col = 1, idx + 1
        sub = test[test["model_name"] == model]
        frac_pos, mean_pred = calibration_curve(
            sub["target"].astype(int),
            sub["predicted_probability"],
            n_bins=10,
            strategy="quantile",
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line={"dash": "dash", "color": REFERENCE_COLOR},
                name="Perfect calibration", showlegend=(idx == 0),
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=mean_pred, y=frac_pos, mode="lines+markers",
                line={"width": 2.5, "color": MODEL_COLOR[model]},
                marker={"size": 8},
                name=MODEL_LABEL[model], showlegend=False,
                hovertemplate="Mean predicted=%{x:.3f}<br>Observed rate=%{y:.3f}<extra>"
                              + MODEL_LABEL[model] + "</extra>",
            ),
            row=row, col=col,
        )
        fig.update_xaxes(range=[-0.02, 1.02], dtick=0.5, row=row, col=col)
        fig.update_yaxes(range=[-0.02, 1.02], dtick=0.5, row=row, col=col)
    if models:
        fig.update_yaxes(title_text="Observed event rate", row=1, col=1)
        # Short enough to stay on one line under a single panel at print scale.
        fig.update_xaxes(title_text="Predicted probability", row=1, col=(len(models) + 1) // 2)
        # Plotly draws axis titles above the base font; at 14 px the shared title fits one panel.
        fig.update_xaxes(title_font_size=14, tickfont_size=14)
        fig.update_yaxes(title_font_size=14, tickfont_size=14)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        # Wide enough that each panel holds its title and the shared axis title on one line.
        width=2000,
        height=820,
        title={
            "text": "Calibration, test site (armd_utsw)",
            "x": 0.5, "xanchor": "center", "font": {"size": 20},
        },
        legend={
            "title": {"text": ""}, "orientation": "h", "yanchor": "top",
            "y": -0.1, "xanchor": "center", "x": 0.5, "font": {"size": 13},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"size": 16, "family": "Arial"},
        margin={"l": 80, "r": 40, "t": 100, "b": 80},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_X, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False)

    static_plotter = _model_static_plotter()
    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig3_calibration",
        static_fallback=lambda fmt, path: static_plotter._write_calibration_static(
            test, fmt, path
        ),
    )


# ── Figure 4 — Cross-split ROC-AUC bar chart ─────────────────────────────────

def fig_cross_split(metrics: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    splits = ["train", "validation", "test"]
    models = [m for m in MODEL_ORDER if m in metrics["model_name"].values]

    fig = go.Figure()
    for split in splits:
        sub = metrics[metrics["split"] == split].set_index("model_name")
        vals = [sub.loc[m, "roc_auc"] if m in sub.index else None for m in models]
        fig.add_trace(go.Bar(
            x=[MODEL_LABEL[m] for m in models],
            y=vals,
            name=SPLIT_LABEL[split],
            marker_color=SPLIT_COLOR[split],
            marker_line={"color": "white", "width": 0.8},
            text=[f"{v:.3f}" if v is not None else "" for v in vals],
            textposition="outside",
            textfont={"size": 12, "color": "#344054"},
        ))

    # The chance line is named in the legend: a label drawn on the line sat on the bars.
    fig.add_hline(y=0.5, line_dash="dot", line_color="#888888")
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line={"dash": "dot", "color": "#888888"}, name="Chance (0.50)",
    ))

    _apply_standard_layout(
        fig,
        title="ROC-AUC by model and site",
        yaxis_title="ROC-AUC",
        legend_title="Split",
    )
    fig.update_layout(barmode="group")
    # Bars start at zero so their heights compare honestly.
    fig.update_yaxes(range=[0, 1.1], dtick=0.2)

    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig4_cross_split_roc",
        static_fallback=lambda fmt, path: _write_cross_split_static(
            metrics, fmt, path
        ),
    )


# ── Figure 5 — Threshold selection (development-validation split) ──────────

def fig_threshold(thresh_metrics: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    validation = thresh_metrics[thresh_metrics["split"] == "validation"].copy()
    models = [m for m in MODEL_ORDER if m in validation["model_name"].values]

    metric_specs = [
        ("precision", "Precision (PPV)"),
        ("recall", "Recall (Sensitivity)"),
        ("f1", "F1 Score"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]

    # Panel titles name the metric, so panels carry no y-axis titles of their own
    # (at print scale they ran into the neighbouring panel).
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[label for _, label in metric_specs],
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 13}

    for i, (metric, label) in enumerate(metric_specs):
        row, col = i // 2 + 1, i % 2 + 1
        for model in models:
            sub = validation[validation["model_name"] == model].sort_values("threshold")
            fig.add_trace(
                go.Scatter(
                    x=sub["threshold"], y=sub[metric], mode="lines+markers",
                    line={"width": 2.5, "color": MODEL_COLOR[model]},
                    marker={"size": 7},
                    name=MODEL_LABEL[model],
                    legendgroup=model,
                    showlegend=(i == 0),
                    hovertemplate="Threshold=%{x:.2f}<br>Value=%{y:.3f}<extra>"
                                  + MODEL_LABEL[model] + "</extra>",
                ),
                row=row, col=col,
            )
            selected_mask = sub.get("selected_for_decision", pd.Series(False, index=sub.index)).fillna(False)
            selected = sub[selected_mask]
            if not selected.empty:
                fig.add_trace(
                    go.Scatter(
                        x=selected["threshold"], y=selected[metric], mode="markers",
                        marker={
                            "size": 12, "color": MODEL_COLOR[model], "symbol": "diamond",
                            "line": {"width": 2, "color": "white"},
                        },
                        name=f"{MODEL_LABEL[model]} (selected)",
                        legendgroup=model,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

    # One legend entry explains the diamonds (previously a wrapped subtitle).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker={"size": 12, "symbol": "diamond", "color": "#667085", "line": {"width": 2, "color": "white"}},
        name="Saved decision threshold",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=FIG_WIDTH,
        height=max(FIG_HEIGHT, 1150),
        title={
            "text": "Threshold performance, validation site (armd_ecuh)",
            "x": 0.5, "xanchor": "center", "font": {"size": 20},
        },
        legend={
            "title": {"text": ""}, "orientation": "h", "yanchor": "top",
            "y": -0.22, "xanchor": "center", "x": 0.5, "font": {"size": 13},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"size": 13, "family": "Arial"},
        margin={"l": 70, "r": 30, "t": 100, "b": 260},
    )
    static_plotter = _model_static_plotter()
    static_plotter.style_threshold_axes(fig, metric_specs, cols=2)
    fig.add_annotation(
        text=static_plotter.threshold_selection_note(validation), name="selection_note",
        x=0.5, y=0, xref="paper", yref="paper", yshift=-70,
        xanchor="center", yanchor="top", showarrow=False, font={"size": 10, "color": "#475467"},
    )
    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig5_threshold_analysis",
        static_fallback=lambda fmt, path: static_plotter._write_threshold_static(
            validation, fmt, path, split="development-validation", metric_specs=metric_specs, cols=2
        ),
    )


# ── Figure 6 — LR top-20 feature coefficients ─────────────────────────────────

def fig_lr_coefficients(coef_df: pd.DataFrame, fig_dir: Path) -> dict[str, Path] | None:
    if coef_df.empty:
        print("  [warn] no LR coefficients — skipping figure 6")
        return None

    # top 10 positive + top 10 negative
    top_pos = coef_df.head(10)
    top_neg = coef_df.tail(10).iloc[::-1]
    plot_df = pd.concat([top_pos, top_neg], ignore_index=True)
    plot_df = plot_df.drop_duplicates("feature")

    colors = ["#1F77B4" if c > 0 else "#D62728" for c in plot_df["coefficient"]]
    text = [f"{v:+.3f}" for v in plot_df["coefficient"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["coefficient"], y=plot_df["feature"].map(lambda name: _axis_label(name, limit=46)), orientation="h",
        marker_color=colors,
        marker_line={"color": "white", "width": 0.5},
        text=text, textposition="outside",
        textfont={"size": 12, "color": "#344054"},
        hovertemplate="%{y}<br>Coefficient=%{x:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=0, line_color="#444444", line_width=1)
    # Dummy zero-data traces to render a two-item colour legend, since the
    # real bar trace's colour varies per-row rather than per-series.
    fig.add_trace(go.Bar(
        x=[None], y=[None], marker_color="#1F77B4",
        name="Increases P(downstream testing)",
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None], marker_color="#D62728",
        name="Decreases P(downstream testing)",
    ))

    # The sign's meaning is in the legend and the caption, so the axis title stays one line.
    _apply_standard_layout(
        fig,
        title="Top 10 positive and negative LR coefficients",
        xaxis_title="Coefficient per unit of MaxAbs-scaled predictor (log-OR)",
        margin={"l": 260, "r": 60, "t": 90, "b": 90},
        legend_title="",
    )
    fig.update_layout(yaxis={"autorange": "reversed"})
    x_min = min(0.0, float(plot_df["coefficient"].min()))
    x_max = max(0.0, float(plot_df["coefficient"].max()))
    # Room on both sides for the value printed beyond each bar end; with less, a
    # long negative bar's value ran into the feature names.
    x_pad = max(0.28 * (x_max - x_min), 0.05)
    fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
    fig.update_traces(cliponaxis=False, selector={"type": "bar"})

    return _write_all_formats(
        fig,
        fig_dir / "supp_pred_fig6_lr_coefficients",
        static_fallback=lambda fmt, path: _write_coefficients_static(
            plot_df, fmt, path
        ),
    )


# ── Tables ─────────────────────────────────────────────────────────────────────

def table_model_comparison(metrics: pd.DataFrame, tab_dir: Path) -> Path:
    """Wide-format comparison table: one row per model, one col per split×metric."""
    rows = []
    for model in MODEL_ORDER:
        if model not in metrics["model_name"].values:
            continue
        row = {"Model": MODEL_LABEL[model]}
        for split in ["train", "validation", "test"]:
            sub = metrics[(metrics["model_name"] == model) & (metrics["split"] == split)]
            if sub.empty:
                row[f"ROC-AUC ({split})"] = ""
                continue
            s = sub.iloc[0]
            row[f"ROC-AUC ({split})"] = f"{s['roc_auc']:.3f}"
        # test-split extras
        test_row = metrics[(metrics["model_name"] == model) & (metrics["split"] == "test")]
        if not test_row.empty:
            t = test_row.iloc[0]
            row["PR-AUC (test)"] = f"{t['pr_auc']:.4f}"
            row["Brier score (test)"] = f"{t['brier_score']:.4f}"
            row["Balanced accuracy (test)"] = f"{t['balanced_accuracy']:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = tab_dir / "supp_pred_table1_model_comparison.csv"
    df.to_csv(out, index=False)
    return out


def table_lr_top_features(coef_df: pd.DataFrame, tab_dir: Path) -> Path:
    """Top-20 LR feature coefficients with OR."""
    if coef_df.empty:
        out = tab_dir / "supp_pred_table2_lr_top_features.csv"
        coef_df.to_csv(out, index=False)
        return out

    n = min(20, len(coef_df))
    top = coef_df.head(n // 2)
    bot = coef_df.tail(n // 2).iloc[::-1]
    export = pd.concat([top, bot], ignore_index=True).drop_duplicates("feature")
    export = export[["feature", "coefficient", "odds_ratio"]].copy()
    export["coefficient"] = export["coefficient"].map(lambda x: f"{x:+.4f}")
    export["odds_ratio"] = export["odds_ratio"].map(lambda x: f"{x:.4f}")
    export.columns = ["Feature", "Coefficient (log-OR)", "Odds Ratio"]

    out = tab_dir / "supp_pred_table2_lr_top_features.csv"
    export.to_csv(out, index=False)
    return out


def table_threshold_decision(thresh_metrics: pd.DataFrame, tab_dir: Path) -> Path:
    """LR threshold decision table for the development-validation split."""
    lr_validation = thresh_metrics[
        (thresh_metrics["model_name"] == "logistic_regression")
        & (thresh_metrics["split"] == "validation")
    ].copy()

    # Representative thresholds plus the selected one, which the manuscript table
    # bolds and quotes -- it must be a row of its own, never a neighbour's.
    keep_thresh = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    if "selected_for_decision" in lr_validation.columns:
        keep_thresh |= set(lr_validation.loc[lr_validation["selected_for_decision"].fillna(False).astype(bool), "threshold"].round(4))
    lr_validation = lr_validation[
        lr_validation["threshold"].round(4).isin({round(value, 4) for value in keep_thresh})
    ].sort_values("threshold")

    export = lr_validation[["threshold", "precision", "recall", "f1", "balanced_accuracy",
                             "roc_auc", "brier_score"]].copy()

    def _fmt(s, decimals=3):
        return s.map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")

    export["threshold"] = export["threshold"].map(lambda x: f"{x:.2f}")
    for col in ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "brier_score"]:
        export[col] = _fmt(export[col])

    export.columns = [
        "Threshold", "Precision", "Recall", "F1",
        "Balanced Acc.", "ROC-AUC", "Brier Score",
    ]

    out = tab_dir / "supp_pred_table3_lr_threshold.csv"
    export.to_csv(out, index=False)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def _describe(paths: dict[str, Path] | None) -> str:
    if not paths:
        return "(skipped)"
    return paths["html"].stem + " [" + ", ".join(sorted(paths)) + "]"


def main(organism: str = "ESCHERICHIA COLI", *, hide_figure_titles: bool = False) -> None:
    with title_policy(hide_figure_titles):
        _main(organism)


def _main(organism: str) -> None:
    print(f"\n=== Supplementary prediction module: {organism} ===\n")

    fig_dir = _fig_dir(organism)
    tab_dir = _tab_dir(organism)

    # ── Load artefacts ────────────────────────────────────────────────────────
    print("Loading prediction artefacts …")
    preds = load_predictions(organism)
    print(f"  Predictions: {len(preds):,} rows, {preds['model_name'].nunique()} models")

    print("Loading metrics …")
    metrics = load_metrics(organism)

    print("Loading threshold metrics …")
    thresh_metrics = load_threshold_metrics(organism)

    # Coefficients must describe the same fitted model as the predictions.
    print("Extracting LR feature coefficients …")
    coef_df = load_lr_coefficients(organism)
    if not coef_df.empty:
        print(f"  Top 3 positive: {coef_df['feature'].head(3).tolist()}")
        print(f"  Top 3 negative: {coef_df['feature'].tail(3).tolist()}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures (.html, .png, .svg, .pdf each) …")

    out = fig_roc_curves(preds, fig_dir)
    print(f"  Figure 1 (ROC curves):          {_describe(out)}")

    out = fig_pr_curves(preds, fig_dir)
    print(f"  Figure 2 (PR curves):           {_describe(out)}")

    out = fig_calibration(preds, fig_dir)
    print(f"  Figure 3 (calibration):         {_describe(out)}")

    out = fig_cross_split(metrics, fig_dir)
    print(f"  Figure 4 (cross-split ROC-AUC): {_describe(out)}")

    out = fig_threshold(thresh_metrics, fig_dir)
    print(f"  Figure 5 (threshold analysis):  {_describe(out)}")

    out = fig_lr_coefficients(coef_df, fig_dir)
    print(f"  Figure 6 (LR coefficients):     {_describe(out)}")

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\nGenerating tables …")

    out = table_model_comparison(metrics, tab_dir)
    print(f"  Table 1 (model comparison):  {out.name}")

    out = table_lr_top_features(coef_df, tab_dir)
    print(f"  Table 2 (LR top features):   {out.name}")

    out = table_threshold_decision(thresh_metrics, tab_dir)
    print(f"  Table 3 (LR threshold):      {out.name}")

    print(f"\nAll supplementary prediction outputs written to:\n  {fig_dir}\n  {tab_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supplementary prediction outputs")
    parser.add_argument("--hide-figure-titles", action="store_true", help="Remove global titles/subtitles while preserving panel labels and figure size.")
    parser.add_argument(
        "--env",
        default="hpc",
        help="Environment config selecting the data root and report directories (default: hpc)",
    )
    parser.add_argument(
        "--organism",
        default="ESCHERICHIA COLI",
        help="Organism name (default: ESCHERICHIA COLI)",
    )
    args = parser.parse_args()
    _configure_layout(args.env)
    main(args.organism, hide_figure_titles=args.hide_figure_titles)
