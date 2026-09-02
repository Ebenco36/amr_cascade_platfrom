"""
Supplementary predictive-modelling comparison — publication figures and tables.

Loads saved prediction artefacts (no retraining except LR for coefficients),
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
    python scripts/run_supplementary_prediction.py [--organism ESCHERICHIA_COLI]

The organism string is normalised to lower-snake for path construction; the
default is ESCHERICHIA_COLI.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
EXPORT_FORMATS = ("html", "png", "pdf")

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


def _org_slug(organism: str) -> str:
    return organism.lower().replace(" ", "_")


def _artifact_dir(organism: str) -> Path:
    return (
        PROJECT_ROOT
        / "data/artifacts/modeling/downstream_testing/combined/organisms"
        / _org_slug(organism)
        / "site__all_models/full"
    )


def _feat_path(organism: str) -> Path:
    return (
        PROJECT_ROOT
        / "data/features/combined/organisms"
        / _org_slug(organism)
        / "model_ready_pair_features.parquet"
    )


def _fig_dir(organism: str) -> Path:
    d = (
        PROJECT_ROOT
        / "outputs/figures/combined/organisms"
        / _org_slug(organism)
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tab_dir(organism: str) -> Path:
    d = (
        PROJECT_ROOT
        / "outputs/tables/combined/organisms"
        / _org_slug(organism)
    )
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


def _write_all_formats(fig: go.Figure, out_stem: Path) -> dict[str, Path]:
    """Write .html (interactive), .png and .pdf (static, via kaleido)."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    width = int(fig.layout.width or FIG_WIDTH)
    height = int(fig.layout.height or FIG_HEIGHT)
    paths: dict[str, Path] = {}
    for fmt in EXPORT_FORMATS:
        path = out_stem.with_suffix(f".{fmt}")
        if fmt == "html":
            pio.write_html(fig, path, include_plotlyjs="cdn", full_html=True)
        else:
            fig.write_image(path, format=fmt, width=width, height=height, scale=2)
        paths[fmt] = path
    return paths


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(organism: str) -> pd.DataFrame:
    """Concatenate saved per-model prediction parquets."""
    art = _artifact_dir(organism)
    dfs = []
    for model in MODEL_ORDER:
        path = art / f"{model}_predictions.parquet"
        if not path.exists():
            print(f"  [warn] missing {path.name} — skipping {model}")
            continue
        df = pd.read_parquet(path, columns=["target", "split", "predicted_probability"])
        df["model_name"] = model
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No prediction parquets found in {art}")
    return pd.concat(dfs, ignore_index=True)


def load_metrics(organism: str) -> pd.DataFrame:
    """Load the pre-computed per-split metrics parquet."""
    path = (
        PROJECT_ROOT
        / "data/artifacts/modeling/downstream_testing/combined/organisms"
        / _org_slug(organism)
        / "site__all_models/metrics.parquet"
    )
    return pd.read_parquet(path)


def load_threshold_metrics(organism: str) -> pd.DataFrame:
    """Load threshold-sweep metrics parquet."""
    path = (
        PROJECT_ROOT
        / "data/artifacts/modeling/downstream_testing/combined/organisms"
        / _org_slug(organism)
        / "site__all_models/threshold_metrics.parquet"
    )
    return pd.read_parquet(path)


# ── LR coefficient extraction ─────────────────────────────────────────────────

_ID_COLS = {
    "anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered",
    "organism", "source_site", "ordering_mode", "culture_description",
    "upstream_antibiotic", "downstream_antibiotic", "pair_direction",
    "target", "downstream_tested", "downstream_eligible", "downstream_intrinsic_resistance",
    "was_positive", "was_positive_numeric",
}

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
    "history_prior_same_organism_any_30d": "Same organism prior (30d)",
    "history_prior_same_organism_any_90d": "Same organism prior (90d)",
    "history_prior_same_organism_any_365d": "Same organism prior (365d)",
    "comorbidity_count": "Total comorbidity count",
}


def _clean_feature_name(raw: str) -> str:
    if raw in _FEATURE_RENAME:
        return _FEATURE_RENAME[raw]
    if raw.startswith("comorb_"):
        name = raw[len("comorb_"):]
        name = name.replace("_", " ").title()
        # shorten very long names
        if len(name) > 45:
            name = name[:43] + "…"
        return f"Comorbidity: {name}"
    return raw.replace("_", " ").title()


def fit_lr_for_coefficients(organism: str) -> pd.DataFrame:
    """
    Re-fit LR on the training split to extract feature coefficients.
    Returns a DataFrame with columns [feature, coefficient, odds_ratio].
    """
    feat_path = _feat_path(organism)
    if not feat_path.exists():
        print(f"  [warn] feature matrix not found at {feat_path} — skipping LR coefficients")
        return pd.DataFrame(columns=["feature", "coefficient", "odds_ratio"])

    # Read only the schema first (metadata only, no data) to build a column
    # allowlist, then use it together with row-group predicate pushdown on
    # source_site. This file can be very large (hundreds of millions of rows
    # across all three sites); a bare pd.read_parquet(feat_path) loads every
    # column for every site into memory before any filtering happens, which
    # is both far more I/O and far more memory than the training-split fit
    # actually needs.
    parquet_file = pq.ParquetFile(feat_path)
    schema_cols = parquet_file.schema.names
    needed_cols = [c for c in schema_cols if c in ("source_site", "target") or c not in _ID_COLS]
    total_rows = parquet_file.metadata.num_rows
    file_size_gb = feat_path.stat().st_size / 1e9

    print(
        f"  Feature matrix on disk: {total_rows:,} rows (all sites), "
        f"{len(schema_cols)} columns, {file_size_gb:.1f} GB compressed"
    )
    # This file is highly compressed on disk (hundreds of mostly-zero
    # comorbidity flags), but pandas decompresses to a DENSE in-memory table:
    # even after column pruning and the source_site filter, materializing
    # hundreds of columns across tens of millions of rows can expand far past
    # the on-disk size and exhaust the job's memory. This table is only used
    # to illustrate LR coefficients, not the primary result, so a large random
    # sample gives statistically stable coefficients at a small, predictable
    # memory footprint — stream row-group batches and keep each row with a
    # fixed probability, rather than materializing the full training split
    # before sampling.
    dataset = ds.dataset(feat_path, format="parquet")
    site_filter = ds.field("source_site") == "armd"
    armd_row_count = dataset.count_rows(filter=site_filter)

    target_sample_rows = 1_000_000
    keep_prob = min(1.0, target_sample_rows / armd_row_count) if armd_row_count else 1.0

    print(
        f"  Loading feature matrix for LR coefficient extraction "
        f"({len(needed_cols)}/{len(schema_cols)} columns; {armd_row_count:,} armd rows on disk, "
        f"sampling ~{keep_prob:.2%} ≈ {min(armd_row_count, target_sample_rows):,} rows) …"
    )

    rng = np.random.default_rng(42)
    sampled_batches = []
    for batch in dataset.to_batches(columns=needed_cols, filter=site_filter):
        if keep_prob >= 1.0:
            sampled_batches.append(batch)
            continue
        keep_mask = pa.array(rng.random(batch.num_rows) < keep_prob)
        kept = batch.filter(keep_mask)
        if kept.num_rows:
            sampled_batches.append(kept)

    # Each batch already carries the needed_cols schema from to_batches()
    # above, so from_batches() can infer it without an explicit schema= arg.
    train = pa.Table.from_batches(sampled_batches).to_pandas()
    # Probabilistic thinning gives an approximate, not exact, sample size;
    # hard-cap in case it overshoots.
    if len(train) > target_sample_rows:
        train = train.sample(n=target_sample_rows, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(train):,} training rows for coefficient extraction.")

    feature_cols = [c for c in train.columns if c not in _ID_COLS]
    # drop columns constant in the training split (uninformative)
    non_const = [c for c in feature_cols if train[c].nunique() > 1]

    cat_cols = [c for c in non_const if str(train[c].dtype) in ("category", "object", "string[python]", "string")]
    num_cols = [c for c in non_const if c not in cat_cols]

    X_train = train[non_const]
    y_train = train["target"].astype(int)

    # Sparse output (matching the production preprocessor in
    # src/amr_cascade_platform/modeling/estimators/preprocessing.py) rather than
    # dense: at this row count, a dense one-hot matrix risks tens of GB of memory
    # for no benefit — lbfgs accepts sparse X natively.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                cat_cols,
            ),
            (
                "num",
                SimpleImputer(strategy="constant", fill_value=0.0),
                num_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    # Hyperparameters match configs/base/modeling.yaml's modeling.logistic_regression
    # block exactly, so this refit is comparable to the LR row in Table S8/S9 (both
    # trace to the same estimator configuration, differing only in that this refit
    # additionally exposes coef_ for interpretability).
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    pipe = Pipeline([("pre", preprocessor), ("lr", model)])

    print(f"  Training rows: {len(X_train):,}  |  raw columns before encoding: {len(non_const)}")
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    fitted_shape = preprocessor.transform(X_train.iloc[:1]).shape[1]
    print(f"  Fit complete in {elapsed:.1f}s  |  encoded design matrix columns: {fitted_shape}")

    # Extract feature names. If cat_cols is empty (no categorical column
    # survived the non-constant filter), the ColumnTransformer never fits the
    # "cat" branch's OneHotEncoder, and get_feature_names_out() would raise
    # NotFittedError on it.
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    else:
        cat_names = []
    all_names = cat_names + num_cols

    coefs = pipe.named_steps["lr"].coef_[0]
    df = pd.DataFrame({"feature_raw": all_names, "coefficient": coefs})
    df["odds_ratio"] = np.exp(df["coefficient"])
    df["feature"] = df["feature_raw"].map(_clean_feature_name)
    df = df.sort_values("coefficient", ascending=False).reset_index(drop=True)
    return df[["feature", "coefficient", "odds_ratio"]]


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
        auc = roc_auc_score(sub["target"], sub["predicted_probability"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line={"width": 3, "color": MODEL_COLOR[model]},
            name=f"{MODEL_LABEL[model]} (AUC = {auc:.3f})",
            hovertemplate="FPR=%{x:.4f}<br>TPR=%{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    _apply_standard_layout(
        fig,
        title="Receiver Operating Characteristic (Test Site, armd_utsw)",
        xaxis_title="False Positive Rate (1 − Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
    )
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01])

    return _write_all_formats(fig, fig_dir / "supp_pred_fig1_roc_curves")


# ── Figure 2 — PR curves (test split) ────────────────────────────────────────

def fig_pr_curves(preds: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = preds[preds["split"] == "test"]
    prevalence = float(test["target"].mean())

    fig = go.Figure()
    fig.add_hline(
        y=prevalence, line_dash="dash", line_color=REFERENCE_COLOR,
        annotation_text=f"No-skill baseline (prevalence = {prevalence:.4f})",
        annotation_position="top left",
    )

    for model in MODEL_ORDER:
        sub = test[test["model_name"] == model]
        if sub.empty:
            continue
        prec, rec, _ = precision_recall_curve(sub["target"], sub["predicted_probability"])
        pr_auc = average_precision_score(sub["target"], sub["predicted_probability"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            line={"width": 3, "color": MODEL_COLOR[model]},
            name=f"{MODEL_LABEL[model]} (PR-AUC = {pr_auc:.4f})",
            hovertemplate="Recall=%{x:.4f}<br>Precision=%{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    subtitle = (
        f"PR-AUC is bounded by prevalence (≈{prevalence:.1%}); "
        "all models exceed the no-skill baseline"
    )
    _apply_standard_layout(
        fig,
        title=f"Precision-Recall Curves (Test Site, armd_utsw)<br><sup>{subtitle}</sup>",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        margin={"l": 80, "r": 40, "t": 110, "b": 90},
    )
    fig.update_xaxes(range=[-0.01, 1.01])
    # Precision is always in [0, 1]. Only zoom toward the baseline for a
    # genuinely rare positive class (prevalence <= 5%, so prevalence*20 <= 1.0
    # stays a valid upper bound) — otherwise use the full range, since a
    # near-balanced positive class (as in this run) would otherwise have its
    # whole curve flattened into a sliver at the bottom of the plot.
    if prevalence <= 0.05:
        fig.update_yaxes(range=[-0.001, max(0.025, prevalence * 20)])
    else:
        fig.update_yaxes(range=[-0.01, 1.01])

    return _write_all_formats(fig, fig_dir / "supp_pred_fig2_pr_curves")


# ── Figure 3 — Calibration plots (test split, 2 × 2 grid) ────────────────────

def fig_calibration(preds: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = preds[preds["split"] == "test"]
    models = [m for m in MODEL_ORDER if not test[test["model_name"] == m].empty]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[MODEL_LABEL[m] for m in models])

    for idx, model in enumerate(models):
        row, col = idx // 2 + 1, idx % 2 + 1
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
        fig.update_xaxes(title_text="Mean predicted probability", row=row, col=col)
        fig.update_yaxes(title_text="Observed event rate", row=row, col=col)

    for idx in range(len(models), 4):
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        title={
            "text": "Calibration Reliability Plots (Test Site, armd_utsw)",
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

    return _write_all_formats(fig, fig_dir / "supp_pred_fig3_calibration")


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

    # Compute the actual train-test gap per model rather than asserting a
    # fixed winner in the title: which family has the smallest gap is a
    # property of the data, not something to hardcode (it need not be LR).
    gap_by_model = {}
    for m in models:
        sub = metrics[metrics["model_name"] == m].set_index("split")
        if "train" in sub.index and "test" in sub.index:
            gap_by_model[m] = float(sub.loc["train", "roc_auc"] - sub.loc["test", "roc_auc"])
    if gap_by_model:
        smallest = min(gap_by_model, key=gap_by_model.get)
        subtitle = (
            f"Train-test gap ranges {min(gap_by_model.values()):.3f}-"
            f"{max(gap_by_model.values()):.3f} across families; smallest for {MODEL_LABEL[smallest]}"
        )
    else:
        subtitle = "Train, validation, and test ROC-AUC by model family"

    fig.add_hline(
        y=0.5, line_dash="dot", line_color="#888888",
        annotation_text="Chance (0.50)", annotation_position="bottom right",
    )

    _apply_standard_layout(
        fig,
        title=f"ROC-AUC Across Train / Validation / Test Splits<br><sup>{subtitle}</sup>",
        yaxis_title="ROC-AUC",
        margin={"l": 80, "r": 40, "t": 110, "b": 90},
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0.4, 1.08])

    return _write_all_formats(fig, fig_dir / "supp_pred_fig4_cross_split_roc")


# ── Figure 5 — Threshold sensitivity (test split, all models) ─────────────────

def fig_threshold(thresh_metrics: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    test = thresh_metrics[thresh_metrics["split"] == "test"].copy()
    models = [m for m in MODEL_ORDER if m in test["model_name"].values]

    metric_specs = [
        ("precision", "Precision (PPV)"),
        ("recall", "Recall (Sensitivity)"),
        ("f1", "F1 Score"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in metric_specs])

    for i, (metric, label) in enumerate(metric_specs):
        row, col = i // 2 + 1, i % 2 + 1
        for model in models:
            sub = test[test["model_name"] == model].sort_values("threshold")
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
        fig.update_xaxes(title_text="Decision threshold", range=[0.04, 0.91], row=row, col=col)
        fig.update_yaxes(title_text=label, row=row, col=col)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=1450,
        height=FIG_HEIGHT,
        title={
            "text": (
                "Threshold Sensitivity Analysis (Test Site, armd_utsw)"
                "<br><sup>Diamond markers show the threshold selected on the validation split</sup>"
            ),
            "x": 0.5, "xanchor": "center", "font": {"size": 20},
        },
        legend={
            "title": {"text": "Model"}, "orientation": "h", "yanchor": "top",
            "y": -0.08, "xanchor": "center", "x": 0.5, "font": {"size": 13},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"size": 16, "family": "Arial"},
        margin={"l": 70, "r": 30, "t": 110, "b": 90},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_X, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False)

    return _write_all_formats(fig, fig_dir / "supp_pred_fig5_threshold_analysis")


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
        x=plot_df["coefficient"], y=plot_df["feature"], orientation="h",
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

    _apply_standard_layout(
        fig,
        title=(
            "Top 10 Positive and Top 10 Negative LR Coefficients"
            "<br><sup>Trained on armd (Stanford) site, full feature set</sup>"
        ),
        xaxis_title="Logistic regression coefficient (log-OR); positive increases downstream-testing probability",
        margin={"l": 260, "r": 60, "t": 110, "b": 90},
        legend_title="",
    )
    fig.update_layout(yaxis={"autorange": "reversed"})
    x_min = min(0.0, float(plot_df["coefficient"].min()))
    x_max = max(0.0, float(plot_df["coefficient"].max()))
    x_pad = max(0.12 * (x_max - x_min), 0.01)
    fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])

    return _write_all_formats(fig, fig_dir / "supp_pred_fig6_lr_coefficients")


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
    """LR threshold decision table for the test split at representative thresholds."""
    lr_test = thresh_metrics[
        (thresh_metrics["model_name"] == "logistic_regression")
        & (thresh_metrics["split"] == "test")
    ].copy()

    keep_thresh = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    lr_test = lr_test[lr_test["threshold"].isin(keep_thresh)].sort_values("threshold")

    export = lr_test[["threshold", "precision", "recall", "f1", "balanced_accuracy",
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


def main(organism: str = "ESCHERICHIA COLI") -> None:
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

    # ── LR coefficients (requires refit on training split) ────────────────────
    print("Extracting LR feature coefficients …")
    coef_df = fit_lr_for_coefficients(organism)
    if not coef_df.empty:
        print(f"  Top 3 positive: {coef_df['feature'].head(3).tolist()}")
        print(f"  Top 3 negative: {coef_df['feature'].tail(3).tolist()}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures (.html, .png, .pdf each) …")

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
    parser.add_argument(
        "--organism",
        default="ESCHERICHIA COLI",
        help="Organism name (default: ESCHERICHIA COLI)",
    )
    args = parser.parse_args()
    main(args.organism)
