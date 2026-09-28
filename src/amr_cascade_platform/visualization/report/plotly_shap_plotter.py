"""Publication figures for adjusted-model SHAP attributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from amr_cascade_platform.core.utils.text import safe_feature_name
from amr_cascade_platform.visualization.report.plotly_exporter import PlotlyFigureExporter


class PlotlyShapPlotter:
    """Export pair-specific global importance and local-attribution summaries."""

    def __init__(
        self,
        exporter: PlotlyFigureExporter,
        template: str,
        width: int,
        height: int,
        top_n: int = 15,
    ) -> None:
        self._exporter = exporter
        self._template = template
        self._width = width
        self._height = height
        self._top_n = top_n

    _PAIR_KEYS = ["upstream_antibiotic", "downstream_antibiotic"]
    # Feature domains as DownstreamTestingShapExplainer.feature_domain names them.
    _DOMAIN_LABELS = {
        "upstream_result": "Upstream result",
        "site": "Site",
        "calendar_time": "Calendar year",
        "specimen_type": "Specimen type",
        "ordering_context": "Ordering context",
        "care_setting": "Care setting (ICU, ED)",
        "prior_antibiotic_exposure": "Prior antibiotic exposure",
        "demographics": "Age and sex",
        "prior_organism_history": "Prior same-genus infection",
        "comorbidity": "Comorbidity",
        "area_deprivation": "Area deprivation",
        "nursing_home": "Nursing-home residence",
        "prior_procedure": "Prior procedures",
        "source_availability": "Covariate-availability indicators",
        "other": "Other",
    }

    @classmethod
    def rank_pairs(
        cls,
        global_importance: pd.DataFrame,
        domain_by_pair: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Pairs ordered by how much the upstream result moves the fitted log-odds.

        The ranking key is the upstream result's mean absolute SHAP value (one
        feature, so no grouping is involved; ties by pair name).
        ``upstream_result_share`` is that domain's share of the pair's total
        attribution from ``domain_by_pair`` (see domain_summary).
        """
        columns = [*cls._PAIR_KEYS, "upstream_result_mean_absolute_shap", "upstream_result_share", "rank"]
        if global_importance.empty:
            return pd.DataFrame(columns=columns)
        pairs = global_importance.loc[:, cls._PAIR_KEYS].drop_duplicates()
        upstream = (
            global_importance.loc[global_importance["feature_domain"].eq("upstream_result")]
            .groupby(cls._PAIR_KEYS, observed=True, sort=False)["mean_absolute_shap"].sum()
            .rename("upstream_result_mean_absolute_shap").reset_index()
        )
        ranking = pairs.merge(upstream, on=cls._PAIR_KEYS, how="left")
        ranking["upstream_result_mean_absolute_shap"] = ranking["upstream_result_mean_absolute_shap"].fillna(0.0)
        if domain_by_pair is not None and not domain_by_pair.empty:
            share = domain_by_pair.loc[
                domain_by_pair["feature_domain"].eq("upstream_result"), [*cls._PAIR_KEYS, "pair_share"]
            ].rename(columns={"pair_share": "upstream_result_share"})
            ranking = ranking.merge(share, on=cls._PAIR_KEYS, how="left")
        else:
            ranking["upstream_result_share"] = np.nan
        ranking = ranking.sort_values(
            ["upstream_result_mean_absolute_shap", *cls._PAIR_KEYS],
            ascending=[False, True, True], kind="mergesort",
        ).reset_index(drop=True)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        return ranking.loc[:, columns]

    @classmethod
    def domain_summary(cls, domain_importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Each pair's attribution split by feature domain, and that split summarised over pairs.

        ``domain_importance`` is the explainer's per-pair domain table, in which a
        domain's SHAP values were summed within each explained row before
        averaging. A pair's domain share is that domain's mean |SHAP| over the
        sum across the pair's domains; a domain absent from a pair's model counts
        as zero, so every pair weighs equally.
        """
        by_pair_columns = [*cls._PAIR_KEYS, "feature_domain", "domain_mean_absolute_shap", "pair_share"]
        overall_columns = [
            "feature_domain", "domain_label", "pairs", "mean_share", "median_share",
            "q25_share", "q75_share", "pairs_where_largest",
        ]
        required = {*cls._PAIR_KEYS, "feature_domain", "mean_absolute_shap"}
        if domain_importance.empty or not required.issubset(domain_importance.columns):
            return pd.DataFrame(columns=by_pair_columns), pd.DataFrame(columns=overall_columns)
        by_pair = (
            domain_importance.groupby([*cls._PAIR_KEYS, "feature_domain"], observed=True, sort=True)["mean_absolute_shap"]
            .sum().rename("domain_mean_absolute_shap").reset_index()
        )
        pair_total = by_pair.groupby(cls._PAIR_KEYS, observed=True)["domain_mean_absolute_shap"].transform("sum")
        by_pair["pair_share"] = by_pair["domain_mean_absolute_shap"] / pair_total.where(pair_total > 0)
        shares = by_pair.pivot_table(
            index=cls._PAIR_KEYS, columns="feature_domain", values="pair_share", aggfunc="sum", fill_value=0.0, observed=True,
        )
        largest = shares.idxmax(axis=1).value_counts()
        overall = pd.DataFrame(
            {
                "feature_domain": shares.columns.astype(str),
                "domain_label": [cls._DOMAIN_LABELS.get(str(domain), str(domain)) for domain in shares.columns],
                "pairs": len(shares),
                "mean_share": shares.mean().to_numpy(),
                "median_share": shares.median().to_numpy(),
                "q25_share": shares.quantile(0.25).to_numpy(),
                "q75_share": shares.quantile(0.75).to_numpy(),
                "pairs_where_largest": largest.reindex(shares.columns, fill_value=0).to_numpy(),
            }
        ).sort_values("mean_share", ascending=False, kind="mergesort").reset_index(drop=True)
        return by_pair.loc[:, by_pair_columns], overall.loc[:, overall_columns]

    def export_domain_summary(
        self,
        overall: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
    ) -> dict[str, Path]:
        """One figure: where the adjusted models' attribution comes from, across all pairs."""
        if overall.empty:
            return {}
        return self._exporter.write(
            self._domain_summary_figure(overall),
            output_dir / "figure_adjusted_model_shap_domain_summary",
            formats,
            static_fallback=lambda fmt, path, data=overall.copy(): self._write_domain_summary_static(data, fmt, path),
            prefer_static_fallback=True,
        )

    def _domain_summary_figure(self, overall: pd.DataFrame) -> go.Figure:
        data = overall.sort_values("mean_share", ascending=True, kind="mergesort")
        pairs = int(data["pairs"].iloc[0])
        figure = go.Figure(
            go.Bar(
                x=100 * data["mean_share"],
                y=data["domain_label"],
                orientation="h",
                marker_color="#2B6F77",
                error_x={
                    "type": "data", "symmetric": False,
                    "array": 100 * (data["q75_share"] - data["mean_share"]).clip(lower=0),
                    "arrayminus": 100 * (data["mean_share"] - data["q25_share"]).clip(lower=0),
                    "color": "#667085",
                },
                customdata=np.column_stack([100 * data["median_share"], data["pairs_where_largest"]]),
                hovertemplate=(
                    "%{y}<br>Mean share=%{x:.1f}%<br>Median share=%{customdata[0]:.1f}%"
                    "<br>Largest domain in %{customdata[1]} pairs<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            template=self._template,
            width=self._width,
            height=max(self._height, 360 + 38 * len(data)),
            title={"text": f"Adjusted-model attribution by feature domain ({pairs} pairs)", "x": 0.5},
            xaxis_title="Share of the pair's mean |SHAP| (%)",
            margin={"l": 260, "r": 60, "t": 90, "b": 90},
            showlegend=False,
        )
        return figure

    def _write_domain_summary_static(self, overall: pd.DataFrame, fmt: str, path: Path) -> None:
        data = overall.sort_values("mean_share", ascending=True, kind="mergesort")
        pairs = int(data["pairs"].iloc[0])
        mean = 100 * data["mean_share"].to_numpy(dtype=float)
        lower = np.clip(mean - 100 * data["q25_share"].to_numpy(dtype=float), 0, None)
        upper = np.clip(100 * data["q75_share"].to_numpy(dtype=float) - mean, 0, None)
        fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(data) + 2.0)))
        ax.barh(data["domain_label"], mean, color="#2B6F77", xerr=[lower, upper], ecolor="#667085", capsize=3)
        ax.set_xlabel("Share of the pair's mean |SHAP| (%)")
        ax.set_title(f"Adjusted-model attribution by feature domain ({pairs} pairs)", fontweight="bold", pad=12)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        fig.text(
            0.5, 0.015,
            "Bars: mean share across pairs; whiskers: interquartile range. Model attribution on the log-odds scale, not a causal effect.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def export_pair_suite(
        self,
        global_importance: pd.DataFrame,
        sampled_values: pd.DataFrame,
        output_dir: Path,
        formats: tuple[str, ...],
        pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Path]:
        """Importance and beeswarm figures for ``pairs`` (default: every pair), in the order given."""
        required = set(self._PAIR_KEYS)
        if global_importance.empty or not required.issubset(global_importance.columns):
            return {}
        outputs: dict[str, Path] = {}
        importance_by_pair = {
            (str(up), str(down)): frame
            for (up, down), frame in global_importance.groupby(self._PAIR_KEYS, observed=True, sort=True)
        }
        # Split the sampled values once: filtering the whole table per pair costs
        # rows x pairs, tens of billions of comparisons on the full cohort.
        values_by_pair = (
            {
                (str(up), str(down)): frame
                for (up, down), frame in sampled_values.groupby(self._PAIR_KEYS, observed=True, sort=False)
            }
            if not sampled_values.empty and required.issubset(sampled_values.columns)
            else {}
        )
        selected = list(importance_by_pair) if pairs is None else [
            (str(up), str(down)) for up, down in pairs if (str(up), str(down)) in importance_by_pair
        ]
        for upstream, downstream in selected:
            importance = importance_by_pair[(upstream, downstream)]
            slug = f"{safe_feature_name(str(upstream))}_to_{safe_feature_name(str(downstream))}"
            outputs.update(
                self._exporter.write(
                    self._importance_figure(importance, str(upstream), str(downstream)),
                    output_dir / f"figure_adjusted_model_shap_importance__{slug}",
                    formats,
                    static_fallback=lambda fmt, path, data=importance.copy(), up=str(upstream), down=str(downstream): self._write_importance_static(
                        data, up, down, fmt, path
                    ),
                    prefer_static_fallback=True,
                )
            )
            pair_values = values_by_pair.get((upstream, downstream), pd.DataFrame())
            if not pair_values.empty:
                outputs.update(
                    self._exporter.write(
                        self._beeswarm_figure(
                            importance,
                            pair_values,
                            str(upstream),
                            str(downstream),
                        ),
                        output_dir / f"figure_adjusted_model_shap_beeswarm__{slug}",
                        formats,
                        static_fallback=lambda fmt, path, imp=importance.copy(), data=pair_values.copy(), up=str(upstream), down=str(downstream): self._write_beeswarm_static(
                            imp, data, up, down, fmt, path
                        ),
                        prefer_static_fallback=True,
                    )
                )
        return outputs

    def _importance_figure(
        self,
        importance: pd.DataFrame,
        upstream: str,
        downstream: str,
    ) -> go.Figure:
        plot_data = (
            importance.sort_values("mean_absolute_shap", ascending=False, kind="stable")
            .head(self._top_n)
            .sort_values("mean_absolute_shap", ascending=True, kind="stable")
            .copy()
        )
        plot_data["feature_label"] = plot_data["feature"].map(self._feature_label)
        figure = go.Figure(
            go.Bar(
                x=plot_data["mean_absolute_shap"],
                y=plot_data["feature_label"],
                orientation="h",
                marker_color="#2B6F77",
                customdata=np.column_stack(
                    [plot_data["feature_domain"], plot_data["mean_signed_shap"]]
                ),
                hovertemplate=(
                    "%{y}<br>Mean |SHAP|=%{x:.3f} log-odds"
                    "<br>Mean signed SHAP=%{customdata[1]:.3f}"
                    "<br>Domain=%{customdata[0]}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            template=self._template,
            width=self._width,
            height=max(self._height, 520 + 30 * len(plot_data)),
            title={
                "text": (
                    "Adjusted-Model SHAP Importance"
                    f"<br><sup>{upstream} → {downstream}; mean absolute contribution to fitted log-odds</sup>"
                ),
                "x": 0.5,
            },
            xaxis_title="Mean absolute SHAP value (log-odds)",
            yaxis_title="Model feature",
            margin={"l": 290, "r": 80, "t": 125, "b": 95},
            showlegend=False,
        )
        return figure

    def _write_importance_static(
        self,
        importance: pd.DataFrame,
        upstream: str,
        downstream: str,
        fmt: str,
        path: Path,
    ) -> None:
        plot_data = (
            importance.sort_values("mean_absolute_shap", ascending=False, kind="stable")
            .head(self._top_n)
            .sort_values("mean_absolute_shap", ascending=True, kind="stable")
        )
        labels = [self._feature_label(feature) for feature in plot_data["feature"]]
        fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.38 * len(plot_data) + 2.2)))
        ax.barh(labels, plot_data["mean_absolute_shap"], color="#2B6F77")
        ax.set_xlabel("Mean absolute SHAP value (log-odds)")
        ax.set_ylabel("Model feature")
        ax.set_title(
            f"Adjusted-Model SHAP Importance\n{upstream} → {downstream}",
            fontweight="bold",
            pad=14,
        )
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        fig.text(
            0.5,
            0.015,
            "Mean absolute contribution to fitted downstream-testing log-odds; model attribution, not a causal effect.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _beeswarm_figure(
        self,
        importance: pd.DataFrame,
        sampled_values: pd.DataFrame,
        upstream: str,
        downstream: str,
    ) -> go.Figure:
        top_features = (
            importance.sort_values("mean_absolute_shap", ascending=False, kind="stable")
            .head(min(self._top_n, 12))["feature"]
            .tolist()
        )
        plot_data = sampled_values.loc[sampled_values["feature"].isin(top_features)].copy()
        ordered_features = list(reversed(top_features))
        feature_position = {feature: index for index, feature in enumerate(ordered_features)}
        # Deterministic vertical jitter preserves density without introducing a
        # visually different scientific figure on each reporting run.
        jitter_code = (
            plot_data["sample_row"].astype(int) * 1103515245
            + plot_data["feature"].map(lambda value: sum(map(ord, str(value))))
        ) % 1000
        plot_data["y_position"] = (
            plot_data["feature"].map(feature_position).astype(float)
            + (jitter_code.astype(float) / 999.0 - 0.5) * 0.54
        )
        color_limit = float(np.nanquantile(np.abs(plot_data["feature_value"]), 0.98))
        color_limit = color_limit if np.isfinite(color_limit) and color_limit > 0 else 1.0
        figure = go.Figure(
            go.Scattergl(
                x=plot_data["shap_value"],
                y=plot_data["y_position"],
                mode="markers",
                marker={
                    "size": 6,
                    "opacity": 0.58,
                    "color": plot_data["feature_value"],
                    "colorscale": "RdBu_r",
                    "cmin": -color_limit,
                    "cmax": color_limit,
                    "colorbar": {"title": "Encoded/scaled<br>model-input value"},
                    "line": {"width": 0},
                },
                customdata=np.column_stack(
                    [plot_data["feature"], plot_data["feature_value"]]
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>SHAP=%{x:.3f} log-odds"
                    "<br>Encoded/scaled model-input value=%{customdata[1]:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_vline(x=0, line_color="#64748B", line_width=1, line_dash="dot")
        figure.update_layout(
            template=self._template,
            width=self._width,
            height=max(self._height, 560 + 34 * len(ordered_features)),
            title={
                "text": (
                    "Adjusted-Model SHAP Distribution"
                    f"<br><sup>{upstream} → {downstream}; positive values increase fitted downstream-testing log-odds</sup>"
                ),
                "x": 0.5,
            },
            xaxis_title="SHAP value (log-odds)",
            yaxis={
                "title": "Model feature",
                "tickmode": "array",
                "tickvals": list(range(len(ordered_features))),
                "ticktext": [self._feature_label(feature) for feature in ordered_features],
                "range": [-0.8, len(ordered_features) - 0.2],
            },
            margin={"l": 290, "r": 150, "t": 125, "b": 95},
        )
        return figure

    def _write_beeswarm_static(
        self,
        importance: pd.DataFrame,
        sampled_values: pd.DataFrame,
        upstream: str,
        downstream: str,
        fmt: str,
        path: Path,
    ) -> None:
        top_features = (
            importance.sort_values("mean_absolute_shap", ascending=False, kind="stable")
            .head(min(self._top_n, 12))["feature"]
            .tolist()
        )
        plot_data = sampled_values.loc[sampled_values["feature"].isin(top_features)].copy()
        ordered_features = list(reversed(top_features))
        feature_position = {feature: index for index, feature in enumerate(ordered_features)}
        jitter_code = (
            plot_data["sample_row"].astype(int) * 1103515245
            + plot_data["feature"].map(lambda value: sum(map(ord, str(value))))
        ) % 1000
        y_position = (
            plot_data["feature"].map(feature_position).astype(float)
            + (jitter_code.astype(float) / 999.0 - 0.5) * 0.54
        )
        color_limit = float(np.nanquantile(np.abs(plot_data["feature_value"]), 0.98))
        color_limit = color_limit if np.isfinite(color_limit) and color_limit > 0 else 1.0

        fig, ax = plt.subplots(figsize=(11.0, max(6.0, 0.48 * len(ordered_features) + 2.4)))
        points = ax.scatter(
            plot_data["shap_value"],
            y_position,
            c=plot_data["feature_value"],
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            s=15,
            alpha=0.55,
            linewidths=0,
            rasterized=fmt in {"pdf", "svg"},
        )
        ax.axvline(0, color="#64748B", linewidth=1, linestyle=":")
        ax.set_yticks(range(len(ordered_features)))
        ax.set_yticklabels([self._feature_label(feature) for feature in ordered_features])
        ax.set_ylim(-0.8, len(ordered_features) - 0.2)
        ax.set_xlabel("SHAP value (log-odds)")
        ax.set_ylabel("Model feature")
        ax.set_title(
            f"Adjusted-Model SHAP Distribution\n{upstream} → {downstream}",
            fontweight="bold",
            pad=14,
        )
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        colorbar = fig.colorbar(points, ax=ax, pad=0.02)
        colorbar.set_label("Encoded/scaled model-input value")
        fig.text(
            0.5,
            0.015,
            "Positive SHAP values increase fitted downstream-testing log-odds; attribution is not causal importance.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _feature_label(feature: object) -> str:
        text = str(feature)
        exact = {
            "upstream_positive": "Upstream Resistant",
            "cov_icu_status": "ICU Status",
            "cov_er_status": "Emergency Department Status",
            "cov_prior_abx_any_90d": "Prior Antibiotic Exposure (90 Days)",
            "cov_prior_same_organism_any_90d": "Prior Same-Organism History (90 Days)",
            "cov_prior_procedure_90d": "Prior Procedure (90 Days)",
        }
        if text in exact:
            return exact[text]
        site_labels = {
            "source_site_armd": "Site: Stanford",
            "source_site_armd_ecuh": "Site: ECU Health",
            "source_site_armd_utsw": "Site: UT Southwestern",
        }
        if text in site_labels:
            return site_labels[text]
        if text.startswith("cov_age_bin_"):
            return "Age: " + text.removeprefix("cov_age_bin_").replace("_plus", "+").replace("_", "-")
        if text.startswith("cov_sex_"):
            return "Sex Category: " + text.removeprefix("cov_sex_").replace("_", " ").title()
        return text.replace("cov_", "").replace("comorb_", "comorbidity: ").replace(
            "proc_", "procedure: "
        ).replace("_", " ").strip().title()
