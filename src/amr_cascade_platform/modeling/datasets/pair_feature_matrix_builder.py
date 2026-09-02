"""Build leak-aware pair-level feature matrices for downstream testing models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.exceptions.custom_exceptions import DataDiscoveryError
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore


@dataclass(frozen=True)
class ModelingDatasetBundle:
    dataframe: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    id_columns: tuple[str, ...]


# drug_pair_episodes.parquet is organism-level fan-out and can run to hundreds of millions
# of rows; these columns repeat far fewer distinct values across all of them -- including
# anon_id/pat_enc_csn_id_coded/order_proc_id_coded/order_time_jittered, which repeat once
# per episode (~1000x on real data), not "once per row" as an earlier version of this
# constant assumed (see cascade_analysis_workflow._PAIR_TABLE_CATEGORICAL_COLUMNS for the
# empirical verification). Loading them as pandas Categorical avoids decoding the same
# handful of strings into a fresh Python object per row (see DatasetStore.read_pandas).
# Not to be confused with ModelingDatasetBundle.categorical_columns, which selects
# sklearn one-hot-encoded feature columns further downstream -- this constant is about
# pandas dtype at load time. NOTE: unlike cascade_analysis_workflow, this builder merges
# drug_pairs against culture_episodes (loaded as plain object dtype) on episode_key_columns
# -- that merge will silently revert these 4 columns back to object dtype (same mechanism
# fixed in CoTestingFilterAnalyzer), so the memory win here is not yet verified end-to-end.
_PAIR_TABLE_DICTIONARY_COLUMNS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "order_time_jittered",
    "organism",
    "source_site",
    "upstream_antibiotic",
    "upstream_susceptibility",
    "downstream_antibiotic",
    "pair_direction",
]


class PairFeatureMatrixBuilder:
    """Create pair-level downstream-testing features from gold and cascade artifacts."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager
        self._dataset_store = DatasetStore(settings)

    def feature_layout(self) -> dict[str, tuple[str, ...]]:
        categorical_columns = (
            "organism",
            "ordering_mode",
            "culture_description",
            "upstream_antibiotic",
            "downstream_antibiotic",
            "upstream_susceptibility",
        )
        numeric_columns = [
            "upstream_positive",
            "was_positive_numeric",
            "downstream_intrinsic_resistance",
        ]
        if self._settings.modeling.include_cascade_features:
            numeric_columns.extend(
                [
                    "train_pair_escalation_ratio",
                    "train_pair_adjusted_odds_ratio",
                    "train_pair_total_support_n",
                    "train_pair_positive_probability",
                    "train_pair_negative_probability",
                    "train_pair_is_retained_edge",
                    "train_upstream_out_degree",
                    "train_upstream_in_degree",
                    "train_upstream_pagerank",
                    "train_upstream_betweenness",
                    "train_downstream_out_degree",
                    "train_downstream_in_degree",
                    "train_downstream_pagerank",
                    "train_downstream_betweenness",
                ]
            )
        id_columns = tuple(self._settings.gold.episode_key_columns) + (
            "pair_direction",
            "target",
        )
        feature_columns = tuple(categorical_columns) + tuple(numeric_columns)
        return {
            "categorical_columns": tuple(categorical_columns),
            "numeric_columns": tuple(numeric_columns),
            "feature_columns": feature_columns,
            "id_columns": id_columns,
        }

    def build(self, scope: str = "combined", organism: str | None = None) -> ModelingDatasetBundle:
        gold_dir = scoped_output_dir(
            root=self._paths.paths.gold,
            scope=scope,
            organism=organism,
        )
        episode_key_columns = list(self._settings.gold.episode_key_columns)
        pair_columns = episode_key_columns + [
            "upstream_antibiotic",
            "upstream_susceptibility",
            "downstream_antibiotic",
            self._settings.modeling.target_column,
            "downstream_eligible",
            "downstream_intrinsic_resistance",
            "pair_direction",
        ]
        pairs = self._load_required(
            gold_dir / "drug_pair_episodes.parquet",
            columns=pair_columns,
            categorical_columns=_PAIR_TABLE_DICTIONARY_COLUMNS,
        )
        episodes = self._load_required(
            gold_dir / "culture_episodes.parquet",
            columns=episode_key_columns + ["ordering_mode", "culture_description", "was_positive"],
        )

        frame = pairs
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame[frame["downstream_eligible"] == 1].copy()

        frame["upstream_positive"] = frame["upstream_susceptibility"].map(self._map_positive)
        frame = frame[frame["upstream_positive"].isin([0, 1])].copy()
        frame["target"] = frame[self._settings.modeling.target_column].astype(int)

        merge_columns = list(self._settings.gold.episode_key_columns) + [
            "ordering_mode",
            "culture_description",
            "was_positive",
        ]
        context = episodes.loc[:, merge_columns].copy()
        frame = frame.merge(
            context,
            on=list(self._settings.gold.episode_key_columns),
            how="left",
        )
        frame["was_positive_numeric"] = pd.to_numeric(frame["was_positive"], errors="coerce").fillna(0).astype(int)

        if self._settings.modeling.include_cascade_features:
            frame = self._merge_training_cascade_features(frame, organism=organism)

        layout = self.feature_layout()
        categorical_columns = layout["categorical_columns"]
        numeric_columns = list(layout["numeric_columns"])

        for column in categorical_columns:
            if column not in frame.columns:
                frame[column] = "UNKNOWN"
        for column in numeric_columns:
            if column not in frame.columns:
                frame[column] = 0.0

        for column in categorical_columns:
            series = frame[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                # Some columns already arrive as Categorical (see
                # _PAIR_TABLE_DICTIONARY_COLUMNS / DatasetStore.read_pandas), whose fixed
                # category set doesn't include "UNKNOWN" yet -- filling NaN with a value
                # outside the existing categories raises TypeError, so it must be added first.
                if "UNKNOWN" not in series.cat.categories:
                    series = series.cat.add_categories(["UNKNOWN"])
                frame[column] = series.fillna("UNKNOWN")
            else:
                frame[column] = series.fillna("UNKNOWN").astype("category")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            if pd.api.types.is_integer_dtype(frame[column]) or (frame[column] % 1 == 0).all():
                frame[column] = pd.to_numeric(frame[column], downcast="integer")
            else:
                frame[column] = pd.to_numeric(frame[column], downcast="float")

        id_columns = layout["id_columns"]
        feature_columns = layout["feature_columns"]
        return ModelingDatasetBundle(
            dataframe=frame,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numeric_columns=tuple(numeric_columns),
            id_columns=id_columns,
        )

    def _merge_training_cascade_features(self, frame: pd.DataFrame, organism: str | None = None) -> pd.DataFrame:
        train_site = self._settings.modeling.train_sites[0]
        artifact_root = scoped_output_dir(
            root=self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir,
            scope="site",
            site=train_site,
            organism=organism,
        )
        edge_report = self._load_required(artifact_root / "edge_report.parquet")
        network_nodes = self._load_required(artifact_root / "network_nodes.parquet")

        if not edge_report.empty:
            pair_features = self._normalize_edge_report_columns(edge_report)
            frame = frame.merge(
                pair_features,
                on=["upstream_antibiotic", "downstream_antibiotic"],
                how="left",
            )

        if not network_nodes.empty:
            upstream_nodes = network_nodes.rename(
                columns={
                    "antibiotic": "upstream_antibiotic",
                    "out_degree": "train_upstream_out_degree",
                    "in_degree": "train_upstream_in_degree",
                    "pagerank": "train_upstream_pagerank",
                    "betweenness": "train_upstream_betweenness",
                }
            )
            downstream_nodes = network_nodes.rename(
                columns={
                    "antibiotic": "downstream_antibiotic",
                    "out_degree": "train_downstream_out_degree",
                    "in_degree": "train_downstream_in_degree",
                    "pagerank": "train_downstream_pagerank",
                    "betweenness": "train_downstream_betweenness",
                }
            )
            frame = frame.merge(upstream_nodes, on="upstream_antibiotic", how="left")
            frame = frame.merge(downstream_nodes, on="downstream_antibiotic", how="left")

        return frame

    def _normalize_edge_report_columns(self, edge_report: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            "positive_probability": "resistant_downstream_test_probability",
            "negative_probability": "susceptible_downstream_test_probability",
            "is_retained_edge": "retained_edge",
        }
        normalized = edge_report.rename(columns=column_map).copy()
        defaults = {
            "adjusted_odds_ratio": 0.0,
            "total_support_n": 0.0,
            "resistant_downstream_test_probability": 0.0,
            "susceptible_downstream_test_probability": 0.0,
            "retained_edge": False,
        }
        for column, default in defaults.items():
            if column not in normalized.columns:
                normalized[column] = default

        pair_features = normalized.loc[
            :,
            [
                "upstream_antibiotic",
                "downstream_antibiotic",
                "escalation_ratio",
                "adjusted_odds_ratio",
                "total_support_n",
                "resistant_downstream_test_probability",
                "susceptible_downstream_test_probability",
                "retained_edge",
            ],
        ].rename(
            columns={
                "escalation_ratio": "train_pair_escalation_ratio",
                "adjusted_odds_ratio": "train_pair_adjusted_odds_ratio",
                "total_support_n": "train_pair_total_support_n",
                "resistant_downstream_test_probability": "train_pair_positive_probability",
                "susceptible_downstream_test_probability": "train_pair_negative_probability",
                "retained_edge": "train_pair_is_retained_edge",
            }
        )
        return pair_features

    def _load_required(
        self,
        path: Path,
        columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if not path.exists():
            raise DataDiscoveryError(f"Required modeling input not found: {path}")
        return self._dataset_store.read_pandas(path, columns=columns, categorical_columns=categorical_columns)

    def _load_optional(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._dataset_store.read_pandas(path)

    def _map_positive(self, value: object) -> int | None:
        if pd.isna(value):
            return None
        if value == self._settings.cascade.upstream_result_positive:
            return 1
        if value == self._settings.cascade.upstream_result_negative:
            return 0
        return None
