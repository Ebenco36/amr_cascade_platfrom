"""Build Sankey-ready pathway flow tables from retained cascade edges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class PathwayFlowBuilder:
    """Create multi-step pathway flow tables for Sankey-style visualization."""

    def export(
        self,
        top_paths: pd.DataFrame,
        retained_edges: pd.DataFrame,
        output_dir: Path,
        top_n: int,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        flow_path = output_dir / "pathway_flows.parquet"
        flow_table = self.build_flow_table(top_paths, retained_edges, top_n)
        flow_table.to_parquet(flow_path, index=False)
        return {"pathway_flows_path": flow_path}

    @staticmethod
    def build_flow_table(
        top_paths: pd.DataFrame,
        retained_edges: pd.DataFrame,
        top_n: int,
    ) -> pd.DataFrame:
        if not top_paths.empty:
            ranked = top_paths.head(top_n).reset_index(drop=True).copy()
            ranked["path_rank"] = ranked.index + 1
            first_stage = ranked.loc[
                :,
                [
                    "path_rank",
                    "path_start",
                    "path_mid",
                    "path_end",
                    "path_escalation_ratio",
                    "path_min_support_n",
                    "edge_1_escalation_ratio",
                ],
            ].rename(
                columns={
                    "path_start": "source",
                    "path_mid": "target",
                    "edge_1_escalation_ratio": "edge_escalation_ratio",
                }
            )
            first_stage["path_start"] = ranked["path_start"].values
            first_stage["stage"] = 1
            second_stage = ranked.loc[
                :,
                [
                    "path_rank",
                    "path_start",
                    "path_mid",
                    "path_end",
                    "path_escalation_ratio",
                    "path_min_support_n",
                    "edge_2_escalation_ratio",
                ],
            ].rename(
                columns={
                    "path_mid": "source",
                    "path_end": "target",
                    "edge_2_escalation_ratio": "edge_escalation_ratio",
                }
            )
            second_stage["path_end"] = ranked["path_end"].values
            second_stage["stage"] = 2
            combined = pd.concat([first_stage, second_stage], ignore_index=True)
            return combined.loc[
                :,
                [
                    "path_rank",
                    "source",
                    "target",
                    "path_start",
                    "path_end",
                    "path_escalation_ratio",
                    "path_min_support_n",
                    "edge_escalation_ratio",
                    "stage",
                ],
            ]

        if retained_edges.empty:
            return pd.DataFrame(
                columns=[
                    "path_rank",
                    "source",
                    "target",
                    "path_start",
                    "path_mid",
                    "path_end",
                    "path_escalation_ratio",
                    "path_min_support_n",
                    "edge_escalation_ratio",
                    "stage",
                ]
            )
        fallback = retained_edges.sort_values(
            by=["escalation_ratio", "total_support_n"],
            ascending=[False, False],
        ).head(top_n).reset_index(drop=True)
        fallback["path_rank"] = fallback.index + 1
        fallback["source"] = fallback["upstream_antibiotic"]
        fallback["target"] = fallback["downstream_antibiotic"]
        fallback["path_start"] = fallback["upstream_antibiotic"]
        fallback["path_mid"] = fallback["downstream_antibiotic"]
        fallback["path_end"] = fallback["downstream_antibiotic"]
        fallback["path_escalation_ratio"] = fallback["escalation_ratio"]
        fallback["path_min_support_n"] = fallback["total_support_n"]
        fallback["edge_escalation_ratio"] = fallback["escalation_ratio"]
        fallback["stage"] = 1
        return fallback.loc[
            :,
            [
                "path_rank",
                "source",
                "target",
                "path_start",
                "path_mid",
                "path_end",
                "path_escalation_ratio",
                "path_min_support_n",
                "edge_escalation_ratio",
                "stage",
            ],
        ]
