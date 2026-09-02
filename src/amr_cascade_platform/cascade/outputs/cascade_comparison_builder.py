"""Compare site-level and combined cascade outputs."""

from __future__ import annotations

from pathlib import Path
from itertools import chain

import pandas as pd


class CascadeComparisonBuilder:
    """Build site-vs-combined edge comparison tables."""

    _EDGE_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "escalation_ratio",
        "adjusted_odds_ratio",
        "total_support_n",
    ]
    _COMPARISON_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "site_escalation_ratio",
        "site_adjusted_odds_ratio",
        "site_total_support_n",
        "combined_escalation_ratio",
        "combined_adjusted_odds_ratio",
        "combined_total_support_n",
        "site",
        "edge_presence",
        "escalation_ratio_delta",
        "adjusted_odds_ratio_delta",
    ]

    def export(
        self,
        artifact_root: Path,
        scope_name: str,
    ) -> dict[str, Path]:
        scope_path = Path(scope_name)
        if scope_path.parts and scope_path.parts[0] == "combined":
            combined_dir = artifact_root / scope_path
            combined_path = combined_dir / "edge_report.parquet"
            if not combined_path.exists():
                return {}
            suffix = Path(*scope_path.parts[1:]) if len(scope_path.parts) > 1 else Path()
            comparison_frames: list[pd.DataFrame] = []
            summary_rows: list[dict[str, object]] = []
            for site_dir in sorted(artifact_root.iterdir()):
                if not site_dir.is_dir():
                    continue
                if site_dir.name in {"combined"} or site_dir.name.endswith("_sensitivity_implied"):
                    continue
                site_path = site_dir / suffix / "edge_report.parquet"
                if not site_path.exists():
                    continue
                comparison = self._compare_pair_tables(
                    site_name=site_dir.name,
                    site_frame=pd.read_parquet(site_path),
                    combined_frame=pd.read_parquet(combined_path),
                )
                comparison_frames.append(comparison)
                summary_rows.extend(self._summarize(site_dir.name, comparison))
            if not comparison_frames:
                return {}
            comparison = pd.DataFrame.from_records(
                chain.from_iterable(frame.to_dict(orient="records") for frame in comparison_frames),
                columns=self._COMPARISON_COLUMNS,
            )
            return self._write_outputs(
                output_dir=combined_dir,
                comparison=comparison,
                summary=pd.DataFrame(summary_rows),
            )

        site_dir = artifact_root / scope_path
        suffix = Path(*scope_path.parts[1:]) if len(scope_path.parts) > 1 else Path()
        combined_path = artifact_root / "combined" / suffix / "edge_report.parquet"
        site_path = site_dir / "edge_report.parquet"
        if scope_name.endswith("_sensitivity_implied") or not site_path.exists() or not combined_path.exists():
            return {}
        comparison = self._compare_pair_tables(
            site_name=scope_name,
            site_frame=pd.read_parquet(site_path),
            combined_frame=pd.read_parquet(combined_path),
        )
        summary = pd.DataFrame(self._summarize(scope_name, comparison))
        return self._write_outputs(site_dir, comparison, summary)

    def _compare_pair_tables(
        self,
        site_name: str,
        site_frame: pd.DataFrame,
        combined_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        site_frame = site_frame.reindex(columns=self._EDGE_COLUMNS)
        combined_frame = combined_frame.reindex(columns=self._EDGE_COLUMNS)
        site_subset = site_frame.loc[
            :,
            self._EDGE_COLUMNS,
        ].rename(
            columns={
                "escalation_ratio": "site_escalation_ratio",
                "adjusted_odds_ratio": "site_adjusted_odds_ratio",
                "total_support_n": "site_total_support_n",
            }
        )
        combined_subset = combined_frame.loc[
            :,
            self._EDGE_COLUMNS,
        ].rename(
            columns={
                "escalation_ratio": "combined_escalation_ratio",
                "adjusted_odds_ratio": "combined_adjusted_odds_ratio",
                "total_support_n": "combined_total_support_n",
            }
        )
        comparison = site_subset.merge(
            combined_subset,
            on=["upstream_antibiotic", "downstream_antibiotic"],
            how="outer",
        )
        comparison["site"] = site_name
        comparison["edge_presence"] = comparison.apply(self._presence_label, axis=1)
        comparison["escalation_ratio_delta"] = (
            comparison["site_escalation_ratio"] - comparison["combined_escalation_ratio"]
        )
        comparison["adjusted_odds_ratio_delta"] = (
            comparison["site_adjusted_odds_ratio"] - comparison["combined_adjusted_odds_ratio"]
        )
        return comparison.reindex(columns=self._COMPARISON_COLUMNS)

    def _summarize(self, site_name: str, comparison: pd.DataFrame) -> list[dict[str, object]]:
        counts = comparison["edge_presence"].value_counts().to_dict()
        return [
            {
                "site": site_name,
                "edge_presence": label,
                "edge_count": int(counts.get(label, 0)),
            }
            for label in ["shared", "site_only", "combined_only"]
        ]

    def _write_outputs(
        self,
        output_dir: Path,
        comparison: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> dict[str, Path]:
        comparison_path = output_dir / "site_vs_combined_edge_comparison.parquet"
        summary_path = output_dir / "site_vs_combined_summary.parquet"
        comparison.to_parquet(comparison_path, index=False)
        summary.to_parquet(summary_path, index=False)
        return {
            "site_vs_combined_edge_comparison_path": comparison_path,
            "site_vs_combined_summary_path": summary_path,
        }

    @staticmethod
    def _presence_label(row: pd.Series) -> str:
        site_present = pd.notna(row.get("site_escalation_ratio"))
        combined_present = pd.notna(row.get("combined_escalation_ratio"))
        if site_present and combined_present:
            return "shared"
        if site_present:
            return "site_only"
        return "combined_only"
