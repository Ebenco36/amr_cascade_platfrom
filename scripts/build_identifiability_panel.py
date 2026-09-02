"""Build the episode x antibiotic eligibility panel for amr_identifiability.

Deliberately bypasses PairFeatureMatrixBuilder / build-features: that path
constructs the 151M-row upstream-drug x downstream-drug cascade table used by
amr_cascade_platform's own escalation modeling, which amr_identifiability does
not need and which does not fit in memory on this machine at E. coli scale.
This script instead merges the already-built episode-level covariates
(baseline/acute/history/comorbidity) directly onto the much smaller
eligible_pairs.parquet (episode x antibiotic) gold table.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.infrastructure.storage.dataset_store import DatasetStore
from amr_cascade_platform.features.builders.acute_feature_builder import AcuteFeatureBuilder
from amr_cascade_platform.features.builders.demographics_feature_builder import DemographicsFeatureBuilder
from amr_cascade_platform.features.builders.history_feature_builder import HistoryFeatureBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGANISM = "ESCHERICHIA COLI"
ORGANISM_SLUG = "e_coli"

RESISTANCE_MAP = {"RESISTANT": 1.0, "INTERMEDIATE": 1.0, "SUSCEPTIBLE": 0.0}


def main() -> None:
    settings = ConfigLoader(PROJECT_ROOT).load("mac")
    paths = PathManager(PROJECT_ROOT, settings)
    store = DatasetStore(settings)

    id_keys = list(settings.platform.id_columns)
    episode_keys = list(settings.gold.episode_key_columns)
    feature_join_keys = id_keys + ["source_site"]

    gold_dir = scoped_output_dir(root=paths.paths.gold, scope="combined", organism=ORGANISM)
    culture_episodes = store.read_pandas(gold_dir / "culture_episodes.parquet")
    eligible_pairs = store.read_pandas(gold_dir / "eligible_pairs.parquet")
    culture_drug_episodes = store.read_pandas(gold_dir / "culture_drug_episodes.parquet")
    print(f"culture_episodes: {len(culture_episodes):,} rows")
    print(f"eligible_pairs:   {len(eligible_pairs):,} rows")

    def filtered_loader(site: str, canonical_table: str) -> pd.DataFrame:
        path = paths.paths.harmonized / "site_aligned" / site / f"{canonical_table}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = store.read_pandas(path)
        if df.empty:
            return df
        keys = culture_episodes.loc[culture_episodes["source_site"] == site, id_keys].drop_duplicates()
        if keys.empty:
            return pd.DataFrame()
        return df.merge(keys, on=id_keys, how="inner")

    print("Building baseline (demographics + ADI) features...")
    baseline = DemographicsFeatureBuilder(settings).build(culture_episodes, filtered_loader)
    print("Building acute (labs/vitals/ward) features...")
    acute = AcuteFeatureBuilder(settings).build(culture_episodes, filtered_loader)
    print("Building history (prior abx/organism/procedures/subtype/nursing home) features...")
    history = HistoryFeatureBuilder(settings).build(culture_episodes, filtered_loader)

    print("Loading pre-built comorbidity aggregates...")
    comorbidity_frames = []
    for site in settings.platform.sites:
        comorbidity_path = (
            paths.paths.interim / site / "feature_matrices" / settings.feature_build.comorbidity_output_filename
        )
        if not comorbidity_path.exists():
            continue
        site_keys = culture_episodes.loc[culture_episodes["source_site"] == site, id_keys].drop_duplicates()
        if site_keys.empty:
            continue
        df = store.read_pandas(comorbidity_path, columns=id_keys + ["comorbidity_count"])
        df = df.merge(site_keys, on=id_keys, how="inner")
        df["source_site"] = site
        comorbidity_frames.append(df)
    comorbidity = (
        pd.concat(comorbidity_frames, ignore_index=True)
        if comorbidity_frames
        else pd.DataFrame(columns=feature_join_keys + ["comorbidity_count"])
    )

    episode_base = culture_episodes.loc[:, episode_keys].drop_duplicates()
    episode_features = episode_base
    for covariates in (baseline, acute, history, comorbidity):
        episode_features = episode_features.merge(
            covariates, on=feature_join_keys, how="left", validate="many_to_one"
        )
    print(f"episode_features: {len(episode_features):,} rows x {len(episode_features.columns)} cols")

    print("Resolving observed susceptibility per (episode, antibiotic)...")
    pair_keys = episode_keys + ["antibiotic"]
    observed = culture_drug_episodes.loc[:, pair_keys + ["susceptibility"]].copy()
    grouped = observed.groupby(pair_keys, dropna=False)["susceptibility"].agg(
        lambda values: _collapse(values.dropna().tolist())
    ).reset_index()

    panel = eligible_pairs.merge(grouped, on=pair_keys, how="left", validate="one_to_one")
    panel["resistance_binary"] = panel["susceptibility"].map(RESISTANCE_MAP)
    panel = panel.merge(episode_features, on=episode_keys, how="left", validate="many_to_one")
    panel = panel.merge(
        culture_episodes.loc[:, episode_keys + ["ordering_mode", "culture_description", "was_positive"]],
        on=episode_keys,
        how="left",
        validate="many_to_one",
    )

    panel = panel.rename(columns={"antibiotic": "antibiotic_canonical", "is_eligible": "eligible"})
    panel["eligible"] = panel["eligible"].astype(bool)
    panel["observed"] = panel["susceptibility"].notna()
    panel["directly_observed"] = panel["observed"].map({True: "yes", False: pd.NA})

    print(f"panel: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"eligible rows: {int(panel['eligible'].sum()):,}")
    print(f"observed rows: {int(panel['observed'].sum()):,}")

    out_dir = scoped_output_dir(root=paths.paths.features, scope="combined", organism=ORGANISM)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eligible_panel_{ORGANISM_SLUG}.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    manifest_path = out_dir / f"eligible_panel_{ORGANISM_SLUG}.manifest.json"
    _write_manifest(
        manifest_path,
        organism=ORGANISM,
        panel=panel,
        source_paths=[
            gold_dir / "culture_episodes.parquet",
            gold_dir / "eligible_pairs.parquet",
            gold_dir / "culture_drug_episodes.parquet",
        ],
    )
    print(f"Manifest -> {manifest_path}")


def _write_manifest(
    manifest_path: Path,
    *,
    organism: str,
    panel: pd.DataFrame,
    source_paths: list[Path],
) -> None:
    """Record build provenance so downstream consumers (amr_identifiability) can tell
    what cascade-platform state a panel was built from, without needing a live import
    or a shared filesystem assumption beyond the one export step itself."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None

    manifest = {
        "organism": organism,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "amr_cascade_platform_commit": commit,
        "amr_cascade_platform_dirty": dirty,
        "panel_rows": len(panel),
        "panel_columns": len(panel.columns),
        "source_files": [
            {
                "path": str(path.relative_to(repo_root)),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
            if path.exists()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _collapse(values: list[str]) -> str | None:
    unique = sorted(set(values))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    evaluable = [v for v in unique if v in {"RESISTANT", "SUSCEPTIBLE"}]
    if len(set(evaluable)) == 1 and evaluable:
        return evaluable[0]
    return "CONFLICTING_RESULT"


if __name__ == "__main__":
    main()
