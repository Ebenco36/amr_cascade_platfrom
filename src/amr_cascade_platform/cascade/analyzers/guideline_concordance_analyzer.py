"""Cascade guideline concordance analysis."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.core.utils.text import normalize_label


class GuidelineConcordanceAnalyzer:
    """Summarize structural alignment and documented guideline concordance for retained cascade pairs."""

    def __init__(self, settings: Settings, path_manager: PathManager) -> None:
        self._settings = settings
        self._paths = path_manager

    def analyze(self, drug_pairs: pd.DataFrame, escalation_results: pd.DataFrame) -> pd.DataFrame:
        if drug_pairs.empty or escalation_results.empty:
            return pd.DataFrame()

        organism_stratified = bool(
            "organism" in escalation_results.columns
            or drug_pairs.get("organism", pd.Series(dtype=object)).dropna().nunique() <= 1
        )
        retained_edges = escalation_results.loc[
            :, [column for column in ["organism", "upstream_antibiotic", "downstream_antibiotic", "escalation_ratio"] if column in escalation_results.columns]
        ].drop_duplicates()
        merge_keys = ["upstream_antibiotic", "downstream_antibiotic"]
        if organism_stratified and "organism" in retained_edges.columns:
            merge_keys = ["organism", *merge_keys]
        edge_level = drug_pairs.merge(retained_edges, on=merge_keys, how="inner")
        if edge_level.empty:
            return pd.DataFrame()

        group_keys = ["upstream_antibiotic", "downstream_antibiotic"]
        if organism_stratified:
            group_keys = ["organism", *group_keys]
        concordance = edge_level.groupby(group_keys, dropna=False, observed=True).agg(
            source_sites=("source_site", lambda values: sorted({str(value) for value in values if pd.notna(value)})),
            eligible_episode_n=("downstream_eligible", "sum"),
            intrinsic_episode_n=("downstream_intrinsic_resistance", "sum"),
            tested_episode_n=("downstream_tested", "sum"),
            escalation_ratio=("escalation_ratio", "first"),
        ).reset_index()
        if not organism_stratified:
            concordance.insert(0, "organism", "ALL_ORGANISMS")
        denominator = concordance["eligible_episode_n"] + concordance["intrinsic_episode_n"]
        concordance["structural_alignment_rate"] = 1 - (concordance["intrinsic_episode_n"] / denominator).fillna(0)
        concordance["structural_concordance"] = concordance["intrinsic_episode_n"].map(
            lambda count: "aligned_with_eligibility_logic" if count == 0 else "contains_structural_conflict"
        )

        if not organism_stratified:
            concordance["documented_rule_match"] = False
            concordance["matching_rule_n"] = 0
            concordance["matching_rule_sites"] = ""
            concordance["matching_rule_examples"] = ""
            concordance["documented_rule_concordance"] = "not_evaluated_for_all_organism_aggregate"
            return concordance

        documented = self._load_documented_rules()
        if documented.empty:
            concordance["documented_rule_match"] = False
            concordance["matching_rule_n"] = 0
            concordance["matching_rule_sites"] = ""
            concordance["matching_rule_examples"] = ""
            concordance["documented_rule_concordance"] = "no_documented_rule_available"
            return concordance

        concordance["organism_key"] = concordance["organism"].map(normalize_label)
        concordance["upstream_key"] = concordance["upstream_antibiotic"].map(normalize_antibiotic_label)
        concordance["downstream_key"] = concordance["downstream_antibiotic"].map(normalize_antibiotic_label)
        matches = concordance.merge(
            documented,
            on=["organism_key", "upstream_key", "downstream_key"],
            how="left",
        )
        summarized = (
            matches.groupby(["organism", "upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True)
            .agg(
                organism_key=("organism_key", "first"),
                source_sites=("source_sites", "first"),
                eligible_episode_n=("eligible_episode_n", "first"),
                intrinsic_episode_n=("intrinsic_episode_n", "first"),
                tested_episode_n=("tested_episode_n", "first"),
                escalation_ratio=("escalation_ratio", "first"),
                structural_alignment_rate=("structural_alignment_rate", "first"),
                structural_concordance=("structural_concordance", "first"),
                matching_rule_n=("rule_text", lambda values: int(pd.Series(values).notna().sum())),
                matching_rule_sites=(
                    "rule_source_site",
                    lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})),
                ),
                matching_rule_examples=(
                    "rule_text",
                    lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v)})),
                ),
            )
            .reset_index()
        )
        summarized["documented_rule_match"] = summarized["matching_rule_n"] > 0
        # "No match" and "no rule exists to match against" look identical downstream
        # (both render as False / 0 / "---") but mean opposite things: the first is an
        # empirical finding (a documented rule exists for this organism and this pair
        # doesn't follow it), the second is a reference-coverage gap (the rule set has
        # nothing for this organism at all, so every pair trivially "fails" to match).
        # Collapsing them made every E. coli row read as a checked-and-failed
        # concordance test when the real reason is that `implied_susceptibility_rules`
        # has no E. coli entries -- distinguishing them keeps the column honest.
        organisms_with_rules = set(documented["organism_key"])
        has_any_rule_for_organism = summarized["organism_key"].isin(organisms_with_rules)
        summarized["documented_rule_concordance"] = "no_matching_documented_rule"
        summarized.loc[summarized["documented_rule_match"], "documented_rule_concordance"] = "matches_documented_rule"
        summarized.loc[~has_any_rule_for_organism, "documented_rule_concordance"] = "no_documented_rule_for_organism"
        return summarized

    def _load_documented_rules(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for site in self._settings.platform.sites:
            frame = self._load_rules_for_site(site)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "organism_key",
                    "upstream_key",
                    "downstream_key",
                    "rule_text",
                    "rule_source_site",
                ]
            )
        documented = pd.concat(frames, ignore_index=True)
        return documented.drop_duplicates(
            ["organism_key", "upstream_key", "downstream_key", "rule_text", "rule_source_site"]
        ).reset_index(drop=True)

    def _load_rules_for_site(self, site: str) -> pd.DataFrame:
        candidates = [
            self._paths.paths.silver / site / "implied_susceptibility_rules.parquet",
            self._paths.paths.bronze / site / "implied_susceptibility_rules.parquet",
            self._paths.paths.sample / site / "implied_susceptibility_rules.csv",
            self._paths.paths.raw / site / "implied_susceptibility_rules.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return pd.DataFrame()
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if frame.empty or "Rule" not in frame.columns or "Antibiotic" not in frame.columns or "Organism" not in frame.columns:
            return pd.DataFrame()
        frame = frame.copy()
        if "source_site" in frame.columns:
            frame["rule_source_site"] = frame["source_site"].fillna(site).astype(str)
        else:
            frame["rule_source_site"] = site
        parsed_rows: list[dict[str, str]] = []
        for row in frame.itertuples(index=False):
            upstream_antibiotics = self._extract_upstream_antibiotics(getattr(row, "Rule", None))
            if not upstream_antibiotics:
                continue
            organism = normalize_label(getattr(row, "Organism", None))
            downstream = normalize_antibiotic_label(getattr(row, "Antibiotic", None))
            if not organism or not downstream:
                continue
            for upstream in upstream_antibiotics:
                parsed_rows.append(
                    {
                        "organism_key": organism,
                        "upstream_key": normalize_antibiotic_label(upstream),
                        "downstream_key": downstream,
                        "rule_text": str(getattr(row, "Rule", "")),
                        "rule_source_site": str(getattr(row, "rule_source_site", site)),
                    }
                )
        return pd.DataFrame(parsed_rows)

    @staticmethod
    def _extract_upstream_antibiotics(rule_text: object) -> list[str]:
        if pd.isna(rule_text):
            return []
        text = str(rule_text).strip()
        if not text:
            return []
        match = re.search(r"^SAME AS\s+(.+?)(?:\s+IF\s+REPORTED|\s+IF\s+SUSCEPTIBLE|\s+IF\s+RESISTANT|$)", text, flags=re.IGNORECASE)
        if not match:
            return []
        payload = match.group(1).strip()
        parts = [part.strip() for part in re.split(r"\s+(?:AND|OR)\s+|,", payload, flags=re.IGNORECASE) if part.strip()]
        return parts
