"""Direction-pure views of cascade edges.

Escalation (ER > 1) and suppression (ER < 1) are opposite observation
behaviours: after an upstream resistant result the downstream drug is observed
more, or less, often than after an upstream susceptible result. They must never
share a graph, matrix, flow diagram or summary statistic, because pooling them
silently turns "strong suppression" into "weak escalation" and makes any
mean/PageRank/centrality over the pooled edges uninterpretable.

Every function here is a pure function of an edge table (one row per ordered
drug pair) and returns tidy tables that renderers draw verbatim. Nothing here
draws or reads files, so the numbers a figure shows can be unit-tested.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
import pandas as pd

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer

LOGGER = logging.getLogger(__name__)

ESCALATION = "escalation"
SUPPRESSION = "suppression"
DIRECTIONS: tuple[str, ...] = (ESCALATION, SUPPRESSION)
VALIDATED_STATUSES: frozenset[str] = CascadeValidationAnalyzer.VALIDATED_STATUSES

PAIR_KEY = ["upstream_antibiotic", "downstream_antibiotic"]


def edge_directions(edges: pd.DataFrame) -> pd.Series:
    """Label each edge ``escalation`` (ER>1), ``suppression`` (0<ER<1) or missing.

    The escalation ratio is authoritative. If the table also carries the
    pipeline's own ``cascade_direction`` column, the two must agree: a
    disagreement means the table was assembled inconsistently and a figure
    drawn from it would be wrong, so this raises rather than picking one.
    """
    ratio = pd.to_numeric(edges["escalation_ratio"], errors="coerce")
    direction = pd.Series(pd.NA, index=edges.index, dtype="object")
    direction[ratio > 1.0] = ESCALATION
    direction[(ratio > 0.0) & (ratio < 1.0)] = SUPPRESSION
    if "cascade_direction" in edges.columns:
        declared = edges["cascade_direction"].astype("object")
        conflict = declared.notna() & direction.notna() & (declared != direction)
        if conflict.any():
            bad = edges.loc[conflict, PAIR_KEY + ["escalation_ratio", "cascade_direction"]].head(5)
            raise ValueError(
                f"{int(conflict.sum())} edge(s) have a cascade_direction that contradicts their "
                f"escalation ratio; refusing to draw direction-separated figures from inconsistent "
                f"data. First offenders:\n{bad.to_string(index=False)}"
            )
    return direction


def canonicalize_edges(edges: pd.DataFrame, canonical_label: Callable[[str | None], str]) -> pd.DataFrame:
    """Map raw drug labels to canonical drug names, keeping one row per canonical pair.

    Source-system spellings of one drug must not appear as two nodes, and two
    spellings of one pair must not be counted twice. If canonicalisation makes
    two rows the same pair, the highest-support row is kept and the collapse is
    logged and recorded in ``result.attrs["n_collapsed_duplicates"]``.
    """
    out = edges.copy()
    for column in PAIR_KEY:
        out[column] = out[column].astype(str).map(canonical_label)
    support = pd.to_numeric(out.get("total_support_n", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    out = (
        out.assign(_support=support)
        .sort_values(["_support"] + PAIR_KEY, ascending=[False, True, True], kind="mergesort")
        .drop_duplicates(PAIR_KEY, keep="first")
        .drop(columns="_support")
        .sort_index()
        .reset_index(drop=True)
    )
    collapsed = len(edges) - len(out)
    if collapsed:
        LOGGER.warning("Collapsed %d edge(s) whose raw labels map to the same canonical drug pair.", collapsed)
    out.attrs["n_collapsed_duplicates"] = collapsed
    return out


def validated_edges(edges: pd.DataFrame, statuses: frozenset[str] | set[str] = VALIDATED_STATUSES) -> pd.DataFrame:
    """Edges whose validation label is in ``statuses``, tagged with ``direction``.

    Edges with an undefined direction (ER exactly 1 or missing) are dropped and
    counted in ``result.attrs["n_undirected"]`` so nothing disappears silently.
    """
    if edges.empty or "validation_status" not in edges.columns:
        empty = edges.head(0).copy()
        empty["direction"] = pd.Series(dtype="object")
        empty.attrs["n_undirected"] = 0
        return empty
    selected = edges.loc[edges["validation_status"].isin(statuses)].copy()
    selected["direction"] = edge_directions(selected)
    n_undirected = int(selected["direction"].isna().sum())
    selected = selected.loc[selected["direction"].notna()].reset_index(drop=True)
    selected.attrs["n_undirected"] = n_undirected
    return selected


def split_by_direction(edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition tagged edges into one table per direction (no overlap, no remainder)."""
    tagged = _tagged(edges)
    parts = {d: tagged.loc[tagged["direction"] == d].reset_index(drop=True) for d in DIRECTIONS}
    if sum(len(p) for p in parts.values()) != int(tagged["direction"].notna().sum()):
        raise AssertionError("Direction split is not a partition of the directed edges.")
    return parts


def tier_edges(edges: pd.DataFrame, tier: str) -> pd.DataFrame:
    """``validated`` = robust + supported; ``robust`` = robust only."""
    if tier == "validated":
        return edges
    if tier == "robust":
        return edges.loc[edges["validation_status"] == "robust"].reset_index(drop=True)
    raise ValueError(f"Unknown tier {tier!r}; expected 'validated' or 'robust'.")


def with_effect_columns(edges: pd.DataFrame) -> pd.DataFrame:
    """Add ``log2_er`` (signed) and ``effect_magnitude`` (|log2 ER|, direction-free strength)."""
    out = edges.copy()
    ratio = pd.to_numeric(out["escalation_ratio"], errors="coerce")
    out["log2_er"] = np.log2(ratio.where(ratio > 0))
    out["effect_magnitude"] = out["log2_er"].abs()
    return out


def display_selection(edges: pd.DataFrame, direction: str, min_fold: float = 1.0) -> pd.DataFrame:
    """Edges of ``direction`` whose effect is at least ``min_fold``-fold from ER = 1.

    A *display* rule, not a validation rule: with tens of thousands of rows per
    pair, permutation/bootstrap/replication validate effects as small as ER 1.05,
    so a figure that ranks by support alone showcases near-null patterns. The
    full validated set always remains in the machine-readable tables; figures
    state this floor and how many patterns it keeps.

    ``min_fold`` = 1.5 keeps escalation with ER >= 1.5 and suppression with
    ER <= 1/1.5; ``min_fold`` <= 1 keeps every validated pattern.
    """
    _require_direction(direction)
    tagged = _tagged(edges)
    pool = with_effect_columns(tagged.loc[tagged["direction"] == direction])
    if min_fold <= 1.0:
        return pool.reset_index(drop=True)
    return pool.loc[pool["effect_magnitude"] >= math.log2(min_fold) - 1e-12].reset_index(drop=True)


def shared_drug_order(
    edges: pd.DataFrame,
    aware_category_of: Callable[[str], str],
    category_sort_key: Callable[[str], int],
) -> list[str]:
    """One drug ordering (AWaRe tier, then name) for every direction-specific figure.

    Built from the union of both directions so an escalation matrix and a
    suppression matrix have identical rows and columns and can be compared cell
    by cell; a drug that only appears in one direction is simply blank in the other.
    """
    drugs = sorted(set(edges["upstream_antibiotic"].astype(str)) | set(edges["downstream_antibiotic"].astype(str)))
    return sorted(drugs, key=lambda drug: (category_sort_key(aware_category_of(drug)), drug))


def bipartite_links(edges: pd.DataFrame, direction: str, *, top_n: int | None = None) -> pd.DataFrame:
    """One row per validated pair of ``direction``: the links of a bipartite flow diagram.

    ``support_n`` is the pair's own eligible episode-pair row count
    (``total_support_n``), taken once per pair. It is never summed across paths
    or pairs that share a drug, so a link's width is exactly the support the
    results tables report for that pair.
    """
    _require_direction(direction)
    tagged = _tagged(edges)
    pool = with_effect_columns(tagged.loc[tagged["direction"] == direction])
    columns = [
        "upstream_antibiotic", "downstream_antibiotic", "support_n", "resistant_support_n",
        "susceptible_support_n", "escalation_ratio", "er_ci_lower", "er_ci_upper", "log2_er",
        "effect_magnitude", "validation_status", "adjusted_odds_ratio",
    ]
    if pool.empty:
        return pd.DataFrame(columns=columns)
    pool["support_n"] = pd.to_numeric(pool["total_support_n"], errors="coerce").fillna(0).astype(int)
    for optional in ("resistant_support_n", "susceptible_support_n", "er_ci_lower", "er_ci_upper", "adjusted_odds_ratio"):
        if optional not in pool.columns:
            pool[optional] = np.nan
    ranked = pool.sort_values(
        ["support_n", "effect_magnitude"] + PAIR_KEY, ascending=[False, False, True, True], kind="mergesort"
    )
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked.loc[:, columns].reset_index(drop=True)


def coverage_summary(links: pd.DataFrame, all_edges: pd.DataFrame, direction: str) -> dict[str, float]:
    """How much of a direction a truncated (top-N) figure actually shows."""
    _require_direction(direction)
    tagged = _tagged(all_edges)
    pool = tagged.loc[tagged["direction"] == direction]
    total_support = float(pd.to_numeric(pool["total_support_n"], errors="coerce").fillna(0).sum())
    shown_support = float(links["support_n"].sum()) if not links.empty else 0.0
    return {
        "n_shown": int(len(links)),
        "n_total": int(len(pool)),
        "support_shown": shown_support,
        "support_total": total_support,
        "support_fraction": (shown_support / total_support) if total_support > 0 else float("nan"),
    }


def matrix_frame(edges: pd.DataFrame, direction: str, order: list[str]) -> pd.DataFrame:
    """Long table of the cells of one direction's upstream x downstream matrix.

    ``strength`` is |log2 ER|: for escalation log2(ER), for suppression
    log2(1/ER). Both are positive and larger means a stronger effect in that
    direction, so a direction's colour scale always runs from "weak" to "strong".
    """
    _require_direction(direction)
    tagged = _tagged(edges)
    pool = with_effect_columns(tagged.loc[tagged["direction"] == direction])
    index_of = {drug: i for i, drug in enumerate(order)}
    missing = (set(pool["upstream_antibiotic"]) | set(pool["downstream_antibiotic"])) - set(index_of)
    if missing:
        raise ValueError(f"Drugs missing from the shared order: {sorted(missing)[:5]}")
    frame = pool.copy()
    frame["row"] = frame["upstream_antibiotic"].map(index_of)
    frame["col"] = frame["downstream_antibiotic"].map(index_of)
    frame["strength"] = frame["effect_magnitude"]
    if frame.duplicated(["row", "col"]).any():
        raise ValueError("Two edges map to the same matrix cell; canonicalise labels first.")
    return frame.reset_index(drop=True)


_NODE_COLUMNS = [
    "antibiotic", "out_degree", "in_degree", "degree", "out_robust", "in_robust",
    "out_weight", "in_weight", "weighted_degree",
]


def directional_graph_tables(edges: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(edges, nodes) of the graph made only of ``direction`` edges.

    Node statistics (degrees, weighted degrees) are computed inside this graph
    alone, never on a graph that also contains the opposite direction.
    """
    _require_direction(direction)
    tagged = _tagged(edges)
    pool = with_effect_columns(tagged.loc[tagged["direction"] == direction]).reset_index(drop=True)
    if pool.empty:
        return pool, pd.DataFrame(columns=_NODE_COLUMNS)
    pool["is_robust"] = pool["validation_status"].eq("robust")
    out_side = pool.groupby("upstream_antibiotic", observed=True).agg(
        out_degree=("upstream_antibiotic", "size"),
        out_weight=("effect_magnitude", "sum"),
        out_robust=("is_robust", "sum"),
    )
    in_side = pool.groupby("downstream_antibiotic", observed=True).agg(
        in_degree=("downstream_antibiotic", "size"),
        in_weight=("effect_magnitude", "sum"),
        in_robust=("is_robust", "sum"),
    )
    nodes = out_side.join(in_side, how="outer").fillna(0)
    nodes.index.name = "antibiotic"
    nodes = nodes.reset_index()
    for column in ("out_degree", "in_degree", "out_robust", "in_robust"):
        nodes[column] = nodes[column].astype(int)
    nodes["degree"] = nodes["out_degree"] + nodes["in_degree"]
    nodes["weighted_degree"] = nodes["out_weight"] + nodes["in_weight"]
    return pool, nodes.loc[:, _NODE_COLUMNS].sort_values("antibiotic").reset_index(drop=True)


def node_roles(edges: pd.DataFrame) -> pd.DataFrame:
    """Per-antibiotic role table, escalation and suppression in separate columns.

    ``escalation_out_degree`` is how many validated escalation patterns the drug
    triggers as the upstream drug; ``suppression_in_degree`` is how many
    validated suppression patterns it is the downstream (suppressed) drug of.
    """
    tagged = _tagged(edges)
    drugs = sorted(set(tagged["upstream_antibiotic"].astype(str)) | set(tagged["downstream_antibiotic"].astype(str)))
    roles = pd.DataFrame({"antibiotic": drugs})
    for direction in DIRECTIONS:
        _, nodes = directional_graph_tables(tagged, direction)
        renamed = nodes.rename(columns={c: f"{direction}_{c}" for c in _NODE_COLUMNS if c != "antibiotic"})
        roles = roles.merge(renamed, on="antibiotic", how="left")
    numeric = [c for c in roles.columns if c != "antibiotic"]
    roles[numeric] = roles[numeric].fillna(0)
    for column in numeric:
        if column.endswith(("_degree", "_robust")) and not column.endswith("weighted_degree"):
            roles[column] = roles[column].astype(int)
    return roles


_AWARE_COLUMNS = [
    "direction", "upstream_aware", "downstream_aware", "aware_direction", "n_patterns",
    "n_robust", "n_supported", "total_support_n", "median_log2_er",
]


def aware_direction_table(
    edges: pd.DataFrame,
    direction: str,
    aware_category_of: Callable[[str], str],
    transition_direction: Callable[[str, str], str],
) -> pd.DataFrame:
    """Validated pattern counts per (upstream AWaRe, downstream AWaRe) for one direction.

    Reports the median signed log2 ER *within the direction* (never a mean over
    both directions) alongside counts, so a cell can be read as "how many
    patterns and how strong", not as an average of opposing effects.
    """
    _require_direction(direction)
    tagged = _tagged(edges)
    pool = with_effect_columns(tagged.loc[tagged["direction"] == direction]).copy()
    if pool.empty:
        return pd.DataFrame(columns=_AWARE_COLUMNS)
    pool["upstream_aware"] = pool["upstream_antibiotic"].map(aware_category_of)
    pool["downstream_aware"] = pool["downstream_antibiotic"].map(aware_category_of)
    pool["aware_direction"] = [
        transition_direction(u, d) for u, d in zip(pool["upstream_aware"], pool["downstream_aware"])
    ]
    pool["is_robust"] = pool["validation_status"].eq("robust").astype(int)
    grouped = pool.groupby(["upstream_aware", "downstream_aware", "aware_direction"], observed=True)
    table = grouped.agg(
        n_patterns=("upstream_antibiotic", "size"),
        n_robust=("is_robust", "sum"),
        total_support_n=("total_support_n", "sum"),
        median_log2_er=("log2_er", "median"),
    ).reset_index()
    table["n_robust"] = table["n_robust"].astype(int)
    table["n_supported"] = table["n_patterns"] - table["n_robust"]
    table.insert(0, "direction", direction)
    return table.loc[:, _AWARE_COLUMNS]


def direction_by_aware_matrix(
    edges: pd.DataFrame,
    aware_category_of: Callable[[str], str],
    transition_direction: Callable[[str, str], str],
) -> pd.DataFrame:
    """Observation direction (rows) x AWaRe direction (columns) validated-pattern counts.

    Keeps two different ideas apart: escalation/suppression is about how
    downstream *observation* changes; upward/lateral/downward is about AWaRe
    *stewardship tier*. They are not synonyms.
    """
    tagged = _tagged(edges)
    pool = tagged.loc[tagged["direction"].notna()].copy()
    pool["aware_direction"] = [
        transition_direction(aware_category_of(u), aware_category_of(d))
        for u, d in zip(pool["upstream_antibiotic"], pool["downstream_antibiotic"])
    ]
    counts = pd.crosstab(pool["direction"], pool["aware_direction"])
    counts = counts.reindex(index=list(DIRECTIONS), fill_value=0)
    for column in ("upward", "lateral", "downward", "unclassified"):
        if column not in counts.columns:
            counts[column] = 0
    return counts.loc[:, ["upward", "lateral", "downward", "unclassified"]]


def circular_layout(
    drugs: list[str],
    aware_category_of: Callable[[str], str],
    category_sort_key: Callable[[str], int],
    *,
    gap_fraction: float = 0.04,
) -> dict[str, tuple[float, float]]:
    """Deterministic node positions on a circle, grouped by AWaRe tier.

    Every direction-specific network uses the same positions so figures can be
    compared by eye. Unlike a force-directed layout this has no seed or physics:
    identical input always gives identical coordinates, and drugs of one AWaRe
    tier form one contiguous arc.
    """
    ordered = sorted(drugs, key=lambda drug: (category_sort_key(aware_category_of(drug)), drug))
    groups: dict[str, list[str]] = {}
    for drug in ordered:
        groups.setdefault(aware_category_of(drug), []).append(drug)
    n_gaps = max(len(groups), 1)
    usable = 2.0 * math.pi * (1.0 - gap_fraction * n_gaps)
    step = usable / max(len(ordered), 1)
    positions: dict[str, tuple[float, float]] = {}
    angle = math.pi / 2.0
    for members in groups.values():
        for drug in members:
            positions[drug] = (math.cos(angle), math.sin(angle))
            angle -= step
        angle -= 2.0 * math.pi * gap_fraction
    return positions


def _tagged(edges: pd.DataFrame) -> pd.DataFrame:
    return edges if "direction" in edges.columns else edges.assign(direction=edge_directions(edges))


def _require_direction(direction: str) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


CONCORDANCE_CATEGORIES: tuple[str, ...] = ("concordant", "attenuated", "reversed", "unavailable")


def with_adjustment_concordance(edges: pd.DataFrame) -> pd.DataFrame:
    """Adds ``adjustment_concordance`` if not already present.

    Mirrors ``CascadeReportBuilder._adjustment_concordance`` exactly (vectorised
    rather than row-wise): ``concordant`` when the adjusted OR is on the same
    side of 1 as the raw ER and its 95% CI excludes 1; ``attenuated`` when it
    stays on the same side but the interval includes 1 (or is missing);
    ``reversed`` when the adjusted point estimate crosses to the opposite side
    of 1 from the raw ER; ``unavailable`` when the raw ER has no clear
    direction or no adjusted OR was estimated. Recomputing here (rather than
    requiring the column) lets this run against edge reports from a pipeline
    version that predates this field, not just the current one.
    """
    edges = edges.copy()
    if "adjustment_concordance" in edges.columns:
        return edges

    raw = pd.to_numeric(edges["escalation_ratio"], errors="coerce")
    adj = pd.to_numeric(edges["adjusted_odds_ratio"], errors="coerce")
    lo = pd.to_numeric(edges.get("adjusted_odds_ratio_ci_lower"), errors="coerce")
    hi = pd.to_numeric(edges.get("adjusted_odds_ratio_ci_upper"), errors="coerce")

    def _direction(ratio: pd.Series) -> pd.Series:
        direction = pd.Series("unavailable", index=ratio.index, dtype="object")
        direction[ratio > 1.0] = "positive"
        direction[(ratio > 0) & (ratio < 1.0)] = "negative"
        direction[ratio == 1.0] = "neutral"
        return direction

    raw_direction = _direction(raw)
    adjusted_direction = _direction(adj)
    has_raw = raw_direction.isin(["positive", "negative"])
    has_adjusted = adj.notna() & has_raw

    reversed_mask = has_adjusted & (adjusted_direction != "neutral") & (adjusted_direction != raw_direction)
    ci_excludes_null = lo.notna() & hi.notna() & ~((lo <= 1.0) & (1.0 <= hi))

    label = pd.Series("unavailable", index=edges.index, dtype="object")
    label[has_adjusted & ~reversed_mask & ci_excludes_null] = "concordant"
    label[has_adjusted & ~reversed_mask & ~ci_excludes_null] = "attenuated"
    label[reversed_mask] = "reversed"
    edges["adjustment_concordance"] = label
    return edges
