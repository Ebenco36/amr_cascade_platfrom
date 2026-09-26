"""Autoregressive model of which antimicrobials an episode reports, and their results.

One culture episode (one organism in one culture order) reports a subset of the
laboratory's antimicrobials, each with a result. The replica needs the joint
structure of those reports, not just their marginals: panel co-reporting,
result-conditioned add-on reporting, and co-resistance are exactly what the
cascade pipeline measures.

The model orders the drugs by how often they are reported and, for drug k,
fits two regularised logistic models on the real episodes:

* reported or not, given the episode context (calendar year, specimen type,
  ordering mode) and the states of drugs 1..k-1;
* the result class among reported episodes (SUSCEPTIBLE, INTERMEDIATE,
  RESISTANT, or a non-interpretive value), given the same predictors.

Drug k is never reported in a calendar year in which the real data never
reported it, so operational availability (a drug first used midway through an
era) carries over to the replica. Only fitted coefficients and aggregate counts
leave the real data; any category below the small-cell threshold is dropped.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

STATE_NOT_REPORTED = 0
STATE_SUSCEPTIBLE = 1
STATE_INTERMEDIATE = 2
STATE_RESISTANT = 3
STATE_OTHER = 4

RESULT_STATES = (STATE_SUSCEPTIBLE, STATE_INTERMEDIATE, STATE_RESISTANT, STATE_OTHER)
RAW_RESULT_SPELLING = {
    STATE_SUSCEPTIBLE: "Susceptible",
    STATE_INTERMEDIATE: "Intermediate",
    STATE_RESISTANT: "Resistant",
}
_STATE_PRIORITY = {"Resistant": STATE_RESISTANT, "Intermediate": STATE_INTERMEDIATE, "Susceptible": STATE_SUSCEPTIBLE}

CONTEXT_COLUMNS = ("year", "culture", "mode")


def episode_states(
    rows: pd.DataFrame,
    episode_columns: list[str],
    drug_column: str = "antibiotic",
    result_column: str = "susceptibility",
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, dict[str, int]]]:
    """Collapse raw AST rows into an episode x drug state matrix.

    Duplicate rows for one episode and drug collapse to the most resistant
    interpretive result (RESISTANT > INTERMEDIATE > SUSCEPTIBLE), or to
    STATE_OTHER when no row carries an interpretive result. Returns the episode
    table, the drug labels ordered by report frequency, the int8 state matrix,
    and, per drug, the raw spellings seen for STATE_OTHER.
    """
    frame = rows.loc[rows[drug_column].ne("Null"), episode_columns + [drug_column, result_column]].copy()
    frame["_state"] = frame[result_column].map(_STATE_PRIORITY).fillna(STATE_OTHER).astype("int8")
    other_spellings: dict[str, dict[str, int]] = {}
    others = frame.loc[frame["_state"].eq(STATE_OTHER)]
    for drug, group in others.groupby(drug_column, sort=False):
        other_spellings[str(drug)] = {str(k): int(v) for k, v in group[result_column].value_counts().items()}

    # Interpretive results win over non-interpretive ones; among interpretive
    # results the most resistant one wins.
    frame["_rank"] = frame["_state"].map({STATE_RESISTANT: 3, STATE_INTERMEDIATE: 2, STATE_SUSCEPTIBLE: 1, STATE_OTHER: 0})
    collapsed = (
        frame.sort_values("_rank")
        .drop_duplicates(subset=episode_columns + [drug_column], keep="last")
        .loc[:, episode_columns + [drug_column, "_state"]]
    )
    episodes = rows.loc[:, episode_columns].drop_duplicates().reset_index(drop=True)
    episodes["_episode_index"] = np.arange(len(episodes))
    collapsed = collapsed.merge(episodes, on=episode_columns, how="inner")
    order = collapsed[drug_column].value_counts()
    drugs = [str(label) for label in order.index]
    drug_index = {label: position for position, label in enumerate(drugs)}
    states = np.zeros((len(episodes), len(drugs)), dtype=np.int8)
    states[
        collapsed["_episode_index"].to_numpy(),
        collapsed[drug_column].map(drug_index).to_numpy(),
    ] = collapsed["_state"].to_numpy(dtype=np.int8)
    return episodes.drop(columns="_episode_index"), drugs, states, other_spellings


def _context_design(context: pd.DataFrame, levels: dict[str, list]) -> np.ndarray:
    blocks = [np.zeros((len(context), 0), dtype=bool)]
    for column in CONTEXT_COLUMNS:
        if not levels.get(column):
            continue
        values = context[column].to_numpy()
        blocks.append(np.stack([values == level for level in levels[column]], axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _previous_features(states: np.ndarray) -> np.ndarray:
    """[reported, resistant, intermediate] indicators per drug, drug-major."""
    n, d = states.shape
    features = np.empty((n, 3 * d), dtype=np.float32)
    features[:, 0::3] = states > STATE_NOT_REPORTED
    features[:, 1::3] = states == STATE_RESISTANT
    features[:, 2::3] = states == STATE_INTERMEDIATE
    return features


def _round(values: np.ndarray) -> list:
    return np.round(np.asarray(values, dtype=float), 4).tolist()


def fit_chain(
    states: np.ndarray,
    drugs: list[str],
    context: pd.DataFrame,
    other_spellings: dict[str, dict[str, int]],
    *,
    min_cell: int,
    regularisation: float = 0.5,
    max_iter: int = 300,
) -> dict:
    """Fit the per-drug report and result models (see module docstring)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    levels = {
        "year": sorted(int(v) for v in pd.unique(context["year"])),
        "culture": sorted(str(v) for v in pd.unique(context["culture"])),
        "mode": sorted(str(v) for v in pd.unique(context["mode"])),
    }
    ctx = _context_design(context, levels)
    previous = _previous_features(states)
    years = context["year"].to_numpy()
    fitted: list[dict] = []
    for k, drug in enumerate(drugs):
        column = states[:, k]
        reported = column > STATE_NOT_REPORTED
        if int(reported.sum()) < min_cell:
            # Too rare to describe without disclosing individual reports; the
            # drug (and anything conditioned on it) is left out of the replica.
            break
        year_counts = pd.Series(years[reported]).value_counts()
        available_years = sorted(int(y) for y in year_counts.index)
        in_window = np.isin(years, available_years)
        design = np.concatenate([ctx, previous[:, : 3 * k]], axis=1)

        y_report = reported[in_window]
        positives, negatives = int(y_report.sum()), int((~y_report).sum())
        if min(positives, negatives) < min_cell:
            report_model = {"kind": "const", "p": round(positives / max(1, positives + negatives), 6)}
        else:
            model = LogisticRegression(C=regularisation, max_iter=max_iter)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(design[in_window], y_report)
            weights = model.coef_[0]
            report_model = {
                "kind": "logit",
                "intercept": round(float(model.intercept_[0]), 4),
                "ctx": _round(weights[: ctx.shape[1]]),
                "prev": _round(weights[ctx.shape[1]:].reshape(k, 3)) if k else [],
            }

        observed_states = column[reported]
        counts = {state: int((observed_states == state).sum()) for state in RESULT_STATES}
        kept = [state for state in RESULT_STATES if counts[state] >= min_cell]
        if not kept:
            kept = [max(counts, key=counts.get)]
        majority = max(kept, key=lambda state: counts[state])
        target = np.where(np.isin(observed_states, kept), observed_states, majority)
        if len(kept) == 1:
            result_model = {"kind": "const", "states": kept, "probs": [1.0]}
        else:
            model = LogisticRegression(C=regularisation, max_iter=max_iter)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(design[reported], target)
            classes = [int(c) for c in model.classes_]
            if len(classes) == 2:
                # Binary sklearn fits describe class 1 against class 0; write
                # them in the same softmax form as the multiclass case.
                coef = np.vstack([np.zeros_like(model.coef_[0]), model.coef_[0]])
                intercepts = np.array([0.0, float(model.intercept_[0])])
            else:
                coef = model.coef_
                intercepts = model.intercept_
            result_model = {
                "kind": "softmax",
                "states": classes,
                "intercepts": _round(intercepts),
                "ctx": [_round(row[: ctx.shape[1]]) for row in coef],
                "prev": [_round(row[ctx.shape[1]:].reshape(k, 3)) if k else [] for row in coef],
            }

        spellings = {
            value: count
            for value, count in other_spellings.get(drug, {}).items()
            if count >= min_cell
        }
        fitted.append(
            {
                "label": drug,
                "available_years": available_years,
                "report": report_model,
                "result": result_model,
                "other_spellings": spellings or {"Null": 1},
            }
        )
    return {"context_levels": levels, "drugs": fitted}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def sample_chain(model: dict, context: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Draw episode x drug states for synthetic episodes from a fitted chain."""
    drugs = model["drugs"]
    n = len(context)
    states = np.zeros((n, len(drugs)), dtype=np.int8)
    if n == 0 or not drugs:
        return states
    levels = model["context_levels"]
    ctx = _context_design(context, levels)
    previous = np.zeros((n, 3 * len(drugs)), dtype=np.float32)
    years = context["year"].to_numpy()
    for k, spec in enumerate(drugs):
        design = np.concatenate([ctx, previous[:, : 3 * k]], axis=1)
        report = spec["report"]
        if report["kind"] == "const":
            probability = np.full(n, float(report["p"]))
        else:
            weights = np.concatenate([np.asarray(report["ctx"], dtype=float), np.asarray(report["prev"], dtype=float).reshape(-1)])
            probability = _sigmoid(report["intercept"] + design @ weights)
        probability = np.where(np.isin(years, spec["available_years"]), probability, 0.0)
        reported = rng.random(n) < probability
        if reported.any():
            result = spec["result"]
            if result["kind"] == "const":
                probabilities = np.asarray(result["probs"], dtype=float)
                drawn = rng.choice(result["states"], size=int(reported.sum()), p=probabilities / probabilities.sum())
            else:
                logits = np.stack(
                    [
                        intercept
                        + design[reported]
                        @ np.concatenate([np.asarray(ctx_w, dtype=float), np.asarray(prev_w, dtype=float).reshape(-1)])
                        for intercept, ctx_w, prev_w in zip(result["intercepts"], result["ctx"], result["prev"], strict=True)
                    ],
                    axis=1,
                )
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                cumulative = probabilities.cumsum(axis=1)
                draws = rng.random(len(cumulative))[:, None]
                chosen = (draws > cumulative).sum(axis=1)
                drawn = np.asarray(result["states"])[np.minimum(chosen, len(result["states"]) - 1)]
            states[reported, k] = drawn
        previous[:, 3 * k] = states[:, k] > STATE_NOT_REPORTED
        previous[:, 3 * k + 1] = states[:, k] == STATE_RESISTANT
        previous[:, 3 * k + 2] = states[:, k] == STATE_INTERMEDIATE
    return states


def fit_marginal(
    states: np.ndarray,
    drugs: list[str],
    context: pd.DataFrame,
    other_spellings: dict[str, dict[str, int]],
    *,
    min_cell: int,
) -> dict:
    """Independent per-drug model for organisms too rare for the chain model."""
    years = context["year"].to_numpy()
    fitted = []
    for k, drug in enumerate(drugs):
        column = states[:, k]
        reported = column > STATE_NOT_REPORTED
        if int(reported.sum()) < min_cell:
            continue
        available_years = sorted(int(y) for y in pd.unique(years[reported]))
        in_window = np.isin(years, available_years)
        observed_states = column[reported]
        counts = {state: int((observed_states == state).sum()) for state in RESULT_STATES}
        kept = [state for state in RESULT_STATES if counts[state] >= min_cell] or [max(counts, key=counts.get)]
        total = sum(counts[state] for state in kept)
        spellings = {v: c for v, c in other_spellings.get(drug, {}).items() if c >= min_cell}
        fitted.append(
            {
                "label": drug,
                "available_years": available_years,
                "report": {"kind": "const", "p": round(float(reported[in_window].mean()), 6)},
                "result": {"kind": "const", "states": kept, "probs": [round(counts[s] / total, 6) for s in kept]},
                "other_spellings": spellings or {"Null": 1},
            }
        )
    levels = {"year": sorted(int(v) for v in pd.unique(years)), "culture": [], "mode": []}
    return {"context_levels": levels, "drugs": fitted}


def pair_statistics(
    states: np.ndarray,
    drugs: list[str],
    *,
    min_cell: int,
    continuity: float = 0.5,
    limit: int = 400,
) -> list[dict]:
    """Naive episode-level escalation ratios and co-reporting for well-supported pairs.

    Used only to compare the replica with the real data: no eligibility or
    availability filtering, just P(downstream reported | upstream R) over
    P(downstream reported | upstream S) with the pipeline's continuity correction.
    """
    reported = states > STATE_NOT_REPORTED
    resistant = states == STATE_RESISTANT
    susceptible = states == STATE_SUSCEPTIBLE
    rows = []
    for u in range(len(drugs)):
        n_r = int(resistant[:, u].sum())
        n_s = int(susceptible[:, u].sum())
        if n_r < min_cell or n_s < min_cell:
            continue
        for d in range(len(drugs)):
            if d == u:
                continue
            k_r = int((resistant[:, u] & reported[:, d]).sum())
            k_s = int((susceptible[:, u] & reported[:, d]).sum())
            if k_r + k_s < min_cell:
                continue
            both = int((reported[:, u] & reported[:, d]).sum())
            p_d_given_u = both / max(1, int(reported[:, u].sum()))
            p_u_given_d = both / max(1, int(reported[:, d].sum()))
            ratio = ((k_r + continuity) / (n_r + 2 * continuity)) / ((k_s + continuity) / (n_s + 2 * continuity))
            rows.append(
                {
                    "upstream": drugs[u],
                    "downstream": drugs[d],
                    "support": n_r + n_s,
                    "escalation_ratio": round(float(ratio), 5),
                    "co_report_min": round(float(min(p_d_given_u, p_u_given_d)), 5),
                }
            )
    rows.sort(key=lambda row: row["support"], reverse=True)
    return rows[:limit]
