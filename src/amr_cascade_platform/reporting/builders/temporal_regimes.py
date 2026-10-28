"""Year-by-year course of each validated directional observation pattern within each site.

Post hoc and descriptive: the pooled escalation ratio (ER) averages over calendar time, and a site
can observe an antibiotic one way for some years and another way later (a cascade rule ended, a panel
changed). For every validated pattern, site and calendar year this module counts the episode-pair rows
of each upstream branch, the share with the downstream antibiotic observed (pR, pS) and the
continuity-corrected ER, then classifies each evaluable year against the pattern's validated direction.

A site-year is evaluable when each branch has at least MIN_BRANCH rows and the downstream antibiotic was
observed in at least MIN_OBSERVED of them (so a year in which it was off or nearly off the panel is not read
as "absent" or "reversed").
Whether the yearly ERs of a site differ by more than sampling would produce is tested with Cochran's Q on
the log ER (delta-method variances of the continuity-corrected shares), Benjamini-Hochberg across series.
Calendar years come from the date-shifted order times (shifts of up to two months at ECU Health and UT
Southwestern; unstated for Stanford).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

MIN_BRANCH = 10        # episode-pair rows per upstream branch for an evaluable site-year
MIN_OBSERVED = 10      # rows with the downstream antibiotic observed (both branches) for an evaluable site-year
PRESENT_FOLD = 1.5     # ER at least this far from 1 in the validated direction: pattern present
ABSENT_FOLD = 1.1      # ER within this factor of 1: no difference that year
MIN_YEARS = 5          # evaluable years needed to describe a pattern's course at a site
FDR = 0.05             # Benjamini-Hochberg level for the across-year heterogeneity test
VALIDATED = ("robust", "supported")
# Patterns shown in the figure and quoted in the text (internal labels of this dataset).
EXAMPLE_PATTERNS = (("GENTAMICIN", "AMIKACIN"), ("CEFTRIAXON", "CEFTAZIDIM"), ("CEFEPIM", "IMIPENEM"),
                    ("CEFTRIAXON", "FOSFOMYCIN"), ("AZTREONAM", "PIPERACILLIN/TAZOBACTAM"))
_COURSE_KEYS = {"present every evaluable year": "every_year", "never present": "never",
                "present in some years, absent or reversed in others": "switching",
                "present in some years, weaker in others": "weaker"}
_COLUMNS = ["source_site", "order_time_jittered", "upstream_antibiotic", "downstream_antibiotic",
            "upstream_susceptibility", "downstream_tested", "downstream_eligible"]
_STATE_ORDER = ("present", "intermediate", "absent", "reversed")


def validated_patterns(validation: pd.DataFrame) -> pd.DataFrame:
    """Validated patterns with their direction (+1 escalation, -1 suppression) from the combined ER."""
    frame = validation.loc[validation["validation_status"].astype(str).str.lower().isin(VALIDATED),
                           ["upstream_antibiotic", "downstream_antibiotic", "validation_status", "observed_escalation_ratio"]].copy()
    frame["upstream_antibiotic"] = frame["upstream_antibiotic"].astype(str)
    frame["downstream_antibiotic"] = frame["downstream_antibiotic"].astype(str)
    frame["direction"] = frame["observed_escalation_ratio"].astype(float).gt(1.0).map({True: 1, False: -1})
    return frame.reset_index(drop=True)


def yearly_counts(pairs_path: Path, patterns: pd.DataFrame, batch_rows: int = 2_000_000) -> pd.DataFrame:
    """nR, nS, kR, kS per pattern, site and calendar year from the eligible, resistant-or-susceptible rows.

    Reads the pair table in batches so memory stays flat on the 100M-row table.
    """
    wanted = set(zip(patterns["upstream_antibiotic"], patterns["downstream_antibiotic"]))
    dataset = ds.dataset(str(pairs_path), format="parquet")
    rows = (ds.field("downstream_eligible") == 1) & ds.field("upstream_susceptibility").isin(["RESISTANT", "SUSCEPTIBLE"])
    keys = ["source_site", "year", "upstream_antibiotic", "downstream_antibiotic", "upstream_susceptibility"]
    parts = []
    for batch in dataset.to_batches(columns=_COLUMNS, filter=rows, batch_size=batch_rows):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        for column in ("source_site", "upstream_antibiotic", "downstream_antibiotic", "upstream_susceptibility"):
            frame[column] = frame[column].astype(str)
        pair = pd.MultiIndex.from_arrays([frame["upstream_antibiotic"], frame["downstream_antibiotic"]])
        frame = frame.loc[pair.isin(wanted)]
        if frame.empty:
            continue
        frame["year"] = pd.to_numeric(frame["order_time_jittered"].astype(str).str[:4], errors="coerce")
        frame = frame.dropna(subset=["year"])
        frame["observed"] = pd.to_numeric(frame["downstream_tested"], errors="coerce").fillna(0).astype(int)
        parts.append(frame.groupby(keys, observed=True)["observed"].agg(n="size", k="sum").reset_index())
    if not parts:
        return pd.DataFrame(columns=["upstream_antibiotic", "downstream_antibiotic", "source_site", "year", "nR", "kR", "nS", "kS"])
    totals = pd.concat(parts, ignore_index=True).groupby(keys, observed=True)[["n", "k"]].sum().reset_index()
    wide = totals.pivot_table(index=["upstream_antibiotic", "downstream_antibiotic", "source_site", "year"],
                              columns="upstream_susceptibility", values=["n", "k"], fill_value=0, aggfunc="sum")
    wide.columns = [f"{stat}{'R' if branch == 'RESISTANT' else 'S'}" for stat, branch in wide.columns]
    for column in ("nR", "kR", "nS", "kS"):
        if column not in wide.columns:
            wide[column] = 0
    wide = wide.reset_index()
    wide["year"] = wide["year"].astype(int)
    return wide[["upstream_antibiotic", "downstream_antibiotic", "source_site", "year", "nR", "kR", "nS", "kS"]]


def classify_years(counts: pd.DataFrame, patterns: pd.DataFrame, continuity: float) -> pd.DataFrame:
    """Per pattern, site and year: shares observed, ER and the state against the validated direction."""
    frame = counts.merge(patterns, on=["upstream_antibiotic", "downstream_antibiotic"], how="inner", validate="many_to_one")
    for column in ("nR", "kR", "nS", "kS"):
        frame[column] = frame[column].astype(int)
    frame["pR"] = frame["kR"] / frame["nR"].where(frame["nR"] > 0)
    frame["pS"] = frame["kS"] / frame["nS"].where(frame["nS"] > 0)
    corrected_r = (frame["kR"] + continuity) / (frame["nR"] + 2 * continuity)
    corrected_s = (frame["kS"] + continuity) / (frame["nS"] + 2 * continuity)
    frame["er"] = corrected_r / corrected_s
    frame["log_er_var"] = ((1 - corrected_r) / (corrected_r * (frame["nR"] + 2 * continuity))
                           + (1 - corrected_s) / (corrected_s * (frame["nS"] + 2 * continuity)))
    frame["evaluable"] = frame["nR"].ge(MIN_BRANCH) & frame["nS"].ge(MIN_BRANCH) & (frame["kR"] + frame["kS"]).ge(MIN_OBSERVED)
    oriented = frame["er"] ** frame["direction"]  # >1 means the validated direction
    state = pd.Series("intermediate", index=frame.index)
    state[oriented.ge(PRESENT_FOLD)] = "present"
    state[oriented.le(1 / PRESENT_FOLD)] = "reversed"
    state[oriented.gt(1 / ABSENT_FOLD) & oriented.lt(ABSENT_FOLD)] = "absent"
    frame["state"] = state.where(frame["evaluable"], "not evaluable")
    return frame.sort_values(["upstream_antibiotic", "downstream_antibiotic", "source_site", "year"]).reset_index(drop=True)


def _course(group: pd.DataFrame) -> pd.Series:
    years = group.loc[group["evaluable"]]
    counts = years["state"].value_counts()
    present = years.loc[years["state"].eq("present"), "year"]
    evaluable = len(years)
    n_present, n_absent, n_reversed = (int(counts.get(s, 0)) for s in ("present", "absent", "reversed"))
    if evaluable < MIN_YEARS:
        course = "fewer than 5 evaluable years"
    elif n_present == evaluable:
        course = "present every evaluable year"
    elif n_present == 0:
        course = "never present"
    elif n_absent + n_reversed > 0:
        course = "present in some years, absent or reversed in others"
    else:
        course = "present in some years, weaker in others"
    # A change year: the first year of a new state when the pattern moves between present and absent or
    # reversed, skipping intermediate years in between (so 2010 present, 2012 intermediate, 2013 absent is 2013).
    clear = years.loc[years["state"].isin(("present", "absent", "reversed"))].sort_values("year")
    sequence = list(zip(clear["year"].astype(int), clear["state"].eq("present")))
    changes = [year for (_, was), (year, now) in zip(sequence, sequence[1:]) if was != now]
    # Cochran's Q: do the yearly log ERs differ by more than their sampling variances allow?
    theta, weight = np.log(years["er"].to_numpy(float)), 1 / years["log_er_var"].to_numpy(float)
    q = float(np.sum(weight * (theta - np.sum(weight * theta) / np.sum(weight)) ** 2)) if evaluable >= 2 else np.nan
    return pd.Series({
        "first_evaluable_year": int(years["year"].min()) if evaluable else None,
        "last_evaluable_year": int(years["year"].max()) if evaluable else None,
        "evaluable_years": evaluable, "present_years": n_present, "absent_years": n_absent,
        "reversed_years": n_reversed, "intermediate_years": int(counts.get("intermediate", 0)),
        "first_present_year": int(present.min()) if len(present) else None,
        "last_present_year": int(present.max()) if len(present) else None,
        "change_years": ";".join(str(y) for y in changes), "course": course,
        "year_heterogeneity_q": q, "year_heterogeneity_p": float(chi2.sf(q, evaluable - 1)) if evaluable >= 2 else np.nan,
    })


def summarise(years: pd.DataFrame) -> pd.DataFrame:
    keys = ["upstream_antibiotic", "downstream_antibiotic", "source_site"]
    meta = years.groupby(keys)[["validation_status", "observed_escalation_ratio", "direction"]].first()
    courses = years.groupby(keys)[["year", "evaluable", "state", "er", "log_er_var"]].apply(_course)
    summary = meta.join(courses).reset_index()
    followed = summary["evaluable_years"].ge(MIN_YEARS) & summary["year_heterogeneity_p"].notna()
    summary["year_heterogeneity_q_value"] = np.nan
    if followed.any():
        summary.loc[followed, "year_heterogeneity_q_value"] = multipletests(
            summary.loc[followed, "year_heterogeneity_p"], method="fdr_bh")[1]
    summary["varies_beyond_sampling"] = summary["year_heterogeneity_q_value"].lt(FDR)
    return summary


def _slug(upstream: str, downstream: str) -> str:
    return f"{upstream}_{downstream}".lower().replace("/", "_").replace(" ", "_")


def _year(value) -> str | None:
    """Years are text, so the paper prints 2017 rather than the integer format 2,017."""
    return None if pd.isna(value) else str(int(value))


def headline_numbers(summary: pd.DataFrame) -> dict[str, object]:
    """Counts quoted in the text: all validated patterns, the strong ones (at least 1.5-fold from ER = 1) and the rest."""
    fold = summary["observed_escalation_ratio"].astype(float).map(lambda er: max(er, 1 / er) if er > 0 else math.inf)
    out: dict[str, object] = {"min_branch": MIN_BRANCH, "min_observed": MIN_OBSERVED, "present_fold": PRESENT_FOLD,
                              "absent_fold": ABSENT_FOLD, "min_years": MIN_YEARS, "fdr": FDR}
    for label, subset in (("validated", summary), ("strong", summary.loc[fold.ge(PRESENT_FOLD)]),
                          ("weak", summary.loc[fold.lt(PRESENT_FOLD)])):
        followed = subset.loc[subset["evaluable_years"].ge(MIN_YEARS)]
        switching = followed["course"].eq("present in some years, absent or reversed in others")
        entry: dict[str, object] = {
            "patterns": int(subset[["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates().shape[0]),
            "patterns_followed": int(followed[["upstream_antibiotic", "downstream_antibiotic"]].drop_duplicates().shape[0]),
            "site_series_followed": int(len(followed)),
            **{key: int(followed["course"].eq(course).sum()) for course, key in _COURSE_KEYS.items()},
            "with_reversed_year": int(followed["reversed_years"].gt(0).sum()),
            "varies_beyond_sampling": int(followed["varies_beyond_sampling"].sum()),
            "switching_and_varies_beyond_sampling": int((switching & followed["varies_beyond_sampling"]).sum()),
        }
        # Do the changes of different patterns fall in the same years within a site?
        sites = {}
        for site, group in followed.loc[switching].groupby("source_site"):
            years = [int(y) for text in group["change_years"].dropna().astype(str) for y in text.split(";") if y]
            if years:
                mode = int(pd.Series(years).mode().min())
                sites[site] = {"series": int(len(group)), "changes": len(years), "mode_year": _year(mode),
                               "changes_within_1_of_mode": int(sum(abs(y - mode) <= 1 for y in years))}
        entry["change_years_by_site"] = sites
        out[label] = entry
    examples = {}
    for upstream, downstream in EXAMPLE_PATTERNS:
        rows = summary.loc[summary["upstream_antibiotic"].eq(upstream) & summary["downstream_antibiotic"].eq(downstream)]
        examples[_slug(upstream, downstream)] = {
            row["source_site"]: {
                "course": _COURSE_KEYS.get(row["course"], "few_years"),
                "evaluable_years": int(row["evaluable_years"]),
                "first_evaluable_year": _year(row["first_evaluable_year"]), "last_evaluable_year": _year(row["last_evaluable_year"]),
                "first_present_year": _year(row["first_present_year"]), "last_present_year": _year(row["last_present_year"]),
                "reversed_years": int(row["reversed_years"]),
            } for _, row in rows.iterrows()}
    out["examples"] = examples
    return out
