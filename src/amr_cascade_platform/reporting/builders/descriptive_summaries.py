"""Descriptive summaries of the analysis set, its observation process and its denominators.

These are the numbers a reader needs before any pattern estimate: who is in the
analysis set, how the organism x antibiotic opportunity grid splits into
ineligible and eligible space, how much of the eligible space was observed and
when, how many results a culture episode carries, what the observed results
were, how often patients recur, and which rows each exclusion rule removed.

Each function takes data frames and returns data frames. ManuscriptTableBuilder
loads the inputs, the report step writes the tables, and the supplementary
builder turns them into the numbers and table bodies the manuscripts quote.
Every function reports all sites first and then each site, so a site-level
number and its pooled counterpart always come from the same code.

Most functions take two lean frames built once by the table builder:

* ``episodes``: one row per culture episode -- ``episode_id``, ``source_site``,
  ``anon_id``, ``episode_time`` (UTC timestamp or NaT) and ``specimen_group``.
* ``opportunities``: one row per episode x antibiotic in the eligibility grid --
  ``episode_id``, ``source_site``, ``availability_era``, ``antibiotic``,
  ``is_intrinsic_resistance``, ``is_eligible``, ``is_observed_tested`` and
  ``availability_support_n``, exactly as the eligibility service wrote them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

from amr_cascade_platform.core.utils.site_labels import ALL_SITES, scope_order, site_label

ALL_ANTIBIOTICS = "__all_antibiotics__"
NOT_RECORDED = "Not recorded"

SPECIMEN_GROUPS = ("Urine", "Blood", "Other or not recorded")
ALL_SPECIMENS = "All specimens"

# The adjusted model's age categories (CascadeCovariateBuilder._age_bin). The
# source extracts record age as bands (18-24, 25-34, ..., 85-89, above 90), which
# the demographics builder turns into band midpoints, so the model's cut points
# at 40 and 65 fall between bands: its categories hold ages 18-44, 45-64 and 65+.
_AGE_GROUPS = (("lt18", "Under 18"), ("18_40", "18–44"), ("41_65", "45–64"), ("66_plus", "65 or older"))
_SPECIMENS = (
    ("urine", "Urine"),
    ("blood", "Blood"),
    ("respiratory", "Respiratory"),
    ("wound_skin_soft_tissue", "Wound, skin or soft tissue"),
    ("gastrointestinal", "Gastrointestinal"),
    ("sterile_fluid", "Sterile body fluid"),
    ("other", "Other"),
)
_ORDERING = (("inpatient", "Inpatient"), ("outpatient", "Outpatient"), ("emergency", "Emergency department"))
# (indicator, record flag, label, no-record label): binary covariates of the adjusted
# model. The flag marks a patient with any record before the culture in that source
# table (the model includes it alongside the indicator), so each indicator is
# reported among episodes with a record and the rest are reported as having none.
_HISTORY = (
    ("cov_prior_abx_any_90d", "cov_prior_abx_available", "Antibiotic exposure", "No medication record"),
    ("cov_prior_same_organism_any_90d", "cov_prior_organism_available", "Infection with the same genus", "No prior culture record"),
    ("cov_nursing_home_90d", "cov_nursing_home_available", "Nursing-home stay", "No nursing-home record"),
    ("cov_prior_procedure_90d", "cov_prior_procedure_available", "Procedure", "No procedure record"),
)
_RESULT_RANK = {"SUSCEPTIBLE": 0, "INTERMEDIATE": 1, "RESISTANT": 2}
_PATIENT_BANDS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, None, "More than 10"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scopes(frame: pd.DataFrame, sites: tuple[str, ...] | list[str]) -> Iterator[tuple[str, pd.DataFrame]]:
    """(scope, rows) for all sites, then each site with rows, in configured order."""
    site_values = frame["source_site"].astype(str) if "source_site" in frame.columns else pd.Series(dtype=str)
    for scope in scope_order(tuple(sites), site_values.unique()):
        if scope == ALL_SITES:
            yield scope, frame
            continue
        part = frame.loc[site_values.eq(scope).to_numpy()]
        if not part.empty:
            yield scope, part


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """A 0/1 column as booleans; an absent column counts as all zero."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int).eq(1)


def _share(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _quartiles(values: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return math.nan, math.nan, math.nan
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return float(median), float(q1), float(q3)


def era_label(years: pd.Series, width: int) -> pd.Series:
    """Calendar year -> the eligibility service's era label, e.g. 2007 -> "2005-2009"."""
    numeric = pd.to_numeric(years, errors="coerce")
    start = (numeric // width) * width
    label = start.astype("Int64").astype("string") + "-" + (start + width - 1).astype("Int64").astype("string")
    return label.fillna(NOT_RECORDED)


def era_display(era: object) -> str:
    """"2005-2009" -> "2005–2009" for tables and figures."""
    return str(era).replace("-", "–")


def specimen_group(category: pd.Series) -> pd.Series:
    """The covariate builder's specimen category -> Urine / Blood / Other or not recorded."""
    text = category.astype("string").fillna("unknown")
    return text.map({"urine": "Urine", "blood": "Blood"}).fillna("Other or not recorded").astype(str)


# ---------------------------------------------------------------------------
# Cohort characteristics by site (Table 1)
# ---------------------------------------------------------------------------


def _count_rows(section: str, rows: list[dict], label: str, n: int, denominator: int | None) -> None:
    rows.append({"section": section, "characteristic": label, "statistic": "n_pct" if denominator is not None else "n",
                 "n": int(n), "denominator": denominator, "share": _share(n, denominator) if denominator is not None else math.nan,
                 "median": math.nan, "q1": math.nan, "q3": math.nan})


def _median_row(section: str, rows: list[dict], label: str, values: pd.Series) -> None:
    median, q1, q3 = _quartiles(values)
    rows.append({"section": section, "characteristic": label, "statistic": "median_iqr", "n": int(values.notna().sum()),
                 "denominator": None, "share": math.nan, "median": median, "q1": q1, "q3": q3})


def _categorical(section: str, rows: list[dict], values: pd.Series, levels: list[tuple[str, str]], n: int) -> None:
    counts = values.astype("string").fillna("unknown").value_counts()
    for code, label in levels:
        _count_rows(section, rows, label, int(counts.get(code, 0)), n)


def _characteristics(part: pd.DataFrame, levels: dict[str, list[tuple[str, str]]]) -> list[dict]:
    rows: list[dict] = []
    n = len(part)
    _count_rows("Analysis set", rows, "Culture episodes", n, None)
    if "anon_id" in part.columns:
        patients = part[["source_site", "anon_id"]].drop_duplicates()
        _count_rows("Analysis set", rows, "Patients", len(patients), None)
    if "cov_age_bin" in part.columns:
        _categorical("Age, years", rows, part["cov_age_bin"], levels["age"], n)
    if "cov_sex" in part.columns:
        _categorical("Sex, source code", rows, part["cov_sex"], levels["sex"], n)
    if "cov_calendar_year" in part.columns:
        _categorical("Calendar period", rows, part["_era"], levels["era"], n)
    if "cov_specimen_type" in part.columns:
        _categorical("Specimen", rows, part["cov_specimen_type"], levels["specimen"], n)
    if "cov_ordering_mode" in part.columns:
        _categorical("Ordering context", rows, part["cov_ordering_mode"], levels["ordering"], n)
    if {"cov_er_status", "cov_er_available"} <= set(part.columns):
        ward = _flag(part, "cov_er_available") | _flag(part, "cov_icu_available")
        evaluable = int(ward.sum())
        _count_rows("Care unit", rows, "Emergency-department presentation", int((_flag(part, "cov_er_status") & _flag(part, "cov_er_available")).sum()), evaluable)
        _count_rows("Care unit", rows, "Intensive care", int((_flag(part, "cov_icu_status") & _flag(part, "cov_icu_available")).sum()), evaluable)
        _count_rows("Care unit", rows, "Ward data not recorded", int((~ward).sum()), n)
    for indicator, available, label, no_record in _HISTORY:
        if {indicator, available} <= set(part.columns):
            record = _flag(part, available)
            section = "In the 90 days before culture"
            _count_rows(section, rows, label, int((_flag(part, indicator) & record).sum()), int(record.sum()))
            _count_rows(section, rows, no_record, int((~record).sum()), n)
    if {"cov_comorbidity_count", "cov_comorbidity_available"} <= set(part.columns):
        available = _flag(part, "cov_comorbidity_available")
        _median_row("Comorbidity and deprivation", rows, "Comorbidity count, median [IQR]", part.loc[available, "cov_comorbidity_count"])
        _count_rows("Comorbidity and deprivation", rows, "Comorbidity data not available", int((~available).sum()), n)
    if {"cov_adi_score", "cov_adi_available"} <= set(part.columns):
        available = _flag(part, "cov_adi_available")
        # Both ADI measures start at 1; the covariate builder writes 0 where the
        # source row lacks the value, so a zero is missing, not a rank.
        score = pd.to_numeric(part.loc[available, "cov_adi_score"], errors="coerce")
        _median_row("Comorbidity and deprivation", rows, "Area Deprivation Index score, median [IQR]", score.where(score.gt(0)))
        if "cov_adi_state_rank" in part.columns:
            rank = pd.to_numeric(part.loc[available, "cov_adi_state_rank"], errors="coerce")
            _median_row("Comorbidity and deprivation", rows, "Area Deprivation Index state rank, median [IQR]", rank.where(rank.gt(0)))
        _count_rows("Comorbidity and deprivation", rows, "Area Deprivation Index not available", int((~available).sum()), n)
    return rows


def _levels(frame: pd.DataFrame, era_order: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Category levels present anywhere, in a fixed order, so every scope has the same rows."""

    def present(column: str, known: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
        values = set(frame[column].astype("string").fillna("unknown")) if column in frame.columns else set()
        levels = [(code, label) for code, label in known if code in values]
        extra = sorted(values - {code for code, _ in known} - {"unknown"})
        levels += [(code, code.replace("_", " ").capitalize()) for code in extra]
        if "unknown" in values:
            levels.append(("unknown", NOT_RECORDED))
        return levels

    sexes = set(frame["cov_sex"].astype("string").fillna("unknown")) if "cov_sex" in frame.columns else set()
    sex_levels = [(code, code.capitalize()) for code in ("female", "male") if code in sexes]
    sex_levels += [(code, f"Code {code.removeprefix('code_')}") for code in sorted(sexes) if code.startswith("code_")]
    sex_levels += [(code, code.replace("_", " ").capitalize()) for code in sorted(sexes - {"female", "male", "unknown"}) if not code.startswith("code_")]
    if "unknown" in sexes:
        sex_levels.append(("unknown", NOT_RECORDED))
    return {
        "age": present("cov_age_bin", _AGE_GROUPS),
        "sex": sex_levels,
        "era": [(era, era_display(era)) for era in era_order],
        "specimen": present("cov_specimen_type", _SPECIMENS),
        "ordering": present("cov_ordering_mode", _ORDERING),
    }


def format_characteristic(row: pd.Series | dict) -> str:
    statistic = row["statistic"]
    if statistic == "n":
        return f"{int(row['n']):,}"
    if statistic == "n_pct":
        if not row["denominator"]:
            return "–"
        return f"{int(row['n']):,} ({100.0 * float(row['share']):.1f})"
    if pd.isna(row["median"]):
        return "–"
    return f"{row['median']:.0f} [{row['q1']:.0f}–{row['q3']:.0f}]"


def cohort_characteristics(
    covariates: pd.DataFrame, sites: tuple[str, ...] | list[str], era_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table 1: the adjusted model's episode-level covariates, per site and for all sites.

    ``covariates`` is CascadeCovariateBuilder's output for the analysis set's
    culture episodes (one row per episode), so the table describes exactly the
    values the adjusted models use, including how often each could not be
    evaluated. Calendar years are grouped into the eligibility service's eras.
    Returns the long table (one row per characteristic and scope, with counts,
    denominators, shares and quartiles) and the display table (one column per
    scope, strings as printed).
    """
    frame = covariates.copy()
    frame["_era"] = era_label(frame.get("cov_calendar_year", pd.Series(index=frame.index, dtype="object")), era_years)
    eras = sorted(era for era in frame["_era"].unique() if era != NOT_RECORDED)
    era_order = eras + ([NOT_RECORDED] if (frame["_era"] == NOT_RECORDED).any() else [])
    levels = _levels(frame, era_order)
    records = []
    for scope, part in _scopes(frame, sites):
        for order, row in enumerate(_characteristics(part, levels)):
            records.append({"scope": scope, "order": order, **row})
    long = pd.DataFrame(records)
    if long.empty:
        return long, pd.DataFrame()
    long["display"] = long.apply(format_characteristic, axis=1)
    scopes = list(dict.fromkeys(long["scope"]))
    display = (
        long.pivot_table(index=["order", "section", "characteristic"], columns="scope", values="display", aggfunc="first")
        .reindex(columns=scopes)
        .reset_index()
        .sort_values("order", kind="mergesort")
        .drop(columns="order")
        .fillna("–")
    )
    display.columns.name = None
    return long.drop(columns="order"), display.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Opportunity space: how the episode x antibiotic grid splits (Table B)
# ---------------------------------------------------------------------------


def opportunity_space(opportunities: pd.DataFrame, sites: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """The episode x antibiotic grid split into intrinsic, not operationally available, and eligible.

    Categories are exclusive and exhaustive: a row is intrinsic when the
    reference lists the organism as intrinsically resistant, otherwise
    eligible when ``is_eligible`` (biologically interpretable and operationally
    available), otherwise not operationally available (no observed result for
    the drug in its site-era stratum at the configured support). Observed rows
    are counted within each category, because an intrinsic drug can still carry
    a recorded result and, under a stricter availability rule, so can an
    unavailable one; the unobserved eligible rows are therefore computed within
    the eligible category, never as eligible minus all observed rows.
    """
    intrinsic = _flag(opportunities, "is_intrinsic_resistance")
    eligible = _flag(opportunities, "is_eligible") & ~intrinsic
    unavailable = ~intrinsic & ~eligible
    observed = _flag(opportunities, "is_observed_tested")
    frame = pd.DataFrame(
        {
            "source_site": opportunities["source_site"].astype(str).to_numpy(),
            "antibiotic": opportunities["antibiotic"].astype(str).to_numpy(),
            "intrinsic": intrinsic.to_numpy(),
            "unavailable": unavailable.to_numpy(),
            "eligible": eligible.to_numpy(),
            "observed": observed.to_numpy(),
        }
    )
    if "episode_id" in opportunities.columns:
        frame["episode_id"] = opportunities["episode_id"].to_numpy()
    rows = []
    for scope, part in _scopes(frame, sites):
        grid = len(part)
        row = {
            "site": scope,
            "episodes": int(part["episode_id"].nunique()) if "episode_id" in part.columns else math.nan,
            "antibiotics": int(part["antibiotic"].nunique()),
            "grid_rows": grid,
            "intrinsic_rows": int(part["intrinsic"].sum()),
            "intrinsic_observed_rows": int((part["intrinsic"] & part["observed"]).sum()),
            "not_available_rows": int(part["unavailable"].sum()),
            "not_available_observed_rows": int((part["unavailable"] & part["observed"]).sum()),
            "eligible_rows": int(part["eligible"].sum()),
            "eligible_observed_rows": int((part["eligible"] & part["observed"]).sum()),
            "eligible_unobserved_rows": int((part["eligible"] & ~part["observed"]).sum()),
            "observed_rows": int(part["observed"].sum()),
        }
        row["observed_ineligible_rows"] = row["intrinsic_observed_rows"] + row["not_available_observed_rows"]
        row["intrinsic_share"] = _share(row["intrinsic_rows"], grid)
        row["not_available_share"] = _share(row["not_available_rows"], grid)
        row["eligible_share"] = _share(row["eligible_rows"], grid)
        row["eligible_observed_share_of_grid"] = _share(row["eligible_observed_rows"], grid)
        row["eligible_unobserved_share_of_grid"] = _share(row["eligible_unobserved_rows"], grid)
        row["observed_share_of_eligible"] = _share(row["eligible_observed_rows"], row["eligible_rows"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exposure of the era-level availability rule (thin support, cold start)
# ---------------------------------------------------------------------------

_CELL = ["source_site", "availability_era", "antibiotic"]


def availability_exposure(
    opportunities: pd.DataFrame,
    episodes: pd.DataFrame,
    sites: tuple[str, ...] | list[str],
    thin_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much of the eligible space rests on thin or out-of-window availability evidence.

    Operational availability is decided per site x era x antibiotic stratum
    from the number of observed results in it (``availability_support_n``).
    Two weaknesses of that era-level rule are counted directly:

    * thin support: eligible opportunities in strata whose support is below
      ``thin_support`` (the stricter availability sensitivity's minimum);
    * outside the observed window: eligible opportunities whose culture was
      ordered before the stratum's first observed result for the drug (cold
      start) or after its last (a drug dropped from the panel mid-era).

    Returns the summary per scope and the per-stratum detail.
    """
    eligible = opportunities.loc[_flag(opportunities, "is_eligible").to_numpy(), ["episode_id", *_CELL, "is_observed_tested", "availability_support_n"]]
    frame = eligible.merge(episodes[["episode_id", "episode_time"]], on="episode_id", how="left", validate="many_to_one")
    observed = _flag(frame, "is_observed_tested")
    window = (
        frame.loc[observed.to_numpy()]
        .groupby(_CELL, observed=True, dropna=False)["episode_time"]
        .agg(first_observed="min", last_observed="max")
        .reset_index()
    )
    frame = frame.merge(window, on=_CELL, how="left", validate="many_to_one")
    support = pd.to_numeric(frame["availability_support_n"], errors="coerce").fillna(0)
    frame["thin"] = support.lt(thin_support).to_numpy()
    known = frame["episode_time"].notna() & frame["first_observed"].notna()
    frame["before_first"] = (known & frame["episode_time"].lt(frame["first_observed"])).to_numpy()
    frame["after_last"] = (known & frame["episode_time"].gt(frame["last_observed"])).to_numpy()
    frame["time_unknown"] = frame["episode_time"].isna().to_numpy()
    frame["observed"] = observed.to_numpy()

    cells = (
        frame.groupby(_CELL, observed=True, dropna=False)
        .agg(
            availability_support_n=("availability_support_n", "first"),
            eligible_n=("episode_id", "size"),
            observed_n=("observed", "sum"),
            first_observed=("first_observed", "first"),
            last_observed=("last_observed", "first"),
            before_first_observed_n=("before_first", "sum"),
            after_last_observed_n=("after_last", "sum"),
        )
        .reset_index()
        .rename(columns={"source_site": "site", "availability_era": "era"})
    )
    cells["thin_support"] = pd.to_numeric(cells["availability_support_n"], errors="coerce").fillna(0).lt(thin_support)

    rows = []
    for scope, part in _scopes(frame, sites):
        n = len(part)
        outside = part["before_first"] | part["after_last"]
        strata = part[_CELL].drop_duplicates()
        thin_strata = part.loc[part["thin"], _CELL].drop_duplicates()
        rows.append(
            {
                "site": scope,
                "thin_support_threshold": int(thin_support),
                "eligible_rows": n,
                "strata": len(strata),
                "thin_strata": len(thin_strata),
                "thin_rows": int(part["thin"].sum()),
                "thin_share": _share(part["thin"].sum(), n),
                "thin_observed_rows": int((part["thin"] & part["observed"]).sum()),
                "before_first_observed_rows": int(part["before_first"].sum()),
                "before_first_observed_share": _share(part["before_first"].sum(), n),
                "after_last_observed_rows": int(part["after_last"].sum()),
                "after_last_observed_share": _share(part["after_last"].sum(), n),
                "outside_observed_window_rows": int(outside.sum()),
                "outside_observed_window_share": _share(outside.sum(), n),
                "thin_or_outside_rows": int((part["thin"] | outside).sum()),
                "thin_or_outside_share": _share((part["thin"] | outside).sum(), n),
                "time_unknown_rows": int(part["time_unknown"].sum()),
            }
        )
    return pd.DataFrame(rows), cells


# ---------------------------------------------------------------------------
# Panel breadth: results per culture episode
# ---------------------------------------------------------------------------


def panel_breadth(
    opportunities: pd.DataFrame, episodes: pd.DataFrame, sites: tuple[str, ...] | list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Antibiotics observed and eligible per culture episode, by site and specimen group.

    Observed counts every antibiotic with an interpretable result in the
    episode; eligible counts the episode's eligible opportunities; the
    observed share pools eligible observed over eligible opportunities.
    Episodes whose only rows were non-interpretive stay in the analysis set
    with zero observed antibiotics and are counted separately. Returns the
    summary and the distribution of observed antibiotics per episode (the
    figure's input).
    """
    observed = _flag(opportunities, "is_observed_tested")
    eligible = _flag(opportunities, "is_eligible")
    per_episode = (
        pd.DataFrame({"episode_id": opportunities["episode_id"].to_numpy(), "observed": observed.to_numpy(),
                      "eligible": eligible.to_numpy(), "eligible_observed": (observed & eligible).to_numpy()})
        .groupby("episode_id")[["observed", "eligible", "eligible_observed"]]
        .sum()
        .reset_index()
    )
    frame = episodes[["episode_id", "source_site", "specimen_group"]].merge(per_episode, on="episode_id", how="left", validate="one_to_one")
    frame[["observed", "eligible", "eligible_observed"]] = frame[["observed", "eligible", "eligible_observed"]].fillna(0).astype(int)

    summary, distribution = [], []
    for scope, part in _scopes(frame, sites):
        for group in (ALL_SPECIMENS, *SPECIMEN_GROUPS):
            rows = part if group == ALL_SPECIMENS else part.loc[part["specimen_group"].eq(group)]
            if rows.empty:
                continue
            observed_median, observed_q1, observed_q3 = _quartiles(rows["observed"])
            eligible_median, eligible_q1, eligible_q3 = _quartiles(rows["eligible"])
            no_result = rows["observed"].eq(0)
            summary.append(
                {
                    "site": scope,
                    "specimen_group": group,
                    "episodes": len(rows),
                    "observed_median": observed_median,
                    "observed_q1": observed_q1,
                    "observed_q3": observed_q3,
                    "observed_mean": float(rows["observed"].mean()),
                    "observed_max": int(rows["observed"].max()),
                    "eligible_median": eligible_median,
                    "eligible_q1": eligible_q1,
                    "eligible_q3": eligible_q3,
                    "episodes_without_result": int(no_result.sum()),
                    "episodes_without_result_share": _share(no_result.sum(), len(rows)),
                    "observed_share_of_eligible": _share(rows["eligible_observed"].sum(), rows["eligible"].sum()),
                }
            )
            counts = rows["observed"].value_counts().sort_index()
            for value, episodes_n in counts.items():
                distribution.append(
                    {"site": scope, "specimen_group": group, "observed_antibiotics": int(value),
                     "episodes": int(episodes_n), "share": _share(episodes_n, len(rows))}
                )
    return pd.DataFrame(summary), pd.DataFrame(distribution)


# ---------------------------------------------------------------------------
# Observation coverage over time
# ---------------------------------------------------------------------------


def coverage_by_era(availability: pd.DataFrame, sites: tuple[str, ...] | list[str], resolver=None) -> pd.DataFrame:
    """Observed share of the eligible space per scope x era, overall and per antibiotic.

    ``availability`` is table U (site x era x antibiotic with ``eligible_n``
    and ``observed_n``). Rows with no eligible opportunity are dropped, so an
    absent (scope, era, antibiotic) means the drug was outside that eligible
    space. ``antibiotic == ALL_ANTIBIOTICS`` rows pool every antibiotic. With a
    ``resolver`` the table also carries each antibiotic's display label and
    WHO AWaRe category.
    """
    if availability.empty:
        return pd.DataFrame()
    table = availability.rename(columns={"site": "source_site"}).copy()
    table["eligible_n"] = pd.to_numeric(table["eligible_n"], errors="coerce").fillna(0).astype(int)
    table["observed_n"] = pd.to_numeric(table["observed_n"], errors="coerce").fillna(0).astype(int)
    table = table.loc[table["eligible_n"].gt(0)]
    rows = []
    for scope, part in _scopes(table, sites):
        per_drug = part.groupby(["era", "antibiotic"], observed=True)[["eligible_n", "observed_n"]].sum().reset_index()
        pooled = part.groupby("era", observed=True)[["eligible_n", "observed_n"]].sum().reset_index().assign(antibiotic=ALL_ANTIBIOTICS)
        for frame in (pooled, per_drug):
            frame = frame.assign(site=scope)
            rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result["observed_share"] = result["observed_n"] / result["eligible_n"]
    result["era_display"] = result["era"].map(era_display)
    columns = ["site", "era", "era_display", "antibiotic", "eligible_n", "observed_n", "observed_share"]
    if resolver is not None:
        drugs = [label for label in result["antibiotic"].unique() if label != ALL_ANTIBIOTICS]
        resolved = {label: resolver.resolve(label) for label in drugs}
        result["antibiotic_display"] = result["antibiotic"].map(
            lambda label: "All antibiotics" if label == ALL_ANTIBIOTICS else _sentence_case(resolved[label].display_label)
        )
        result["aware_category"] = result["antibiotic"].map(lambda label: "" if label == ALL_ANTIBIOTICS else resolved[label].aware_category)
        columns[4:4] = ["antibiotic_display", "aware_category"]
    return result.loc[:, columns]


# ---------------------------------------------------------------------------
# Cumulative antibiogram
# ---------------------------------------------------------------------------


def first_isolate_episodes(episodes: pd.DataFrame) -> pd.Series:
    """Episode ids of each patient's first culture episode per calendar year (CLSI M39).

    Episodes without a usable timestamp cannot be ordered and are left out.
    """
    timed = episodes.loc[episodes["episode_time"].notna(), ["episode_id", "source_site", "anon_id", "episode_time"]].copy()
    if timed.empty:
        return pd.Series(dtype="int64")
    timed["year"] = timed["episode_time"].dt.year
    timed = timed.sort_values(["episode_time", "episode_id"], kind="mergesort")
    return timed.drop_duplicates(["source_site", "anon_id", "year"], keep="first")["episode_id"]


def antibiogram(
    results: pd.DataFrame,
    episodes: pd.DataFrame,
    sites: tuple[str, ...] | list[str],
    resolver,
    min_tested: int,
) -> pd.DataFrame:
    """Observed S / I / R per antibiotic and scope, among episodes with a result for the drug.

    ``results`` holds the observed episode-drug results (``episode_id``,
    ``antibiotic``, ``susceptibility``). An episode with more than one result
    for a drug contributes its most resistant one (R over I over S) and is
    counted as conflicting when the results differ; the primary
    directional analysis instead excludes such an episode from the upstream
    resistant-versus-susceptible contrast. The first-isolate columns restrict
    to each patient's first culture episode per calendar year (CLSI M39).
    ``sparse`` marks fewer than ``min_tested`` tested episodes, below which
    CLSI M39 advises against reporting a percentage.
    """
    frame = results.loc[:, ["episode_id", "antibiotic", "susceptibility"]].copy()
    frame["rank"] = frame["susceptibility"].astype(str).str.upper().map(_RESULT_RANK)
    frame = frame.loc[frame["rank"].notna()]
    per = (
        frame.groupby(["episode_id", "antibiotic"], observed=True)["rank"]
        .agg(rank="max", distinct="nunique")
        .reset_index()
        .merge(episodes[["episode_id", "source_site"]], on="episode_id", how="left", validate="many_to_one")
    )
    per["first_isolate"] = per["episode_id"].isin(set(first_isolate_episodes(episodes)))
    rows = []
    for scope, part in _scopes(per, sites):
        for antibiotic, drug in part.groupby("antibiotic", observed=True):
            tested = len(drug)
            first = drug.loc[drug["first_isolate"]]
            counts = drug["rank"].value_counts()
            s, i, r = (int(counts.get(value, 0)) for value in (0, 1, 2))
            first_r = int(first["rank"].eq(2).sum())
            rows.append(
                {
                    "site": scope,
                    "antibiotic": str(antibiotic),
                    "tested_n": tested,
                    "susceptible_n": s,
                    "intermediate_n": i,
                    "resistant_n": r,
                    "susceptible_share": _share(s, tested),
                    "intermediate_share": _share(i, tested),
                    "resistant_share": _share(r, tested),
                    "conflicting_n": int(drug["distinct"].gt(1).sum()),
                    "first_isolate_tested_n": len(first),
                    "first_isolate_resistant_n": first_r,
                    "first_isolate_resistant_share": _share(first_r, len(first)),
                    "sparse": tested < min_tested,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    resolved = {label: resolver.resolve(label) for label in table["antibiotic"].unique()}
    table["antibiotic_display"] = table["antibiotic"].map(lambda label: _sentence_case(resolved[label].display_label))
    table["aware_category"] = table["antibiotic"].map(lambda label: resolved[label].aware_category)
    return _aware_sorted(table, "resistant_share")


def _sentence_case(label: str) -> str:
    text = str(label).strip().lower()
    return text[:1].upper() + text[1:]


_AWARE_ORDER = ("Access", "Watch", "Reserve", "Not Set", "Unclassified")


def _aware_sorted(table: pd.DataFrame, share_column: str) -> pd.DataFrame:
    """AWaRe category, drugs with too few results pooled last, then the all-sites value (highest first)."""
    pooled = table.loc[table["site"].eq(ALL_SITES)].set_index("antibiotic")
    rank = {name: index for index, name in enumerate(_AWARE_ORDER)}
    ordered = table.assign(
        _aware=table["aware_category"].map(rank).fillna(len(_AWARE_ORDER)),
        _sparse=table["antibiotic"].map(pooled["sparse"]).fillna(True).astype(bool) if "sparse" in pooled else False,
        _pooled=table["antibiotic"].map(pooled[share_column]),
        _scope=table["site"].map({scope: index for index, scope in enumerate(dict.fromkeys(table["site"]))}),
    )
    ordered = ordered.sort_values(
        ["_aware", "_sparse", "_pooled", "antibiotic_display", "_scope"], ascending=[True, True, False, True, True], kind="mergesort"
    )
    return ordered.drop(columns=["_aware", "_sparse", "_pooled", "_scope"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Episodes per patient
# ---------------------------------------------------------------------------


def episodes_per_patient(episodes: pd.DataFrame, sites: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """How many culture episodes each patient contributes, per scope.

    A patient is a (site, anonymised id) pair: identifiers are not linked
    across the three sources, so a person seen at two sites counts twice.
    """
    rows = []
    for scope, part in _scopes(episodes, sites):
        per_patient = part.groupby(["source_site", "anon_id"], observed=True, dropna=False).size()
        patients = len(per_patient)
        median, q1, q3 = _quartiles(per_patient)
        row = {
            "site": scope,
            "patients": patients,
            "episodes": int(per_patient.sum()),
            "episodes_per_patient_mean": float(per_patient.mean()) if patients else math.nan,
            "episodes_per_patient_median": median,
            "episodes_per_patient_q1": q1,
            "episodes_per_patient_q3": q3,
            "episodes_per_patient_max": int(per_patient.max()) if patients else 0,
            "patients_with_multiple": int(per_patient.gt(1).sum()),
            "patients_with_multiple_share": _share(per_patient.gt(1).sum(), patients),
            "episodes_from_patients_with_multiple_share": _share(per_patient[per_patient.gt(1)].sum(), per_patient.sum()),
        }
        for low, high, label in _PATIENT_BANDS:
            mask = per_patient.ge(low) & (per_patient.le(high) if high is not None else True)
            key = label.replace("–", "_").replace(" ", "_").lower()
            row[f"patients_{key}"] = int(mask.sum())
            row[f"patients_{key}_share"] = _share(mask.sum(), patients)
        rows.append(row)
    return pd.DataFrame(rows)


PATIENT_BAND_LABELS = tuple(label for _, _, label in _PATIENT_BANDS)


def patient_band_key(label: str) -> str:
    return label.replace("–", "_").replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Row exclusions by reason
# ---------------------------------------------------------------------------

# (block, stage, how the count is obtained). Every stage is a count of the unit
# named by its block; each block's rows reconcile arithmetically (checked below).
EXCLUSION_STAGES = (
    ("AST rows", "raw_rows", "Rows in the source AST extract (all organisms)"),
    ("AST rows", "not_ingested_rows", "Rows not read at ingestion"),
    ("AST rows", "exact_duplicate_rows", "Exact duplicate rows removed at cleaning"),
    ("AST rows", "not_harmonized_rows", "Rows not carried into the harmonised cohort"),
    ("AST rows", "other_organism_rows", "Rows for other organisms"),
    ("AST rows", "organism_rows", "Rows for the analysed organism"),
    ("AST rows", "no_result_rows", "No result recorded"),
    ("AST rows", "other_non_interpretive_rows", "Result other than R, S or I (e.g. INCONCLUSIVE)"),
    ("AST rows", "excluded_assay_rows", "ESBL phenotypic confirmation assays"),
    ("AST rows", "interpretable_rows", "Interpretable results (R, S or I)"),
    ("Episode–antibiotic results", "repeated_rows", "Repeated rows for the same episode, antibiotic and result"),
    ("Episode–antibiotic results", "conflicting_extra_rows", "Additional rows of conflicting results"),
    ("Episode–antibiotic results", "observed_episode_drugs", "Observed episode–antibiotic results"),
    ("Episode–antibiotic results", "conflicting_episode_drugs", "of which with conflicting results"),
    ("Culture episodes", "culture_episodes", "Culture episodes"),
    ("Culture episodes", "episodes_without_result", "of which with no interpretable result"),
)


def exclusion_flow(
    ledger: dict[str, dict[str, int]],
    raw_rows: dict[str, int],
    cleaning: dict[str, dict[str, int]],
    sites: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Every AST row from the source extract to the observed episode-drug results, by reason.

    ``ledger`` is the gold build's per-site ``observation_ledger``; ``raw_rows``
    the source extract's row count per site; ``cleaning`` the silver cohort
    metadata per site (``rows_before``, ``rows_after``,
    ``duplicates_removed``). Returns the long table (block, stage, label,
    scope, count) and a list of reconciliation failures (empty when every
    block adds up); a failure means a row left the flow without a reason.
    """
    records, problems = [], []
    per_site: dict[str, dict[str, int]] = {}
    for site in scope_order(tuple(sites), ledger)[1:]:
        entry = ledger.get(site)
        if not entry:
            continue
        clean = cleaning.get(site, {})
        raw = int(raw_rows.get(site, 0) or 0)
        rows_before = int(clean.get("rows_before", raw) or 0)
        rows_after = int(clean.get("rows_after", entry.get("cohort_rows", 0)) or 0)
        counts = {
            "raw_rows": raw,
            "not_ingested_rows": raw - rows_before,
            "exact_duplicate_rows": int(clean.get("duplicates_removed", rows_before - rows_after) or 0),
            "not_harmonized_rows": rows_after - int(entry.get("cohort_rows", rows_after)),
            "other_organism_rows": int(entry["cohort_rows"]) - int(entry["organism_rows"]),
            "organism_rows": int(entry["organism_rows"]),
            "no_result_rows": int(entry["no_result_rows"]),
            "other_non_interpretive_rows": int(entry["other_non_interpretive_rows"]),
            "excluded_assay_rows": int(entry["excluded_assay_rows"]),
            "interpretable_rows": int(entry["interpretable_rows"]),
            "repeated_rows": int(entry["identical_rows_collapsed"]) + int(entry["same_result_duplicate_rows"]),
            "conflicting_extra_rows": int(entry["conflicting_result_rows"]) - int(entry["conflicting_episode_drugs"]),
            "observed_episode_drugs": int(entry["observed_episode_drugs"]),
            "conflicting_episode_drugs": int(entry["conflicting_episode_drugs"]),
            "culture_episodes": int(entry["culture_episodes"]),
            "episodes_without_result": int(entry["episodes_without_result"]),
        }
        checks = {
            "AST rows": counts["raw_rows"] - counts["not_ingested_rows"] - counts["exact_duplicate_rows"] - counts["not_harmonized_rows"]
            - counts["other_organism_rows"] - counts["organism_rows"],
            "organism rows": counts["organism_rows"] - counts["no_result_rows"] - counts["other_non_interpretive_rows"]
            - counts["excluded_assay_rows"] - counts["interpretable_rows"],
            "episode-antibiotic results": counts["interpretable_rows"] - counts["repeated_rows"] - counts["conflicting_extra_rows"]
            - counts["observed_episode_drugs"],
        }
        problems += [f"{site}: {name} do not reconcile (difference {value:,})" for name, value in checks.items() if value != 0]
        per_site[site] = counts
    if not per_site:
        return pd.DataFrame(), problems
    pooled = {stage: sum(counts[stage] for counts in per_site.values()) for _, stage, _ in EXCLUSION_STAGES}
    for order, (block, stage, label) in enumerate(EXCLUSION_STAGES):
        for scope, counts in [(ALL_SITES, pooled), *per_site.items()]:
            records.append({"order": order, "block": block, "stage": stage, "label": label, "site": scope, "count": int(counts[stage])})
    return pd.DataFrame(records), problems


def display_columns(scopes: list[str]) -> list[str]:
    return [site_label(scope) for scope in scopes]
