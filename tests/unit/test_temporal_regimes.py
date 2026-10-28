"""Year-by-year course of validated patterns: counts, evaluable years, states, change years and the
across-year heterogeneity test."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.reporting.builders import temporal_regimes as tr

ROOT = Path(__file__).resolve().parents[2]


def _rows(site, year, up, down, upstream_result, n, observed, eligible=1):
    return [{"source_site": site, "order_time_jittered": f"{year}-06-01 10:00:00", "upstream_antibiotic": up,
             "downstream_antibiotic": down, "upstream_susceptibility": upstream_result,
             "downstream_tested": int(i < observed), "downstream_eligible": eligible} for i in range(n)]


def _validation(*rows):
    return pd.DataFrame([{"upstream_antibiotic": u, "downstream_antibiotic": d, "validation_status": status,
                          "observed_escalation_ratio": er} for u, d, status, er in rows])


@pytest.fixture()
def pairs(tmp_path):
    rows = []
    # A validated escalation pattern at one site: present 2010-2011, absent 2012-2013.
    for year, (r, s) in {2010: (18, 2), 2011: (17, 3), 2012: (20, 20), 2013: (20, 20)}.items():
        rows += _rows("a", year, "GEN", "AMI", "RESISTANT", 20, r) + _rows("a", year, "GEN", "AMI", "SUSCEPTIBLE", 20, s)
    rows += _rows("a", 2010, "GEN", "AMI", "INTERMEDIATE", 50, 50)          # not R or S: left out
    rows += _rows("a", 2010, "GEN", "AMI", "RESISTANT", 50, 50, eligible=0)  # not eligible: left out
    rows += _rows("a", 2010, "X", "Y", "RESISTANT", 20, 20)                 # not validated: left out
    rows += _rows("a", 2014, "GEN", "AMI", "RESISTANT", 5, 5) + _rows("a", 2014, "GEN", "AMI", "SUSCEPTIBLE", 20, 0)
    path = tmp_path / "pairs.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_yearly_counts_keep_eligible_resistant_or_susceptible_rows_of_validated_pairs(pairs):
    patterns = tr.validated_patterns(_validation(("GEN", "AMI", "robust", 3.0), ("X", "Y", "mixed", 2.0)))
    counts = tr.yearly_counts(pairs, patterns, batch_rows=37)  # small batches: totals must still add up
    assert set(zip(counts["upstream_antibiotic"], counts["downstream_antibiotic"])) == {("GEN", "AMI")}
    y2010 = counts.loc[counts["year"].eq(2010)].iloc[0]
    assert (y2010["nR"], y2010["kR"], y2010["nS"], y2010["kS"]) == (20, 18, 20, 2)
    assert counts["year"].tolist() == [2010, 2011, 2012, 2013, 2014]


def test_states_change_years_and_courses(pairs):
    patterns = tr.validated_patterns(_validation(("GEN", "AMI", "robust", 3.0)))
    years = tr.classify_years(tr.yearly_counts(pairs, patterns), patterns, 0.5)
    states = dict(zip(years["year"], years["state"]))
    assert states == {2010: "present", 2011: "present", 2012: "absent", 2013: "absent", 2014: "not evaluable"}
    er = years.loc[years["year"].eq(2010), "er"].iloc[0]
    assert er == pytest.approx((18.5 / 21) / (2.5 / 21))
    summary = tr.summarise(years)
    row = summary.iloc[0]
    assert row["evaluable_years"] == 4 and row["course"] == "fewer than 5 evaluable years"
    assert row["change_years"] == "2012" and row["last_present_year"] == 2011
    assert row["year_heterogeneity_p"] < 1e-3


def test_suppression_direction_is_oriented():
    counts = pd.DataFrame({"upstream_antibiotic": "AZT", "downstream_antibiotic": "PIP", "source_site": "a",
                           "year": [2010, 2011, 2012], "nR": 100, "kR": [20, 100, 100], "nS": 100, "kS": [100, 100, 20]})
    patterns = tr.validated_patterns(_validation(("AZT", "PIP", "robust", 0.6)))
    assert patterns["direction"].tolist() == [-1]
    years = tr.classify_years(counts, patterns, 0.5)
    assert years["state"].tolist() == ["present", "absent", "reversed"]


def _series(states_by_year):
    """Counts that give each year the requested state for an escalation pattern."""
    k = {"present": (60, 10), "absent": (30, 30), "intermediate": (36, 30), "reversed": (10, 60)}
    return pd.DataFrame([{"upstream_antibiotic": "U", "downstream_antibiotic": "D", "source_site": "a", "year": y,
                          "nR": 100, "kR": k[s][0], "nS": 100, "kS": k[s][1]} for y, s in states_by_year.items()])


def test_change_years_skip_intermediate_years_and_courses_are_named():
    patterns = tr.validated_patterns(_validation(("U", "D", "robust", 2.0)))
    years = tr.classify_years(_series({2010: "present", 2011: "present", 2012: "intermediate", 2013: "absent",
                                       2014: "absent", 2015: "present"}), patterns, 0.5)
    assert years["state"].tolist() == ["present", "present", "intermediate", "absent", "absent", "present"]
    row = tr.summarise(years).iloc[0]
    assert row["change_years"] == "2013;2015"
    assert row["course"] == "present in some years, absent or reversed in others"
    assert bool(row["varies_beyond_sampling"])
    steady = tr.summarise(tr.classify_years(_series({y: "present" for y in range(2010, 2016)}), patterns, 0.5)).iloc[0]
    assert steady["course"] == "present every evaluable year" and steady["change_years"] == ""
    assert not bool(steady["varies_beyond_sampling"])  # identical years: no heterogeneity


def test_rarely_observed_downstream_year_is_not_evaluable():
    counts = pd.DataFrame({"upstream_antibiotic": "U", "downstream_antibiotic": "D", "source_site": "a",
                           "year": [2010], "nR": [500], "kR": [0], "nS": [500], "kS": [tr.MIN_OBSERVED - 1]})
    patterns = tr.validated_patterns(_validation(("U", "D", "robust", 2.0)))
    assert tr.classify_years(counts, patterns, 0.5)["state"].tolist() == ["not evaluable"]


def test_headline_numbers_count_followed_series():
    patterns = tr.validated_patterns(_validation(("U", "D", "robust", 2.0)))
    years = tr.classify_years(_series({2010: "present", 2011: "present", 2012: "absent", 2013: "absent", 2014: "absent"}),
                              patterns, 0.5)
    numbers = tr.headline_numbers(tr.summarise(years))
    assert numbers["strong"]["site_series_followed"] == 1 and numbers["strong"]["switching"] == 1
    assert numbers["strong"]["change_years_by_site"]["a"] == {"series": 1, "changes": 1, "mode_year": "2012",
                                                              "changes_within_1_of_mode": 1}
    assert set(numbers["examples"]) == {tr._slug(u, d) for u, d in tr.EXAMPLE_PATTERNS}


def test_script_parses_its_options():
    spec = importlib.util.spec_from_file_location("run_temporal_regimes", ROOT / "scripts" / "run_temporal_regimes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.FIGURE_PATTERNS == tr.EXAMPLE_PATTERNS
