"""Supplementary tables and quoted numbers for the two manuscripts.

Every table here is recomputed from the run's own artefacts, so no supplementary
number is typed by hand. Each function takes data frames and returns data frames;
``SupplementaryTableBuilder`` only locates the inputs, calls the functions, and
writes the results. Where a table restates a stored result (a validation label, a
confounding-strength value, a primary adjusted odds ratio), the function
recomputes it and raises if it disagrees, so inconsistent inputs fail loudly
instead of producing a table that contradicts the main results.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amr_cascade_platform.cascade.analyzers.cascade_validation_analyzer import CascadeValidationAnalyzer
from amr_cascade_platform.cascade.analyzers.retained_edge_analyzer import RetainedEdgeAnalyzer
from amr_cascade_platform.cascade.statistics.downstream_testing_regression import DownstreamTestingRegression
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.config.config_models import Settings
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.statistics.e_values import e_value_from_odds_ratio_interval
from amr_cascade_platform.core.utils.antibiotic_names import normalize_antibiotic_label
from amr_cascade_platform.core.utils.organism_names import normalize_organism_label
from amr_cascade_platform.core.utils.scopes import scoped_output_dir
from amr_cascade_platform.core.utils.site_labels import site_label
from amr_cascade_platform.core.utils.text import normalize_label, safe_feature_name
from amr_cascade_platform.visualization.report.antibiotic_classification import AntibioticClassificationResolver

PAIR = ["upstream_antibiotic", "downstream_antibiotic"]
LABELS = ("robust", "supported", "mixed", "insufficient")
VALIDATED = ("robust", "supported")

_NON_ESTIMABLE_NOTES = {
    "quasi_separation": "Non-estimable: adjusted OR outside 1/1000 to 1000 (quasi-separation)",
    "zero_variance_outcome_or_exposure": "Non-estimable: no variation in downstream observation or upstream result",
    "outcome_events_below_model_dimension": "Non-estimable: fewer observed or unobserved downstream episodes than model parameters",
    "model_fit_failed": "Non-estimable: model did not converge",
    "design_matrix_unavailable": "Non-estimable: no usable design matrix",
}


def _pair_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose pair columns are plain strings, so merges never mix dtypes."""
    out = frame.copy()
    for column in PAIR:
        out[column] = out[column].astype(str)
    return out


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["upstream_antibiotic"].astype(str), frame["downstream_antibiotic"].astype(str)))


def _label_counts(statuses: pd.Series) -> dict[str, int]:
    unexpected = sorted(set(statuses.dropna().astype(str)) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected validation labels: {unexpected}")
    return {label: int(statuses.eq(label).sum()) for label in LABELS}


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


# ---------------------------------------------------------------------------
# Supplementary Table S4: validation labels
# ---------------------------------------------------------------------------


def validation_summary(validation_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    counts = _label_counts(validation_results["validation_status"])
    validated = validation_results.loc[validation_results["validation_status"].isin(VALIDATED)]
    direction = validated["cascade_direction"].astype(str)
    robust = validated["validation_status"].eq("robust")
    row = {
        "retained_patterns": len(validation_results),
        **counts,
        "validated": len(validated),
        "validated_share": len(validated) / len(validation_results) if len(validation_results) else math.nan,
        "validated_escalation": int(direction.eq("escalation").sum()),
        "validated_suppression": int(direction.eq("suppression").sum()),
        "robust_escalation": int((robust & direction.eq("escalation")).sum()),
        "robust_suppression": int((robust & direction.eq("suppression")).sum()),
    }
    if row["validated_escalation"] + row["validated_suppression"] != row["validated"]:
        raise ValueError("Validated patterns without an escalation/suppression direction.")
    ratios = validated["observed_escalation_ratio"].astype(float)
    summary = dict(row)
    summary["escalation_ratio"] = {
        name: {"min": float(values.min()), "median": float(values.median()), "max": float(values.max())}
        for name, values in ((name, ratios[direction.eq(name)]) for name in ("escalation", "suppression"))
        if len(values)
    }
    return pd.DataFrame([row]), summary


def downstream_event_profile(edge_report: pd.DataFrame) -> dict[str, object]:
    """How many validated patterns rest on few downstream-observed episodes."""
    validated = edge_report.loc[edge_report["validation_status"].isin(VALIDATED)]
    events = validated["resistant_tested_n"] + validated["susceptible_tested_n"]
    one_branch_empty = validated[["resistant_tested_n", "susceptible_tested_n"]].min(axis=1).eq(0)
    robust = validated["validation_status"].eq("robust")
    profile: dict[str, object] = {"validated": len(validated)}
    for ceiling in (1, 5, 10, 25):
        profile[f"events_at_most_{ceiling}"] = int(events.le(ceiling).sum())
        profile[f"robust_events_at_most_{ceiling}"] = int((events.le(ceiling) & robust).sum())
    profile["one_branch_without_events"] = int(one_branch_empty.sum())
    profile["robust_one_branch_without_events"] = int((one_branch_empty & robust).sum())
    return profile


# ---------------------------------------------------------------------------
# Supplementary Table S6: support thresholds
# ---------------------------------------------------------------------------


def support_threshold_sensitivity(
    escalation_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recompute the retained set at alternative support thresholds.

    Labels are the primary run's labels. A pair retained only at an alternative
    threshold was never validated and is counted as such, not given a label.
    The co-testing screen is held at its primary setting.
    """
    primary_total = settings.cascade.min_total_support
    primary_result = settings.cascade.min_result_support
    grid: list[tuple[int, int]] = [(int(total), primary_result) for total in settings.cascade.sensitivity_min_total_supports]
    grid += [(primary_total, int(result)) for result in settings.cascade.sensitivity_min_result_supports]
    grid.append((primary_total, primary_result))
    grid = sorted(dict.fromkeys(grid), key=lambda item: (item[1] != primary_result, item[0], item[1]))

    labels = _pair_keys(validation_results).set_index(PAIR)["validation_status"]
    primary_keys = set(labels.index)
    validated_keys = set(labels.index[labels.isin(VALIDATED)])
    analyzer = RetainedEdgeAnalyzer(settings)
    rows = []
    for total, result in grid:
        retained = analyzer.analyze(escalation_results, pd.DataFrame(), min_total_support=total, min_result_support=result)
        keys = _key_set(retained) if not retained.empty else set()
        shared = keys & primary_keys
        is_primary = (total, result) == (primary_total, primary_result)
        if is_primary and keys != primary_keys:
            raise ValueError(
                "The primary support thresholds do not reproduce the validated candidate set "
                f"({len(keys ^ primary_keys)} pairs differ); escalation and validation results are from different runs."
            )
        counts = _label_counts(labels.loc[sorted(shared)]) if shared else dict.fromkeys(LABELS, 0)
        rows.append(
            {
                "min_total_support": total,
                "min_result_support": result,
                "primary": is_primary,
                "retained_patterns": len(keys),
                "retained_in_primary_set": len(shared),
                "added_not_validated": len(keys - primary_keys),
                "dropped_from_primary_set": len(primary_keys - keys),
                "validated_dropped": len(validated_keys - keys),
                **counts,
            }
        )
    table = pd.DataFrame(rows)
    return table, {"rows": table.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Supplementary Table S7: patient-cluster bootstrap
# ---------------------------------------------------------------------------


def patient_cluster_table(
    patient_cluster: pd.DataFrame,
    validation_results: pd.DataFrame,
    resolver: AntibioticClassificationResolver,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validation = _pair_keys(validation_results)
    robust_keys = _key_set(validation.loc[validation["validation_status"].eq("robust")])
    patient = _pair_keys(patient_cluster)
    if _key_set(patient) != robust_keys:
        raise ValueError(
            "The patient-cluster results do not cover exactly the robust patterns "
            f"({len(_key_set(patient) ^ robust_keys)} pairs differ); rerun the patient-cluster analysis for this run."
        )
    table = patient.merge(
        validation[PAIR + ["bootstrap_sign_stability"]].rename(columns={"bootstrap_sign_stability": "episode_bootstrap_sign_stability"}),
        on=PAIR,
        how="left",
        validate="one_to_one",
    )
    table["upstream_abbreviation"] = table["upstream_antibiotic"].map(resolver.get_abbreviation)
    table["downstream_abbreviation"] = table["downstream_antibiotic"].map(resolver.get_abbreviation)
    table["episode_below_threshold"] = table["episode_bootstrap_sign_stability"].lt(threshold)
    table["patient_below_threshold"] = table["patient_bootstrap_sign_stability"].lt(threshold)
    table = table.sort_values(
        ["patient_bootstrap_sign_stability", "episode_bootstrap_sign_stability", *PAIR], kind="mergesort"
    ).reset_index(drop=True)
    table = table[
        [
            "upstream_antibiotic",
            "downstream_antibiotic",
            "upstream_abbreviation",
            "downstream_abbreviation",
            "observed_escalation_ratio",
            "episode_bootstrap_sign_stability",
            "patient_bootstrap_sign_stability",
            "n_rows",
            "n_patients",
            "episode_below_threshold",
            "patient_below_threshold",
        ]
    ]
    episode = table["episode_bootstrap_sign_stability"]
    patient_values = table["patient_bootstrap_sign_stability"]
    both_vary = len(table) >= 3 and episode.nunique() > 1 and patient_values.nunique() > 1
    lowest = table.head(5)
    summary = {
        "patterns": len(table),
        "threshold": threshold,
        "episode_mean": float(episode.mean()),
        "episode_median": float(episode.median()),
        "episode_min": float(episode.min()),
        "patient_mean": float(patient_values.mean()),
        "patient_median": float(patient_values.median()),
        "patient_min": float(patient_values.min()),
        "episode_below_threshold": int(table["episode_below_threshold"].sum()),
        "patient_below_threshold": int(table["patient_below_threshold"].sum()),
        "pearson_r": float(np.corrcoef(episode, patient_values)[0, 1]) if both_vary else math.nan,
        "lowest_patient_stability": lowest[PAIR + ["observed_escalation_ratio", "patient_bootstrap_sign_stability"]].to_dict(orient="records"),
    }
    return table, summary


def episode_multiplicity(culture_episodes: pd.DataFrame) -> dict[str, object]:
    """Patients (site, anon_id) with more than one culture episode."""
    per_patient = culture_episodes.groupby(["source_site", "anon_id"], observed=True, dropna=False).size()
    multiple = per_patient.gt(1)
    return {
        "patients": int(len(per_patient)),
        "culture_episodes": int(per_patient.sum()),
        "patients_with_multiple_episodes": int(multiple.sum()),
        "share_patients_with_multiple_episodes": float(multiple.mean()),
        "share_episodes_from_patients_with_multiple": float(per_patient[multiple].sum() / per_patient.sum()),
        "max_episodes_per_patient": int(per_patient.max()),
    }


def study_period(order_times: pd.Series) -> dict[str, object]:
    """First and last culture month of the analysis set, e.g. "December 1999" and "March 2025"."""
    times = pd.to_datetime(order_times, errors="coerce", format="mixed", utc=True).dropna()
    if times.empty:
        return {}
    return {"first_month": times.min().strftime("%B %Y"), "last_month": times.max().strftime("%B %Y"),
            "first_year": int(times.min().year), "last_year": int(times.max().year)}


def observation_coverage_numbers(coverage: pd.DataFrame) -> dict[str, object]:
    """Share of the eligible opportunity space observed, per scope, and its spread across antibiotics.

    ``coverage`` is table W (one row per antibiotic and scope, scope
    ``all_sites`` or a site code) written by the report step.
    """
    totals = coverage.groupby("site", sort=False)[["eligible_n", "observed_n"]].sum()
    numbers: dict[str, object] = {
        str(site): {
            "eligible_n": int(row.eligible_n),
            "observed_n": int(row.observed_n),
            "observed_share": float(row.observed_n / row.eligible_n),
        }
        for site, row in totals.iterrows()
    }
    shares = coverage.loc[coverage["site"].eq("all_sites"), "observed_share"].astype(float)
    q1, median, q3 = shares.quantile([0.25, 0.5, 0.75])
    numbers["antibiotics"] = int(len(shares))
    numbers["drug_share_median"] = float(median)
    numbers["drug_share_q1"] = float(q1)
    numbers["drug_share_q3"] = float(q3)
    return numbers


def organism_label_counts(gold_metadata: dict, organism: str) -> dict[str, int]:
    """Culture episodes the gold layer pooled into the organism from annotated labels.

    A species cohort includes raw labels that annotate a resistance phenotype or
    biotype of the same species (e.g. "ESBL ESCHERICHIA COLI"); the gold build
    records how many culture episodes each raw label contributed.
    """
    if "organism_labels" not in gold_metadata:
        raise ValueError("gold metadata has no organism_labels; rebuild the gold layer")
    labels = {str(label): int(count) for label, count in gold_metadata["organism_labels"].items()}
    canonical = normalize_label(organism)
    annotated = {label: count for label, count in labels.items() if label != canonical}
    return {
        "annotated_label_episodes": int(sum(annotated.values())),
        "annotated_labels": len(annotated),
    }


# ---------------------------------------------------------------------------
# Supplementary Table S8: heterogeneity threshold
# ---------------------------------------------------------------------------


def _relabel(frame: pd.DataFrame, settings: Settings, *, q_column: str, ignore_heterogeneity: bool) -> list[str]:
    analyzer = CascadeValidationAnalyzer(settings)
    return [
        analyzer._classify_validation(
            permutation_p_value=float(row[q_column]),
            bootstrap_sign_stability=float(row["bootstrap_sign_stability"]),
            site_replication_n=int(row["site_replication_n"]),
            site_direction_agreement_rate=float(row["site_direction_agreement_rate"]),
            temporal_direction_agreement=bool(row["temporal_direction_agreement"]),
            observed_er=float(row["observed_escalation_ratio"]),
            site_i_squared=math.nan if ignore_heterogeneity else float(row["site_i_squared"]),
            site_pooled_log_er=float(row["site_pooled_log_er"]),
            site_pooled_se=float(row["site_pooled_se"]),
        )
        for _, row in frame.iterrows()
    ]


def _with_i_squared_threshold(settings: Settings, threshold: float) -> Settings:
    validation = dataclasses.replace(settings.cascade.validation, i_squared_threshold=float(threshold))
    return dataclasses.replace(settings, cascade=dataclasses.replace(settings.cascade, validation=validation))


def _primary_site_failure(row: pd.Series, settings: Settings) -> str:
    """Which part of the pooled-site criterion a pattern failed under the primary rule."""
    if pd.isna(row["site_i_squared"]) or pd.isna(row["site_pooled_se"]) or not row["site_pooled_se"] > 0:
        return "no_pooled_estimate"
    reasons = []
    if row["site_replication_n"] < settings.cascade.validation.minimum_replicated_sites:
        reasons.append("too_few_sites")
    if row["site_i_squared"] > settings.cascade.validation.i_squared_threshold:
        reasons.append("i_squared_above_threshold")
    if abs(row["site_pooled_log_er"] / row["site_pooled_se"]) < 1.96:
        reasons.append("pooled_z_below_1.96")
    er = row["observed_escalation_ratio"]
    if not ((er > 1 and row["site_pooled_log_er"] > 0) or (er < 1 and row["site_pooled_log_er"] < 0)):
        reasons.append("pooled_direction_mismatch")
    return "+".join(reasons) or "none"


def heterogeneity_sensitivity(
    validation_results: pd.DataFrame,
    settings: Settings,
    alternative_threshold: float = 75.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = _pair_keys(validation_results).reset_index(drop=True)
    q_column = "permutation_fdr_q_value_two_sided" if "permutation_fdr_q_value_two_sided" in frame.columns else "permutation_fdr_q_value"
    primary_threshold = float(settings.cascade.validation.i_squared_threshold)
    policies = [
        (f"I2 <= {primary_threshold:g}% (primary)", settings, False),
        (f"I2 <= {alternative_threshold:g}%", _with_i_squared_threshold(settings, alternative_threshold), False),
        ("No heterogeneity check (direction agreement only)", settings, True),
    ]
    stored = frame["validation_status"].astype(str)
    rows, transitions = [], []
    relabelled: dict[str, pd.Series] = {}
    for name, policy_settings, ignore in policies:
        labels = pd.Series(_relabel(frame, policy_settings, q_column=q_column, ignore_heterogeneity=ignore), index=frame.index)
        relabelled[name] = labels
        if name == policies[0][0]:
            mismatched = int(labels.ne(stored).sum())
            if mismatched:
                raise ValueError(
                    f"Relabelling with the primary settings changes {mismatched} stored labels; "
                    "the validation results were produced with different settings or code."
                )
        changed = labels.ne(stored)
        rows.append({"policy": name, **_label_counts(labels), "changed_from_primary": int(changed.sum())})
        moves = pd.DataFrame({"from_status": stored[changed], "to_status": labels[changed]})
        for (source, target), n in moves.value_counts().sort_index().items():
            transitions.append({"policy": name, "from_status": source, "to_status": target, "patterns": int(n)})

    no_check = relabelled[policies[2][0]]
    upgraded = frame.loc[stored.eq("supported") & no_check.eq("robust")]
    i_squared = upgraded["site_i_squared"].dropna()
    failure = (
        upgraded.apply(_primary_site_failure, axis=1, settings=settings) if not upgraded.empty else pd.Series(dtype=str)
    )
    summary = {
        "q_value_column": q_column,
        "rows": rows,
        "supported_to_robust_without_heterogeneity_check": len(upgraded),
        "their_i_squared_median": float(i_squared.median()) if len(i_squared) else math.nan,
        "their_i_squared_min": float(i_squared.min()) if len(i_squared) else math.nan,
        "their_i_squared_max": float(i_squared.max()) if len(i_squared) else math.nan,
        "their_primary_site_failure": failure.value_counts().to_dict(),
    }
    return (
        pd.DataFrame(rows),
        pd.DataFrame(transitions, columns=["policy", "from_status", "to_status", "patterns"]),
        summary,
    )


# ---------------------------------------------------------------------------
# Supplementary Table S9: panel-bundling threshold
# ---------------------------------------------------------------------------


def pbi_sensitivity(
    cotesting_probabilities: pd.DataFrame,
    validation_results: pd.DataFrame,
    settings: Settings,
    stricter_threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Apply the co-testing screen's own rule at a stricter threshold to the retained pairs."""
    primary_threshold = float(settings.cascade.cotesting_probability_threshold)
    min_support = int(settings.cascade.min_total_support)
    probabilities = _pair_keys(cotesting_probabilities)[
        PAIR + ["p_downstream_given_upstream", "p_upstream_given_downstream", "support_n", "reverse_support_n"]
    ]
    table = _pair_keys(validation_results)[PAIR + ["validation_status"]].merge(
        probabilities, on=PAIR, how="left", validate="one_to_one"
    )
    table["panel_bundling_index"] = table[["p_downstream_given_upstream", "p_upstream_given_downstream"]].min(
        axis=1, skipna=False
    )

    def screened(threshold: float) -> pd.Series:
        return (
            table["p_downstream_given_upstream"].ge(threshold)
            & table["p_upstream_given_downstream"].ge(threshold)
            & table["support_n"].ge(min_support)
            & table["reverse_support_n"].ge(min_support)
        )

    table["excluded_at_primary"] = screened(primary_threshold)
    table["excluded_at_stricter"] = screened(stricter_threshold)
    if table["excluded_at_primary"].any():
        raise ValueError(
            f"{int(table['excluded_at_primary'].sum())} retained pairs meet the primary co-testing exclusion rule; "
            "the screen was not applied to these results."
        )
    bands = [
        (
            f"{stricter_threshold:g} <= PBI < {primary_threshold:g} with support >= {min_support} in both directions "
            "(excluded only at the stricter threshold)",
            table["excluded_at_stricter"],
        ),
        ("Retained at both thresholds", ~table["excluded_at_stricter"]),
    ]
    rows = []
    for name, mask in bands:
        counts = _label_counts(table.loc[mask, "validation_status"])
        rows.append(
            {
                "band": name,
                "pairs": int(mask.sum()),
                "robust": counts["robust"],
                "supported": counts["supported"],
                "mixed_or_insufficient": counts["mixed"] + counts["insufficient"],
            }
        )
    exempt = table["panel_bundling_index"].ge(primary_threshold) & ~table["excluded_at_primary"]
    summary = {
        "primary_threshold": primary_threshold,
        "stricter_threshold": stricter_threshold,
        "min_support_each_direction": min_support,
        "retained_pairs": len(table),
        "pbi_undefined_reverse_not_assessed": int(table["panel_bundling_index"].isna().sum()),
        "pbi_at_or_above_primary_but_support_below_minimum": int(exempt.sum()),
        "max_pbi_among_retained": float(table["panel_bundling_index"].max()),
        "excluded_only_at_stricter": int(table["excluded_at_stricter"].sum()),
        "excluded_only_at_stricter_share": float(table["excluded_at_stricter"].mean()) if len(table) else math.nan,
        "bands": rows,
    }
    return table, pd.DataFrame(rows), summary


def pair_flow(
    cotesting_probabilities: pd.DataFrame,
    cotesting_pairs: pd.DataFrame,
    escalation_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    settings: Settings,
) -> dict[str, object]:
    """Ordered-pair counts at each step from eligible rows to the retained candidate set."""
    counts = escalation_results[["total_support_n", "positive_support_n", "negative_support_n"]]
    passing = (
        counts["total_support_n"].ge(settings.cascade.min_total_support)
        & counts["positive_support_n"].ge(settings.cascade.min_result_support)
        & counts["negative_support_n"].ge(settings.cascade.min_result_support)
    )
    events = escalation_results["positive_tested_n"].fillna(0) + escalation_results["negative_tested_n"].fillna(0)
    no_event = passing & events.lt(settings.cascade.min_total_tested_events_for_retention)
    flow = {
        "pairs_with_eligible_rows": len(cotesting_probabilities),
        "removed_by_cotesting_screen": len(cotesting_pairs),
        "with_both_upstream_branches": len(escalation_results),
        "meeting_support": int(passing.sum()),
        "below_downstream_event_minimum": int(no_event.sum()),
        "retained": len(validation_results),
    }
    if flow["meeting_support"] - flow["below_downstream_event_minimum"] != flow["retained"]:
        raise ValueError("Support and event counts do not reproduce the retained set; the inputs come from different runs.")
    if flow["with_both_upstream_branches"] > flow["pairs_with_eligible_rows"] - flow["removed_by_cotesting_screen"]:
        raise ValueError("More pairs reached estimation than were left after the co-testing screen.")
    return flow


def pbi_distribution(cotesting_probabilities: pd.DataFrame, settings: Settings, stricter_threshold: float = 0.90) -> dict[str, object]:
    """Where the screen's threshold falls in the distribution of assessed pairs' PBI."""
    frame = cotesting_probabilities.copy()
    pbi = frame[["p_downstream_given_upstream", "p_upstream_given_downstream"]].min(axis=1, skipna=False)
    assessed = pbi.notna()
    removed = frame["flagged"].fillna(False).astype(bool)
    kept = assessed & ~removed
    primary = float(settings.cascade.cotesting_probability_threshold)
    return {
        "assessed_pairs": int(assessed.sum()),
        "one_direction_pairs": int((~assessed).sum()),
        "removed": int(removed.sum()),
        "kept": int(kept.sum()),
        "kept_between_stricter_and_primary": int((kept & pbi.ge(stricter_threshold) & pbi.lt(primary)).sum()),
        "kept_at_or_above_primary": int((kept & pbi.ge(primary)).sum()),
        "max_kept_pbi": float(pbi[kept].max()) if kept.any() else math.nan,
        "min_removed_pbi": float(pbi[removed].min()) if removed.any() else math.nan,
    }


# ---------------------------------------------------------------------------
# Supplementary Table S10: confounding-strength calibration
# ---------------------------------------------------------------------------


def confounding_strength_table(
    edge_report: pd.DataFrame,
    adjusted_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validated = _pair_keys(edge_report.loc[edge_report["validation_status"].isin(VALIDATED)])
    reasons = _pair_keys(adjusted_results)[PAIR + ["non_estimable_reason"]]
    table = validated.merge(reasons, on=PAIR, how="left", validate="one_to_one")
    table = table.rename(
        columns={
            "adjusted_odds_ratio_ci_lower": "adjusted_or_ci_lower",
            "adjusted_odds_ratio_ci_upper": "adjusted_or_ci_upper",
            "adjusted_or_ci_e_value": "confounding_strength",
        }
    )
    expected = np.asarray(
        [e_value_from_odds_ratio_interval(low, high) for low, high in zip(table["adjusted_or_ci_lower"], table["adjusted_or_ci_upper"])],
        dtype=float,
    )
    stored = table["confounding_strength"].to_numpy(dtype=float)
    agree = np.isclose(stored, expected, rtol=1e-9, atol=0) | (np.isnan(stored) & np.isnan(expected))
    if not agree.all():
        raise ValueError(f"{int((~agree).sum())} confounding-strength values do not match the stated formula.")

    def note(row: pd.Series) -> str:
        reason = row["non_estimable_reason"]
        if isinstance(reason, str) and reason:
            return _NON_ESTIMABLE_NOTES.get(reason, f"Non-estimable: {reason}")
        if pd.isna(row["adjusted_odds_ratio"]):
            return "Non-estimable"
        if pd.isna(row["adjusted_or_ci_lower"]):
            return "Confidence interval not finite"
        return ""

    table["notes"] = table.apply(note, axis=1)
    table["_label_order"] = table["validation_status"].map({"robust": 0, "supported": 1})
    table = table.sort_values(
        ["_label_order", "confounding_strength", *PAIR],
        ascending=[True, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    columns = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "validation_status",
        "cascade_direction",
        "adjusted_odds_ratio",
        "adjusted_or_ci_lower",
        "adjusted_or_ci_upper",
        "confounding_strength",
        "notes",
    ]
    table = table[columns].reset_index(drop=True)
    estimable = table["adjusted_odds_ratio"].notna()
    strength = table.loc[estimable, "confounding_strength"]
    summary = {
        "validated": len(table),
        "estimable": int(estimable.sum()),
        "ci_excludes_null": int(strength.gt(1).sum()),
        "confounding_strength_median": float(strength.median()) if len(strength) else math.nan,
        "confounding_strength_max": float(strength.max()) if len(strength) else math.nan,
    }
    return table, summary


def adjusted_model_estimability(
    adjusted_results: pd.DataFrame, validation_results: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    validated = _key_set(validation_results.loc[validation_results["validation_status"].isin(VALIDATED)])
    frame = _pair_keys(adjusted_results)
    frame = frame.loc[[key in validated for key in zip(frame["upstream_antibiotic"], frame["downstream_antibiotic"])]]
    if _key_set(frame) != validated or len(frame) != len(validated):
        raise ValueError("The adjusted results do not cover exactly the validated patterns.")
    reason = frame["non_estimable_reason"].where(frame["non_estimable_reason"].notna(), "estimable")
    statuses = ["estimable", *_NON_ESTIMABLE_NOTES]
    unexpected = sorted(set(reason) - set(statuses))
    counts = reason.value_counts().reindex(statuses + unexpected, fill_value=0)
    table = counts.rename_axis("status").reset_index(name="patterns")
    table["share"] = table["patterns"] / len(frame) if len(frame) else math.nan
    return table, {"validated": len(frame), "not_estimable": int(len(frame) - counts["estimable"]), **{str(k): int(v) for k, v in counts.items()}}


# ---------------------------------------------------------------------------
# Supplementary Table S11: cross-site replication
# ---------------------------------------------------------------------------


def _same_direction(first: float, second: float) -> bool:
    if pd.isna(first) or pd.isna(second) or first == 1.0 or second == 1.0:
        return False
    return (first > 1.0) == (second > 1.0)


def _consistency(with_estimate: int, agreeing: int) -> str:
    if with_estimate < 2:
        return "Insufficient"
    return "Yes" if agreeing == with_estimate else "No"


def cross_site_replication(
    validation_results: pd.DataFrame,
    site_edge_reports: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Site agreement under one definition for both measures.

    A site agrees when its escalation ratio lies on the same side of 1 as the
    combined estimate. Pooled: site strata inside the combined analysis (the
    counts behind site_replication_n and site_direction_agreement_rate).
    Independent: each site's own analysis, where a site has an estimate only
    when the pattern cleared that site's retention rules.
    """
    table = _pair_keys(validation_results.loc[validation_results["validation_status"].isin(VALIDATED)])
    table = table[
        PAIR
        + [
            "validation_status",
            "cascade_direction",
            "observed_escalation_ratio",
            "site_replication_n",
            "site_direction_agreement_rate",
        ]
    ]
    with_estimate = table["site_replication_n"].fillna(0).astype(int).to_numpy()
    rate = table["site_direction_agreement_rate"].fillna(0.0).to_numpy(dtype=float)
    agreeing = np.rint(with_estimate * rate).astype(int)
    if not np.allclose(agreeing, with_estimate * rate, atol=1e-6):
        raise ValueError("site_direction_agreement_rate is not a whole number of sites.")
    table["pooled_sites_with_estimate"] = with_estimate
    table["pooled_sites_agreeing"] = agreeing
    table["pooled_consistent"] = [_consistency(n, a) for n, a in zip(with_estimate, agreeing)]

    site_columns = []
    for site, report in site_edge_reports.items():
        column = f"er_{site}"
        site_columns.append(column)
        site_er = _pair_keys(report)[PAIR + ["escalation_ratio"]].rename(columns={"escalation_ratio": column})
        table = table.merge(site_er, on=PAIR, how="left", validate="one_to_one")
    independent = table[site_columns]
    table["independent_sites_with_estimate"] = independent.notna().sum(axis=1).astype(int)
    table["independent_sites_agreeing"] = [
        int(sum(_same_direction(combined, value) for value in row))
        for combined, row in zip(table["observed_escalation_ratio"], independent.itertuples(index=False))
    ]
    table["independent_consistent"] = [
        _consistency(n, a) for n, a in zip(table["independent_sites_with_estimate"], table["independent_sites_agreeing"])
    ]
    table = table.drop(columns=["site_replication_n", "site_direction_agreement_rate"]).rename(
        columns={"observed_escalation_ratio": "combined_escalation_ratio"}
    )

    site_count = len(site_edge_reports)

    def measure(prefix: str) -> dict[str, object]:
        n = table[f"{prefix}_sites_with_estimate"]
        agree = table[f"{prefix}_sites_agreeing"]
        two_or_more = n.ge(2)
        all_agree = two_or_more & agree.eq(n)
        every_site = n.eq(site_count)
        result: dict[str, object] = {
            "patterns_with_estimates_at_1_or_more_sites": int(n.ge(1).sum()),
            "patterns_with_estimates_at_2_or_more_sites": int(two_or_more.sum()),
            "patterns_with_estimates_at_all_sites": int(every_site.sum()),
            "patterns_agreeing_at_2_or_more_sites": int(agree.ge(2).sum()),
            "share_agreeing_at_2_or_more_sites": float(agree.ge(2).mean()) if len(table) else math.nan,
            "patterns_agreeing_at_all_sites": int(agree.eq(site_count).sum()),
            "patterns_agreeing_at_exactly_2_sites": int(agree.eq(2).sum()),
            "patterns_agreeing_at_1_or_no_site": int(agree.le(1).sum()),
            "patterns_all_sites_agree_among_2_or_more": int(all_agree.sum()),
            "share_all_agree_among_2_or_more": float(all_agree.sum() / two_or_more.sum()) if two_or_more.any() else math.nan,
        }
        for label in VALIDATED:
            subset = every_site & table["validation_status"].eq(label)
            agreeing = subset & agree.eq(n)
            result[f"{label}_with_estimates_at_all_sites"] = int(subset.sum())
            result[f"{label}_all_agree_at_all_sites"] = int(agreeing.sum())
            result[f"{label}_share_all_agree_at_all_sites"] = float(agreeing.sum() / subset.sum()) if subset.any() else math.nan
        return result

    summary = {
        "validated": len(table),
        "sites": list(site_edge_reports),
        "pooled": measure("pooled"),
        "independent": measure("independent"),
    }
    return table.reset_index(drop=True), summary


# ---------------------------------------------------------------------------
# Ridge-penalty and continuity-correction sensitivity
# ---------------------------------------------------------------------------


def ridge_penalty_table(
    ridge: pd.DataFrame,
    edge_report: pd.DataFrame,
    primary_penalty: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    long = _pair_keys(ridge)
    long["ridge_penalty"] = long["ridge_penalty"].astype(float)
    penalties = sorted(long["ridge_penalty"].unique())
    if primary_penalty not in penalties:
        raise ValueError(f"The ridge sensitivity does not include the primary penalty {primary_penalty:g}.")
    validated = _pair_keys(edge_report.loc[edge_report["validation_status"].isin(VALIDATED)])
    primary = long.loc[long["ridge_penalty"].eq(primary_penalty)].merge(
        validated[PAIR + ["adjusted_odds_ratio"]].rename(columns={"adjusted_odds_ratio": "reported"}),
        on=PAIR,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not primary["_merge"].eq("both").all():
        raise ValueError("The ridge sensitivity does not cover exactly the validated patterns.")
    refit = primary["adjusted_odds_ratio"].to_numpy(float)
    reported = primary["reported"].to_numpy(float)
    agree = np.isclose(refit, reported, rtol=1e-6, atol=0) | (np.isnan(refit) & np.isnan(reported))
    if not agree.all():
        raise ValueError(
            f"The refit at the primary penalty differs from the reported adjusted OR for {int((~agree).sum())} patterns."
        )

    wide = validated[PAIR + ["validation_status"]].copy()
    for penalty in penalties:
        tag = f"lambda_{penalty:g}"
        subset = long.loc[long["ridge_penalty"].eq(penalty)]
        wide = wide.merge(
            subset[
                PAIR
                + ["adjusted_odds_ratio", "adjusted_odds_ratio_ci_lower", "adjusted_odds_ratio_ci_upper", "non_estimable_reason"]
            ].rename(
                columns={
                    "adjusted_odds_ratio": f"adjusted_or_{tag}",
                    "adjusted_odds_ratio_ci_lower": f"ci_lower_{tag}",
                    "adjusted_odds_ratio_ci_upper": f"ci_upper_{tag}",
                    "non_estimable_reason": f"non_estimable_{tag}",
                }
            ),
            on=PAIR,
            how="left",
            validate="one_to_one",
        )
    or_columns = [f"adjusted_or_lambda_{p:g}" for p in penalties]
    ors = wide[or_columns].astype(float)
    estimable_all = ors.notna().all(axis=1)
    same_direction = estimable_all & (ors.gt(1).all(axis=1) | ors.lt(1).all(axis=1))
    lower = wide[[f"ci_lower_lambda_{p:g}" for p in penalties]].to_numpy(float)
    upper = wide[[f"ci_upper_lambda_{p:g}" for p in penalties]].to_numpy(float)
    excludes = (lower > 1) | (upper < 1)
    same_significance = estimable_all & pd.Series(excludes.all(axis=1) | ~excludes.any(axis=1), index=wide.index)
    log_or = np.log(ors.loc[estimable_all])
    primary_column = f"adjusted_or_lambda_{primary_penalty:g}"
    max_shift = (
        float(log_or.sub(log_or[primary_column], axis=0).abs().to_numpy().max()) if estimable_all.any() else math.nan
    )
    summary = {
        "penalties": penalties,
        "primary_penalty": primary_penalty,
        "validated": len(wide),
        "estimable_at_every_penalty": int(estimable_all.sum()),
        "estimable_at_no_penalty": int(ors.isna().all(axis=1).sum()),
        "same_direction_at_every_penalty": int(same_direction.sum()),
        "same_ci_excludes_null_status_at_every_penalty": int(same_significance.sum()),
        "max_abs_log_or_change_from_primary": max_shift,
        "direction_changes": wide.loc[estimable_all & ~same_direction, PAIR + or_columns].to_dict(orient="records"),
    }
    return wide, summary


def continuity_correction_summary(table_r: pd.DataFrame) -> dict[str, object]:
    columns = [column for column in table_r.columns if column.startswith("er_lambda_")]
    ers = table_r[columns].astype(float)
    same_direction = ers.gt(1).all(axis=1) | ers.lt(1).all(axis=1)
    fold = ers.max(axis=1) / ers.min(axis=1)
    return {
        "patterns": len(table_r),
        "lambdas": [column.removeprefix("er_lambda_") for column in columns],
        "same_direction_at_every_lambda": int(same_direction.sum()),
        "fold_range_min": float(fold.min()),
        "fold_range_median": float(fold.median()),
        "fold_range_max": float(fold.max()),
    }


def era_stratified_table(era: pd.DataFrame, validation_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Site-by-era permutation p-values and degeneracy diagnostics for the validated patterns."""
    validation = _pair_keys(validation_results)
    validated_keys = _key_set(validation.loc[validation["validation_status"].isin(VALIDATED)])
    frame = _pair_keys(era)
    if _key_set(frame) != validated_keys or len(frame) != len(validated_keys):
        raise ValueError(
            "The era-stratified results do not cover exactly the validated patterns; rerun the era-stratified analysis for this run."
        )
    primary_p = "permutation_p_value_two_sided" if "permutation_p_value_two_sided" in validation.columns else "permutation_p_value"
    primary_q = "permutation_fdr_q_value_two_sided" if "permutation_fdr_q_value_two_sided" in validation.columns else "permutation_fdr_q_value"
    table = frame.merge(
        validation[PAIR + ["validation_status", primary_p, primary_q]].rename(
            columns={primary_p: "site_only_p_two_sided", primary_q: "site_only_q_two_sided"}
        ),
        on=PAIR,
        how="left",
        validate="one_to_one",
    )
    table = table[
        [
            "upstream_antibiotic",
            "downstream_antibiotic",
            "validation_status",
            "observed_escalation_ratio",
            "site_only_p_two_sided",
            "site_only_q_two_sided",
            "permutation_p_value_two_sided_era_stratified",
            "era_stratified_n_strata",
            "era_stratified_n_strata_with_both_labels",
            "era_stratified_n_permutable_rows",
            "era_stratified_fraction_fixed_rows",
        ]
    ].sort_values(["permutation_p_value_two_sided_era_stratified", *PAIR], kind="mergesort", na_position="last")
    p_era = table["permutation_p_value_two_sided_era_stratified"].astype(float)
    fixed = table["era_stratified_fraction_fixed_rows"].astype(float)
    summary = {
        "patterns": len(table),
        "era_p_at_most_0.05": int(p_era.le(0.05).sum()),
        "era_p_missing": int(p_era.isna().sum()),
        "fraction_fixed_rows_median": float(fixed.median()) if fixed.notna().any() else math.nan,
        "fraction_fixed_rows_max": float(fixed.max()) if fixed.notna().any() else math.nan,
        "patterns_with_any_fixed_rows": int(fixed.gt(0).sum()),
    }
    return table.reset_index(drop=True), summary


# ---------------------------------------------------------------------------
# Supplementary Table S2: antibiotic panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntrinsicReference:
    eucast: frozenset[tuple[str, str]]
    supplemental: dict[tuple[str, str], str]

    @classmethod
    def load(cls, reference_dir: Path, settings: Settings) -> IntrinsicReference:
        primary = pd.read_csv(reference_dir / settings.platform.reference_files["intrinsic_resistance"], sep=";")
        eucast = frozenset(
            zip(primary["microorganism"].map(normalize_label), primary["antibiotic"].map(normalize_antibiotic_label))
        )
        supplemental: dict[tuple[str, str], str] = {}
        name = settings.platform.reference_files.get("supplemental_intrinsic_resistance")
        if name and (reference_dir / name).exists():
            extra = pd.read_csv(reference_dir / name)
            notes = extra["note"] if "note" in extra.columns else pd.Series([""] * len(extra))
            for organism, antibiotic, note in zip(extra["microorganism"], extra["antibiotic"], notes):
                key = (normalize_label(organism), normalize_antibiotic_label(antibiotic))
                supplemental[key] = "" if pd.isna(note) else str(note).strip()
        return cls(eucast, supplemental)

    def reason(self, organism: str, antibiotic: str) -> str | None:
        key = (normalize_organism_label(organism), normalize_antibiotic_label(antibiotic))
        if key in self.eucast:
            return "Intrinsically resistant (EUCAST expected resistant phenotype)"
        if key in self.supplemental:
            note = self.supplemental[key]
            return "Biologically ineligible (curated supplemental reference" + (f": {note})" if note else ")")
        return None


def antibiotic_panel(
    culture_drug_episodes: pd.DataFrame,
    escalation_results: pd.DataFrame,
    cotesting_pairs: pd.DataFrame,
    validation_results: pd.DataFrame,
    organism: str,
    resolver: AntibioticClassificationResolver,
    reference: IntrinsicReference,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, object]]:
    observed = culture_drug_episodes.loc[
        culture_drug_episodes["susceptibility"].isin(settings.gold.observed_result_values)
    ]
    counts = observed.groupby(observed["antibiotic"].astype(str)).size()

    def drugs(frame: pd.DataFrame) -> set[str]:
        if frame.empty:
            return set()
        return set(frame["upstream_antibiotic"].astype(str)) | set(frame["downstream_antibiotic"].astype(str))

    candidates = drugs(escalation_results)
    support = escalation_results.loc[
        escalation_results["total_support_n"].ge(settings.cascade.min_total_support)
        & escalation_results["positive_support_n"].ge(settings.cascade.min_result_support)
        & escalation_results["negative_support_n"].ge(settings.cascade.min_result_support)
    ]
    supported_drugs = drugs(support)
    events = support["positive_tested_n"].fillna(0) + support["negative_tested_n"].fillna(0)
    with_events = drugs(support.loc[events.ge(settings.cascade.min_total_tested_events_for_retention)])
    screened = drugs(cotesting_pairs)
    retained = drugs(validation_results)

    rows = []
    for antibiotic, n in counts.items():
        resolved = resolver.resolve(antibiotic)
        ineligible = reference.reason(organism, antibiotic)
        if antibiotic in retained:
            reason = ""
        elif ineligible:
            reason = ineligible
        elif antibiotic not in candidates:
            reason = (
                "All candidate pairs removed by the co-testing screen"
                if antibiotic in screened
                else "No pair with both resistant and susceptible upstream results"
            )
        elif antibiotic not in supported_drugs:
            reason = "Below minimum candidate-pair support threshold"
        elif antibiotic not in with_events:
            reason = (
                "No downstream-observed episode in any support-passing pair"
                if settings.cascade.min_total_tested_events_for_retention <= 1
                else f"Fewer than {settings.cascade.min_total_tested_events_for_retention} downstream-observed episodes in every support-passing pair"
            )
        else:
            raise ValueError(f"{antibiotic} passes every retention rule but is not retained; retention logic has changed.")
        rows.append(
            {
                "antibiotic": antibiotic,
                "full_name": resolved.display_label,
                "abbreviation": resolved.abbreviation,
                "aware_category": resolved.aware_category,
                "antibiotic_class": resolved.antibiotic_class,
                "observed_results": int(n),
                "retained": "Yes" if antibiotic in retained else "No",
                "reason_not_retained": reason,
            }
        )
    table = pd.DataFrame(rows).sort_values("full_name", kind="mergesort").reset_index(drop=True)
    collisions = table.groupby("abbreviation")["full_name"].nunique()
    if collisions.gt(1).any():
        raise ValueError(f"Abbreviations shared by different antibiotics: {sorted(collisions.index[collisions.gt(1)])}")
    summary = {
        "observed_antibiotics": len(table),
        "retained_antibiotics": int(table["retained"].eq("Yes").sum()),
        "not_retained_count": int(table["retained"].eq("No").sum()),
        "not_retained": table.loc[
            table["retained"].eq("No"), ["antibiotic", "observed_results", "reason_not_retained"]
        ].to_dict(orient="records"),
        "abbreviation_collisions": sorted(collisions.index[collisions.gt(1)].tolist()),
        "unclassified": sorted(table.loc[table["aware_category"].eq("Unclassified"), "antibiotic"].tolist()),
    }
    return table, summary


# ---------------------------------------------------------------------------
# Companion paper (prevalence) supplementary tables
# ---------------------------------------------------------------------------


def _display_named(diagnostics: pd.DataFrame, resolver: AntibioticClassificationResolver) -> pd.DataFrame:
    """Rename the diagnostics table's raw drug labels to the display names the prevalence table uses."""
    named = diagnostics.copy()
    named["drug"] = named["Drug"].map(lambda value: resolver.canonical_display_label(resolver.normalize_label(value)))
    if not named["drug"].is_unique:
        raise ValueError("Two diagnostics rows map to the same drug name.")
    return named


def prevalence_diagnostics_table(
    prevalence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    resolver: AntibioticClassificationResolver,
) -> pd.DataFrame:
    """Supplementary Table S2 of the prevalence paper, one row per analysed drug."""
    named = _display_named(diagnostics, resolver)
    if set(named["drug"]) != set(prevalence["drug"]):
        raise ValueError("The prevalence and diagnostics tables cover different drugs.")
    merged = prevalence.merge(
        named[["drug", "Observed n", "Binary-evaluable tested n"]],
        on="drug",
        how="left",
        validate="one_to_one",
    )
    if merged["Binary-evaluable tested n"].isna().any():
        raise ValueError("Prevalence diagnostics are missing drugs present in the prevalence table.")
    if not merged["tested_n"].astype(int).eq(merged["Binary-evaluable tested n"].astype(int)).all():
        raise ValueError("Binary-evaluable counts differ between the prevalence and diagnostics tables.")
    table = merged.rename(
        columns={
            "Observed n": "observed_n",
            "tested_n": "binary_evaluable_n",
            "unknown_binary_outcome_n": "unknown_binary_n",
            "naive_prevalence_pct": "naive_resistant_pct",
            "prevalence_lower_bound_pct": "lower_bound_resistant_pct",
            "prevalence_upper_bound_pct": "upper_bound_resistant_pct",
            "mnar_lambda0_prevalence_pct": "mnar_lambda0_resistant_pct",
        }
    )
    columns = [
        "drug",
        "eligible_n",
        "observed_n",
        "binary_evaluable_n",
        "unknown_binary_n",
        "naive_resistant_pct",
        "lower_bound_resistant_pct",
        "upper_bound_resistant_pct",
        "mnar_lambda0_resistant_pct",
    ]
    return table[columns].sort_values("drug", kind="mergesort").reset_index(drop=True)


def smd_balance_table(
    diagnostics: pd.DataFrame, resolver: AntibioticClassificationResolver
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Supplementary Table S3 of the prevalence paper."""
    table = _display_named(diagnostics, resolver).rename(
        columns={
            "Mean abs SMD (observed vs unobserved)": "mean_abs_smd_observed_vs_unobserved",
            "Mean abs SMD (cascade evaluable vs unobserved)": "mean_abs_smd_triggered_vs_unobserved",
            "Mean abs SMD (independent evaluable vs unobserved)": "mean_abs_smd_independent_vs_unobserved",
            "Flagged covariates (observed vs unobserved)": "flagged_covariates_observed_vs_unobserved",
        }
    )
    columns = [
        "drug",
        "mean_abs_smd_observed_vs_unobserved",
        "mean_abs_smd_triggered_vs_unobserved",
        "mean_abs_smd_independent_vs_unobserved",
        "flagged_covariates_observed_vs_unobserved",
    ]
    table = table[columns].sort_values("drug", kind="mergesort").reset_index(drop=True)
    largest = table[columns[1:4]].astype(float).max(axis=1)
    summary = {
        "drugs": len(table),
        "drugs_with_a_mean_abs_smd_of_at_least_0.10": int(largest.ge(0.10).sum()),
        "largest_mean_abs_smd_min": float(largest.min()) if len(table) else math.nan,
        "largest_mean_abs_smd_max": float(largest.max()) if len(table) else math.nan,
        "drug_with_smallest_largest_mean_abs_smd": table.loc[largest.idxmin(), "drug"] if len(table) else None,
        "drug_with_largest_mean_abs_smd": table.loc[largest.idxmax(), "drug"] if len(table) else None,
    }
    return table, summary


# ---------------------------------------------------------------------------
# LaTeX table bodies
# ---------------------------------------------------------------------------
# Table bodies are written as macros into generated_numbers.tex and expanded in the
# papers with \runrows{key}{columns}. A macro expands cleanly inside an alignment,
# whereas \input leaves non-expandable tokens behind that start an empty row.

_LATEX_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "/": r"\slash ",
    "β": r"$\beta$",
    "ß": r"\ss{}",
    "—": "---",
    "–": "--",
    "…": r"\ldots{}",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "µ": r"$\mu$",
    "μ": r"$\mu$",
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "→": r"\,$\to$\,",
    "·": r"$\cdot$",
}


def latex_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    escaped = "".join(_LATEX_CHARACTERS.get(character, character) for character in str(value))
    unsupported = sorted({character for character in escaped if ord(character) > 127})
    if unsupported:
        raise ValueError(f"No LaTeX mapping for {unsupported} in {value!r}.")
    return escaped


def latex_int(value: object) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def latex_float(value: object, digits: int) -> str:
    return "---" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def latex_pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _rows(cells: list[list[str]]) -> str:
    return "".join(" & ".join(row) + " \\\\\n" for row in cells)


def s2_fragment(panel: pd.DataFrame) -> str:
    return _rows(
        [
            [latex_text(r.full_name), latex_text(r.abbreviation), latex_text(r.aware_category), latex_text(r.antibiotic_class), r.retained, latex_text(r.reason_not_retained)]
            for r in panel.itertuples(index=False)
        ]
    )


_FLOW_STAGES = [
    "raw_ast_rows",
    "culture_episodes",
    "observed_episode_drug_rows",
    "eligible_episode_drug_rows",
    "binary_upstream_pair_rows",
]


def s3_fragment(flow: pd.DataFrame, sites: list[str]) -> str:
    wide = flow.pivot(index="site", columns="stage", values="row_count")
    missing = [stage for stage in _FLOW_STAGES if stage not in wide.columns] + [site for site in sites if site not in wide.index]
    if missing:
        raise ValueError(f"Row-flow table lacks {missing}; rerun the report step with the current code.")
    wide = wide.loc[sites, _FLOW_STAGES].astype("int64")
    body = _rows([[latex_text(site_label(site)), *[latex_int(v) for v in wide.loc[site]]] for site in sites])
    total = ["\\textbf{Total}", *[f"\\textbf{{{latex_int(v)}}}" for v in wide.sum()]]
    return body + "\\midrule\n" + _rows([total])


def s4_fragment(summary: dict[str, object]) -> str:
    row = ["Combined \\textit{E. coli}", latex_int(summary["retained_patterns"]), latex_int(summary["robust"]), latex_int(summary["supported"]), latex_int(summary["validated"]), latex_pct(summary["validated_share"])]
    return _rows([row])


def s6_fragment(table: pd.DataFrame) -> str:
    rows = []
    for r in table.itertuples(index=False):
        marker = "\\textbf" if r.primary else ""
        cells = [latex_int(r.min_total_support), latex_int(r.min_result_support), latex_int(r.retained_patterns), latex_int(r.retained_in_primary_set), latex_int(r.added_not_validated), latex_int(r.robust), latex_int(r.supported), latex_int(r.mixed), latex_int(r.insufficient)]
        rows.append([f"{marker}{{{cell}}}" if marker else cell for cell in cells])
    return _rows(rows)


def s7_fragment(table: pd.DataFrame) -> str:
    return _rows(
        [
            [latex_text(r.upstream_abbreviation), latex_text(r.downstream_abbreviation), latex_float(r.episode_bootstrap_sign_stability, 3), latex_float(r.patient_bootstrap_sign_stability, 3), latex_int(r.n_patients)]
            for r in table.itertuples(index=False)
        ]
    )


def s8_fragment(table: pd.DataFrame, primary_threshold: float, alternative_threshold: float) -> str:
    labels = [
        f"$I^2\\le{primary_threshold:g}\\%$ (primary)",
        f"$I^2\\le{alternative_threshold:g}\\%$",
        "No heterogeneity check (direction agreement only)",
    ]
    if len(table) != len(labels):
        raise ValueError("Unexpected number of heterogeneity policies.")
    return _rows([[label, latex_int(r.robust), latex_int(r.supported), latex_int(r.mixed), latex_int(r.insufficient)] for label, r in zip(labels, table.itertuples(index=False))])


def transition_sentence(transitions: pd.DataFrame, policy: str) -> str:
    """Plain-text list of label changes under one policy, for \\runnum in captions."""
    moves = transitions.loc[transitions["policy"].eq(policy)].sort_values("patterns", ascending=False, kind="mergesort")
    if moves.empty:
        return "no pattern changes label"
    parts = [f"{r.patterns:,} {'moves' if r.patterns == 1 else 'move'} from {r.from_status} to {r.to_status}" for r in moves.itertuples(index=False)]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def s9_fragment(summary_table: pd.DataFrame, primary: float, stricter: float, min_support: int) -> str:
    labels = [
        f"${stricter:.2f}\\le\\mathrm{{PBI}}<{primary:.2f}$, support $\\ge{min_support}$ in both directions (excluded only at the stricter threshold)",
        "Retained at both thresholds",
    ]
    return _rows([[label, latex_int(r.pairs), latex_int(r.robust), latex_int(r.supported), latex_int(r.mixed_or_insufficient)] for label, r in zip(labels, summary_table.itertuples(index=False))])


_MODEL_COMPARISON_COLUMNS = [
    "ROC-AUC (train)",
    "ROC-AUC (validation)",
    "ROC-AUC (test)",
    "PR-AUC (test)",
    "Brier score (test)",
    "Balanced accuracy (test)",
]


def prediction_summary(metrics: pd.DataFrame) -> dict[str, object]:
    """Test-site prevalence, each model's selected threshold, and ROC-AUC by split."""
    test = metrics.loc[metrics["split"].eq("test")]
    prevalence = test["prevalence"].astype(float).unique()
    if len(prevalence) != 1:
        raise ValueError("Models disagree on the test-site prevalence; the metrics come from different data.")
    return {
        "test_prevalence": float(prevalence[0]),
        "selected_threshold": {r.model_name: float(r.decision_threshold) for r in test.itertuples(index=False)},
        "roc_auc": {
            model: {r.split: float(r.roc_auc) for r in group.itertuples(index=False)}
            for model, group in metrics.groupby("model_name")
        },
    }


def s12_fragment(comparison: pd.DataFrame, test_prevalence: float) -> str:
    values = comparison[_MODEL_COMPARISON_COLUMNS].apply(pd.to_numeric, errors="coerce")
    best = values["ROC-AUC (test)"].max()
    rows = []
    for model, row in zip(comparison["Model"], values.itertuples(index=False)):
        cells = [latex_float(value, 3) for value in row]
        if row[2] == best:
            cells[2] = f"\\textbf{{{cells[2]}}}"
        rows.append([latex_text(model), *cells])
    baseline = ["No-skill baseline", "0.500", "0.500", "0.500", latex_float(test_prevalence, 3), "---", "0.500"]
    return _rows(rows) + "\\midrule\n" + _rows([baseline])


def s13_fragment(thresholds: pd.DataFrame, selected: float) -> str:
    rows = []
    for row in thresholds.itertuples(index=False):
        cells = [latex_text(value) for value in row]
        if abs(float(row[0]) - selected) < 1e-9:
            cells = [f"\\textbf{{{cell}}}" for cell in cells]
        rows.append(cells)
    return _rows(rows)


def s14_fragment(coefficients: pd.DataFrame) -> str:
    coefficient = coefficients["Coefficient (log-OR)"].astype(float)
    odds = coefficients["Odds Ratio"].astype(float)

    def block(mask: pd.Series, heading: str) -> str:
        cells = [
            [latex_text(feature), f"${'+' if value >= 0 else '-'}${abs(value):.3f}", f"{ratio:.3f}"]
            for feature, value, ratio in zip(coefficients.loc[mask, "Feature"], coefficient[mask], odds[mask])
        ]
        return f"\\multicolumn{{3}}{{l}}{{\\textit{{{heading}}}}} \\\\\n" + _rows(cells)

    positive, negative = coefficient.ge(0), coefficient.lt(0)
    return (
        block(positive, f"Top {int(positive.sum())} positive (increase downstream-observation probability)")
        + "\\midrule\n"
        + block(negative, f"Top {int(negative.sum())} negative (decrease downstream-observation probability)")
    )


def numbers_tex(numbers: dict[str, object], provenance: str, table_rows: dict[str, str] | None = None) -> str:
    """One LaTeX macro per quoted number, read by the manuscripts' \runnum{key}.

    Integers and strings are defined once under their dotted key; decimals are
    defined under key|0 ... key|3, key|pct0 ... key|pct2 and, with an explicit
    $+$/$-$ sign, key|signed0 ... key|signed2, so the text chooses its precision.
    Keys absent from this run print as a red marker in the paper.
    """
    lines = [f"% Generated by SupplementaryTableBuilder from {provenance}. Do not edit.\n"]

    def define(key: str, value: str) -> None:
        lines.append(f"\\expandafter\\def\\csname amrnum:{key}\\endcsname{{{value}}}\n")

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                walk(f"{prefix}.{name}" if prefix else str(name), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{prefix}.{index}", item)
        elif isinstance(value, (bool, np.bool_)) or value is None:
            return
        elif isinstance(value, (int, np.integer)):
            define(prefix, latex_int(value))
        elif isinstance(value, (float, np.floating)):
            if math.isfinite(float(value)):
                for digits in range(4):
                    define(f"{prefix}|{digits}", f"{float(value):,.{digits}f}".replace(",", "{,}"))
                for digits in range(3):
                    define(f"{prefix}|pct{digits}", latex_pct(float(value), digits))
                for digits in range(3):
                    define(f"{prefix}|signed{digits}", _signed(value, digits))
        else:
            define(prefix, latex_text(value))

    walk("", numbers)
    for key, rows in (table_rows or {}).items():
        lines.append(f"\\expandafter\\def\\csname amrrows:{key}\\endcsname{{%\n{rows}}}\n")
    return "".join(lines)


def classification_reference_check(resolver: AntibioticClassificationResolver) -> dict[str, object]:
    owners = resolver.reference_label_owners()
    shared = {label: sorted(rows) for label, rows in owners.items() if len(rows) > 1}
    if shared:
        raise ValueError(f"Reference spellings claimed by more than one antibiotic: {dict(list(shared.items())[:5])}")
    return {"antibiotics": len({row for rows in owners.values() for row in rows}), "spellings": len(owners)}


# ---------------------------------------------------------------------------
# Descriptive summaries (report-step tables) as quoted numbers and table bodies
# ---------------------------------------------------------------------------

_SCOPE_COLUMNS = ("all_sites", "armd", "armd_ecuh", "armd_utsw")


def _scopes_present(frame: pd.DataFrame, column: str = "site") -> list[str]:
    seen = list(dict.fromkeys(frame[column].astype(str)))
    return (["all_sites"] if "all_sites" in seen else []) + [scope for scope in seen if scope != "all_sites"]


def _count_pct(count: object, share: object, digits: int = 1) -> str:
    if count is None or pd.isna(count):
        return "---"
    if share is None or pd.isna(share):
        return latex_int(count)
    return f"{latex_int(count)} ({100 * float(share):.{digits}f})"


def _pct(value: object, digits: int = 1) -> str:
    return "---" if value is None or pd.isna(value) else f"{100 * float(value):.{digits}f}"


def _median_iqr(median: object, q1: object, q3: object, digits: int = 0) -> str:
    if median is None or pd.isna(median):
        return "---"
    return f"{float(median):.{digits}f} [{float(q1):.{digits}f}--{float(q3):.{digits}f}]"


def _records_by_scope(table: pd.DataFrame, columns: list[str], key: str = "site") -> dict[str, dict[str, object]]:
    return {
        str(row[key]): {column: row[column] for column in columns if column in table.columns}
        for _, row in table.iterrows()
    }


def cohort_numbers(long: pd.DataFrame) -> dict[str, object]:
    """Table 1 values keyed scope -> section_characteristic slug (n, denominator, share, quartiles)."""
    numbers: dict[str, object] = {}
    for scope, part in long.groupby("scope", sort=False):
        entries = {}
        for row in part.itertuples(index=False):
            slug = safe_feature_name(f"{row.section} {row.characteristic}").lower()
            entry = {"n": int(row.n)}
            if not pd.isna(row.denominator):
                entry["denominator"] = int(row.denominator)
                entry["share"] = float(row.share) if not pd.isna(row.share) else math.nan
            if not pd.isna(row.median):
                entry.update({"median": float(row.median), "q1": float(row.q1), "q3": float(row.q3)})
            entries[slug] = entry
        numbers[str(scope)] = entries
    return numbers


def cohort_fragment(display: pd.DataFrame, scopes: list[str]) -> str:
    """Table 1 body: one column per scope; section headings as spanning italic rows."""
    width = 1 + len(scopes)
    lines, current = [], None
    for row in display.itertuples(index=False):
        values = [latex_text(getattr(row, scope)) if scope in display.columns else "---" for scope in scopes]
        if row.section == "Analysis set":
            lines.append(" & ".join([latex_text(row.characteristic), *values]) + " \\\\\n")
            continue
        if row.section != current:
            current = row.section
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{latex_text(row.section)}}}}} \\\\\n")
        lines.append(" & ".join([f"\\quad {latex_text(row.characteristic)}", *values]) + " \\\\\n")
    return "".join(lines)


_OPPORTUNITY_NUMBER_COLUMNS = [
    "episodes", "antibiotics", "grid_rows", "intrinsic_rows", "intrinsic_observed_rows", "not_available_rows",
    "not_available_observed_rows", "eligible_rows", "eligible_observed_rows", "eligible_unobserved_rows", "observed_rows",
    "observed_ineligible_rows", "intrinsic_share", "not_available_share", "eligible_share", "observed_share_of_eligible",
]


def opportunity_space_fragment(table: pd.DataFrame) -> str:
    rows = []
    for row in table.itertuples(index=False):
        label = f"\\textbf{{{latex_text(site_label(row.site))}}}" if row.site == "all_sites" else latex_text(site_label(row.site))
        rows.append([
            label,
            latex_int(row.grid_rows),
            _count_pct(row.intrinsic_rows, row.intrinsic_share),
            _count_pct(row.not_available_rows, row.not_available_share),
            _count_pct(row.eligible_rows, row.eligible_share),
            _count_pct(row.eligible_observed_rows, row.observed_share_of_eligible),
        ])
    return _rows(rows)


_FLOW_KIND = {
    "raw_rows": "total", "organism_rows": "total", "interpretable_rows": "total", "observed_episode_drugs": "total",
    "culture_episodes": "total", "conflicting_episode_drugs": "subset", "episodes_without_result": "subset",
}
_FLOW_OPTIONAL = {"not_ingested_rows", "not_harmonized_rows"}


def exclusion_flow_numbers(flow: pd.DataFrame) -> tuple[dict[str, object], list[str]]:
    """Counts per scope and stage, and the reconciliation failures (none when every row is accounted for)."""
    numbers: dict[str, object] = {}
    problems = []
    for scope, part in flow.groupby("site", sort=False):
        counts = {str(stage): int(count) for stage, count in zip(part["stage"], part["count"])}
        numbers[str(scope)] = counts
        identities = {
            "AST rows": counts["raw_rows"] - counts["not_ingested_rows"] - counts["exact_duplicate_rows"]
            - counts["not_harmonized_rows"] - counts["other_organism_rows"] - counts["organism_rows"],
            "organism rows": counts["organism_rows"] - counts["no_result_rows"] - counts["other_non_interpretive_rows"]
            - counts["excluded_assay_rows"] - counts["interpretable_rows"],
            "episode-antibiotic results": counts["interpretable_rows"] - counts["repeated_rows"]
            - counts["conflicting_extra_rows"] - counts["observed_episode_drugs"],
        }
        problems += [f"exclusion flow does not reconcile at {scope}: {name} off by {value:,}" for name, value in identities.items() if value]
    return numbers, problems


def exclusion_flow_fragment(flow: pd.DataFrame, scopes: list[str]) -> str:
    width = 1 + len(scopes)
    wide = flow.pivot_table(index=["order", "block", "stage", "label"], columns="site", values="count", aggfunc="first").reset_index()
    wide = wide.sort_values("order", kind="mergesort")
    lines, current = [], None
    for row in wide.itertuples(index=False):
        values = [getattr(row, scope) if scope in wide.columns else math.nan for scope in scopes]
        if row.stage in _FLOW_OPTIONAL and all(pd.isna(value) or value == 0 for value in values):
            continue
        if row.block != current:
            current = row.block
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{latex_text(row.block)}}}}} \\\\\n")
        kind = _FLOW_KIND.get(row.stage, "excluded")
        label = latex_text(row.label)
        if kind == "total":
            label = f"\\textbf{{{label}}}"
        elif kind == "excluded":
            label = f"\\quad $-$ {label}"
        else:
            label = f"\\quad\\quad {label}"
        cells = ["---" if pd.isna(value) else latex_int(value) for value in values]
        lines.append(" & ".join([label, *cells]) + " \\\\\n")
    return "".join(lines)


_EXPOSURE_COLUMNS = [
    "thin_support_threshold", "eligible_rows", "strata", "thin_strata", "thin_rows", "thin_share", "before_first_observed_rows",
    "before_first_observed_share", "after_last_observed_rows", "after_last_observed_share", "outside_observed_window_rows",
    "outside_observed_window_share", "thin_or_outside_rows", "thin_or_outside_share", "time_unknown_rows",
]


def availability_exposure_fragment(table: pd.DataFrame) -> str:
    rows = []
    for row in table.itertuples(index=False):
        label = f"\\textbf{{{latex_text(site_label(row.site))}}}" if row.site == "all_sites" else latex_text(site_label(row.site))
        rows.append([
            label,
            latex_int(row.eligible_rows),
            _count_pct(row.thin_rows, row.thin_share),
            _count_pct(row.before_first_observed_rows, row.before_first_observed_share),
            _count_pct(row.after_last_observed_rows, row.after_last_observed_share),
            _count_pct(row.thin_or_outside_rows, row.thin_or_outside_share),
        ])
    return _rows(rows)


_BREADTH_COLUMNS = [
    "episodes", "observed_median", "observed_q1", "observed_q3", "observed_mean", "observed_max", "eligible_median",
    "eligible_q1", "eligible_q3", "episodes_without_result", "episodes_without_result_share", "observed_share_of_eligible",
]


def panel_breadth_numbers(table: pd.DataFrame) -> dict[str, object]:
    numbers: dict[str, dict[str, object]] = {}
    for row in table.to_dict(orient="records"):
        numbers.setdefault(str(row["site"]), {})[safe_feature_name(str(row["specimen_group"])).lower()] = {
            column: row[column] for column in _BREADTH_COLUMNS
        }
    return numbers


def panel_breadth_fragment(table: pd.DataFrame) -> str:
    rows = []
    previous = None
    for row in table.itertuples(index=False):
        scope = latex_text(site_label(row.site)) if row.site != previous else ""
        if row.site != previous and row.site == "all_sites":
            scope = f"\\textbf{{{scope}}}"
        previous = row.site
        rows.append([
            scope,
            latex_text(row.specimen_group),
            latex_int(row.episodes),
            _median_iqr(row.observed_median, row.observed_q1, row.observed_q3),
            _median_iqr(row.eligible_median, row.eligible_q1, row.eligible_q3),
            _pct(row.observed_share_of_eligible),
            latex_int(row.episodes_without_result),
        ])
    return _rows(rows)


def coverage_by_era_numbers(table: pd.DataFrame) -> dict[str, object]:
    pooled = table.loc[table["antibiotic"].eq("__all_antibiotics__")]
    numbers: dict[str, dict[str, object]] = {}
    for row in pooled.to_dict(orient="records"):
        numbers.setdefault(str(row["site"]), {})[safe_feature_name(str(row["era"])).lower()] = {
            "eligible_n": int(row["eligible_n"]), "observed_n": int(row["observed_n"]), "observed_share": float(row["observed_share"]),
        }
    return numbers


def coverage_by_era_fragment(table: pd.DataFrame, scopes: list[str]) -> str:
    pooled = table.loc[table["antibiotic"].eq("__all_antibiotics__")]
    rows = []
    for era in sorted(pooled["era"].astype(str).unique()):
        cells = [latex_text(str(era).replace("-", "\u2013"))]
        for scope in scopes:
            match = pooled.loc[pooled["site"].eq(scope) & pooled["era"].astype(str).eq(era)]
            cells.append("---" if match.empty else f"{_pct(match['observed_share'].iloc[0])} ({latex_int(match['eligible_n'].iloc[0])})")
        rows.append(cells)
    return _rows(rows)


def observed_results(antibiogram: pd.DataFrame) -> dict[str, object]:
    """Interpretations among observed episode-antibiotic results, all sites (most resistant per episode and drug)."""
    pooled = antibiogram.loc[antibiogram["site"].eq("all_sites")]
    total = int(pooled["tested_n"].sum())
    counts = {name: int(pooled[f"{name}_n"].sum()) for name in ("susceptible", "intermediate", "resistant")}
    return {
        "results": total,
        **{f"{name}_share": (count / total if total else math.nan) for name, count in counts.items()},
        "conflicting": int(pooled["conflicting_n"].sum()),
        "antibiotics": int(pooled["antibiotic"].nunique()),
    }


def antibiogram_fragment(table: pd.DataFrame, sites: list[str]) -> str:
    """Pooled S/I/R, each site's resistant share, and the first-isolate resistant share; AWaRe headings."""
    pooled = table.loc[table["site"].eq("all_sites")]
    per_site = table.set_index(["site", "antibiotic"])
    width = 6 + len(sites)  # antibiotic, tested, S, I, R, one resistant share per site, first isolate
    lines, current = [], None
    for row in pooled.itertuples(index=False):
        category = row.aware_category if isinstance(row.aware_category, str) and row.aware_category else "Not classified"
        if category != current:
            current = category
            heading = {"Not Set": "Not classified", "Unclassified": "Not classified"}.get(category, category)
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{latex_text(heading)}}}}} \\\\\n")
        marker = "$^{\\dagger}$" if bool(row.sparse) else ""
        cells = [
            f"\\quad {latex_text(row.antibiotic_display)}{marker}",
            latex_int(row.tested_n),
            _pct(row.susceptible_share),
            _pct(row.intermediate_share),
            _pct(row.resistant_share),
        ]
        for site in sites:
            if (site, row.antibiotic) in per_site.index:
                site_row = per_site.loc[(site, row.antibiotic)]
                value = _pct(site_row["resistant_share"])
                cells.append(f"{value}$^{{\\dagger}}$" if bool(site_row["sparse"]) else value)
            else:
                cells.append("---")
        cells.append(_pct(row.first_isolate_resistant_share))
        lines.append(" & ".join(cells) + " \\\\\n")
    return "".join(lines)


_PER_PATIENT_COLUMNS = [
    "patients", "episodes", "episodes_per_patient_mean", "episodes_per_patient_median", "episodes_per_patient_q1",
    "episodes_per_patient_q3", "episodes_per_patient_max", "patients_with_multiple", "patients_with_multiple_share",
    "episodes_from_patients_with_multiple_share",
]


def episodes_per_patient_fragment(table: pd.DataFrame) -> str:
    rows = []
    for row in table.itertuples(index=False):
        label = f"\\textbf{{{latex_text(site_label(row.site))}}}" if row.site == "all_sites" else latex_text(site_label(row.site))
        rows.append([
            label,
            latex_int(row.patients),
            latex_int(row.episodes),
            f"{float(row.episodes_per_patient_mean):.2f}",
            _median_iqr(row.episodes_per_patient_median, row.episodes_per_patient_q1, row.episodes_per_patient_q3),
            latex_int(row.episodes_per_patient_max),
            _count_pct(row.patients_with_multiple, row.patients_with_multiple_share),
            _pct(row.episodes_from_patients_with_multiple_share),
        ])
    return _rows(rows)


# ---------------------------------------------------------------------------
# Main-text tables and paragraphs
# ---------------------------------------------------------------------------

_RESULTS_FLOW_STAGES = [
    "raw_ast_rows", "culture_episodes", "observed_episode_drug_rows", "eligible_episode_drug_rows", "eligible_directed_pair_rows",
]


def row_flow_numbers(flow: pd.DataFrame, sites: list[str]) -> dict[str, object]:
    wide = flow.pivot(index="site", columns="stage", values="row_count")
    numbers = {site: {stage: int(wide.loc[site, stage]) for stage in wide.columns} for site in sites if site in wide.index}
    numbers["total"] = {stage: int(wide.loc[[site for site in sites if site in wide.index], stage].sum()) for stage in wide.columns}
    return numbers


def results_flow_fragment(flow: pd.DataFrame, sites: list[str]) -> str:
    wide = flow.pivot(index="site", columns="stage", values="row_count")
    missing = [stage for stage in _RESULTS_FLOW_STAGES if stage not in wide.columns] + [site for site in sites if site not in wide.index]
    if missing:
        raise ValueError(f"Row-flow table lacks {missing}; rerun the report step with the current code.")
    wide = wide.loc[sites, _RESULTS_FLOW_STAGES].astype("int64")
    body = _rows([[latex_text(site_label(site)), *[latex_int(v) for v in wide.loc[site]]] for site in sites])
    total = ["\\textbf{Total}", *[f"\\textbf{{{latex_int(v)}}}" for v in wide.sum()]]
    return body + "\\midrule\n" + _rows([total])


def _conservative_order(edges: pd.DataFrame, direction: str) -> pd.DataFrame:
    """The directional forests' ranking: both branches observed first, then the conservative 95% limit."""
    tested = edges[["resistant_tested_n", "susceptible_tested_n"]].apply(pd.to_numeric, errors="coerce")
    bound = pd.to_numeric(edges["er_ci_lower" if direction == "escalation" else "er_ci_upper"], errors="coerce")
    signed = np.log2(bound.where(bound > 0))
    ranked = edges.assign(
        _both=tested.gt(0).all(axis=1),
        _rank=(signed if direction == "escalation" else -signed).fillna(-np.inf),
        _magnitude=np.abs(np.log2(pd.to_numeric(edges["escalation_ratio"], errors="coerce"))),
    )
    return ranked.sort_values(["_both", "_rank", "_magnitude", "total_support_n"], ascending=[False, False, False, False], kind="mergesort")


def selected_patterns(edge_report: pd.DataFrame, escalation_n: int = 4, suppression_n: int = 3) -> pd.DataFrame:
    """The main-text example patterns: the top robust patterns per direction under the forests' ranking.

    Supported patterns fill a direction only when it has too few robust ones.
    """
    validated = edge_report.loc[edge_report["validation_status"].isin(VALIDATED)].copy()
    chosen = []
    for direction, count in (("escalation", escalation_n), ("suppression", suppression_n)):
        pool = validated.loc[validated["cascade_direction"].astype(str).eq(direction)]
        ordered = pd.concat([
            _conservative_order(pool.loc[pool["validation_status"].eq("robust")], direction),
            _conservative_order(pool.loc[pool["validation_status"].eq("supported")], direction),
        ])
        chosen.append(ordered.head(count))
    return pd.concat(chosen, ignore_index=True)


def _ratio(value: object) -> str:
    if value is None or pd.isna(value):
        return "---"
    value = float(value)
    if value >= 10:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def patterns_fragment(patterns: pd.DataFrame, resolver: AntibioticClassificationResolver) -> str:
    rows = []
    for index, row in enumerate(patterns.itertuples(index=False)):
        if index and row.cascade_direction != patterns["cascade_direction"].iloc[index - 1]:
            rows.append(None)
        rows.append([
            latex_text(_display(resolver, row.upstream_antibiotic)),
            latex_text(_display(resolver, row.downstream_antibiotic)),
            latex_float(row.resistant_downstream_test_probability, 3),
            latex_float(row.susceptible_downstream_test_probability, 3),
            _ratio(row.escalation_ratio),
            _ratio(row.adjusted_odds_ratio),
            latex_int(row.total_support_n),
            latex_text(row.validation_status),
        ])
    body = ""
    for cells in rows:
        body += "\\midrule\n" if cells is None else _rows([cells])
    return body


def _display(resolver: AntibioticClassificationResolver, label: object) -> str:
    text = str(resolver.resolve(str(label)).display_label).strip().lower()
    return text[:1].upper() + text[1:]


def adjusted_direction_summary(adjusted_results: pd.DataFrame, validation_results: pd.DataFrame) -> dict[str, object]:
    """Direction agreement between each estimable adjusted OR and its pattern's unadjusted escalation ratio."""
    validated = _pair_keys(validation_results.loc[validation_results["validation_status"].isin(VALIDATED)])
    frame = validated[PAIR + ["cascade_direction", "observed_escalation_ratio"]].merge(
        _pair_keys(adjusted_results)[PAIR + ["adjusted_odds_ratio", "non_estimable_reason"]], on=PAIR, how="left", validate="one_to_one"
    )
    odds = pd.to_numeric(frame["adjusted_odds_ratio"], errors="coerce")
    estimable = frame["non_estimable_reason"].isna() & odds.notna() & np.isfinite(odds) & odds.gt(0)
    escalation = frame["cascade_direction"].astype(str).eq("escalation")
    consistent = estimable & ((escalation & odds.gt(1)) | (~escalation & odds.lt(1)))
    summary: dict[str, object] = {
        "validated": len(frame),
        "estimable": int(estimable.sum()),
        "consistent": int(consistent.sum()),
        "consistent_share": float(consistent.sum() / estimable.sum()) if estimable.any() else math.nan,
        "discordant": int((estimable & ~consistent).sum()),
    }
    for name, mask in (("escalation", escalation), ("suppression", ~escalation)):
        values = odds[estimable & mask]
        summary[name] = {
            "estimable": len(values),
            "median_or": float(values.median()) if len(values) else math.nan,
            "min_or": float(values.min()) if len(values) else math.nan,
            "max_or": float(values.max()) if len(values) else math.nan,
        }
    return summary


_NON_ESTIMABLE_PHRASES = {
    "quasi_separation": "non-estimable because of quasi-complete separation",
    "zero_variance_outcome_or_exposure": "non-estimable because downstream observation or the upstream result does not vary",
    "outcome_events_below_model_dimension": "non-estimable because it has fewer observed or unobserved downstream episodes than model parameters",
    "model_fit_failed": "non-estimable because the model did not converge",
    "design_matrix_unavailable": "non-estimable because no usable design matrix could be built",
}


def zero_branch_example(edge_report: pd.DataFrame, adjusted_results: pd.DataFrame, resolver: AntibioticClassificationResolver) -> dict[str, object]:
    """Validated patterns with no downstream-observed episode in one upstream branch, and the best-supported example."""
    validated = _pair_keys(edge_report.loc[edge_report["validation_status"].isin(VALIDATED)])
    tested = validated[["resistant_tested_n", "susceptible_tested_n"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    empty = tested.eq(0).any(axis=1)
    summary: dict[str, object] = {"validated": len(validated), "with_empty_branch": int(empty.sum())}
    if empty.any():
        example = validated.loc[empty].sort_values(["total_support_n", *PAIR], ascending=[False, True, True], kind="mergesort").iloc[0]
        branch = "susceptible" if float(example["susceptible_tested_n"]) == 0 else "resistant"
        reasons = _pair_keys(adjusted_results).set_index(PAIR)["non_estimable_reason"]
        reason = reasons.get((str(example["upstream_antibiotic"]), str(example["downstream_antibiotic"])))
        summary["example"] = {
            "upstream": _display(resolver, example["upstream_antibiotic"]),
            "downstream": _display(resolver, example["downstream_antibiotic"]),
            "empty_branch": branch,
            "adjusted": "estimable" if reason is None or pd.isna(reason) else _NON_ESTIMABLE_PHRASES.get(str(reason), "non-estimable"),
        }
    return summary


def _arrow(transition: str) -> str:
    return str(transition).replace(" -> ", " \u2192 ")


def aware_summary(transitions: pd.DataFrame) -> dict[str, object]:
    """Validated patterns by WHO AWaRe tier movement, and every transition group largest first."""
    table = transitions.copy()
    table["validated_edge_n"] = pd.to_numeric(table["validated_edge_n"], errors="coerce").fillna(0).astype(int)
    table = table.sort_values(["validated_edge_n", "aware_transition"], ascending=[False, True], kind="mergesort")
    directions = table.groupby("aware_direction")["validated_edge_n"].sum()
    groups = [f"{_arrow(row.aware_transition)} ({row.validated_edge_n:,})" for row in table.itertuples(index=False)]
    ratios = [
        f"{row.support_weighted_mean_escalation_ratio:.2f} for {_arrow(row.aware_transition)}"
        for row in table.itertuples(index=False) if pd.notna(row.support_weighted_mean_escalation_ratio)
    ]
    return {
        "validated": int(table["validated_edge_n"].sum()),
        **{name: 0 for name in ("upward", "lateral", "downward", "unclassified")},
        **{str(name): int(count) for name, count in directions.items()},
        "groups": len(table),
        "largest_group": _arrow(table["aware_transition"].iloc[0]) if len(table) else "",
        "largest_group_n": int(table["validated_edge_n"].iloc[0]) if len(table) else 0,
        "groups_sentence": _join(groups),
        "weighted_er_sentence": _join(ratios),
    }


def _join(parts: list[str]) -> str:
    if not parts:
        return "none"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def prevalence_summary(prevalence: pd.DataFrame, negligible_pp: float = 0.1) -> dict[str, object]:
    """Distribution of the naive-minus-MNAR(lambda=0) shift, naive prevalence, and bound width across drugs.

    The shift is naive minus the eligible-denominator MNAR(0) estimate, in
    percentage points: positive when the tested-only figure is the higher one.
    """
    shift = pd.to_numeric(prevalence["mnar_lambda0_shift_from_naive_pct"], errors="coerce")
    naive = pd.to_numeric(prevalence["naive_prevalence_pct"], errors="coerce")
    width = pd.to_numeric(prevalence["prevalence_upper_bound_pct"], errors="coerce") - pd.to_numeric(prevalence["prevalence_lower_bound_pct"], errors="coerce")
    estimated = shift.notna()
    return {
        "drugs": len(prevalence),
        "with_mnar_estimate": int(estimated.sum()),
        "shift_median_pp": float(shift.median()),
        "shift_min_pp": float(shift.min()),
        "shift_max_pp": float(shift.max()),
        "naive_lower": int(shift.le(-negligible_pp).sum()),
        "naive_higher": int(shift.ge(negligible_pp).sum()),
        "negligible": int((estimated & shift.abs().lt(negligible_pp)).sum()),
        "negligible_threshold_pp": float(negligible_pp),
        "naive_min_pct": float(naive.min()),
        "naive_max_pct": float(naive.max()),
        "naive_median_pct": float(naive.median()),
        "bound_width_median_pp": float(width.median()),
        "bound_width_max_pp": float(width.max()),
    }


def prevalence_top(prevalence: pd.DataFrame, top_n: int) -> pd.DataFrame:
    shift = pd.to_numeric(prevalence["mnar_lambda0_shift_from_naive_pct"], errors="coerce")
    return (
        prevalence.assign(_abs=shift.abs())
        .sort_values(["_abs", "eligible_n"], ascending=[False, False], kind="mergesort")
        .head(top_n)
        .drop(columns="_abs")
    )


def _signed(value: object, digits: int = 1) -> str:
    """$+$1.2 / $-$1.2, and an unsigned zero when the value rounds to zero at this precision."""
    if value is None or pd.isna(value):
        return "---"
    magnitude = f"{abs(float(value)):,.{digits}f}".replace(",", "{,}")
    if float(magnitude.replace("{,}", "")) == 0:
        return magnitude
    return f"${'+' if float(value) > 0 else '-'}${magnitude}"


def prevalence_fragment(top: pd.DataFrame, resolver: AntibioticClassificationResolver) -> str:
    return _rows([
        [
            latex_text(_display(resolver, row.drug)),
            latex_float(row.naive_prevalence_pct, 1),
            latex_float(row.prevalence_lower_bound_pct, 1),
            latex_float(row.prevalence_upper_bound_pct, 1),
            latex_int(row.unknown_binary_outcome_n),
            latex_float(row.mnar_lambda0_prevalence_pct, 1),
            _signed(row.mnar_lambda0_shift_from_naive_pct),
            latex_float(row.rho_independent_vs_cascade, 2),
            latex_float(row.cascade_trigger_fraction, 2),
        ]
        for row in top.itertuples(index=False)
    ])


_MODEL_DISPLAY = {"logistic_regression": "logistic regression", "random_forest": "random forest", "xgboost": "XGBoost"}


def prediction_details(metrics: pd.DataFrame, thresholds: pd.DataFrame | None, coefficients: pd.DataFrame | None) -> dict[str, object]:
    """Ranges and gaps the prediction text quotes, the selected LR threshold row, and the strongest LR features."""
    details: dict[str, object] = {}
    auc = metrics.pivot_table(index="model_name", columns="split", values="roc_auc", aggfunc="first")
    details["roc_auc_range"] = {split: {"min": float(auc[split].min()), "max": float(auc[split].max())} for split in auc.columns}
    if "test" in auc.columns:
        best = auc["test"].idxmax()
        details["test_best_model"] = _MODEL_DISPLAY.get(str(best), str(best).replace("_", " "))
        details["test_gap_to_best"] = {str(model): float(auc.loc[best, "test"] - value) for model, value in auc["test"].items()}
        details["train_to_test_gap"] = {str(model): float(auc.loc[model, "train"] - auc.loc[model, "test"]) for model in auc.index if "train" in auc.columns}
    if thresholds is not None and not thresholds.empty:
        numeric = thresholds.apply(pd.to_numeric, errors="coerce")
        selected = metrics.loc[metrics["split"].eq("test") & metrics["model_name"].eq("logistic_regression"), "decision_threshold"]
        if len(selected):
            row = numeric.iloc[(numeric["Threshold"] - float(selected.iloc[0])).abs().argmin()]
            details["lr_at_selected"] = {"threshold": float(row["Threshold"]), "precision": float(row["Precision"]), "recall": float(row["Recall"]), "f1": float(row["F1"])}
        high = numeric.loc[(numeric["Threshold"] - 0.7).abs().lt(1e-9)]
        if len(high):
            row = high.iloc[0]
            details["lr_at_070"] = {"precision": float(row["Precision"]), "recall": float(row["Recall"]), "f1": float(row["F1"])}
    if coefficients is not None and not coefficients.empty:
        ranked = pd.DataFrame({"feature": coefficients["Feature"].astype(str), "odds_ratio": pd.to_numeric(coefficients["Odds Ratio"], errors="coerce")})
        ranked = ranked.dropna(subset=["odds_ratio"]).sort_values("odds_ratio", ascending=False, kind="mergesort")
        details["lr_top_positive"] = ranked.head(3).to_dict(orient="records")
        details["lr_top_negative"] = ranked.tail(3).iloc[::-1].to_dict(orient="records")
    return details


# ---------------------------------------------------------------------------
# Availability-denominator sensitivity runs against the primary run
# ---------------------------------------------------------------------------


def availability_sensitivity(
    primary_validation: pd.DataFrame,
    primary_eligibility: pd.DataFrame,
    primary_settings: Settings,
    runs: dict[str, tuple[Settings, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Each availability-denominator run against the primary run.

    ``*_eligibility`` holds ``is_eligible`` and ``is_observed_tested`` from the
    run's gold eligibility table; ``*_validation`` the run's validation labels.
    A pattern is kept when validated in both runs, dropped when validated only
    in the primary run (not retained, or retained with a non-validated label),
    and added when validated only in the sensitivity run.
    """
    primary_labels = _pair_keys(primary_validation).set_index(PAIR)["validation_status"]
    primary_validated = set(primary_labels.index[primary_labels.isin(VALIDATED)])
    primary_eligible = int(pd.to_numeric(primary_eligibility["is_eligible"], errors="coerce").fillna(0).eq(1).sum())

    def row_for(label: str, settings: Settings, validation: pd.DataFrame, eligibility: pd.DataFrame, primary: bool) -> dict[str, object]:
        eligible = pd.to_numeric(eligibility["is_eligible"], errors="coerce").fillna(0).eq(1)
        observed = pd.to_numeric(eligibility["is_observed_tested"], errors="coerce").fillna(0).eq(1)
        labels = _pair_keys(validation).set_index(PAIR)["validation_status"]
        validated = set(labels.index[labels.isin(VALIDATED)])
        counts = _label_counts(labels)
        shared = sorted(set(labels.index) & set(primary_labels.index))
        moved = sum(1 for key in shared if labels.get(key) != primary_labels.get(key))
        return {
            "run": label,
            "primary": primary,
            "availability_min_observed": int(settings.gold.eligibility.availability_min_observed),
            "availability_era_years": int(settings.gold.eligibility.availability_era_years),
            "eligible_opportunities": int(eligible.sum()),
            "eligible_share_of_primary": _json_share(int(eligible.sum()), primary_eligible),
            "observed_share_of_eligible": _json_share(int((eligible & observed).sum()), int(eligible.sum())),
            "retained_patterns": len(labels),
            **counts,
            "validated": len(validated),
            "validated_kept": len(validated & primary_validated),
            "validated_dropped": len(primary_validated - validated),
            "validated_added": len(validated - primary_validated),
            "retained_in_both_label_changed": moved,
        }

    rows = [row_for("primary", primary_settings, primary_validation, primary_eligibility, True)]
    rows += [row_for(label, settings, validation, eligibility, False) for label, (settings, validation, eligibility) in runs.items()]
    table = pd.DataFrame(rows)
    summary = {row["run"]: {key: value for key, value in row.items() if key != "run"} for row in rows}
    return table, summary


def _json_share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def availability_sensitivity_fragment(table: pd.DataFrame) -> str:
    rows = []
    for row in table.itertuples(index=False):
        label = f"Support $\\ge${row.availability_min_observed}, {row.availability_era_years}-year era"
        if row.primary:
            label = f"\\textbf{{{label} (primary)}}"
        rows.append([
            label,
            latex_int(row.eligible_opportunities),
            _pct(row.eligible_share_of_primary),
            _pct(row.observed_share_of_eligible),
            latex_int(row.retained_patterns),
            latex_int(row.robust),
            latex_int(row.supported),
            "---" if row.primary else latex_int(row.validated_kept),
            "---" if row.primary else latex_int(row.validated_dropped),
            "---" if row.primary else latex_int(row.validated_added),
        ])
    return _rows(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SupplementaryBuildResult:
    output_dir: Path
    written: dict[str, Path] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    numbers: dict[str, object] = field(default_factory=dict)


class SupplementaryTableBuilder:
    """Locate a run's artefacts and write every generated supplementary table."""

    NUMBERS_FILENAME = "supplementary_numbers.json"
    # Rows of the main-text prevalence table (paper 2, Table 2): largest absolute MNAR(0) shift first.
    PREVALENCE_TABLE_ROWS = 10
    LATEX_DIRNAME = "latex"

    def __init__(self, settings: Settings, path_manager: PathManager, classification_path: Path | None = None) -> None:
        self._settings = settings
        self._paths = path_manager
        self._resolver = AntibioticClassificationResolver(
            classification_path or path_manager.project_root / "data" / "antibiotic_classification_complete.csv"
        )

    def output_dir(self, organism: str) -> Path:
        return self._tables_dir(organism) / "supplementary"

    def _tables_dir(self, organism: str) -> Path:
        scope_dir = scoped_output_dir(Path("."), "combined", organism=organism)
        return self._paths.project_root / self._settings.reporting.tables_dir / scope_dir

    def _modeling_dir(self, organism: str) -> Path:
        root = self._paths.paths.artifacts / self._settings.modeling.output_dir / self._settings.modeling.task_name
        return scoped_output_dir(root, "combined", organism=organism)

    def _cascade_dir(self, organism: str, site: str | None = None) -> Path:
        root = self._paths.paths.artifacts / self._settings.cascade.outputs.result_dir
        return scoped_output_dir(root, "site" if site else "combined", site=site, organism=organism)

    def _episode_times(self, path: Path) -> pd.Series:
        """The culture episodes' order times, or an empty series when the gold table does not carry them."""
        column = self._settings.gold.eligibility.availability_time_column
        if not path.exists() or column not in pq.read_schema(path).names:
            return pd.Series(dtype="object")
        return pd.read_parquet(path, columns=[column])[column]

    @staticmethod
    def _gold_started(gold_dir: Path) -> float | None:
        """When the gold layer an output derives from was written (its earliest file)."""
        times = [path.stat().st_mtime for path in gold_dir.glob("*.parquet")] if gold_dir.exists() else []
        return min(times) if times else None

    @staticmethod
    def _read(
        path: Path,
        missing: list[str],
        columns: list[str] | None = None,
        not_before: float | None = None,
    ) -> pd.DataFrame | dict | None:
        """Read one input, or record it as missing when absent or older than its gold layer.

        Every input is produced after the gold layer it derives from, so an older file is
        left over from an earlier run and must not be mixed with this run's results.
        """
        if not path.exists():
            missing.append(str(path))
            return None
        if not_before is not None and path.stat().st_mtime < not_before:
            missing.append(f"{path} (older than this run's gold layer)")
            return None
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_parquet(path, columns=columns)

    def build(self, organism: str) -> SupplementaryBuildResult:
        settings = self._settings
        cascade_dir = self._cascade_dir(organism)
        gold_dir = scoped_output_dir(self._paths.paths.gold, "combined", organism=organism)
        tables_dir = self._tables_dir(organism)
        result = SupplementaryBuildResult(output_dir=self.output_dir(organism))
        missing = result.missing_inputs
        numbers = result.numbers
        tables: dict[str, pd.DataFrame] = {}
        run_started = self._gold_started(gold_dir)

        def read(path: Path, missing_inputs: list[str], columns: list[str] | None = None, not_before: float | None = run_started) -> pd.DataFrame | dict | None:
            return self._read(path, missing_inputs, columns=columns, not_before=not_before)


        validation = read(cascade_dir / "validation_results.parquet", missing)
        edge_report = read(cascade_dir / "edge_report.parquet", missing)
        escalation = read(cascade_dir / "escalation_results.parquet", missing)
        adjusted = read(cascade_dir / "adjusted_results.parquet", missing)
        cotesting_pairs = read(cascade_dir / "cotesting_pairs.parquet", missing)
        cotesting_probabilities = read(cascade_dir / "cotesting_probabilities.parquet", missing)
        patient_cluster = read(cascade_dir / "patient_cluster_bootstrap_sensitivity.parquet", missing)
        era = read(cascade_dir / "era_stratified_permutation_sensitivity.parquet", missing)
        ridge = read(cascade_dir / "ridge_penalty_sensitivity.parquet", missing)
        culture_episodes = read(gold_dir / "culture_episodes.parquet", missing, columns=["source_site", "anon_id"], not_before=None)
        gold_metadata_path = self._paths.paths.metadata / "datasets" / "gold" / f"gold_build_combined__{safe_feature_name(organism)}.json"
        gold_metadata = read(gold_metadata_path, missing)
        if gold_metadata is not None and "organism_labels" not in gold_metadata:
            # Written by gold builds before organism-label pooling was recorded.
            missing.append(f"{gold_metadata_path} (no organism_labels; rebuild the gold layer)")
            gold_metadata = None
        culture_drug = read(gold_dir / "culture_drug_episodes.parquet", missing, columns=["antibiotic", "susceptibility"], not_before=None)
        table_r = read(tables_dir / "table_r_correction_sensitivity.csv", missing)
        flow = read(tables_dir / "table_a_data_quality_flow.csv", missing)
        model_comparison = read(tables_dir / "supp_pred_table1_model_comparison.csv", missing)
        lr_coefficients = read(tables_dir / "supp_pred_table2_lr_top_features.csv", missing)
        lr_thresholds = read(tables_dir / "supp_pred_table3_lr_threshold.csv", missing)
        model_metrics = read(self._modeling_dir(organism) / "site__all_models" / "metrics.parquet", missing)
        prevalence = read(tables_dir / "table_k_prevalence_shift.csv", missing)
        coverage = read(tables_dir / "table_w_observation_coverage.csv", missing)
        diagnostics = read(tables_dir / "table_k_prevalence_shift_diagnostics.csv", missing)
        cohort_long = read(tables_dir / "table_v_cohort_characteristics_long.csv", missing)
        cohort_display = read(tables_dir / "table_v_cohort_characteristics.csv", missing)
        opportunity = read(tables_dir / "table_b_eligibility.csv", missing)
        exclusion = read(tables_dir / "table_a_exclusion_flow.csv", missing)
        exposure = read(tables_dir / "table_u_availability_exposure.csv", missing)
        breadth = read(tables_dir / "table_w_panel_breadth.csv", missing)
        coverage_era = read(tables_dir / "table_w_observation_coverage_by_era.csv", missing)
        antibiogram_table = read(tables_dir / "table_k_antibiogram.csv", missing)
        per_patient = read(tables_dir / "table_t_episodes_per_patient.csv", missing)
        aware_transitions = read(tables_dir / "table_m_aware_transition_summary.csv", missing)
        site_reports = {
            site: read(
                self._cascade_dir(organism, site) / "edge_report.parquet",
                missing,
                not_before=self._gold_started(scoped_output_dir(self._paths.paths.gold, "site", site=site, organism=organism)),
            )
            for site in settings.platform.sites
        }

        fragments: dict[str, str] = {}
        numbers["antibiotic_reference"] = classification_reference_check(self._resolver)
        if validation is not None:
            tables["table_s4_validation_summary.csv"], numbers["validation"] = validation_summary(validation)
            fragments["s4"] = s4_fragment(numbers["validation"])
            (
                tables["table_s8_heterogeneity_threshold_sensitivity.csv"],
                tables["table_s8_heterogeneity_label_transitions.csv"],
                numbers["heterogeneity"],
            ) = heterogeneity_sensitivity(validation, settings)
            policies = tables["table_s8_heterogeneity_threshold_sensitivity.csv"]
            fragments["s8"] = s8_fragment(policies, settings.cascade.validation.i_squared_threshold, 75.0)
            transitions = tables["table_s8_heterogeneity_label_transitions.csv"]
            numbers["heterogeneity"]["alternative_transitions"] = transition_sentence(transitions, policies["policy"].iloc[1])
            numbers["heterogeneity"]["no_check_transitions"] = transition_sentence(transitions, policies["policy"].iloc[2])
        if flow is not None:
            fragments["s3"] = s3_fragment(flow, list(settings.platform.sites))
        if edge_report is not None:
            numbers["downstream_events"] = downstream_event_profile(edge_report)
        if validation is not None and escalation is not None:
            tables["table_s6_support_threshold_sensitivity.csv"], numbers["support_thresholds"] = (
                support_threshold_sensitivity(escalation, validation, settings)
            )
            fragments["s6"] = s6_fragment(tables["table_s6_support_threshold_sensitivity.csv"])
        if validation is not None and patient_cluster is not None:
            tables["table_s7_patient_cluster_bootstrap_sensitivity.csv"], numbers["patient_cluster"] = patient_cluster_table(
                patient_cluster, validation, self._resolver, settings.cascade.validation.bootstrap_sign_stability_threshold
            )
            fragments["s7"] = s7_fragment(tables["table_s7_patient_cluster_bootstrap_sensitivity.csv"])
        if culture_episodes is not None:
            numbers["episode_multiplicity"] = episode_multiplicity(culture_episodes)
            numbers["study_period"] = study_period(self._episode_times(gold_dir / "culture_episodes.parquet"))
        if gold_metadata is not None:
            numbers["cohort"] = organism_label_counts(gold_metadata, organism)
        if coverage is not None:
            numbers["observation_coverage"] = observation_coverage_numbers(coverage)
        if cotesting_probabilities is not None:
            numbers["pbi_distribution"] = pbi_distribution(cotesting_probabilities, settings)
            if cotesting_pairs is not None and escalation is not None and validation is not None:
                numbers["pair_flow"] = pair_flow(cotesting_probabilities, cotesting_pairs, escalation, validation, settings)
        if validation is not None and cotesting_probabilities is not None:
            (
                tables["table_s9_pbi_sensitivity.csv"],
                tables["table_s9_pbi_sensitivity_summary.csv"],
                numbers["pbi"],
            ) = pbi_sensitivity(cotesting_probabilities, validation, settings)
            fragments["s9"] = s9_fragment(
                tables["table_s9_pbi_sensitivity_summary.csv"],
                numbers["pbi"]["primary_threshold"],
                numbers["pbi"]["stricter_threshold"],
                numbers["pbi"]["min_support_each_direction"],
            )
        if edge_report is not None and adjusted is not None:
            tables["table_s10_confounding_strength_calibration.csv"], numbers["confounding_strength"] = (
                confounding_strength_table(edge_report, adjusted)
            )
        if adjusted is not None and validation is not None:
            tables["table_adjusted_model_estimability.csv"], numbers["adjusted_model_estimability"] = (
                adjusted_model_estimability(adjusted, validation)
            )
        if validation is not None and all(report is not None for report in site_reports.values()):
            tables["table_s11_cross_site_replication.csv"], numbers["cross_site"] = cross_site_replication(
                validation, site_reports
            )
        if ridge is not None and edge_report is not None:
            tables["table_s16_ridge_penalty_sensitivity.csv"], numbers["ridge_penalty"] = ridge_penalty_table(
                ridge, edge_report, DownstreamTestingRegression._RIDGE_PENALTY
            )
        if table_r is not None:
            numbers["continuity_correction"] = continuity_correction_summary(table_r)
        if era is not None and validation is not None:
            tables["table_s15_era_stratified_permutation_sensitivity.csv"], numbers["era_stratified"] = era_stratified_table(
                era, validation
            )
        if all(frame is not None for frame in (culture_drug, escalation, cotesting_pairs, validation)):
            reference = IntrinsicReference.load(self._paths.paths.reference, settings)
            tables["table_s2_antibiotic_panel_reference.csv"], numbers["antibiotic_panel"] = antibiotic_panel(
                culture_drug, escalation, cotesting_pairs, validation, organism, self._resolver, reference, settings
            )
            fragments["s2"] = s2_fragment(tables["table_s2_antibiotic_panel_reference.csv"])
        if model_metrics is not None:
            numbers["prediction"] = prediction_summary(model_metrics)
            if model_comparison is not None:
                fragments["s12"] = s12_fragment(model_comparison, numbers["prediction"]["test_prevalence"])
            if lr_thresholds is not None:
                fragments["s13"] = s13_fragment(lr_thresholds, numbers["prediction"]["selected_threshold"]["logistic_regression"])
        if lr_coefficients is not None:
            fragments["s14"] = s14_fragment(lr_coefficients)
        if prevalence is not None and diagnostics is not None:
            tables["table_s2_prevalence_diagnostics.csv"] = prevalence_diagnostics_table(prevalence, diagnostics, self._resolver)
            tables["table_s3_smd_balance_summary.csv"], numbers["smd_balance"] = smd_balance_table(diagnostics, self._resolver)

        sites = list(settings.platform.sites)
        if cohort_long is not None and cohort_display is not None:
            numbers["cohort_characteristics"] = cohort_numbers(cohort_long)
            fragments["cohort_by_site"] = cohort_fragment(cohort_display, [s for s in ("all_sites", *sites) if s in cohort_display.columns])
        if opportunity is not None:
            numbers["opportunity_space"] = _records_by_scope(opportunity, _OPPORTUNITY_NUMBER_COLUMNS)
            fragments["opportunity_space"] = opportunity_space_fragment(opportunity)
        if exclusion is not None:
            numbers["exclusion_flow"], problems = exclusion_flow_numbers(exclusion)
            missing.extend(problems)
            fragments["exclusion_flow"] = exclusion_flow_fragment(exclusion, _scopes_present(exclusion))
        if exposure is not None:
            numbers["availability_exposure"] = _records_by_scope(exposure, _EXPOSURE_COLUMNS)
            fragments["availability_exposure"] = availability_exposure_fragment(exposure)
        if breadth is not None:
            numbers["panel_breadth"] = panel_breadth_numbers(breadth)
            fragments["panel_breadth"] = panel_breadth_fragment(breadth)
        if coverage_era is not None:
            numbers["coverage_by_era"] = coverage_by_era_numbers(coverage_era)
            fragments["coverage_by_era"] = coverage_by_era_fragment(coverage_era, _scopes_present(coverage_era))
        if antibiogram_table is not None:
            numbers["observed_results"] = {**observed_results(antibiogram_table), "min_tested": settings.reporting.antibiogram_min_tested}
            fragments["antibiogram"] = antibiogram_fragment(antibiogram_table, sites)
        if per_patient is not None:
            numbers["episodes_per_patient"] = _records_by_scope(per_patient, _PER_PATIENT_COLUMNS)
            fragments["episodes_per_patient"] = episodes_per_patient_fragment(per_patient)
        if flow is not None:
            numbers["row_flow"] = row_flow_numbers(flow, sites)
            fragments["results_flow"] = results_flow_fragment(flow, sites)
        if edge_report is not None:
            patterns = selected_patterns(edge_report)
            fragments["main_patterns"] = patterns_fragment(patterns, self._resolver)
            numbers["main_patterns"] = {
                "escalation": int(patterns["cascade_direction"].eq("escalation").sum()),
                "suppression": int(patterns["cascade_direction"].eq("suppression").sum()),
            }
        if adjusted is not None and validation is not None:
            numbers["adjusted_direction"] = adjusted_direction_summary(adjusted, validation)
        if edge_report is not None and adjusted is not None:
            numbers["zero_branch"] = zero_branch_example(edge_report, adjusted, self._resolver)
        if aware_transitions is not None:
            numbers["aware"] = aware_summary(aware_transitions)
        if prevalence is not None:
            top = prevalence_top(prevalence, self.PREVALENCE_TABLE_ROWS)
            numbers["prevalence_summary"] = {**prevalence_summary(prevalence), "shown": len(top)}
            fragments["prevalence_top"] = prevalence_fragment(top, self._resolver)
        if model_metrics is not None and "prediction" in numbers:
            numbers["prediction"].update(prediction_details(model_metrics, lr_thresholds, lr_coefficients))
        configured_runs = settings.reporting.availability_sensitivity_environments
        if configured_runs and validation is not None:
            runs = {}
            for label, environment in configured_runs:
                run_settings = ConfigLoader(self._paths.project_root).load(environment)
                run_paths = PathManager(self._paths.project_root, run_settings)
                run_gold = scoped_output_dir(run_paths.paths.gold, "combined", organism=organism)
                run_cascade = scoped_output_dir(
                    run_paths.paths.artifacts / run_settings.cascade.outputs.result_dir, "combined", organism=organism
                )
                run_validation = self._read(run_cascade / "validation_results.parquet", missing, not_before=self._gold_started(run_gold))
                run_eligibility = self._read(run_gold / "eligible_pairs.parquet", missing, columns=["is_eligible", "is_observed_tested"])
                if run_validation is not None and run_eligibility is not None:
                    runs[label] = (run_settings, run_validation, run_eligibility)
            primary_eligibility = read(gold_dir / "eligible_pairs.parquet", missing, columns=["is_eligible", "is_observed_tested"], not_before=None)
            if primary_eligibility is not None and len(runs) == len(configured_runs):
                tables["table_availability_sensitivity.csv"], numbers["availability_sensitivity"] = availability_sensitivity(
                    validation, primary_eligibility, settings, runs
                )
                fragments["availability_sensitivity"] = availability_sensitivity_fragment(tables["table_availability_sensitivity.csv"])

        # This directory holds only this builder's output; clearing it means a table that
        # could not be rebuilt this time is absent rather than silently left from a past run.
        result.output_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(result.output_dir.rglob("*"), reverse=True):
            if stale.is_file():
                stale.unlink()
        for name, frame in tables.items():
            path = result.output_dir / name
            frame.to_csv(path, index=False)
            result.written[name] = path
        latex_dir = result.output_dir / self.LATEX_DIRNAME
        latex_dir.mkdir(exist_ok=True)
        numbers["missing_inputs"] = list(missing)
        numbers_macros = latex_dir / "generated_numbers.tex"
        numbers_macros.write_text(numbers_tex(_json_ready(numbers), f"{cascade_dir} and {tables_dir}", fragments), encoding="utf-8")
        result.written[numbers_macros.name] = numbers_macros
        numbers_path = result.output_dir / self.NUMBERS_FILENAME
        numbers_path.write_text(json.dumps(_json_ready(numbers), indent=2, sort_keys=True), encoding="utf-8")
        result.written[self.NUMBERS_FILENAME] = numbers_path
        return result
