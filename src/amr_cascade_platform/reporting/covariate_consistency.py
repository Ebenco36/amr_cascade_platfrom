"""Reconcile the availability figure's exact counts with the cohort table."""

from __future__ import annotations

import pandas as pd

from amr_cascade_platform.core.utils.site_labels import site_label


def covariate_count_errors(profile: pd.DataFrame, cohort: pd.DataFrame) -> list[str]:
    """Check counts, not rounded display strings. Labs/vitals are record-only diagnostics."""
    required = {"scope", "covariate", "episodes", "recorded_n", "not_recorded_n", "evaluable"}
    if not required <= set(profile) or not {"scope", "section", "characteristic", "n"} <= set(cohort):
        return ["Covariate consistency inputs lack required columns"]
    if profile.empty or cohort.empty:
        return ["Covariate consistency inputs are empty"]
    if profile.duplicated(["scope", "covariate"]).any():
        return ["Duplicate scope/covariate rows in availability counts"]
    if cohort.duplicated(["scope", "section", "characteristic"]).any():
        return ["Duplicate scope/section/characteristic rows in cohort table"]

    missing_rows = {
        "Age": ("Age, years", "Not recorded"),
        "Sex": ("Sex, source code", "Not recorded"),
        "Ordering context": ("Ordering context", "Not recorded"),
        "Specimen type": ("Specimen", "Not recorded"),
        "Medication record before culture": ("In the 90 days before culture", "No medication record"),
        "Earlier culture record": ("In the 90 days before culture", "No prior culture record"),
        "Nursing-home record": ("In the 90 days before culture", "No nursing-home record"),
        "Procedure record": ("In the 90 days before culture", "No procedure record"),
        "Comorbidity data": ("Comorbidity and deprivation", "Comorbidity data not available"),
        "Area Deprivation Index": ("Comorbidity and deprivation", "Area Deprivation Index not available"),
        "Area Deprivation Index state rank": ("Comorbidity and deprivation", "Area Deprivation Index state rank not available"),
    }
    errors = []
    cohort = cohort.copy()
    cohort["scope"] = cohort.scope.map(site_label)
    expected_scopes = set(cohort.loc[cohort.characteristic.eq("Culture episodes"), "scope"])
    if set(profile.scope) != expected_scopes:
        errors.append("Availability figure and cohort table cover different scopes")
    for scope in expected_scopes:
        table = cohort.loc[cohort.scope.eq(scope)].set_index(["section", "characteristic"])
        cells = profile.loc[profile.scope.eq(scope)].set_index("covariate")
        expected_n = float(table.loc[("Analysis set", "Culture episodes"), "n"])
        if not cells.episodes.eq(expected_n).all():
            errors.append(f"{scope}: availability denominator differs from cohort episodes")
        if not cells.evaluable.eq(True).all():
            errors.append(f"{scope}: availability contains unevaluable inputs")
        if not (cells.recorded_n + cells.not_recorded_n).eq(cells.episodes).all():
            errors.append(f"{scope}: recorded and missing counts do not partition episodes")
        if (cells.recorded_n.lt(0) | cells.not_recorded_n.lt(0)).any():
            errors.append(f"{scope}: negative availability count")
        for name, key in missing_rows.items():
            if name not in cells.index:
                errors.append(f"{scope}: availability omits {name}")
                continue
            if key not in table.index:
                # Categorical 'Not recorded' rows are omitted when no episode is missing.
                if key[1] == "Not recorded" and key[0] in table.index.get_level_values(0):
                    expected = 0
                else:
                    errors.append(f"{scope}: cohort table omits {key}")
                    continue
            else:
                expected = float(table.loc[key, "n"])
            actual = float(cells.loc[name, "not_recorded_n"])
            if actual != expected:
                errors.append(f"{scope}, {name}: figure missing={actual:g}, cohort missing={expected:g}")
        dated = "Comorbidity data, all starts dated"
        undated_key = ("Comorbidity and deprivation", "Condition with unrecorded start")
        absent_key = missing_rows["Comorbidity data"]
        if dated not in cells.index or undated_key not in table.index or absent_key not in table.index:
            errors.append(f"{scope}: comorbidity dating comparison is incomplete")
        else:
            expected = float(table.loc[undated_key, "n"]) + float(table.loc[absent_key, "n"])
            if float(cells.loc[dated, "not_recorded_n"]) != expected:
                errors.append(f"{scope}: dated-comorbidity figure disagrees with cohort counts")
    return errors
