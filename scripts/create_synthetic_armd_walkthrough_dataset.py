"""Create a small ARMD-like three-site tutorial dataset.

The generated files are educational fixtures, not research data. They mimic the
three ARMD-family raw layouts closely enough for notebooks to demonstrate:

* site-specific file naming and timestamp-column differences,
* repeated AST rows per culture episode,
* selective downstream testing after upstream resistance,
* the gap between observed tested rows and eligible organism-drug opportunities.

Outputs are written outside ``data/raw`` by default so real data are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SiteSpec:
    site: str
    time_column: str
    cohort_file: str
    demographics_file: str
    ward_file: str
    exposure_file: str
    subtype_exposure_file: str
    comorbidity_file: str
    microbial_resistance_file: str
    labs_file: str
    vitals_file: str
    adi_file: str
    prior_med_file: str
    prior_procedures_file: str
    nursing_home_file: str
    prior_organism_file: str
    prior_organism_days_column: str
    implied_files: bool = False


SITE_SPECS = (
    SiteSpec(
        site="armd",
        time_column="order_time_jittered_utc",
        cohort_file="microbiology_cultures_cohort.csv",
        demographics_file="microbiology_cultures_demographics.csv",
        ward_file="microbiology_cultures_ward_info.csv",
        exposure_file="microbiology_cultures_antibiotic_class_exposure.csv",
        subtype_exposure_file="microbiology_cultures_antibiotic_subtype_exposure.csv",
        comorbidity_file="microbiology_cultures_comorbidity.csv",
        microbial_resistance_file="microbiology_cultures_microbial_resistance.csv",
        labs_file="microbiology_cultures_labs.csv",
        vitals_file="microbiology_cultures_vitals.csv",
        adi_file="microbiology_cultures_adi_scores.csv",
        prior_med_file="microbiology_cultures_prior_med.csv",
        prior_procedures_file="microbiology_cultures_priorprocedures.csv",
        nursing_home_file="microbiology_cultures_nursing_home_visits.csv",
        prior_organism_file="microbiology_culture_prior_infecting_organism.csv",
        prior_organism_days_column="prior_infecting_organism_days_to_culutre",
        implied_files=True,
    ),
    SiteSpec(
        site="armd_ecuh",
        time_column="order_time_jittered",
        cohort_file="00_microbiology_cultures_cohort.csv",
        demographics_file="05_microbiology_cultures_demographics.csv",
        ward_file="012_microbiology_cultures_ward_info.csv",
        exposure_file="02_microbiology_cultures_antibiotic_class_exposure.csv",
        subtype_exposure_file="03_microbiology_cultures_antibiotic_subtype_exposure.csv",
        comorbidity_file="04_microbiology_cultures_comorbidity.csv",
        microbial_resistance_file="07_microbiology_cultures_microbial_resistance.csv",
        labs_file="06_microbiology_cultures_labs.csv",
        vitals_file="013_microbiology_cultures_vitals.csv",
        adi_file="01_microbiology_cultures_adi_scores.csv",
        prior_med_file="010_microbiology_cultures_prior_med.csv",
        prior_procedures_file="011_microbiology_cultures_prior_procedures.csv",
        nursing_home_file="08_microbiology_cultures_nursing_home_visits.csv",
        prior_organism_file="09_microbiology_cultures_prior_infecting_organism.csv",
        prior_organism_days_column="prior_infecting_organism_days_to_culture",
    ),
    SiteSpec(
        site="armd_utsw",
        time_column="order_time_jittered",
        cohort_file="microbiology_cultures_cohort.csv",
        demographics_file="microbiology_cultures_demographics.csv",
        ward_file="microbiology_cultures_ward_info.csv",
        exposure_file="microbiology_cultures_antibiotic_class_exposure.csv",
        subtype_exposure_file="microbiology_cultures_antibiotic_subtype_exposure.csv",
        comorbidity_file="microbiology_cultures_comorbidity.csv",
        microbial_resistance_file="microbiology_cultures_microbial_resistance.csv",
        labs_file="microbiology_cultures_labs.csv",
        vitals_file="microbiology_cultures_vitals.csv",
        adi_file="microbiology_cultures_adi_scores.csv",
        prior_med_file="microbiology_cultures_prior_med.csv",
        prior_procedures_file="microbiology_cultures_prior_procedures.csv",
        nursing_home_file="microbiology_cultures_nursing_home_visits.csv",
        prior_organism_file="microbiology_cultures_prior_infecting_organism.csv",
        prior_organism_days_column="prior_infecting_organism_days_to_culture",
    ),
)


ANTIBIOTICS = (
    "CEFTRIAXONE",
    "CIPROFLOXACIN",
    "FOSFOMYCIN",
    "MEROPENEM",
    "NITROFURANTOIN",
    "AMPICILLIN",
)


def _episode_ids(site: str, index: int) -> dict[str, str]:
    return {
        "anon_id": f"{site}_P{index // 3:04d}",
        "pat_enc_csn_id_coded": f"{site}_E{index:05d}",
        "order_proc_id_coded": f"{site}_O{index:05d}",
    }


def _resistance_call(rng: np.random.Generator, probability: float) -> str:
    draw = rng.random()
    if draw < probability:
        return "Resistant"
    if draw < probability + 0.04:
        return "Intermediate"
    return "Susceptible"


def _site_testing_profile(spec: SiteSpec, severity: int, ceftriaxone_result: str) -> dict[str, float]:
    """Return per-drug testing probabilities for one episode.

    The profile intentionally creates a visible cascade: fosfomycin and meropenem
    are much more likely to be observed when ceftriaxone is resistant.
    """

    cef_resistant = ceftriaxone_result == "Resistant"
    site_shift = {"armd": 0.00, "armd_ecuh": 0.04, "armd_utsw": -0.03}[spec.site]
    severe_shift = 0.06 if severity else 0.0
    return {
        "CEFTRIAXONE": 0.98,
        "CIPROFLOXACIN": 0.72 + site_shift,
        "NITROFURANTOIN": 0.55 if spec.site != "armd_utsw" else 0.35,
        "AMPICILLIN": 0.48 + site_shift,
        "FOSFOMYCIN": (0.78 if cef_resistant else 0.08) + severe_shift,
        "MEROPENEM": (0.42 if cef_resistant else 0.04) + severe_shift,
    }


def _generate_site(spec: SiteSpec, episodes_per_site: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=11 * list(s.site for s in SITE_SPECS).index(spec.site))

    cohort_rows: list[dict[str, object]] = []
    demographics_rows: list[dict[str, object]] = []
    ward_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    subtype_exposure_rows: list[dict[str, object]] = []
    comorbidity_rows: list[dict[str, object]] = []
    microbial_resistance_rows: list[dict[str, object]] = []
    labs_rows: list[dict[str, object]] = []
    vitals_rows: list[dict[str, object]] = []
    adi_rows: list[dict[str, object]] = []
    prior_med_rows: list[dict[str, object]] = []
    prior_procedure_rows: list[dict[str, object]] = []
    nursing_home_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    implied_rows: list[dict[str, object]] = []

    for i in range(episodes_per_site):
        ids = _episode_ids(spec.site, i)
        timestamp = base_date + pd.Timedelta(days=int(i % 45), hours=int(i % 8))
        time_value = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        keys = {**ids, spec.time_column: time_value}
        organism = "ESCHERICHIA COLI" if rng.random() < 0.82 else "KLEBSIELLA PNEUMONIAE"
        severity = int(rng.random() < {"armd": 0.28, "armd_ecuh": 0.34, "armd_utsw": 0.24}[spec.site])
        prior_abx = int(rng.random() < (0.32 + 0.18 * severity))
        cef_prob = 0.13 + 0.12 * severity + 0.06 * prior_abx
        cipro_prob = 0.18 + 0.10 * severity + 0.08 * prior_abx

        ceftriaxone_result = _resistance_call(rng, cef_prob)
        cipro_result = _resistance_call(rng, cipro_prob)
        probs = _site_testing_profile(spec, severity, ceftriaxone_result)

        for antibiotic in ANTIBIOTICS:
            if rng.random() > min(max(probs[antibiotic], 0.0), 1.0):
                continue
            if antibiotic == "CEFTRIAXONE":
                susceptibility = ceftriaxone_result
            elif antibiotic == "CIPROFLOXACIN":
                susceptibility = cipro_result
            elif antibiotic in {"FOSFOMYCIN", "MEROPENEM"}:
                base_resistance = 0.07 + 0.18 * (ceftriaxone_result == "Resistant") + 0.05 * severity
                susceptibility = _resistance_call(rng, base_resistance)
            else:
                susceptibility = _resistance_call(rng, 0.16 + 0.08 * severity + 0.04 * prior_abx)
            cohort_rows.append(
                {
                    **keys,
                    "ordering_mode": "Inpatient" if severity else "Outpatient",
                    "culture_description": "Urine Culture" if i % 3 else "Blood Culture",
                    "was_positive": 1,
                    "organism": organism,
                    "antibiotic": antibiotic,
                    "susceptibility": susceptibility,
                }
            )
            if susceptibility == "Resistant":
                microbial_resistance_rows.append(
                    {
                        **keys,
                        "organism": organism,
                        "antibiotic": antibiotic,
                        "resistant_time_to_culturetime": int(rng.integers(0, 4)),
                    }
                )

        demographics_rows.append({**keys, "age": rng.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "90+"]), "gender": int(rng.random() < 0.52)})
        ward_rows.append({**keys, "hosp_ward_IP": severity, "hosp_ward_OP": 1 - severity, "hosp_ward_ER": int(rng.random() < 0.18), "hosp_ward_ICU": severity})
        adi_rows.append({**keys, "adi_score": int(rng.integers(1, 101)), "adi_state_rank": int(rng.integers(1, 11))})

        wbc = 7.5 + 3.0 * severity + rng.normal(0, 1.1)
        cr = 0.9 + 0.35 * severity + rng.normal(0, 0.12)
        lactate = 1.2 + 0.9 * severity + rng.normal(0, 0.25)
        labs_rows.append(
            {
                **ids,
                "Period_Day": 0,
                "Q25_wbc": round(wbc - 0.8, 2),
                "Q75_wbc": round(wbc + 0.8, 2),
                "median_wbc": round(wbc, 2),
                "Q25_neutrophils": round(58 + 8 * severity, 2),
                "Q75_neutrophils": round(70 + 8 * severity, 2),
                "median_neutrophils": round(64 + 8 * severity, 2),
                "Q25_lymphocytes": 12.0,
                "Q75_lymphocytes": 26.0,
                "median_lymphocytes": 18.0,
                "Q25_hgb": 10.5,
                "Q75_hgb": 13.5,
                "median_hgb": 12.1,
                "Q25_plt": 180.0,
                "Q75_plt": 310.0,
                "median_plt": 240.0,
                "Q25_na": 134.0,
                "Q75_na": 141.0,
                "median_na": 138.0,
                "Q25_hco3": 20.0,
                "Q75_hco3": 27.0,
                "median_hco3": 24.0,
                "Q25_bun": 10.0,
                "Q75_bun": 28.0,
                "median_bun": 18.0,
                "Q25_cr": round(cr - 0.1, 2),
                "Q75_cr": round(cr + 0.1, 2),
                "median_cr": round(cr, 2),
                "Q25_lactate": round(lactate - 0.2, 2),
                "Q75_lactate": round(lactate + 0.2, 2),
                "median_lactate": round(lactate, 2),
                "Q25_procalcitonin": round(0.08 + 0.15 * severity, 2),
                "Q75_procalcitonin": round(0.18 + 0.25 * severity, 2),
                "median_procalcitonin": round(0.12 + 0.20 * severity, 2),
                "first_procalcitonin": round(0.10 + 0.18 * severity, 2),
                "last_procalcitonin": round(0.12 + 0.22 * severity, 2),
                "first_lactate": round(lactate, 2),
                "last_lactate": round(lactate - 0.1, 2),
                "first_cr": round(cr, 2),
                "last_cr": round(cr + 0.02, 2),
                "first_bun": 18.0,
                "last_bun": 19.0,
                "first_hco3": 24.0,
                "last_hco3": 24.5,
                "first_na": 138.0,
                "last_na": 139.0,
                "first_plt": 240.0,
                "last_plt": 245.0,
                "first_hgb": 12.1,
                "last_hgb": 12.0,
                "first_lymphocytes": 18.0,
                "last_lymphocytes": 18.5,
                "first_neutrophils": round(64 + 8 * severity, 2),
                "last_neutrophils": round(63 + 8 * severity, 2),
                "first_wbc": round(wbc, 2),
                "last_wbc": round(wbc - 0.2, 2),
            }
        )
        vitals_rows.append(
            {
                **ids,
                "Q25_heartrate": 75 + 12 * severity,
                "Q75_heartrate": 92 + 18 * severity,
                "median_heartrate": 84 + 15 * severity,
                "Q25_resprate": 14 + 2 * severity,
                "Q75_resprate": 20 + 4 * severity,
                "median_resprate": 17 + 3 * severity,
                "Q25_temp": 36.4 + 0.3 * severity,
                "Q75_temp": 37.2 + 0.7 * severity,
                "median_temp": 36.8 + 0.5 * severity,
                "Q25_sysbp": 106 - 8 * severity,
                "Q75_sysbp": 132 - 4 * severity,
                "median_sysbp": 120 - 6 * severity,
                "Q25_diasbp": 58,
                "Q75_diasbp": 76,
                "median_diasbp": 67,
                "first_diasbp": 67,
                "last_diasbp": 68,
                "first_sysbp": 120 - 6 * severity,
                "last_sysbp": 121 - 5 * severity,
                "first_temp": 36.8 + 0.5 * severity,
                "last_temp": 36.7 + 0.3 * severity,
                "first_resprate": 17 + 3 * severity,
                "last_resprate": 16 + 2 * severity,
                "first_heartrate": 84 + 15 * severity,
                "last_heartrate": 82 + 11 * severity,
            }
        )
        if prior_abx:
            medication_days = int(rng.integers(1, 80))
            exposure_rows.append({**keys, "medication_category": "antibiotic", "medication_name": "ceftriaxone", "antibiotic_class": "cephalosporin", "time_to_culturetime": medication_days})
            subtype_exposure_rows.append({**keys, "medication_category": "antibiotic", "medication_name": "ceftriaxone", "antibiotic_subtype": "third_generation_cephalosporin", "antibiotic_subtype_category": "cephalosporin", "medication_time_to_culturetime": medication_days})
            prior_med_rows.append({**keys, "medication_name": "ceftriaxone", "medication_time_to_culturetime": medication_days, "medication_category": "antibiotic"})
        if rng.random() < 0.42:
            comorbidity_rows.append({**keys, "comorbidity_component": rng.choice(["diabetes", "renal_failure", "malignancy"]), "comorbidity_component_start_days_culture": 365, "comorbidity_component_end_days_culture": "Null"})
        if rng.random() < 0.18:
            prior_procedure_rows.append({**keys, "procedure_description": rng.choice(["dialysis", "central_line", "mechanical_ventilation"]), "procedure_time_to_culturetime": int(rng.integers(1, 60))})
        if rng.random() < 0.10:
            nursing_home_rows.append({**keys, "nursing_home_visit_culture": int(rng.integers(1, 180))})
        if rng.random() < 0.22:
            prior_rows.append({**keys, "prior_organism": organism, spec.prior_organism_days_column: int(rng.integers(14, 365))})
        if spec.implied_files and rng.random() < 0.08:
            implied_rows.append(
                {
                    **keys,
                    "organism": organism,
                    "antibiotic": "CEFTRIAXONE",
                    "susceptibility": "Susceptible",
                    "implied_susceptibility": "Susceptible",
                    "implied_rule": "tutorial_rule",
                }
            )

    return {
        spec.cohort_file: pd.DataFrame(cohort_rows),
        spec.demographics_file: pd.DataFrame(demographics_rows),
        spec.ward_file: pd.DataFrame(ward_rows),
        spec.exposure_file: pd.DataFrame(exposure_rows),
        spec.subtype_exposure_file: pd.DataFrame(subtype_exposure_rows),
        spec.comorbidity_file: pd.DataFrame(comorbidity_rows),
        spec.microbial_resistance_file: pd.DataFrame(microbial_resistance_rows),
        spec.labs_file: pd.DataFrame(labs_rows),
        spec.vitals_file: pd.DataFrame(vitals_rows),
        spec.adi_file: pd.DataFrame(adi_rows),
        spec.prior_med_file: pd.DataFrame(prior_med_rows),
        spec.prior_procedures_file: pd.DataFrame(prior_procedure_rows),
        spec.nursing_home_file: pd.DataFrame(nursing_home_rows),
        spec.prior_organism_file: pd.DataFrame(prior_rows),
        "microbiology_cultures_implied_susceptibility.csv": pd.DataFrame(implied_rows),
        "implied_susceptibility_rules.csv": pd.DataFrame(
            [{"Organism": "ESCHERICHIA COLI", "Antibiotic": "CEFTRIAXONE", "Rule": "Tutorial implied susceptibility example"}]
        ),
    }


def _normalise_site_frame(df: pd.DataFrame, spec: SiteSpec) -> pd.DataFrame:
    frame = df.copy()
    if spec.time_column != "order_time_jittered":
        frame = frame.rename(columns={spec.time_column: "order_time_jittered"})
    frame["source_site"] = spec.site
    return frame


def _build_expected_outputs(raw_by_site: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    cohort_frames = []
    for spec in SITE_SPECS:
        cohort_frames.append(_normalise_site_frame(raw_by_site[spec.site][spec.cohort_file], spec))
    observed = pd.concat(cohort_frames, ignore_index=True)
    observed["binary_result"] = observed["susceptibility"].map({"Resistant": "R", "Susceptible": "S"})

    episode_keys = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered", "organism", "source_site"]
    episodes = observed[episode_keys].drop_duplicates()
    antibiotic_universe = sorted(observed["antibiotic"].dropna().unique().tolist())
    eligible = episodes.assign(_key=1).merge(pd.DataFrame({"antibiotic": antibiotic_universe, "_key": 1}), on="_key").drop(columns="_key")
    tested = observed[episode_keys + ["antibiotic"]].drop_duplicates().assign(is_observed_tested=1)
    eligible = eligible.merge(tested, on=episode_keys + ["antibiotic"], how="left")
    eligible["is_observed_tested"] = eligible["is_observed_tested"].fillna(0).astype(int)
    eligible["is_biologically_eligible"] = 1
    eligible["is_operationally_available"] = 1
    eligible["is_eligible"] = 1

    upstream = observed.loc[observed["binary_result"].isin(["R", "S"]), episode_keys + ["antibiotic", "binary_result"]].rename(
        columns={"antibiotic": "upstream_antibiotic", "binary_result": "upstream_result"}
    )
    downstream = eligible[episode_keys + ["antibiotic", "is_observed_tested"]].rename(
        columns={"antibiotic": "downstream_antibiotic", "is_observed_tested": "downstream_tested"}
    )
    pairs = upstream.merge(downstream, on=episode_keys, how="inner")
    pairs = pairs[pairs["upstream_antibiotic"] != pairs["downstream_antibiotic"]].copy()

    edge_rows = []
    for (upstream_drug, downstream_drug), group in pairs.groupby(["upstream_antibiotic", "downstream_antibiotic"], observed=True):
        r = group[group["upstream_result"] == "R"]
        s = group[group["upstream_result"] == "S"]
        if len(r) < 5 or len(s) < 5:
            continue
        p_r = (int(r["downstream_tested"].sum()) + 0.5) / (len(r) + 1.0)
        p_s = (int(s["downstream_tested"].sum()) + 0.5) / (len(s) + 1.0)
        edge_rows.append(
            {
                "upstream_antibiotic": upstream_drug,
                "downstream_antibiotic": downstream_drug,
                "n_R": len(r),
                "n_S": len(s),
                "p_R": p_r,
                "p_S": p_s,
                "escalation_ratio": p_r / p_s,
                "interpretation": "tutorial cascade candidate" if p_r / p_s > 2 else "weak/no cascade",
            }
        )
    cascade_edges = pd.DataFrame(edge_rows).sort_values("escalation_ratio", ascending=False).reset_index(drop=True)

    prevalence_rows = []
    observed_binary = observed[observed["binary_result"].isin(["R", "S"])].copy()
    for antibiotic, elig_group in eligible.groupby("antibiotic", observed=True):
        tested_group = observed_binary[observed_binary["antibiotic"] == antibiotic]
        eligible_n = len(elig_group)
        tested_n = len(tested_group)
        resistant_n = int((tested_group["binary_result"] == "R").sum())
        if tested_n == 0:
            continue
        prevalence_rows.append(
            {
                "antibiotic": antibiotic,
                "eligible_n": eligible_n,
                "tested_n": tested_n,
                "unknown_binary_outcome_n": eligible_n - tested_n,
                "pi_naive": resistant_n / tested_n,
                "pi_lower": resistant_n / eligible_n,
                "pi_upper": (resistant_n + eligible_n - tested_n) / eligible_n,
                "denominator_inflation": resistant_n / tested_n - resistant_n / eligible_n,
            }
        )
    prevalence = pd.DataFrame(prevalence_rows).sort_values("denominator_inflation", ascending=False).reset_index(drop=True)

    row_flow = []
    for spec in SITE_SPECS:
        site_observed = _normalise_site_frame(raw_by_site[spec.site][spec.cohort_file], spec)
        site_episodes = site_observed[episode_keys].drop_duplicates()
        row_flow.append(
            {
                "site": spec.site,
                "raw_ast_rows": len(site_observed),
                "culture_episodes": len(site_episodes),
                "observed_episode_drug_rows": len(site_observed),
                "eligible_episode_drug_opportunities": len(site_episodes) * len(antibiotic_universe),
            }
        )

    return {
        "combined_observed_ast.csv": observed,
        "expected_gold_eligible_opportunities.csv": eligible,
        "expected_directional_pair_rows.csv": pairs,
        "expected_cascade_edges.csv": cascade_edges,
        "expected_prevalence_shift.csv": prevalence,
        "expected_row_flow.csv": pd.DataFrame(row_flow),
    }


def create_dataset(output_root: Path, outputs_root: Path, episodes_per_site: int, seed: int) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    raw_by_site: dict[str, dict[str, pd.DataFrame]] = {}
    for offset, spec in enumerate(SITE_SPECS):
        site_dir = output_root / spec.site
        site_dir.mkdir(parents=True, exist_ok=True)
        tables = _generate_site(spec, episodes_per_site=episodes_per_site, seed=seed + offset * 997)
        raw_by_site[spec.site] = tables
        for filename, df in tables.items():
            if filename in {"microbiology_cultures_implied_susceptibility.csv", "implied_susceptibility_rules.csv"} and not spec.implied_files:
                continue
            df.to_csv(site_dir / filename, index=False)

    expected = _build_expected_outputs(raw_by_site)
    for filename, df in expected.items():
        df.to_csv(outputs_root / filename, index=False)

    manifest = {
        "generated_by": Path(__file__).name,
        "seed": seed,
        "episodes_per_site": episodes_per_site,
        "raw_root": str(output_root),
        "outputs_root": str(outputs_root),
        "sites": [spec.site for spec in SITE_SPECS],
        "files_by_site": {
            spec.site: sorted(path.name for path in (output_root / spec.site).glob("*.csv"))
            for spec in SITE_SPECS
        },
        "expected_outputs": sorted(expected.keys()),
        "note": "Educational synthetic data only; not real patient data and not suitable for scientific inference.",
    }
    (outputs_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "tutorial_synthetic_raw")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "data" / "tutorial_synthetic_outputs")
    parser.add_argument("--episodes-per-site", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_dataset(
        output_root=args.output_root,
        outputs_root=args.outputs_root,
        episodes_per_site=args.episodes_per_site,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
