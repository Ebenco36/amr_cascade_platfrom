"""Write a synthetic replica of the raw ARMD-family extracts from a profile.

Every file keeps its raw name, column order and spellings (timestamp format,
missing-value token, integer-vs-float writing, float32 artefacts), and every
row is keyed to a synthetic culture order, so the files join exactly as the
real ones do. Nothing is copied from the real data: patients, encounters,
orders and identifiers are generated, identifiers keep the raw character
layout but start with "Z" (letters) or "9" (digit-only), and all values are
drawn from the profile's aggregates and fitted models.

Scale: ``scale`` is the fraction of each site's real culture orders to
generate. Every per-order distribution (organisms per positive culture, AST
panel, covariate rows per order) is kept, except that comorbidity rows per
order can be capped (``comorbidity_rows_cap``) because the Stanford extract
averages several hundred rows per order.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from amr_cascade_platform.data.replica import ast_chain_model as chain
from amr_cascade_platform.data.replica.raw_formats import (
    dominant_missing_token,
    generate_identifiers,
    histogram_draw,
    render_numbers,
    render_times,
    sample_from_quantiles,
    weighted_choice,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS = ("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded")
TIME_COLUMNS = ("order_time_jittered", "order_time_jittered_utc")
_SEP = "\x1f"


@dataclass(frozen=True)
class GenerationOptions:
    scale: float = 0.03
    seed: int = 20260924
    comorbidity_rows_cap: int = 60
    sites: tuple[str, ...] | None = None


@dataclass
class _Orders:
    """One row per synthetic culture order, plus its patient's attributes."""

    frame: pd.DataFrame
    patients: pd.DataFrame


class ReplicaGenerator:
    """Generate every raw file of every profiled site (see module docstring)."""

    def __init__(self, profile: dict, options: GenerationOptions) -> None:
        if not 0.0 < options.scale <= 1.0:
            raise ValueError("scale must be in (0, 1].")
        self._profile = profile
        self._options = options
        self._used_ids: dict[str, set[str]] = {column: set() for column in KEY_COLUMNS}

    # ------------------------------------------------------------------ public
    def write(self, *, output_root: Path, reference_source: Path, force: bool) -> dict:
        raw_dir = output_root / "raw"
        self._refuse_real_data(output_root, reference_source)
        if raw_dir.exists():
            if not force:
                raise FileExistsError(f"{raw_dir} already exists; pass force=True to replace it.")
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True)
        reference_dir = output_root / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(reference_source.iterdir()):
            if source.is_file():
                shutil.copy2(source, reference_dir / source.name)

        sites = self._options.sites or tuple(self._profile["sites"])
        manifest: dict = {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "generator": "amr_cascade_platform.data.replica.replica_generator",
            "options": asdict(self._options),
            "profile_generated_at_utc": self._profile.get("generated_at_utc"),
            "profile_sha256": hashlib.sha256(json.dumps(self._profile, sort_keys=True).encode()).hexdigest(),
            "note": (
                "Synthetic data generated from an aggregate profile of data/raw. No record, identifier "
                "or value is copied from the real extracts; not for scientific inference."
            ),
            "row_counts": {},
            "files": {},
            "fidelity": {},
        }
        for offset, site in enumerate(sites):
            rng = np.random.default_rng([self._options.seed, offset])
            site_dir = raw_dir / site
            site_dir.mkdir(parents=True)
            counts, files, fidelity = self._write_site(site, site_dir, rng)
            manifest["row_counts"][site] = counts
            manifest["files"][site] = files
            manifest["fidelity"][site] = fidelity
        (output_root / "REPLICA_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _refuse_real_data(output_root: Path, reference_source: Path) -> None:
        target = (output_root / "raw").resolve()
        real = (reference_source.resolve().parent / "raw").resolve()
        if target == real or real in target.parents:
            raise ValueError(f"Refusing to write the replica over the real raw data at {real}.")

    # ------------------------------------------------------------------- sites
    def _write_site(self, site: str, site_dir: Path, rng: np.random.Generator) -> tuple[dict, dict, dict]:
        profile = self._profile["sites"][site]
        orders = self._build_orders(profile, rng)
        cohort, fidelity = self._build_cohort(site, profile, orders, rng)
        tables: dict[str, pd.DataFrame] = {"cohort": cohort}
        builders = {
            "demographics": self._build_demographics,
            "ward_info": self._build_ward_info,
            "adi_scores": self._build_adi_scores,
            "labs": self._build_measures,
            "vitals": self._build_measures,
            "comorbidity": self._build_comorbidity,
            "prior_procedures": self._build_prior_procedures,
            "nursing_home_visits": self._build_nursing_home,
            "prior_infecting_organism": self._build_prior_organisms,
            "microbial_resistance": self._build_microbial_resistance,
        }
        for canonical, builder in builders.items():
            if canonical in profile["tables"]:
                tables[canonical] = builder(canonical, profile["tables"][canonical], orders, rng)
        if "prior_med" in profile["tables"]:
            tables.update(self._build_medications(profile["tables"], orders, rng))
        if "implied_susceptibility" in profile["tables"]:
            tables["implied_susceptibility"] = self._build_implied(profile["tables"]["implied_susceptibility"], cohort, rng)
        if "implied_susceptibility_rules" in profile["tables"]:
            rules = profile["tables"]["implied_susceptibility_rules"]
            tables["implied_susceptibility_rules"] = pd.DataFrame(rules["values"]["rows"], columns=rules["header"])

        counts: dict[str, int] = {}
        files: dict[str, dict] = {}
        for canonical, frame in tables.items():
            spec = profile["tables"][canonical]
            frame = frame.loc[:, spec["header"]]
            path = site_dir / spec["filename"]
            frame.to_csv(path, index=False)
            counts[canonical] = int(len(frame))
            files[canonical] = {"filename": spec["filename"], "rows": int(len(frame)), "sha256": _sha256(path)}
        logger.info("Wrote %s: %d orders, %d cohort rows", site, len(orders.frame), len(cohort))
        return counts, files, fidelity

    # ------------------------------------------------------------------ orders
    def _build_orders(self, profile: dict, rng: np.random.Generator) -> _Orders:
        cohort = profile["cohort"]
        target = max(1, int(round(self._options.scale * cohort["orders"])))
        encounters_per_patient = cohort["encounters_per_patient"]
        orders_per_encounter = cohort["orders_per_encounter"]
        mean_orders = _hist_mean(encounters_per_patient) * _hist_mean(orders_per_encounter)
        patients_n = int(np.ceil(1.2 * target / mean_orders)) + 10
        encounter_counts = histogram_draw(encounters_per_patient, patients_n, rng)
        patient_of_encounter = np.repeat(np.arange(patients_n), encounter_counts)
        order_counts = histogram_draw(orders_per_encounter, len(patient_of_encounter), rng)
        encounter_of_order = np.repeat(np.arange(len(patient_of_encounter)), order_counts)
        encounter_of_order = encounter_of_order[:target]
        target = len(encounter_of_order)
        encounters_used = int(encounter_of_order.max()) + 1
        patient_of_encounter = patient_of_encounter[:encounters_used]
        patients_used = int(patient_of_encounter.max()) + 1

        # Encounter start times follow the real year and time-of-day
        # distributions; orders within an encounter share its time or follow
        # after the real within-encounter gaps. Times are whole minutes.
        context = pd.DataFrame(cohort["order_context"])
        year_weights = context.groupby("year")["count"].sum()
        years = weighted_choice(list(year_weights.index), year_weights.to_numpy(), encounters_used, rng).astype(int)
        minute_weights = np.asarray(cohort["minute_of_day"], dtype=float)
        minutes = rng.choice(1440, size=encounters_used, p=minute_weights / minute_weights.sum())
        day_of_year = rng.integers(0, 365, size=encounters_used)
        encounter_start = (
            pd.to_datetime(pd.Series(years).astype(str) + "-01-01")
            + pd.to_timedelta(day_of_year, unit="D")
            + pd.to_timedelta(minutes, unit="m")
        ).to_numpy()

        position = _position_within_group(encounter_of_order)
        same_time = rng.random(target) < cohort["same_time_within_encounter_share"]
        gaps = np.where(
            (position > 0) & ~same_time,
            np.rint(sample_from_quantiles(cohort["within_encounter_gap_minutes"], target, rng)),
            0.0,
        )
        offsets = pd.Series(gaps).groupby(encounter_of_order).cumsum().to_numpy()
        stamps = encounter_start[encounter_of_order] + pd.to_timedelta(offsets, unit="m").to_numpy()

        patient_of_order = patient_of_encounter[encounter_of_order]
        shapes = cohort["id_shapes"]
        anon = generate_identifiers(patients_used, shapes["anon_id"], rng, used=self._used_ids["anon_id"])
        csn = generate_identifiers(encounters_used, shapes["pat_enc_csn_id_coded"], rng, used=self._used_ids["pat_enc_csn_id_coded"])
        order_ids = generate_identifiers(target, shapes["order_proc_id_coded"], rng, used=self._used_ids["order_proc_id_coded"])

        frame = pd.DataFrame(
            {
                "anon_id": np.asarray(anon, dtype=object)[patient_of_order],
                "pat_enc_csn_id_coded": np.asarray(csn, dtype=object)[encounter_of_order],
                "order_proc_id_coded": order_ids,
                "_stamp": pd.to_datetime(stamps),
                "_patient": patient_of_order,
            }
        )
        frame["year"] = frame["_stamp"].dt.year

        # Specimen type, ordering mode and culture positivity, jointly, given the year.
        grouped = {year: group for year, group in context.groupby("year")}
        overall = context.groupby(["culture", "mode", "positive"])["count"].sum().reset_index()
        culture = np.empty(target, dtype=object)
        mode = np.empty(target, dtype=object)
        positive = np.empty(target, dtype=object)
        for year, index in frame.groupby("year").indices.items():
            table = grouped.get(year, overall)
            pick = rng.choice(len(table), size=len(index), p=table["count"].to_numpy() / table["count"].sum())
            culture[index] = table["culture"].to_numpy()[pick]
            mode[index] = table["mode"].to_numpy()[pick]
            positive[index] = table["positive"].astype(str).to_numpy()[pick]
        frame["culture"] = culture
        frame["mode"] = mode
        frame["positive"] = positive
        patients = pd.DataFrame({"_patient": np.arange(patients_used)})
        return _Orders(frame=frame, patients=patients)

    # ------------------------------------------------------------------ cohort
    def _build_cohort(
        self,
        site: str,
        profile: dict,
        orders: _Orders,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, dict]:
        cohort = profile["cohort"]
        frame = orders.frame
        positive_index = np.flatnonzero(frame["positive"].to_numpy() == "1")
        with_organism = positive_index[rng.random(len(positive_index)) >= cohort["positive_without_organism_share"]]

        organism_context = pd.DataFrame(cohort["organism_context"])
        by_context = {key: group for key, group in organism_context.groupby(["culture", "year"])}
        by_culture = {key: group.groupby("organism")["count"].sum() for key, group in organism_context.groupby("culture")}
        overall = organism_context.groupby("organism")["count"].sum()
        per_order = histogram_draw(cohort["organisms_per_positive_order"], len(with_organism), rng)
        episode_order: list[int] = []
        episode_organism: list[str] = []
        for order_index, wanted in zip(with_organism, per_order, strict=True):
            key = (frame.at[order_index, "culture"], int(frame.at[order_index, "year"]))
            if key in by_context:
                counts = by_context[key].groupby("organism")["count"].sum()
            else:
                counts = by_culture.get(key[0], overall)
            chosen = rng.choice(len(counts), size=min(int(wanted), len(counts)), replace=False, p=counts.to_numpy() / counts.sum())
            for position in chosen:
                episode_order.append(int(order_index))
                episode_organism.append(str(counts.index[position]))
        episodes = pd.DataFrame({"_order": episode_order, "organism": episode_organism})
        episodes = episodes.loc[episodes["organism"].isin(profile["ast"])].reset_index(drop=True)

        duplicate = cohort["ast_rows_per_episode_drug"]
        extra_share = duplicate["extra_row_share"]
        discordant_given_extra = min(1.0, duplicate["discordant_group_share"] / extra_share) if extra_share > 0 else 0.0

        parts: list[pd.DataFrame] = []
        fidelity: dict = {}
        for organism, group in episodes.groupby("organism", sort=True):
            model = profile["ast"][organism]
            context = pd.DataFrame(
                {
                    "year": frame["year"].to_numpy()[group["_order"].to_numpy()],
                    "culture": frame["culture"].to_numpy()[group["_order"].to_numpy()],
                    "mode": frame["mode"].to_numpy()[group["_order"].to_numpy()],
                }
            )
            states = chain.sample_chain(model, context, rng)
            labels = np.asarray([drug["label"] for drug in model["drugs"]], dtype=object)
            if organism in profile.get("diagnostics", {}):
                fidelity[organism] = _fidelity(profile["diagnostics"][organism], states, list(labels))
            episode_index, drug_index = np.nonzero(states)
            result = states[episode_index, drug_index]
            spelled = np.empty(len(result), dtype=object)
            for state, spelling in chain.RAW_RESULT_SPELLING.items():
                spelled[result == state] = spelling
            other = np.flatnonzero(result == chain.STATE_OTHER)
            for position in other:
                spellings = model["drugs"][drug_index[position]]["other_spellings"]
                spelled[position] = weighted_choice(list(spellings), list(spellings.values()), 1, rng)[0]
            rows = pd.DataFrame(
                {
                    "_order": group["_order"].to_numpy()[episode_index],
                    "organism": organism,
                    "antibiotic": labels[drug_index] if len(drug_index) else np.empty(0, dtype=object),
                    "susceptibility": spelled,
                }
            )
            reported_any = np.zeros(len(group), dtype=bool)
            reported_any[episode_index] = True
            no_ast = pd.DataFrame(
                {"_order": group["_order"].to_numpy()[~reported_any], "organism": organism, "antibiotic": "Null", "susceptibility": "Null"}
            )
            parts.extend([rows, no_ast])
        ast_rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["_order", "organism", "antibiotic", "susceptibility"])

        # Repeated rows for the same episode and drug, some with a conflicting
        # result, at the real rates (the pipeline collapses or excludes them).
        tested = ast_rows.loc[ast_rows["antibiotic"].ne("Null")]
        repeat = tested.loc[rng.random(len(tested)) < extra_share].copy()
        conflict = rng.random(len(repeat)) < discordant_given_extra
        interpretive = repeat["susceptibility"].isin(["Susceptible", "Resistant"])
        flip = conflict & interpretive.to_numpy()
        repeat.loc[flip, "susceptibility"] = np.where(
            repeat.loc[flip, "susceptibility"].eq("Susceptible"), "Resistant", "Susceptible"
        )
        ast_rows = pd.concat([ast_rows, repeat], ignore_index=True)

        without_organism = np.setdiff1d(positive_index, episodes["_order"].to_numpy())
        blank_orders = np.concatenate([np.flatnonzero(frame["positive"].to_numpy() != "1"), without_organism])
        blank = pd.DataFrame({"_order": blank_orders, "organism": "Null", "antibiotic": "Null", "susceptibility": "Null"})
        rows = pd.concat([ast_rows, blank], ignore_index=True)
        rows = self._attach_keys(rows, orders, profile["tables"]["cohort"])
        rows["ordering_mode"] = frame["mode"].to_numpy()[rows["_order"].to_numpy()]
        rows["culture_description"] = frame["culture"].to_numpy()[rows["_order"].to_numpy()]
        rows["was_positive"] = frame["positive"].to_numpy()[rows["_order"].to_numpy()]
        rows = self._order_rows(rows, cohort["grouped_by_order"], rng)
        return rows, fidelity

    # -------------------------------------------------------- per-order tables
    def _covered_rows(self, spec: dict, orders: _Orders, rng: np.random.Generator, *, cap: int = 0) -> np.ndarray:
        """Order index of every row: covered orders repeated by their row counts."""
        n = len(orders.frame)
        covered = np.flatnonzero(rng.random(n) < float(spec.get("coverage", 1.0)))
        counts = histogram_draw(spec["rows_per_order"], len(covered), rng) if spec.get("rows_per_order") else np.ones(len(covered), dtype=int)
        if cap > 0:
            counts = np.minimum(counts, cap)
        return np.repeat(covered, counts)

    def _attach_keys(self, rows: pd.DataFrame, orders: _Orders, spec: dict) -> pd.DataFrame:
        index = rows["_order"].to_numpy()
        frame = orders.frame
        for column in KEY_COLUMNS:
            if column in spec["header"]:
                rows[column] = frame[column].to_numpy()[index]
        time_column = next((c for c in TIME_COLUMNS if c in spec["header"]), None)
        if time_column is not None:
            format_key = spec.get("time_format") or "space_naive"
            cache_column = f"_time_{format_key}"
            if cache_column not in frame.columns:
                frame[cache_column] = render_times(frame["_stamp"], format_key).to_numpy()
            rows[time_column] = frame[cache_column].to_numpy()[index]
        return rows

    def _finish(self, rows: pd.DataFrame, orders: _Orders, spec: dict, rng: np.random.Generator) -> pd.DataFrame:
        rows = self._attach_keys(rows, orders, spec)
        duplicate_share = float(spec.get("duplicate_row_share", 0.0))
        if duplicate_share > 0 and len(rows):
            # The real per-order row counts already include exact duplicate
            # rows, so duplicates replace rows rather than being added: a
            # selected row becomes a copy of the last original row of its order.
            rows = rows.sort_values("_order", kind="stable").reset_index(drop=True)
            later = _position_within_group(rows["_order"].to_numpy()) > 0
            copy = later & (rng.random(len(rows)) < duplicate_share)
            if copy.any():
                source = pd.Series(np.where(copy, np.nan, np.arange(len(rows)))).ffill().to_numpy(dtype=int)
                columns = [column for column in rows.columns if column != "_order"]
                rows.loc[copy, columns] = rows.loc[source[copy], columns].to_numpy()
        return self._order_rows(rows, spec.get("grouped_by_order", True), rng)

    @staticmethod
    def _order_rows(rows: pd.DataFrame, grouped: bool, rng: np.random.Generator) -> pd.DataFrame:
        if grouped:
            rows = rows.sort_values("_order", kind="stable")
        else:
            rows = rows.iloc[rng.permutation(len(rows))]
        return rows.reset_index(drop=True)

    @staticmethod
    def _missing_token(spec: dict, column: str) -> str:
        return dominant_missing_token(spec.get("missing", {}).get(column, {}))

    def _render(self, spec: dict, column: str, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        numeric = spec.get("numeric", {}).get(column, {"kind": "int"})
        if numeric.get("kind") == "text":
            numeric = {"kind": "int"}
        return render_numbers(np.asarray(values, dtype=float), numeric, rng, self._missing_token(spec, column))

    def _patient_draw(self, orders: _Orders, counts: dict[str, int], rng: np.random.Generator) -> np.ndarray:
        """One value per patient, repeated for each of the patient's orders."""
        per_patient = weighted_choice(list(counts), list(counts.values()), len(orders.patients), rng)
        return per_patient[orders.frame["_patient"].to_numpy()]

    def _build_demographics(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        age = self._patient_draw(orders, values["age"], rng)
        gender = self._patient_draw(orders, values["gender"], rng)
        rows["age"] = age[rows["_order"].to_numpy()]
        rows["gender"] = gender[rows["_order"].to_numpy()]
        return self._finish(rows, orders, spec, rng)

    def _build_ward_info(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        flags = values["flags"]
        combos = pd.Series(values["mode_flag_combos"])
        parsed = combos.index.to_series().str.split("|", expand=True).reset_index(drop=True)
        parsed.columns = ["mode", *flags]
        parsed["count"] = combos.to_numpy()
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        mode = orders.frame["mode"].to_numpy()[rows["_order"].to_numpy()]
        chosen = np.empty(len(rows), dtype=int)
        for value in np.unique(mode):
            index = np.flatnonzero(mode == value)
            table = parsed.loc[parsed["mode"].eq(value)]
            if table.empty:
                table = parsed
            weights = table["count"].to_numpy(dtype=float)
            chosen[index] = table.index.to_numpy()[rng.choice(len(table), size=len(index), p=weights / weights.sum())]
        for flag in flags:
            rows[flag] = parsed[flag].to_numpy()[chosen]
        return self._finish(rows, orders, spec, rng)

    def _build_adi_scores(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        pairs = self._patient_draw(orders, values["score_rank"], rng)[rows["_order"].to_numpy()]
        rows["adi_score"] = [pair.split("|")[0] for pair in pairs]
        rows["adi_state_rank"] = [pair.split("|")[1] for pair in pairs]
        return self._finish(rows, orders, spec, rng)

    def _build_measures(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        measures = values["measures"]
        identical = rng.random(len(orders.frame)) < float(values.get("identical_rows_share", 1.0))
        patterns = values.get("period_patterns")
        if patterns and "Period_Day" in spec["header"]:
            covered = np.flatnonzero(rng.random(len(orders.frame)) < float(spec.get("coverage", 1.0)))
            chosen = weighted_choice(list(patterns), list(patterns.values()), len(covered), rng)
            period_lists = [pattern.split("|") for pattern in chosen]
            order_index = np.repeat(covered, [len(p) for p in period_lists])
            periods = np.concatenate([np.asarray(p, dtype=object) for p in period_lists]) if period_lists else np.empty(0, dtype=object)
        else:
            order_index = self._covered_rows(spec, orders, rng)
            periods = None
        rows = pd.DataFrame({"_order": order_index})
        if periods is not None:
            rows["Period_Day"] = periods
        # Rows of one order either repeat the same values (one summary written
        # once per window) or are independent sparse shards, at the real rates.
        first_row = _position_within_group(order_index) == 0
        source_row = np.arange(len(rows))
        repeat_values = identical[order_index] & ~first_row
        source_row[repeat_values] = pd.Series(np.arange(len(rows))).where(first_row).ffill().to_numpy(dtype=int)[repeat_values]
        n = len(rows)
        for name, stats in measures.items():
            median = np.where(rng.random(n) < stats["median_present"], sample_from_quantiles(stats["median"], n, rng), np.nan)
            quartiles = ~np.isnan(median) & (rng.random(n) < stats["quartiles_present_given_median"])
            lower = median - np.abs(sample_from_quantiles(stats["lower_spread"], n, rng))
            upper = median + np.abs(sample_from_quantiles(stats["upper_spread"], n, rng))
            first = np.where(
                np.isnan(median),
                sample_from_quantiles(stats["first_alone"], n, rng),
                median + sample_from_quantiles(stats["first_minus_median"], n, rng),
            )
            last = np.where(
                np.isnan(median),
                sample_from_quantiles(stats["last_alone"], n, rng),
                median + sample_from_quantiles(stats["last_minus_median"], n, rng),
            )
            first = np.where(rng.random(n) < stats["first_present"], first, np.nan)
            last = np.where(rng.random(n) < stats["last_present"], last, np.nan)
            columns = {
                f"median_{name}": median,
                f"Q25_{name}": np.where(quartiles, lower, np.nan),
                f"Q75_{name}": np.where(quartiles, upper, np.nan),
                f"first_{name}": first,
                f"last_{name}": last,
            }
            for column, data in columns.items():
                if column in spec["header"]:
                    rows[column] = self._render(spec, column, data, rng)[source_row]
        for column in spec["header"]:
            if column not in rows.columns and column not in KEY_COLUMNS and column not in TIME_COLUMNS:
                rows[column] = self._missing_token(spec, column)
        return self._finish(rows, orders, spec, rng)

    def _build_comorbidity(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        components = pd.Series(values["components"]).sort_values(ascending=False)
        order_index = self._covered_rows(spec, orders, rng, cap=self._options.comorbidity_rows_cap)
        counts = np.bincount(order_index, minlength=len(orders.frame))
        # A patient's orders share one comorbidity list: each order carries the
        # first k components of the patient's list.
        patient = orders.frame["_patient"].to_numpy()
        needed = pd.Series(counts).groupby(patient).max()
        probabilities = components.to_numpy(dtype=float) / components.sum()
        names = components.index.to_numpy(dtype=object)
        patient_lists = {
            int(p): names[rng.choice(len(names), size=min(int(k), len(names)), replace=False, p=probabilities)]
            for p, k in needed.items()
            if k > 0
        }
        component = np.empty(len(order_index), dtype=object)
        position = _position_within_group(order_index)
        for row, (order, k) in enumerate(zip(order_index, position, strict=True)):
            listing = patient_lists[int(patient[order])]
            component[row] = listing[k % len(listing)]
        rows = pd.DataFrame({"_order": order_index, "comorbidity_component": component})
        rows = rows.drop_duplicates(["_order", "comorbidity_component"])
        n = len(rows)
        rows["comorbidity_component_start_days_culture"] = self._render(
            spec, "comorbidity_component_start_days_culture", self._days(values["start_days"], n, rng), rng
        )
        rows["comorbidity_component_end_days_culture"] = self._render(
            spec, "comorbidity_component_end_days_culture", self._days(values["end_days"], n, rng), rng
        )
        return self._finish(rows, orders, spec, rng)

    @staticmethod
    def _days(summary: dict, n: int, rng: np.random.Generator) -> np.ndarray:
        values = np.rint(sample_from_quantiles(summary["quantiles"], n, rng)) if summary["quantiles"] else np.full(n, np.nan)
        values = np.where(rng.random(n) < summary.get("zero_share", 0.0), 0.0, values)
        return np.where(rng.random(n) < summary.get("missing_share", 0.0), np.nan, values)

    def _build_medications(self, tables: dict, orders: _Orders, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
        prior = tables["prior_med"]
        medications = pd.Series(prior["values"]["medications"])
        order_index = self._covered_rows(prior, orders, rng)
        drawn = weighted_choice(list(medications.index), medications.to_numpy(), len(order_index), rng)
        name, category = zip(*(value.split(_SEP) for value in drawn), strict=True) if len(drawn) else ((), ())
        exposures = pd.DataFrame(
            {"_order": order_index, "medication_name": list(name), "medication_category": list(category)}
        )
        exposures["_days"] = self._days(prior["values"]["days"], len(exposures), rng)
        result: dict[str, pd.DataFrame] = {}
        prior_rows = exposures.copy()
        prior_rows["medication_time_to_culturetime"] = self._render(prior, "medication_time_to_culturetime", prior_rows["_days"].to_numpy(), rng)
        result["prior_med"] = self._finish(prior_rows, orders, prior, rng)
        # The class and subtype extracts list each exposure once: they are
        # built from the distinct prior-medication rows (the real Stanford
        # prior_med extract repeats ~40% of its rows; the other two do not).
        exposures = result["prior_med"].drop_duplicates(["_order", "medication_name", "medication_category", "_days"])
        exposures = exposures.loc[:, ["_order", "medication_name", "medication_category", "_days"]]

        mapping_specs = {
            "antibiotic_class_exposure": ("antibiotic_class",),
            "antibiotic_subtype_exposure": ("antibiotic_subtype", "antibiotic_subtype_category"),
        }
        for canonical, targets in mapping_specs.items():
            if canonical not in tables:
                continue
            spec = tables[canonical]
            mapping = pd.Series(spec["values"]["mapping"])
            parts = mapping.index.to_series().str.split(_SEP, expand=True)
            parts.columns = ["medication_name", "medication_category", *targets]
            parts["count"] = mapping.to_numpy()
            best = parts.sort_values("count", ascending=False).drop_duplicates(["medication_name", "medication_category"])
            rows = exposures.merge(best.drop(columns="count"), on=["medication_name", "medication_category"], how="inner")
            days_column = next(c for c in spec["header"] if c.lower() in {"time_to_culturetime", "medication_time_to_culturetime"})
            rows[days_column] = self._render(spec, days_column, rows["_days"].to_numpy(), rng)
            result[canonical] = self._finish(rows.drop(columns="_days"), orders, spec, rng)
        return result

    def _build_prior_procedures(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        procedures = values["procedures"]
        rows["procedure_description"] = weighted_choice(list(procedures), list(procedures.values()), len(rows), rng)
        rows["procedure_time_to_culturetime"] = self._render(
            spec, "procedure_time_to_culturetime", self._days(values["days"], len(rows), rng), rng
        )
        return self._finish(rows, orders, spec, rng)

    def _build_nursing_home(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        rows["nursing_home_visit_culture"] = self._render(
            spec, "nursing_home_visit_culture", self._days(spec["values"]["value"], len(rows), rng), rng
        )
        return self._finish(rows, orders, spec, rng)

    def _build_prior_organisms(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        organisms = values["organisms"]
        rows["prior_organism"] = weighted_choice(list(organisms), list(organisms.values()), len(rows), rng)
        days_column = next(c for c in spec["header"] if c.startswith("prior_infecting_organism_days_to_cul"))
        rows[days_column] = self._render(spec, days_column, self._days(values["days"], len(rows), rng), rng)
        return self._finish(rows, orders, spec, rng)

    def _build_microbial_resistance(self, canonical: str, spec: dict, orders: _Orders, rng: np.random.Generator) -> pd.DataFrame:
        values = spec["values"]
        rows = pd.DataFrame({"_order": self._covered_rows(spec, orders, rng)})
        pairs = values["organism_antibiotic"]
        drawn = weighted_choice(list(pairs), list(pairs.values()), len(rows), rng)
        empty = rng.random(len(rows)) < values["empty_row_share"]
        organism = np.array([value.split(_SEP)[0] for value in drawn], dtype=object)
        antibiotic = np.array([value.split(_SEP)[1] for value in drawn], dtype=object)
        days = self._days(values["days"], len(rows), rng)
        organism[empty] = self._missing_token(spec, "organism")
        antibiotic[empty] = self._missing_token(spec, "antibiotic")
        days[empty] = np.nan
        rows["organism"] = organism
        rows["antibiotic"] = antibiotic
        rows["resistant_time_to_culturetime"] = self._render(spec, "resistant_time_to_culturetime", days, rng)
        return self._finish(rows, orders, spec, rng)

    def _build_implied(self, spec: dict, cohort: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Implied-susceptibility rows for a share of positive episodes.

        In the real extract a covered episode repeats each of its reported
        results (antibiotic spelled as in the cohort, implied value missing) and
        adds rule-implied drugs (lower-case names, reported value missing). The
        repeated rows are copied from the synthetic cohort; the implied rows are
        drawn from the profiled inclusion shares and value distribution.
        """
        values = spec["values"]
        missing = self._missing_token(spec, "implied_susceptibility")
        positives = cohort.loc[cohort["organism"].ne("Null")]
        episodes = positives.drop_duplicates(["_order", "organism"])[["_order", *KEY_COLUMNS, "organism"]]
        # Organisms the real extract never covers (or covers too rarely to
        # profile) get no implied rows, rather than an average.
        coverage = values["coverage_by_organism"]
        keep = rng.random(len(episodes)) < episodes["organism"].map(coverage).fillna(0.0).to_numpy()
        episodes = episodes.loc[keep]

        reported = positives.loc[positives["antibiotic"].ne("Null")].merge(
            episodes[["_order", "organism"]], on=["_order", "organism"], how="inner"
        )
        reported = reported.drop_duplicates(["_order", "organism", "antibiotic"])
        repeated = reported[["_order", *KEY_COLUMNS, "organism", "antibiotic", "susceptibility"]].copy()
        repeated["implied_susceptibility"] = missing

        inclusion = pd.DataFrame(values["inclusion"], columns=["organism", "antibiotic", "share"])
        inclusion = inclusion.loc[inclusion["antibiotic"].str.islower()]
        joint = pd.Series(values["values"])
        joint_parts = joint.index.to_series().str.split(_SEP, expand=True)
        joint_parts.columns = ["antibiotic", "susceptibility", "implied_susceptibility"]
        joint_parts["count"] = joint.to_numpy()
        frames = [repeated]
        for organism, group in episodes.groupby("organism"):
            shares = inclusion.loc[inclusion["organism"].eq(organism)].set_index("antibiotic")["share"]
            for antibiotic, share in shares.items():
                chosen = group.loc[rng.random(len(group)) < share]
                options = joint_parts.loc[joint_parts["antibiotic"].eq(antibiotic)]
                if chosen.empty or options.empty:
                    continue
                weights = options["count"].to_numpy(dtype=float)
                pick = rng.choice(len(options), size=len(chosen), p=weights / weights.sum())
                frame = chosen.copy()
                frame["antibiotic"] = antibiotic
                frame["susceptibility"] = options["susceptibility"].to_numpy()[pick]
                frame["implied_susceptibility"] = options["implied_susceptibility"].to_numpy()[pick]
                frames.append(frame)
        rows = pd.concat(frames, ignore_index=True)
        return self._order_rows(rows, spec.get("grouped_by_order", True), rng)

def _position_within_group(group: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... within each run of equal consecutive group labels."""
    if len(group) == 0:
        return np.empty(0, dtype=int)
    starts = np.r_[True, group[1:] != group[:-1]]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(len(group)), 0))
    return np.arange(len(group)) - run_start


def _hist_mean(histogram: dict[str, int]) -> float:
    values = np.array([int(k) for k in histogram], dtype=float)
    weights = np.array(list(histogram.values()), dtype=float)
    return float((values * weights).sum() / weights.sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fidelity(real: dict, states: np.ndarray, labels: list[str]) -> dict:
    """Compare the replica's AST structure with the real data's (same statistics)."""
    from scipy.stats import spearmanr

    min_cell = 5
    replica_rates = {}
    reported = states > chain.STATE_NOT_REPORTED
    for k, label in enumerate(labels):
        interpretive = np.isin(states[:, k], (chain.STATE_SUSCEPTIBLE, chain.STATE_INTERMEDIATE, chain.STATE_RESISTANT))
        replica_rates[label] = (
            float(reported[:, k].mean()) if len(states) else np.nan,
            float((states[:, k] == chain.STATE_RESISTANT).sum() / max(1, interpretive.sum())),
        )
    matched = [(row, replica_rates[row["drug"]]) for row in real["drug_rates"] if row["drug"] in replica_rates]
    report_real = np.array([row["report_rate"] for row, _ in matched])
    report_replica = np.array([rates[0] for _, rates in matched])
    resistant_real = np.array([row["resistant_share"] for row, _ in matched])
    resistant_replica = np.array([rates[1] for _, rates in matched])

    replica_pairs = {
        (row["upstream"], row["downstream"]): row
        for row in chain.pair_statistics(states, labels, min_cell=min_cell, limit=100_000)
    }
    pairs = [(row, replica_pairs[(row["upstream"], row["downstream"])]) for row in real["pairs"] if (row["upstream"], row["downstream"]) in replica_pairs]
    real_log_er = np.log([row["escalation_ratio"] for row, _ in pairs]) if pairs else np.empty(0)
    replica_log_er = np.log([row["escalation_ratio"] for _, row in pairs]) if pairs else np.empty(0)
    real_screen = np.array([row["co_report_min"] >= 0.95 for row, _ in pairs])
    replica_screen = np.array([row["co_report_min"] >= 0.95 for _, row in pairs])

    def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 3:
            return None
        return round(float(spearmanr(a, b).statistic), 4)

    return {
        "replica_episodes": int(len(states)),
        "real_episodes": int(real["episodes"]),
        "drugs_compared": len(matched),
        "report_rate_spearman": _spearman(report_real, report_replica),
        "report_rate_mean_abs_diff": round(float(np.mean(np.abs(report_real - report_replica))), 4) if matched else None,
        "resistant_share_spearman": _spearman(resistant_real, resistant_replica),
        "resistant_share_mean_abs_diff": round(float(np.mean(np.abs(resistant_real - resistant_replica))), 4) if matched else None,
        "pairs_compared": len(pairs),
        "log_escalation_ratio_spearman": _spearman(real_log_er, replica_log_er),
        "co_report_screen_agreement": round(float((real_screen == replica_screen).mean()), 4) if pairs else None,
    }
