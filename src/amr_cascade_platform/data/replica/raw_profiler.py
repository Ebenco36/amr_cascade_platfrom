"""Aggregate profile of the raw ARMD-family extracts.

The profile is everything the replica generator needs to write synthetic files
that look like ``data/raw`` to the pipeline: file names and column order, the
spelling of timestamps, missing values and numbers, value vocabularies, row
counts per culture order, and (for the AST cohort) fitted models of which drugs
an episode reports and what they show.

Disclosure control: the profile holds no record-level data. Identifiers are
reduced to their character-class shape, categories seen fewer than
``min_cell`` times are dropped, numeric columns are summarised as quantile
grids, and the AST structure is carried by regularised model coefficients.

Reading cost: each file is streamed in column batches with pyarrow. Files above
``oversized_bytes`` (the Stanford comorbidity extract, ~20 GB) are read only
from the head; that extract's rows are in no particular order, so the head is a
fair sample, and its per-order row counts are estimated (see
``_estimate_oversized``).
"""

from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from amr_cascade_platform.data.replica import ast_chain_model as chain
from amr_cascade_platform.data.replica.raw_formats import (
    MISSING_TOKENS,
    detect_time_format,
    id_shape,
    numeric_spec,
    quantile_grid,
    suppressed_counts,
)

PROFILE_SCHEMA_VERSION = 1
KEY_COLUMNS = ("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded")
TIME_COLUMNS = ("order_time_jittered", "order_time_jittered_utc")
ROWS_PER_ORDER_CAP = 1000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileOptions:
    min_cell: int = 11
    chain_min_episodes: int = 1000
    sample_rows: int = 250_000
    oversized_bytes: int = 5_000_000_000
    oversized_head_rows: int = 2_000_000
    regularisation: float = 0.5
    diagnostic_organisms: tuple[str, ...] = ("ESCHERICHIA COLI",)
    sites: tuple[str, ...] | None = None
    seed: int = 20260924


@dataclass
class _Scan:
    """Accumulated statistics of one streamed file."""

    filename: str
    header: list[str]
    size_bytes: int
    rows: int = 0
    estimated: bool = False
    missing: dict[str, Counter] = field(default_factory=dict)
    order_rows: Counter = field(default_factory=Counter)
    runs: int = 0
    row_hashes: list[np.ndarray] = field(default_factory=list)
    samples: list[pd.DataFrame] = field(default_factory=list)
    extra: list[pd.DataFrame] = field(default_factory=list)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _bytes_per_row(path: Path, probe_bytes: int = 20_000_000) -> float:
    with path.open("rb") as handle:
        block = handle.read(probe_bytes)
    lines = max(1, block.count(b"\n") - 1)
    return len(block) / lines


def _stream(path: Path, columns: list[str], *, max_rows: int | None = None):
    convert = pa_csv.ConvertOptions(
        include_columns=columns,
        column_types={column: pa.string() for column in columns},
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )
    reader = pa_csv.open_csv(
        path,
        read_options=pa_csv.ReadOptions(block_size=64 << 20),
        convert_options=convert,
    )
    seen = 0
    for batch in reader:
        if max_rows is not None and seen >= max_rows:
            break
        if max_rows is not None and seen + batch.num_rows > max_rows:
            batch = batch.slice(0, max_rows - seen)
        seen += batch.num_rows
        yield batch


class RawProfiler:
    """Build the replica profile from ``raw_root`` (see module docstring)."""

    def __init__(self, raw_root: Path, table_map: dict[str, dict[str, str]], options: ProfileOptions) -> None:
        self._raw_root = raw_root
        self._table_map = table_map
        self._options = options
        self._rng = np.random.default_rng(options.seed)

    # ------------------------------------------------------------------ public
    def profile(self) -> dict:
        sites = self._options.sites or tuple(self._table_map)
        profile: dict = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "options": {
                "min_cell": self._options.min_cell,
                "chain_min_episodes": self._options.chain_min_episodes,
                "sample_rows": self._options.sample_rows,
                "oversized_bytes": self._options.oversized_bytes,
                "oversized_head_rows": self._options.oversized_head_rows,
                "regularisation": self._options.regularisation,
            },
            "disclosure_control": (
                "No record-level values: identifiers reduced to character-class shapes, categories "
                f"below {self._options.min_cell} dropped, numeric columns as quantile grids, AST "
                "structure as regularised model coefficients."
            ),
            "sites": {},
        }
        for site in sites:
            started = time.monotonic()
            logger.info("Profiling site %s", site)
            profile["sites"][site] = self._profile_site(site)
            logger.info("Profiled %s in %.0fs", site, time.monotonic() - started)
        self._estimate_oversized(profile)
        return profile

    # ------------------------------------------------------------------- sites
    def _profile_site(self, site: str) -> dict:
        tables = self._table_map[site]
        site_dir = self._raw_root / site
        cohort_path = site_dir / tables["cohort"]
        cohort = pd.read_csv(cohort_path, dtype=str, keep_default_na=False)
        time_column = next(column for column in TIME_COLUMNS if column in cohort.columns)
        cohort_profile, order_context = self._profile_cohort(site, cohort, time_column)
        site_profile: dict = {
            "time_column": time_column,
            "tables": {
                "cohort": {
                    "filename": tables["cohort"],
                    "header": list(cohort.columns),
                    "size_bytes": cohort_path.stat().st_size,
                    "rows": len(cohort),
                    "time_format": detect_time_format(cohort[time_column].head(5000)),
                    "missing": self._missing_counts(cohort),
                },
            },
            "cohort": cohort_profile,
        }
        site_profile["ast"], site_profile["diagnostics"] = self._profile_ast(site, cohort, time_column, order_context)
        cohort_orders = order_context["order_proc_id_coded"]
        positive_episodes = cohort.loc[cohort["was_positive"].eq("1") & cohort["organism"].ne("Null"), ["order_proc_id_coded", "organism"]].drop_duplicates()
        order_mode = dict(zip(order_context["order_proc_id_coded"], order_context["mode"], strict=True))
        del cohort
        for canonical, filename in tables.items():
            if canonical == "cohort":
                continue
            path = site_dir / filename
            if not path.exists():
                continue
            started = time.monotonic()
            summary = self._profile_table(
                canonical,
                path,
                cohort_orders=cohort_orders,
                order_mode=order_mode,
                positive_episodes=positive_episodes,
            )
            site_profile["tables"][canonical] = summary
            logger.info("  %s/%s profiled in %.0fs", site, filename, time.monotonic() - started)
        return site_profile

    # ------------------------------------------------------------------ cohort
    def _profile_cohort(self, site: str, cohort: pd.DataFrame, time_column: str) -> tuple[dict, pd.DataFrame]:
        min_cell = self._options.min_cell
        order_columns = list(KEY_COLUMNS) + [time_column]
        orders = cohort.drop_duplicates(order_columns)[order_columns + ["ordering_mode", "culture_description", "was_positive"]].copy()
        stamps = pd.to_datetime(orders[time_column], errors="coerce", utc=True, format="mixed")
        orders["year"] = stamps.dt.year.astype("Int64")
        orders["minute"] = (stamps.dt.hour * 60 + stamps.dt.minute).astype("Int64")
        orders["_stamp"] = stamps
        orders = orders.dropna(subset=["year"])
        orders["year"] = orders["year"].astype(int)
        orders = orders.rename(columns={"culture_description": "culture", "ordering_mode": "mode", "was_positive": "positive"})

        per_patient_encounters = orders.groupby("anon_id")["pat_enc_csn_id_coded"].nunique()
        per_encounter_orders = orders.groupby(["anon_id", "pat_enc_csn_id_coded"]).size()
        encounter_sorted = orders.sort_values(["anon_id", "pat_enc_csn_id_coded", "_stamp"])
        gaps = encounter_sorted.groupby(["anon_id", "pat_enc_csn_id_coded"])["_stamp"].diff().dropna()
        gap_minutes = gaps.dt.total_seconds() / 60.0
        context_counts = (
            orders.groupby(["year", "culture", "mode", "positive"]).size().rename("count").reset_index()
        )
        context_counts = context_counts.loc[context_counts["count"] >= min_cell]

        positives = cohort.loc[cohort["was_positive"].eq("1")]
        positive_orders = positives.drop_duplicates(order_columns)
        with_organism = positives.loc[positives["organism"].ne("Null")]
        organisms_per_order = with_organism.drop_duplicates(order_columns + ["organism"]).groupby(order_columns).size()
        positive_without_organism = 1.0 - (
            with_organism.drop_duplicates(order_columns).shape[0] / max(1, positive_orders.shape[0])
        )
        episode_table = with_organism.drop_duplicates(order_columns + ["organism"]).merge(
            orders[order_columns + ["year", "culture"]], on=order_columns, how="inner"
        )
        organism_context = (
            episode_table.groupby(["culture", "year", "organism"]).size().rename("count").reset_index()
        )
        organism_context = organism_context.loc[organism_context["count"] >= min_cell]

        # Duplicate AST rows: same episode and drug more than once, possibly with
        # different results (the pipeline collapses concordant duplicates and
        # drops discordant groups).
        tested = with_organism.loc[with_organism["antibiotic"].ne("Null")]
        group_columns = order_columns + ["organism", "antibiotic"]
        group_sizes = tested.groupby(group_columns).size()
        group_results = tested.groupby(group_columns)["susceptibility"].nunique()
        exact_duplicates = int(tested.duplicated().sum())

        row_keys = cohort["anon_id"] + "|" + cohort["order_proc_id_coded"]
        runs = int((row_keys != row_keys.shift()).sum())
        cohort_profile = {
            "orders": int(len(orders)),
            "patients": int(orders["anon_id"].nunique()),
            "encounters": int(orders.groupby(["anon_id", "pat_enc_csn_id_coded"]).ngroups),
            "encounters_per_patient": self._histogram(per_patient_encounters, cap=200),
            "orders_per_encounter": self._histogram(per_encounter_orders, cap=100),
            "same_time_within_encounter_share": round(float((gap_minutes == 0).mean()) if len(gap_minutes) else 1.0, 4),
            "within_encounter_gap_minutes": quantile_grid(gap_minutes[gap_minutes > 0]),
            "order_context": context_counts.to_dict(orient="records"),
            "minute_of_day": [
                int(v) if v >= min_cell else 0
                for v in np.bincount(orders["minute"].dropna().astype(int), minlength=1440)[:1440]
            ],
            "organisms_per_positive_order": self._histogram(organisms_per_order, cap=10),
            "positive_without_organism_share": round(float(positive_without_organism), 5),
            "organism_context": organism_context.to_dict(orient="records"),
            "ast_rows_per_episode_drug": {
                "groups": int(len(group_sizes)),
                "extra_row_share": round(float((group_sizes - 1).sum() / max(1, len(group_sizes))), 6),
                "discordant_group_share": round(float((group_results > 1).sum() / max(1, len(group_sizes))), 6),
                "exact_duplicate_row_share": round(exact_duplicates / max(1, len(tested)), 6),
            },
            "grouped_by_order": bool(runs <= 1.05 * orders.shape[0]),
            "id_shapes": {column: self._id_shapes(orders[column]) for column in KEY_COLUMNS},
        }
        order_context = orders[order_columns + ["year", "culture", "mode", "positive"]]
        return cohort_profile, order_context

    def _profile_ast(
        self,
        site: str,
        cohort: pd.DataFrame,
        time_column: str,
        order_context: pd.DataFrame,
    ) -> tuple[dict, dict]:
        options = self._options
        order_columns = list(KEY_COLUMNS) + [time_column]
        rows = cohort.loc[cohort["was_positive"].eq("1") & cohort["organism"].ne("Null")]
        episode_counts = rows.drop_duplicates(order_columns + ["organism"])["organism"].value_counts()
        models: dict[str, dict] = {}
        diagnostics: dict[str, dict] = {}
        context_lookup = order_context.set_index(order_columns)[["year", "culture", "mode"]]
        for organism, count in episode_counts.items():
            if count < options.min_cell:
                continue
            started = time.monotonic()
            organism_rows = rows.loc[rows["organism"].eq(organism)]
            episodes, drugs, states, others = chain.episode_states(organism_rows, order_columns + ["organism"])
            context = context_lookup.reindex(pd.MultiIndex.from_frame(episodes[order_columns])).reset_index(drop=True)
            keep = context["year"].notna().to_numpy()
            episodes, states, context = episodes.loc[keep].reset_index(drop=True), states[keep], context.loc[keep].reset_index(drop=True)
            context["year"] = context["year"].astype(int)
            if count >= options.chain_min_episodes or organism in options.diagnostic_organisms:
                model = chain.fit_chain(
                    states, drugs, context, others,
                    min_cell=options.min_cell, regularisation=options.regularisation,
                )
                model["kind"] = "chain"
            else:
                model = chain.fit_marginal(states, drugs, context, others, min_cell=options.min_cell)
                model["kind"] = "marginal"
            model["episodes"] = int(len(episodes))
            models[str(organism)] = model
            if organism in options.diagnostic_organisms:
                diagnostics[str(organism)] = self.ast_diagnostics(states, drugs)
            logger.info(
                "  %s %s: %s model, %d episodes, %d drugs (%.0fs)",
                site, organism, model["kind"], len(episodes), len(model["drugs"]), time.monotonic() - started,
            )
        return models, diagnostics

    def ast_diagnostics(self, states: np.ndarray, drugs: list[str]) -> dict:
        """Per-drug report and resistance rates plus naive pair statistics."""
        min_cell = self._options.min_cell
        reported = states > chain.STATE_NOT_REPORTED
        rates = []
        for k, drug in enumerate(drugs):
            n_reported = int(reported[:, k].sum())
            if n_reported < min_cell:
                continue
            interpretive = np.isin(states[:, k], (chain.STATE_SUSCEPTIBLE, chain.STATE_INTERMEDIATE, chain.STATE_RESISTANT))
            rates.append(
                {
                    "drug": drug,
                    "report_rate": round(n_reported / len(states), 5),
                    "resistant_share": round(float((states[:, k] == chain.STATE_RESISTANT).sum() / max(1, interpretive.sum())), 5),
                }
            )
        return {
            "episodes": int(len(states)),
            "drug_rates": rates,
            "pairs": chain.pair_statistics(states, drugs, min_cell=min_cell),
        }

    # ------------------------------------------------------------ other tables
    def _profile_table(
        self,
        canonical: str,
        path: Path,
        *,
        cohort_orders: pd.Series,
        order_mode: dict[str, str],
        positive_episodes: pd.DataFrame,
    ) -> dict:
        header = _read_header(path)
        size = path.stat().st_size
        oversized = size > self._options.oversized_bytes
        scan = _Scan(filename=path.name, header=header, size_bytes=size, estimated=oversized)
        estimated_rows = size / _bytes_per_row(path)
        sample_fraction = min(1.0, self._options.sample_rows / max(1.0, estimated_rows))
        if oversized:
            sample_fraction = min(1.0, self._options.sample_rows / self._options.oversized_head_rows)
        value_columns = [c for c in header if c not in KEY_COLUMNS and c not in TIME_COLUMNS]
        previous_key = None
        for batch in _stream(path, header, max_rows=self._options.oversized_head_rows if oversized else None):
            frame = batch.to_pandas()
            scan.rows += len(frame)
            for column in header:
                counter = scan.missing.setdefault(column, Counter())
                values = frame[column]
                for token in MISSING_TOKENS:
                    counter[token] += int((values == token).sum())
            if "order_proc_id_coded" in frame.columns:
                keys = frame["order_proc_id_coded"]
                scan.order_rows.update(keys.value_counts().to_dict())
                changes = int((keys != keys.shift()).sum())
                if previous_key is not None and keys.iloc[0] == previous_key:
                    changes -= 1
                scan.runs += changes
                previous_key = keys.iloc[-1]
                scan.row_hashes.append(pd.util.hash_pandas_object(frame, index=False).to_numpy())
                if canonical in {"labs", "vitals"}:
                    extra = pd.DataFrame(
                        {
                            "order": keys.to_numpy(),
                            "value_hash": pd.util.hash_pandas_object(frame[[c for c in value_columns if c != "Period_Day"]], index=False).to_numpy(),
                        }
                    )
                    if "Period_Day" in frame.columns:
                        extra["period"] = frame["Period_Day"].to_numpy()
                    scan.extra.append(extra)
            take = self._rng.random(len(frame)) < sample_fraction
            if take.any():
                scan.samples.append(frame.loc[take])
        sample = pd.concat(scan.samples, ignore_index=True) if scan.samples else pd.DataFrame(columns=header)
        summary: dict = {
            "filename": scan.filename,
            "header": header,
            "size_bytes": size,
            "rows": int(round(estimated_rows)) if oversized else scan.rows,
            "estimated": oversized,
            "time_format": next(
                (detect_time_format(sample[c].head(5000)) for c in TIME_COLUMNS if c in sample.columns), None
            ),
            "missing": {column: dict(counter) for column, counter in scan.missing.items()},
            "numeric": {column: numeric_spec(sample[column]) for column in value_columns if column in sample.columns},
        }
        if scan.order_rows:
            cohort_set = set(cohort_orders)
            covered = [count for order, count in scan.order_rows.items() if order in cohort_set]
            summary["coverage"] = round(len(covered) / max(1, len(cohort_set)), 5)
            summary["rows_per_order"] = self._histogram(pd.Series(covered), cap=ROWS_PER_ORDER_CAP)
            summary["grouped_by_order"] = bool(scan.runs <= 1.05 * len(scan.order_rows))
            hashes = np.concatenate(scan.row_hashes) if scan.row_hashes else np.empty(0, dtype=np.uint64)
            summary["duplicate_row_share"] = round(1.0 - len(np.unique(hashes)) / max(1, len(hashes)), 6)
        summarise = getattr(self, f"_summarise_{canonical}", None)
        if summarise is not None:
            summary["values"] = summarise(
                sample,
                scan=scan,
                sample_fraction=sample_fraction,
                order_mode=order_mode,
                positive_episodes=positive_episodes,
                path=path,
            )
        return summary

    # Table-specific value summaries. Each receives the uniform row sample.
    def _counts(self, series: pd.Series, sample_fraction: float) -> dict[str, int]:
        scaled = (series.value_counts() / max(sample_fraction, 1e-12)).round().astype(int)
        raw = series.value_counts()
        keep = (scaled >= self._options.min_cell) & (raw >= 3 if sample_fraction < 1.0 else raw >= 1)
        return {str(k): int(v) for k, v in scaled[keep].items()}

    def _summarise_demographics(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        return {"age": self._counts(sample["age"], f), "gender": self._counts(sample["gender"], f)}

    def _summarise_ward_info(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        order_mode = kwargs["order_mode"]
        flags = ["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
        combos = sample["order_proc_id_coded"].map(order_mode).fillna("Null") + "|" + sample[flags].agg("|".join, axis=1)
        return {"flags": flags, "mode_flag_combos": self._counts(combos, f)}

    def _summarise_adi_scores(self, sample: pd.DataFrame, **kwargs) -> dict:
        # Score and state rank are drawn together: a missing score always comes
        # with a missing rank, and a patient keeps one pair across orders.
        f = kwargs["sample_fraction"]
        return {"score_rank": self._counts(sample["adi_score"] + "|" + sample["adi_state_rank"], f)}

    def _measure_summary(self, sample: pd.DataFrame, header: list[str]) -> dict:
        measures: dict[str, dict] = {}
        names = sorted({c.split("_", 1)[1] for c in header if c.startswith(("median_", "Q25_", "Q75_", "first_", "last_"))})
        for name in names:
            def col(prefix: str, measure: str = name) -> pd.Series:
                column = f"{prefix}_{measure}"
                if column not in sample.columns:
                    return pd.Series(np.nan, index=sample.index)
                return pd.to_numeric(sample[column].where(~sample[column].isin(MISSING_TOKENS)), errors="coerce")
            median, q25, q75, first, last = col("median"), col("Q25"), col("Q75"), col("first"), col("last")
            has_median = median.notna()
            measures[name] = {
                "median_present": round(float(has_median.mean()), 5),
                "quartiles_present_given_median": round(float((q25.notna() & q75.notna())[has_median].mean()) if has_median.any() else 0.0, 5),
                "first_present": round(float(first.notna().mean()), 5),
                "last_present": round(float(last.notna().mean()), 5),
                "median": quantile_grid(median),
                "lower_spread": quantile_grid((median - q25).clip(lower=0)),
                "upper_spread": quantile_grid((q75 - median).clip(lower=0)),
                "first_minus_median": quantile_grid(first - median),
                "last_minus_median": quantile_grid(last - median),
                "first_alone": quantile_grid(first),
                "last_alone": quantile_grid(last),
            }
        return measures

    def _per_order_patterns(self, scan: _Scan) -> dict:
        if not scan.extra:
            return {}
        extra = pd.concat(scan.extra, ignore_index=True)
        grouped = extra.groupby("order")
        multi = grouped.size() > 1
        identical = grouped["value_hash"].nunique().eq(1)
        result = {"identical_rows_share": round(float(identical[multi].mean()) if multi.any() else 1.0, 5)}
        if "period" in extra.columns:
            patterns = grouped["period"].agg(lambda values: "|".join(sorted(values, key=lambda v: (len(v), v))))
            result["period_patterns"] = suppressed_counts(patterns.value_counts(), self._options.min_cell)
        return result

    def _summarise_labs(self, sample: pd.DataFrame, **kwargs) -> dict:
        return {"measures": self._measure_summary(sample, list(sample.columns)), **self._per_order_patterns(kwargs["scan"])}

    def _summarise_vitals(self, sample: pd.DataFrame, **kwargs) -> dict:
        return {"measures": self._measure_summary(sample, list(sample.columns)), **self._per_order_patterns(kwargs["scan"])}

    def _days_summary(self, series: pd.Series) -> dict:
        present = series[~series.isin(MISSING_TOKENS)]
        numeric = pd.to_numeric(present, errors="coerce").dropna()
        return {
            "missing_share": round(1.0 - len(present) / max(1, len(series)), 5),
            "zero_share": round(float((numeric == 0).mean()) if len(numeric) else 0.0, 5),
            "quantiles": quantile_grid(numeric),
        }

    def _summarise_comorbidity(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        return {
            "components": self._counts(sample["comorbidity_component"], f),
            "start_days": self._days_summary(sample["comorbidity_component_start_days_culture"]),
            "end_days": self._days_summary(sample["comorbidity_component_end_days_culture"]),
        }

    def _summarise_prior_med(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        pairs = sample["medication_name"] + "\x1f" + sample["medication_category"]
        return {"medications": self._counts(pairs, f), "days": self._days_summary(sample["medication_time_to_culturetime"])}

    def _summarise_antibiotic_class_exposure(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        mapping = sample["medication_name"] + "\x1f" + sample["medication_category"] + "\x1f" + sample["antibiotic_class"]
        return {"mapping": self._counts(mapping, f), "days": self._days_summary(sample["time_to_culturetime"])}

    def _summarise_antibiotic_subtype_exposure(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        mapping = (
            sample["medication_name"] + "\x1f" + sample["medication_category"] + "\x1f"
            + sample["antibiotic_subtype"] + "\x1f" + sample["antibiotic_subtype_category"]
        )
        days_column = next(c for c in sample.columns if c.lower() == "medication_time_to_culturetime")
        return {"mapping": self._counts(mapping, f), "days": self._days_summary(sample[days_column])}

    def _summarise_prior_procedures(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        return {
            "procedures": self._counts(sample["procedure_description"], f),
            "days": self._days_summary(sample["procedure_time_to_culturetime"]),
        }

    def _summarise_nursing_home_visits(self, sample: pd.DataFrame, **kwargs) -> dict:
        return {"value": self._days_summary(sample["nursing_home_visit_culture"])}

    def _summarise_prior_infecting_organism(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        days_column = next(c for c in sample.columns if c.startswith("prior_infecting_organism_days_to_cul"))
        return {"organisms": self._counts(sample["prior_organism"], f), "days": self._days_summary(sample[days_column])}

    def _summarise_microbial_resistance(self, sample: pd.DataFrame, **kwargs) -> dict:
        f = kwargs["sample_fraction"]
        empty = sample["organism"].isin(MISSING_TOKENS) & sample["antibiotic"].isin(MISSING_TOKENS)
        present = sample.loc[~empty]
        pairs = present["organism"] + "\x1f" + present["antibiotic"]
        return {
            "empty_row_share": round(float(empty.mean()), 5),
            "organism_antibiotic": self._counts(pairs, f),
            "days": self._days_summary(present["resistant_time_to_culturetime"]),
        }

    def _summarise_implied_susceptibility(self, sample: pd.DataFrame, **kwargs) -> dict:
        path: Path = kwargs["path"]
        positive_episodes: pd.DataFrame = kwargs["positive_episodes"]
        full = pd.read_csv(path, dtype=str, keep_default_na=False)
        episodes = full.drop_duplicates(["order_proc_id_coded", "organism"])
        covered = episodes.merge(positive_episodes, on=["order_proc_id_coded", "organism"], how="inner")
        coverage = covered.groupby("organism").size() / positive_episodes.groupby("organism").size()
        per_episode = full.groupby(["order_proc_id_coded", "organism"])["antibiotic"].nunique()
        inclusion = (
            full.drop_duplicates(["order_proc_id_coded", "organism", "antibiotic"])
            .groupby(["organism", "antibiotic"]).size().rename("episodes").reset_index()
        )
        inclusion = inclusion.merge(episodes.groupby("organism").size().rename("organism_episodes").reset_index(), on="organism")
        inclusion = inclusion.loc[inclusion["episodes"] >= self._options.min_cell]
        inclusion["share"] = (inclusion["episodes"] / inclusion["organism_episodes"]).round(5)
        values = full["antibiotic"] + "\x1f" + full["susceptibility"] + "\x1f" + full["implied_susceptibility"]
        return {
            "coverage_by_organism": {
                str(k): round(float(v), 5)
                for k, v in coverage.dropna().items()
                if positive_episodes["organism"].eq(k).sum() >= self._options.min_cell
            },
            "antibiotics_per_episode": self._histogram(per_episode, cap=200),
            "inclusion": inclusion[["organism", "antibiotic", "share"]].to_dict(orient="records"),
            "values": suppressed_counts(values.value_counts(), self._options.min_cell),
        }

    def _summarise_implied_susceptibility_rules(self, sample: pd.DataFrame, **kwargs) -> dict:
        # A published reference table (organism, antibiotic, rule), not patient data.
        rules = pd.read_csv(kwargs["path"], dtype=str, keep_default_na=False)
        return {"rows": rules.to_dict(orient="records")}

    # ----------------------------------------------------------------- helpers
    def _missing_counts(self, frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {column: {token: int((frame[column] == token).sum()) for token in MISSING_TOKENS} for column in frame.columns}

    def _histogram(self, counts: pd.Series, *, cap: int) -> dict[str, int]:
        values = counts.astype(int).clip(upper=cap)
        histogram = values.value_counts().sort_index()
        return {str(k): int(v) for k, v in histogram.items() if v >= self._options.min_cell}

    def _id_shapes(self, values: pd.Series) -> list[list]:
        sample = values.drop_duplicates()
        if len(sample) > 50_000:
            sample = sample.sample(n=50_000, random_state=0)
        shapes = sample.map(id_shape).value_counts(normalize=True)
        return [[shape, round(float(share), 5)] for shape, share in shapes.head(8).items()]

    def _estimate_oversized(self, profile: dict) -> None:
        """Per-order row counts for files read only from the head.

        The head of a randomly ordered file shows which orders have rows but not
        how many rows each has, so the coverage and the shape of the per-order
        distribution are borrowed from the same table at the other sites and the
        distribution is rescaled to the file's estimated total row count.
        """
        for site, site_profile in profile["sites"].items():
            for canonical, summary in site_profile["tables"].items():
                if not summary.get("estimated"):
                    continue
                donors = [
                    other["tables"][canonical]
                    for name, other in profile["sites"].items()
                    if name != site and canonical in other["tables"] and not other["tables"][canonical].get("estimated")
                ]
                if not donors:
                    continue
                coverage = float(np.mean([d["coverage"] for d in donors]))
                pooled: Counter = Counter()
                for donor in donors:
                    total = sum(donor["rows_per_order"].values())
                    for value, count in donor["rows_per_order"].items():
                        pooled[int(value)] += count / total
                mean_pooled = sum(v * w for v, w in pooled.items()) / sum(pooled.values())
                orders = site_profile["cohort"]["orders"]
                target_mean = summary["rows"] / max(1.0, coverage * orders)
                scale = target_mean / mean_pooled
                rescaled: Counter = Counter()
                for value, weight in pooled.items():
                    rescaled[max(1, int(round(value * scale)))] += weight
                total_weight = sum(rescaled.values())
                summary["coverage"] = round(coverage, 5)
                summary["rows_per_order"] = {
                    str(v): int(round(1e6 * w / total_weight)) for v, w in sorted(rescaled.items()) if w > 0
                }
                summary["rows_per_order_note"] = (
                    f"estimated: coverage and distribution shape from {len(donors)} other site(s), "
                    f"rescaled to {summary['rows']:,} estimated rows (mean {target_mean:.1f} per covered order)"
                )


def load_table_map(reference_dir: Path) -> dict[str, dict[str, str]]:
    import yaml

    with (reference_dir / "canonical_table_map.yaml").open("r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("sites", {})
