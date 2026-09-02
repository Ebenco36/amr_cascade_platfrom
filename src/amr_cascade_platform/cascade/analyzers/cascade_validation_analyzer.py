"""Validate retained cascade edges against null, stability, and replication checks."""

from __future__ import annotations

import gc
import logging
import math
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from amr_cascade_platform.core.config.config_models import Settings

logger = logging.getLogger(__name__)


class CascadeValidationAnalyzer:
    """Stress-test retained edges so reported cascades are less likely to be artifacts."""

    # Single source of truth for what counts as a "validated" edge downstream
    # (network/sankey/chord tiering, primary-cascade tables, prevalence
    # cross-referencing). _classify_validation() below is what actually
    # produces these four status strings; every other module that needs to
    # know which ones count as validated should import this constant rather
    # than re-declaring the {"robust", "supported"} literal.
    VALIDATED_STATUSES: frozenset[str] = frozenset({"robust", "supported"})

    _VALIDATION_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "cascade_direction",
        "observed_escalation_ratio",
        "permutation_p_value",
        "permutation_p_value_two_sided",
        "permutation_fdr_q_value",
        "permutation_fdr_supported",
        "permutation_fdr_q_value_two_sided",
        "permutation_fdr_supported_two_sided",
        "permutation_null_median_er",
        "permutation_null_q05_er",
        "permutation_null_q95_er",
        "bootstrap_median_er",
        "bootstrap_er_ci_lower",
        "bootstrap_er_ci_upper",
        "bootstrap_sign_stability",
        "bootstrap_retention_rate",
        "site_replication_n",
        "site_direction_agreement_rate",
        "site_median_er",
        "site_pooled_log_er",
        "site_pooled_se",
        "site_i_squared",
        "site_cochran_q",
        "site_heterogeneity_p",
        "between_site_permutation_p_value",
        "between_site_permutation_fdr_q_value",
        "between_site_permutation_supported",
        "temporal_early_er",
        "temporal_late_er",
        "temporal_n_early",
        "temporal_n_late",
        "temporal_direction_agreement",
        "temporal_log_ratio_delta",
        "validation_status",
    ]

    # Columns written by each shard: raw p-values only; FDR columns and final
    # validation_status are added by merge_shards() after all shards complete.
    _SHARD_COLUMNS = [
        "upstream_antibiotic",
        "downstream_antibiotic",
        "cascade_direction",
        "observed_escalation_ratio",
        "permutation_p_value",
        "permutation_p_value_two_sided",
        "permutation_null_median_er",
        "permutation_null_q05_er",
        "permutation_null_q95_er",
        "bootstrap_median_er",
        "bootstrap_er_ci_lower",
        "bootstrap_er_ci_upper",
        "bootstrap_sign_stability",
        "bootstrap_retention_rate",
        "site_replication_n",
        "site_direction_agreement_rate",
        "site_median_er",
        "site_pooled_log_er",
        "site_pooled_se",
        "site_i_squared",
        "site_cochran_q",
        "site_heterogeneity_p",
        "between_site_permutation_p_value",
        "temporal_early_er",
        "temporal_late_er",
        "temporal_n_early",
        "temporal_n_late",
        "temporal_direction_agreement",
        "temporal_log_ratio_delta",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.cascade.validation

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(self, drug_pairs: pd.DataFrame, retained_edges: pd.DataFrame) -> pd.DataFrame:
        """Validate all retained edges in a single sequential run.

        For large edge sets (>~200 edges at B=1000) this may exceed SLURM
        wall-time limits.  Use :meth:`analyze_shard` + :meth:`merge_shards`
        to split the work across multiple jobs.
        """
        if not self._config.enabled or drug_pairs.empty or retained_edges.empty:
            return pd.DataFrame(columns=self._VALIDATION_COLUMNS)

        prepared = self._prepare_pairs(drug_pairs)
        if prepared.empty:
            return pd.DataFrame(columns=self._VALIDATION_COLUMNS)

        total_edges = len(retained_edges)
        logger.info(
            "Cascade validation starting: %d retained edges | permutation_iterations=%d "
            "bootstrap_iterations=%d",
            total_edges, self._config.permutation_iterations, self._config.bootstrap_iterations,
        )

        rows = self._run_edge_loop(prepared, retained_edges, total_edges, global_offset=0)
        if not rows:
            logger.info("Cascade validation complete: 0 edges validated.")
            return pd.DataFrame(columns=self._VALIDATION_COLUMNS)

        return self.merge_shards([pd.DataFrame(rows)])

    def analyze_shard(
        self,
        drug_pairs: pd.DataFrame,
        retained_edges: pd.DataFrame,
        shard_index: int,
        shard_total: int,
    ) -> pd.DataFrame:
        """Validate one slice of retained_edges without applying BH-FDR.

        Parameters
        ----------
        drug_pairs:
            Full filtered drug-pair episodes (same as passed to :meth:`analyze`).
        retained_edges:
            Full retained-edge table.  The shard determines its own slice.
        shard_index:
            Zero-based shard number (0 … shard_total-1).
        shard_total:
            Total number of shards the edge set is divided into.

        Returns
        -------
        DataFrame with :attr:`_SHARD_COLUMNS` — raw p-values, no FDR columns.
        Pass a list of shard frames to :meth:`merge_shards` to produce the
        final validated result.
        """
        if not self._config.enabled or drug_pairs.empty or retained_edges.empty:
            return pd.DataFrame(columns=self._SHARD_COLUMNS)

        prepared = self._prepare_pairs(drug_pairs)
        if prepared.empty:
            return pd.DataFrame(columns=self._SHARD_COLUMNS)

        total_edges = len(retained_edges)
        edges_per_shard = math.ceil(total_edges / shard_total)
        start = shard_index * edges_per_shard
        end = min(start + edges_per_shard, total_edges)

        if start >= total_edges:
            logger.info(
                "Shard %d/%d: no edges to process (total=%d, start=%d)",
                shard_index, shard_total, total_edges, start,
            )
            return pd.DataFrame(columns=self._SHARD_COLUMNS)

        shard_edges = retained_edges.iloc[start:end].reset_index(drop=True)
        logger.info(
            "Cascade validation shard %d/%d: edges %d-%d of %d | "
            "permutation_iterations=%d bootstrap_iterations=%d",
            shard_index, shard_total, start, end - 1, total_edges,
            self._config.permutation_iterations, self._config.bootstrap_iterations,
        )

        rows = self._run_edge_loop(prepared, shard_edges, total_edges, global_offset=start)
        if not rows:
            logger.info(
                "Shard %d/%d complete: 0 edges validated (all keys missing from grouped data).",
                shard_index, shard_total,
            )
            return pd.DataFrame(columns=self._SHARD_COLUMNS)

        result = pd.DataFrame(rows)
        logger.info(
            "Shard %d/%d complete: %d edges validated.",
            shard_index, shard_total, len(result),
        )
        return result.reindex(columns=self._SHARD_COLUMNS)

    def merge_shards(self, shards: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate shard frames and apply global BH-FDR.

        This is the merge step that must run after all ``analyze_shard`` jobs
        complete.  It replicates the post-loop FDR block from ``analyze`` but
        operates over the combined set of edges from all shards.

        Parameters
        ----------
        shards:
            List of DataFrames produced by :meth:`analyze_shard` (or a single
            DataFrame produced by the internal ``_run_edge_loop`` call inside
            :meth:`analyze`).

        Returns
        -------
        DataFrame with the full :attr:`_VALIDATION_COLUMNS` schema, BH-FDR
        applied globally, and ``validation_status`` derived from FDR q-values.
        """
        non_empty = [s for s in shards if not s.empty]
        if not non_empty:
            return pd.DataFrame(columns=self._VALIDATION_COLUMNS)

        combined = pd.concat(non_empty, ignore_index=True)

        # Apply BH-FDR simultaneously across all edges for both permutation tests.
        # The provisional validation_status written during the edge loop used the raw
        # two-sided p-value; the authoritative classification here uses the two-sided
        # FDR q-value (permutation_fdr_q_value_two_sided) as the primary criterion,
        # I² heterogeneity, and the pooled log-ER z-score.
        thresh = self._config.permutation_p_value_threshold
        combined["permutation_fdr_q_value"] = self._benjamini_hochberg(combined["permutation_p_value"])
        combined["permutation_fdr_supported"] = (
            combined["permutation_fdr_q_value"].notna()
            & combined["permutation_fdr_q_value"].le(thresh)
        )
        # Parallel two-sided variant, not yet wired into validation_status below:
        # reported alongside the one-sided-in-observed-direction q-value that
        # currently governs the authoritative label, so the two can be compared
        # directly before deciding whether to switch which one governs.
        combined["permutation_fdr_q_value_two_sided"] = self._benjamini_hochberg(
            combined["permutation_p_value_two_sided"]
        )
        combined["permutation_fdr_supported_two_sided"] = (
            combined["permutation_fdr_q_value_two_sided"].notna()
            & combined["permutation_fdr_q_value_two_sided"].le(thresh)
        )
        combined["between_site_permutation_fdr_q_value"] = self._benjamini_hochberg(
            combined["between_site_permutation_p_value"]
        )
        combined["between_site_permutation_supported"] = (
            combined["between_site_permutation_fdr_q_value"].notna()
            & combined["between_site_permutation_fdr_q_value"].le(thresh)
        )
        combined["validation_status"] = [
            self._classify_validation(
                permutation_p_value=float(q),
                bootstrap_sign_stability=float(bs),
                site_replication_n=int(sn),
                site_direction_agreement_rate=float(sa),
                temporal_direction_agreement=bool(ta),
                observed_er=float(er),
                site_i_squared=float(i2),
                site_pooled_log_er=float(pl),
                site_pooled_se=float(ps),
            )
            for q, bs, sn, sa, ta, er, i2, pl, ps in zip(
                combined["permutation_fdr_q_value_two_sided"],
                combined["bootstrap_sign_stability"],
                combined["site_replication_n"],
                combined["site_direction_agreement_rate"],
                combined["temporal_direction_agreement"],
                combined["observed_escalation_ratio"],
                combined["site_i_squared"],
                combined["site_pooled_log_er"],
                combined["site_pooled_se"],
            )
        ]
        status_counts = combined["validation_status"].value_counts().to_dict()
        logger.info(
            "Cascade validation complete: %d edges | status counts (FDR-corrected): %s",
            len(combined), status_counts,
        )
        return combined.reindex(columns=self._VALIDATION_COLUMNS)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_edge_loop(
        self,
        prepared: pd.DataFrame,
        edges: pd.DataFrame,
        total_edges: int,
        global_offset: int,
    ) -> list[dict[str, object]]:
        """Iterate over *edges* and compute per-edge validation statistics.

        Parameters
        ----------
        prepared:
            Output of :meth:`_prepare_pairs`.
        edges:
            The slice of retained edges to process (may be the full set or a shard).
        total_edges:
            Total edge count across ALL shards — used only for ETA logging.
        global_offset:
            Index of the first edge in *edges* within the full retained-edge
            table — used to log ``[global_idx/total_edges]`` rather than a
            shard-local index.
        """
        rows: list[dict[str, object]] = []
        grouped = prepared.groupby(
            ["upstream_antibiotic", "downstream_antibiotic"], dropna=False, observed=True
        )
        run_start = time.monotonic()

        for local_idx, edge in enumerate(edges.itertuples(index=False), start=1):
            global_idx = global_offset + local_idx
            key = (edge.upstream_antibiotic, edge.downstream_antibiotic)
            if key not in grouped.groups:
                continue
            subset = grouped.get_group(key).reset_index(drop=True)
            observed_er = self._coerce_ratio(getattr(edge, "escalation_ratio", math.nan))

            t0 = time.monotonic()
            permutation = self._permutation_summary(subset, observed_er, *key)
            t1 = time.monotonic()
            bootstrap = self._bootstrap_summary(subset, observed_er, *key)
            t2 = time.monotonic()
            site = self._site_replication_summary(subset, observed_er)
            t3 = time.monotonic()
            temporal = self._temporal_replication_summary(subset, observed_er)
            t4 = time.monotonic()
            between_site = self._between_site_permutation_summary(subset, observed_er, *key)
            t5 = time.monotonic()

            cascade_direction = "escalation" if pd.notna(observed_er) and observed_er >= 1.0 else (
                "suppression" if pd.notna(observed_er) else None
            )
            # provisional status using raw two-sided p-value; overwritten by merge_shards() with the two-sided FDR q-value
            status = self._classify_validation(
                permutation_p_value=permutation["permutation_p_value_two_sided"],
                bootstrap_sign_stability=bootstrap["bootstrap_sign_stability"],
                site_replication_n=site["site_replication_n"],
                site_direction_agreement_rate=site["site_direction_agreement_rate"],
                temporal_direction_agreement=temporal["temporal_direction_agreement"],
                observed_er=observed_er,
                site_i_squared=site.get("site_i_squared", math.nan),
                site_pooled_log_er=site.get("site_pooled_log_er", math.nan),
                site_pooled_se=site.get("site_pooled_se", math.nan),
            )
            rows.append(
                {
                    "upstream_antibiotic": key[0],
                    "downstream_antibiotic": key[1],
                    "cascade_direction": cascade_direction,
                    "observed_escalation_ratio": observed_er,
                    **permutation,
                    **bootstrap,
                    **site,
                    **temporal,
                    **between_site,
                }
            )

            elapsed = time.monotonic() - run_start
            avg_per_edge = elapsed / local_idx
            eta = avg_per_edge * (total_edges - global_idx)
            logger.info(
                "[%d/%d] %s -> %s | perm=%.1fs boot=%.1fs site=%.1fs temporal=%.1fs btwn=%.1fs "
                "| raw_p_status=%s | elapsed=%s eta=%s",
                global_idx, total_edges, key[0], key[1],
                t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                status, self._format_duration(elapsed), self._format_duration(eta),
            )

        return rows

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _prepare_pairs(self, drug_pairs: pd.DataFrame) -> pd.DataFrame:
        frame = drug_pairs.copy()
        if self._settings.cascade.require_downstream_eligible and "downstream_eligible" in frame.columns:
            frame = frame.loc[frame["downstream_eligible"] == 1].copy()
        frame["upstream_result_group"] = frame["upstream_susceptibility"].map(self._map_result_group)
        frame = frame.loc[frame["upstream_result_group"].isin(["positive", "negative"])].copy()
        if "order_time_jittered" in frame.columns:
            # astype(object) first: pd.to_datetime on a Categorical column can, depending on
            # category cardinality, return Categorical dtype instead of datetime64 (observed
            # empirically -- a large-cardinality column converts correctly, a small one
            # doesn't). _temporal_replication_summary below does `<=` comparisons on this
            # column, which raises unconditionally for an unordered Categorical. Casting to
            # object first guarantees to_datetime always receives a plain array. format="mixed"
            # matches denominator construction because the combined ARMD extracts contain
            # site-specific timestamp string formats.
            frame["_event_time"] = pd.to_datetime(
                frame["order_time_jittered"].astype(object), errors="coerce", utc=True, format="mixed"
            )
        else:
            frame["_event_time"] = pd.NaT
        if "source_site" not in frame.columns:
            frame["source_site"] = "unknown"
        frame["downstream_tested"] = pd.to_numeric(frame["downstream_tested"], errors="coerce").fillna(0).astype(int)
        return frame

    def _permutation_summary(
        self,
        subset: pd.DataFrame,
        observed_er: float,
        upstream_antibiotic: str,
        downstream_antibiotic: str,
    ) -> dict[str, float]:
        iterations = self._config.permutation_iterations
        if iterations <= 0 or pd.isna(observed_er):
            return {
                "permutation_p_value": math.nan,
                "permutation_p_value_two_sided": math.nan,
                "permutation_null_median_er": math.nan,
                "permutation_null_q05_er": math.nan,
                "permutation_null_q95_er": math.nan,
            }

        rng = np.random.default_rng(self._edge_seed(upstream_antibiotic, downstream_antibiotic, salt=17))
        frame = subset.loc[:, ["source_site", "upstream_result_group", "downstream_tested"]].copy()
        null_values: list[float] = []
        for i in range(iterations):
            permuted = frame.copy()
            permuted["upstream_result_group"] = self._shuffle_within_strata(
                values=frame["upstream_result_group"],
                strata=frame["source_site"],
                rng=rng,
                episode_keys=subset,
            )
            null_values.append(self._compute_escalation_ratio(permuted))
            del permuted
            if i % 50 == 49:
                gc.collect()

        return {
            "permutation_p_value": self._empirical_p_value(observed_er, null_values),
            "permutation_p_value_two_sided": self._empirical_p_value_two_sided(observed_er, null_values),
            "permutation_null_median_er": self._safe_quantile(null_values, 0.5),
            "permutation_null_q05_er": self._safe_quantile(null_values, 0.05),
            "permutation_null_q95_er": self._safe_quantile(null_values, 0.95),
        }

    def _bootstrap_summary(
        self,
        subset: pd.DataFrame,
        observed_er: float,
        upstream_antibiotic: str,
        downstream_antibiotic: str,
    ) -> dict[str, float]:
        iterations = self._config.bootstrap_iterations
        if iterations <= 0 or pd.isna(observed_er):
            return {
                "bootstrap_median_er": math.nan,
                "bootstrap_er_ci_lower": math.nan,
                "bootstrap_er_ci_upper": math.nan,
                "bootstrap_sign_stability": math.nan,
                "bootstrap_retention_rate": math.nan,
            }

        rng = np.random.default_rng(self._edge_seed(upstream_antibiotic, downstream_antibiotic, salt=31))
        boot_values: list[float] = []
        for i in range(iterations):
            resampled = self._resample_within_sites(subset, rng)
            boot_values.append(self._compute_escalation_ratio(resampled))
            del resampled
            if i % 50 == 49:
                gc.collect()

        valid = self._finite_array(boot_values)
        if valid.size == 0:
            return {
                "bootstrap_median_er": math.nan,
                "bootstrap_er_ci_lower": math.nan,
                "bootstrap_er_ci_upper": math.nan,
                "bootstrap_sign_stability": math.nan,
                "bootstrap_retention_rate": math.nan,
            }
        positive_effect = observed_er >= 1.0
        sign_stability = float((valid >= 1.0).mean()) if positive_effect else float((valid <= 1.0).mean())
        retention_rate = (
            float((valid >= self._settings.cascade.retained_min_escalation_ratio).mean())
            if positive_effect
            else float((valid <= 1 / max(self._settings.cascade.retained_min_escalation_ratio, 1e-9)).mean())
        )
        return {
            "bootstrap_median_er": float(np.median(valid)),
            "bootstrap_er_ci_lower": float(np.quantile(valid, 0.025)),
            "bootstrap_er_ci_upper": float(np.quantile(valid, 0.975)),
            "bootstrap_sign_stability": sign_stability,
            "bootstrap_retention_rate": retention_rate,
        }

    def _site_replication_summary(self, subset: pd.DataFrame, observed_er: float) -> dict[str, float]:
        """Per-site ERs with DerSimonian-Laird random-effects pooling and I² heterogeneity."""
        λ = float(self._settings.cascade.continuity_correction)
        valid_site_ers: list[float] = []
        site_log_ers: list[float] = []
        site_ses: list[float] = []
        matching_direction = 0
        for _, site_subset in subset.groupby("source_site", dropna=False, observed=True):
            summary = self._compute_ratio_summary(site_subset)
            if not summary["passes_support_threshold"] or pd.isna(summary["escalation_ratio"]):
                continue
            site_er = float(summary["escalation_ratio"])
            valid_site_ers.append(site_er)
            if self._same_direction(observed_er, site_er):
                matching_direction += 1
            if site_er <= 0:
                continue
            k_R = int(summary["positive_tested_n"])
            n_R = int(summary["positive_support_n"])
            k_S = int(summary["negative_tested_n"])
            n_S = int(summary["negative_support_n"])
            se2 = self._log_er_variance(k_R, n_R, k_S, n_S, λ)
            if pd.notna(se2) and se2 > 0:
                site_log_ers.append(math.log(site_er))
                site_ses.append(math.sqrt(se2))

        site_replication_n = len(valid_site_ers)
        agreement_rate = (
            float(matching_direction / site_replication_n)
            if site_replication_n > 0
            else math.nan
        )
        dl = self._dersimonian_laird(site_log_ers, site_ses)
        return {
            "site_replication_n": site_replication_n,
            "site_direction_agreement_rate": agreement_rate,
            "site_median_er": float(np.median(valid_site_ers)) if valid_site_ers else math.nan,
            **dl,
        }

    def _between_site_permutation_summary(
        self,
        subset: pd.DataFrame,
        observed_er: float,
        upstream_antibiotic: str,
        downstream_antibiotic: str,
    ) -> dict[str, float]:
        """Between-site permutation: shuffle upstream-result/downstream-tested values across
        site-sized blocks and recompute the site-pooled ER. `subset` is one row per episode for
        this pair (PairGenerator excludes discordant duplicates upstream), so permuting rows is
        equivalent to permuting episodes here -- but the permutation itself operates on the two
        derived numpy arrays, not on an episode-level object, so nothing else about an episode
        travels with it into the shuffle."""
        if not self._config.between_site_permutation_enabled:
            return {"between_site_permutation_p_value": math.nan}

        iterations = self._config.permutation_iterations
        sites = list(subset["source_site"].dropna().unique())

        if iterations <= 0 or pd.isna(observed_er) or len(sites) < 2:
            return {"between_site_permutation_p_value": math.nan}

        rng = np.random.default_rng(
            self._edge_seed(upstream_antibiotic, downstream_antibiotic, salt=53)
        )
        λ = float(self._settings.cascade.continuity_correction)
        min_result = self._settings.cascade.min_result_support
        min_total = self._settings.cascade.min_total_support

        # The site-pooled statistic below slices contiguous site blocks. Enforce
        # that ordering here rather than relying on upstream parquet/merge order.
        subset_ordered = subset.sort_values("source_site", kind="mergesort").reset_index(drop=True)
        site_sizes = np.array(
            [int((subset_ordered["source_site"] == s).sum()) for s in sites], dtype=np.intp
        )
        total_n = int(site_sizes.sum())

        is_positive = (subset_ordered["upstream_result_group"].to_numpy() == "positive")
        tested = subset_ordered["downstream_tested"].to_numpy(dtype=float)

        def _dl_pooled_log_er_stat(is_pos: np.ndarray, test_arr: np.ndarray) -> float:
            log_ers: list[float] = []
            ses: list[float] = []
            start = 0
            for size in site_sizes:
                end = int(start + size)
                sp = is_pos[start:end]
                st = test_arr[start:end]
                n_R = int(sp.sum())
                n_S = int((~sp).sum())
                if n_R < min_result or n_S < min_result or (n_R + n_S) < min_total:
                    start = end
                    continue
                k_R = int(st[sp].sum())
                k_S = int(st[~sp].sum())
                p_R = (k_R + λ) / (n_R + 2.0 * λ)
                p_S = (k_S + λ) / (n_S + 2.0 * λ)
                if p_S > 0:
                    er = p_R / p_S
                    if er > 0:
                        se2 = self._log_er_variance(k_R, n_R, k_S, n_S, λ)
                        if pd.notna(se2) and se2 > 0:
                            log_ers.append(math.log(er))
                            ses.append(math.sqrt(se2))
                start = end
            if len(log_ers) < 2:
                return math.nan
            dl = self._dersimonian_laird(log_ers, ses)
            return dl["site_pooled_log_er"]

        observed_dl_log_er = _dl_pooled_log_er_stat(is_positive, tested)
        if pd.isna(observed_dl_log_er):
            return {"between_site_permutation_p_value": math.nan}
        observed_dl_er = math.exp(observed_dl_log_er)

        null_values: list[float] = []
        for i in range(iterations):
            shuffled = rng.permutation(total_n)
            null_log_er = _dl_pooled_log_er_stat(is_positive[shuffled], tested[shuffled])
            null_values.append(math.exp(null_log_er) if pd.notna(null_log_er) else math.nan)
            if i % 50 == 49:
                gc.collect()

        return {
            "between_site_permutation_p_value": self._empirical_p_value(observed_dl_er, null_values),
        }

    def _temporal_replication_summary(self, subset: pd.DataFrame, observed_er: float) -> dict[str, object]:
        """Temporal replication using first and last thirds (gap in middle avoids leakage)."""
        _nan = {
            "temporal_early_er": math.nan,
            "temporal_late_er": math.nan,
            "temporal_n_early": 0,
            "temporal_n_late": 0,
            "temporal_direction_agreement": False,
            "temporal_log_ratio_delta": math.nan,
        }
        dated = subset.loc[subset["_event_time"].notna()].copy()
        if dated.empty or dated["_event_time"].nunique() < 2:
            return _nan
        n = len(dated)
        if n < 15:  # cheap pre-filter only: guarantees 5 rows/third, NOT that either third
            return _nan  # is estimable -- _compute_ratio_summary below still applies the real
            # min_total_support/min_result_support gates per third, so early_er/late_er are
            # only ever non-NaN around n>=3*min_total_support (~75 with default config), not n=15.
        sd = dated.sort_values("_event_time").reset_index(drop=True)
        t = n // 3
        early = sd.iloc[:t]
        late = sd.iloc[n - t:]
        early_summary = self._compute_ratio_summary(early)
        late_summary = self._compute_ratio_summary(late)
        early_er = (
            self._coerce_ratio(early_summary["escalation_ratio"])
            if early_summary["passes_support_threshold"] else math.nan
        )
        late_er = (
            self._coerce_ratio(late_summary["escalation_ratio"])
            if late_summary["passes_support_threshold"] else math.nan
        )
        direction_agreement = bool(
            pd.notna(early_er)
            and pd.notna(late_er)
            and self._same_direction(early_er, late_er)
            and self._same_direction(observed_er, early_er)
        )
        log_ratio_delta = (
            abs(math.log(early_er) - math.log(late_er))
            if pd.notna(early_er) and early_er > 0 and pd.notna(late_er) and late_er > 0
            else math.nan
        )
        return {
            "temporal_early_er": early_er,
            "temporal_late_er": late_er,
            "temporal_n_early": int(len(early)),
            "temporal_n_late": int(len(late)),
            "temporal_direction_agreement": direction_agreement,
            "temporal_log_ratio_delta": log_ratio_delta,
        }

    def _compute_ratio_summary(self, frame: pd.DataFrame) -> dict[str, float | bool]:
        positive = frame.loc[frame["upstream_result_group"] == "positive", "downstream_tested"]
        negative = frame.loc[frame["upstream_result_group"] == "negative", "downstream_tested"]
        positive_support_n = int(len(positive))
        negative_support_n = int(len(negative))
        total_support_n = positive_support_n + negative_support_n
        positive_probability = float(positive.mean()) if positive_support_n > 0 else math.nan
        negative_probability = float(negative.mean()) if negative_support_n > 0 else math.nan
        correction = float(self._settings.cascade.continuity_correction)
        positive_tested_n = int(positive.sum()) if positive_support_n > 0 else 0
        negative_tested_n = int(negative.sum()) if negative_support_n > 0 else 0
        if positive_support_n > 0 and negative_support_n > 0:
            positive_corrected_probability = (positive_tested_n + correction) / (positive_support_n + 2.0 * correction)
            negative_corrected_probability = (negative_tested_n + correction) / (negative_support_n + 2.0 * correction)
            escalation_ratio = (
                positive_corrected_probability / negative_corrected_probability
                if negative_corrected_probability > 0
                else math.nan
            )
        else:
            escalation_ratio = math.nan
        passes_support_threshold = (
            total_support_n >= self._settings.cascade.min_total_support
            and positive_support_n >= self._settings.cascade.min_result_support
            and negative_support_n >= self._settings.cascade.min_result_support
        )
        return {
            "positive_support_n": positive_support_n,
            "negative_support_n": negative_support_n,
            "positive_tested_n": positive_tested_n,
            "negative_tested_n": negative_tested_n,
            "total_support_n": total_support_n,
            "escalation_ratio": escalation_ratio,
            "passes_support_threshold": passes_support_threshold,
        }

    def _compute_escalation_ratio(self, frame: pd.DataFrame) -> float:
        summary = self._compute_ratio_summary(frame)
        return self._coerce_ratio(summary["escalation_ratio"])

    def _classify_validation(
        self,
        permutation_p_value: float,
        bootstrap_sign_stability: float,
        site_replication_n: int,
        site_direction_agreement_rate: float,
        temporal_direction_agreement: bool,
        observed_er: float,
        site_i_squared: float = math.nan,
        site_pooled_log_er: float = math.nan,
        site_pooled_se: float = math.nan,
    ) -> str:
        permutation_supported = (
            pd.notna(permutation_p_value)
            and permutation_p_value <= self._config.permutation_p_value_threshold
        )
        stability_supported = (
            pd.notna(bootstrap_sign_stability)
            and bootstrap_sign_stability >= self._config.bootstrap_sign_stability_threshold
        )
        # Prefer I²-based replication when available: low heterogeneity AND
        # significant pooled log ER (z ≥ 1.96).  Fall back to binary concordance
        # rate when I² could not be computed (< 2 sites with valid SE estimates).
        if (
            pd.notna(site_i_squared)
            and pd.notna(site_pooled_se) and site_pooled_se > 0
            and pd.notna(site_pooled_log_er)
        ):
            pooled_z = abs(site_pooled_log_er / site_pooled_se)
            pooled_direction_matches = (
                (observed_er > 1.0 and site_pooled_log_er > 0)
                or (observed_er < 1.0 and site_pooled_log_er < 0)
            )
            site_replicated = (
                site_replication_n >= self._config.minimum_replicated_sites
                and site_i_squared <= self._config.i_squared_threshold
                and pooled_z >= 1.96
                and pooled_direction_matches
            )
        else:
            site_replicated = (
                site_replication_n >= self._config.minimum_replicated_sites
                and pd.notna(site_direction_agreement_rate)
                and site_direction_agreement_rate >= self._config.minimum_site_direction_agreement_rate
            )
        replication_supported = site_replicated or temporal_direction_agreement
        if permutation_supported and stability_supported and replication_supported:
            return "robust"
        if permutation_supported and stability_supported:
            return "supported"
        if permutation_supported or stability_supported or replication_supported:
            return "mixed"
        return "insufficient"

    @staticmethod
    def _log_er_variance(k_R: int, n_R: int, k_S: int, n_S: int, λ: float) -> float:
        """Delta-method variance of log ER using Laplace-corrected proportions."""
        if n_R <= 0 or n_S <= 0:
            return math.nan
        term_R = 1.0 / (k_R + λ) - 1.0 / (n_R + 2.0 * λ)
        term_S = 1.0 / (k_S + λ) - 1.0 / (n_S + 2.0 * λ)
        var = term_R + term_S
        return var if var > 0 else math.nan

    @staticmethod
    def _dersimonian_laird(log_ers: list[float], ses: list[float]) -> dict[str, float]:
        """DerSimonian-Laird random-effects meta-analysis returning pooled estimate and I²."""
        _empty: dict[str, float] = {
            "site_pooled_log_er": math.nan,
            "site_pooled_se": math.nan,
            "site_i_squared": math.nan,
            "site_cochran_q": math.nan,
            "site_heterogeneity_p": math.nan,
        }
        k = len(log_ers)
        if k < 2:
            return _empty
        y = np.array(log_ers, dtype=float)
        s = np.array(ses, dtype=float)
        w = 1.0 / (s ** 2)
        w_sum = float(w.sum())
        y_fixed = float((w * y).sum() / w_sum)
        Q = float((w * (y - y_fixed) ** 2).sum())
        df = k - 1
        c = float(w_sum - (w ** 2).sum() / w_sum)
        tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
        w_star = 1.0 / (s ** 2 + tau2)
        theta_star = float((w_star * y).sum() / w_star.sum())
        se_star = float(1.0 / math.sqrt(float(w_star.sum())))
        i_squared = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0
        het_p = float(_chi2.sf(Q, df))
        return {
            "site_pooled_log_er": theta_star,
            "site_pooled_se": se_star,
            "site_i_squared": i_squared,
            "site_cochran_q": Q,
            "site_heterogeneity_p": het_p,
        }

    def _shuffle_within_strata(
        self,
        values: pd.Series,
        strata: pd.Series,
        rng: np.random.Generator,
        episode_keys: pd.DataFrame | None = None,
    ) -> pd.Series:
        episode_key_columns = [
            col for col in self._settings.gold.episode_key_columns
            if episode_keys is not None and col in episode_keys.columns
        ]
        # Operate on a numpy array to avoid per-element pandas overhead.
        # values/strata/episode_keys share RangeIndex(0..n-1), so index labels == iloc positions.
        result = values.to_numpy(copy=True)
        for _, stratum_index in strata.groupby(strata, dropna=False, observed=True).groups.items():
            locs = stratum_index.to_numpy()
            if not episode_key_columns or episode_keys is None:
                local = result[locs].copy()
                rng.shuffle(local)
                result[locs] = local
                continue
            stratum_ep_keys = episode_keys.iloc[locs]
            # .groups.values() yields Index objects whose values are index labels of stratum_ep_keys.
            # Those labels are a subset of 0..n-1 and serve directly as numpy positions.
            episode_groups: list[np.ndarray] = [
                grp.to_numpy()
                for grp in stratum_ep_keys.groupby(
                    episode_key_columns, dropna=False, sort=False, observed=True
                ).groups.values()
            ]
            if not episode_groups:
                continue
            first_pos = np.array([grp[0] for grp in episode_groups])
            episode_labels = result[first_pos].copy()
            rng.shuffle(episode_labels)
            sizes = np.fromiter((len(grp) for grp in episode_groups), dtype=np.intp, count=len(episode_groups))
            result[np.concatenate(episode_groups)] = np.repeat(episode_labels, sizes)
        return pd.Series(result, index=values.index)

    def _resample_within_sites(self, subset: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        episode_key_columns = [
            column
            for column in self._settings.gold.episode_key_columns
            if column in subset.columns
        ]
        if subset.empty:
            return subset.copy()
        # Collect subset-level iloc positions for all selected rows across every site, then
        # issue a single iloc at the end — avoids pd.concat on per-site frames each iteration.
        # subset has RangeIndex(0..n-1) so .index values are iloc positions.
        selected: list[np.ndarray] = []
        for _, site_subset in subset.groupby("source_site", dropna=False, sort=False, observed=True):
            if site_subset.empty:
                continue
            if episode_key_columns:
                # .groups.values() yields Index objects whose label values are positions in subset.
                episode_groups: list[np.ndarray] = [
                    grp.to_numpy()
                    for grp in site_subset.groupby(
                        episode_key_columns, dropna=False, sort=False, observed=True
                    ).groups.values()
                ]
                if not episode_groups:
                    continue
                n_ep = len(episode_groups)
                take = rng.integers(0, n_ep, size=n_ep)
                selected.append(np.concatenate([episode_groups[i] for i in take]))
            else:
                site_pos = site_subset.index.to_numpy()
                take = rng.integers(0, len(site_pos), size=len(site_pos))
                selected.append(site_pos[take])
        if not selected:
            return subset.iloc[[]]
        return subset.iloc[np.concatenate(selected)].reset_index(drop=True)

    def _empirical_p_value(self, observed_er: float, null_values: list[float]) -> float:
        valid = self._finite_array(null_values)
        if valid.size == 0 or pd.isna(observed_er):
            return math.nan
        if observed_er >= 1.0:
            exceedances = int((valid >= observed_er).sum())
        else:
            exceedances = int((valid <= observed_er).sum())
        return float((exceedances + 1) / (valid.size + 1))

    def _empirical_p_value_two_sided(self, observed_er: float, null_values: list[float]) -> float:
        """Two-sided permutation p-value on the log-ER scale.

        _empirical_p_value above always tests the tail matching the observed
        direction (upper tail when ER>=1, lower when ER<1), decided from the
        same data being tested; treating its threshold as if it were a
        pre-registered one-sided test is anti-conservative, since either an
        extreme escalation or an extreme suppression would each independently
        clear it. This measures |log ER| against the two-sided null instead,
        so the direction played no part in selecting the test.
        """
        valid = self._finite_array(null_values)
        valid = valid[valid > 0]
        if valid.size == 0 or pd.isna(observed_er) or observed_er <= 0:
            return math.nan
        observed_abs_log_er = abs(math.log(observed_er))
        exceedances = int((np.abs(np.log(valid)) >= observed_abs_log_er).sum())
        return float((exceedances + 1) / (valid.size + 1))

    @staticmethod
    def _safe_quantile(values: list[float], quantile: float) -> float:
        valid = CascadeValidationAnalyzer._finite_array(values)
        if valid.size == 0:
            return math.nan
        return float(np.quantile(valid, quantile))

    @staticmethod
    def _finite_array(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array[np.isfinite(array)]

    @staticmethod
    def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(p_values, errors="coerce")
        valid = numeric.dropna()
        q_values = pd.Series(math.nan, index=p_values.index, dtype=float)
        if valid.empty:
            return q_values
        ordered = valid.sort_values(kind="mergesort")
        m = len(ordered)
        adjusted = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1, dtype=float)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        q_values.loc[ordered.index] = adjusted
        return q_values

    @staticmethod
    def _coerce_ratio(value: object) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(numeric) if pd.notna(numeric) else math.nan

    @staticmethod
    def _same_direction(first: float, second: float) -> bool:
        if pd.isna(first) or pd.isna(second):
            return False
        if first == 1.0 or second == 1.0:
            return False
        return (first > 1.0 and second > 1.0) or (first < 1.0 and second < 1.0)

    def _map_result_group(self, value: object) -> str | None:
        if pd.isna(value):
            return None
        if value == self._settings.cascade.upstream_result_positive:
            return "positive"
        if value == self._settings.cascade.upstream_result_negative:
            return "negative"
        return None

    def _edge_seed(self, upstream_antibiotic: str, downstream_antibiotic: str, *, salt: int) -> int:
        encoded = f"{self._config.random_seed}|{salt}|{upstream_antibiotic}|{downstream_antibiotic}".encode("utf-8")
        return zlib.crc32(encoded)
