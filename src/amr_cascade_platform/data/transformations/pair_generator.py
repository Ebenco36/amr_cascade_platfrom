"""Generate directional drug-pair episodes."""

from __future__ import annotations

import logging

import pandas as pd

from amr_cascade_platform.core.config.config_models import Settings

logger = logging.getLogger(__name__)


class PairGenerator:
    """Generate ordered pair opportunities using observed upstream tests and eligible downstream drugs.

    The unit of analysis is one episode evaluated for one ordered (upstream, downstream) drug pair.
    When an episode contains multiple upstream test results for the same drug (e.g. two culture
    records), the merge can produce duplicate (episode, upstream_antibiotic, downstream_antibiotic)
    rows.  These are resolved with the following rule, applied before any cascade estimation:

    * **Same-result duplicates** (all rows in the group share the same upstream susceptibility):
      collapse to one row (keep first).  Downstream testing outcome is identical across these rows
      so the choice is inconsequential.

    * **Discordant duplicates** (rows in the group have mixed upstream susceptibility, e.g. one
      RESISTANT and one SUSCEPTIBLE record for the same episode/drug/pair): **exclude the entire
      group** from the primary contrast.  Resolving discordant ambiguity by picking either branch
      would arbitrarily inflate that branch's conditional testing probability.  The exclusion is
      logged and the excluded count is available for audit.

    This rule is conservative: it never moves a pair from one susceptibility branch to another.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        culture_drug_episodes: pd.DataFrame,
        eligibility_space: pd.DataFrame,
    ) -> pd.DataFrame:
        episode_keys = list(self._settings.gold.episode_key_columns)
        if culture_drug_episodes.empty or eligibility_space.empty:
            return pd.DataFrame(
                columns=[
                    *episode_keys,
                    "upstream_antibiotic",
                    "upstream_susceptibility",
                    "downstream_antibiotic",
                    "downstream_tested",
                    "downstream_eligible",
                    "downstream_intrinsic_resistance",
                    "pair_direction",
                ]
            )

        upstream = culture_drug_episodes.loc[
            :, episode_keys + ["antibiotic", "susceptibility"]
        ].rename(
            columns={
                "antibiotic": "upstream_antibiotic",
                "susceptibility": "upstream_susceptibility",
            }
        )
        downstream = eligibility_space.loc[
            :,
            episode_keys
            + ["antibiotic", "is_observed_tested", "is_eligible", "is_intrinsic_resistance"],
        ].rename(
            columns={
                "antibiotic": "downstream_antibiotic",
                "is_observed_tested": "downstream_tested",
                "is_eligible": "downstream_eligible",
                "is_intrinsic_resistance": "downstream_intrinsic_resistance",
            }
        )
        pairs = upstream.merge(downstream, on=episode_keys, how="inner")
        pairs = pairs[pairs["upstream_antibiotic"] != pairs["downstream_antibiotic"]].copy()
        pairs["pair_direction"] = pairs["upstream_antibiotic"] + " -> " + pairs["downstream_antibiotic"]

        # Enforce one row per (episode, upstream_antibiotic, downstream_antibiotic).
        pair_keys = episode_keys + ["upstream_antibiotic", "downstream_antibiotic"]
        n_before = len(pairs)

        is_any_dup = pairs.duplicated(subset=pair_keys, keep=False)
        if is_any_dup.any():
            non_dup = pairs[~is_any_dup]
            dup = pairs[is_any_dup].copy()

            # Count distinct susceptibility values within each duplicate group
            n_suscept = dup.groupby(pair_keys, sort=False)["upstream_susceptibility"].transform("nunique")

            # Concordant: all rows share the same susceptibility → keep first occurrence
            concordant = dup[n_suscept == 1].drop_duplicates(subset=pair_keys, keep="first")

            # Discordant: mixed susceptibility within the group → exclude entirely
            discordant = dup[n_suscept > 1]
            n_discordant_pairs = int(discordant.loc[:, pair_keys].drop_duplicates().shape[0])
            n_discordant_rows = int(len(discordant))
            n_concordant_collapsed = int(len(dup[n_suscept == 1]) - len(concordant))

            pairs = pd.concat([non_dup, concordant], ignore_index=True)

            logger.warning(
                "PairGenerator deduplication: %d same-result duplicate row(s) collapsed; "
                "%d discordant episode/upstream/downstream pair(s) (%d row(s)) excluded from primary contrast "
                "(%.4f%% of pre-dedup total). Discordant pairs are excluded to avoid "
                "arbitrary susceptibility-branch assignment that would bias ER.",
                n_concordant_collapsed,
                n_discordant_pairs,
                n_discordant_rows,
                100.0 * (n_concordant_collapsed + n_discordant_rows) / n_before,
            )

        return pairs.reset_index(drop=True)
