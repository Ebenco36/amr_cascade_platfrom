"""Unit tests for PairGenerator.

Tests enforce the unit-of-analysis contract:
  - one row per (episode_keys, upstream_antibiotic, downstream_antibiotic)
  - same-result duplicates are collapsed to one row
  - discordant duplicates (mixed susceptibility) are excluded entirely
  - self-pairs (upstream == downstream) are never emitted
"""

from pathlib import Path

import pandas as pd
import pytest

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.transformations.pair_generator import PairGenerator


@pytest.fixture(scope="module")
def generator() -> PairGenerator:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    return PairGenerator(settings)


def _episode(anon_id: str, antibiotic: str, susceptibility: str) -> dict:
    return {
        "anon_id": anon_id,
        "pat_enc_csn_id_coded": "1",
        "order_proc_id_coded": "2",
        "order_time_jittered": "2024-01-01",
        "organism": "E COLI",
        "source_site": "armd",
        "antibiotic": antibiotic,
        "susceptibility": susceptibility,
    }


def _downstream(anon_id: str, antibiotic: str, tested: int = 0) -> dict:
    return {
        "anon_id": anon_id,
        "pat_enc_csn_id_coded": "1",
        "order_proc_id_coded": "2",
        "order_time_jittered": "2024-01-01",
        "organism": "E COLI",
        "source_site": "armd",
        "antibiotic": antibiotic,
        "is_observed_tested": tested,
        "is_eligible": 1,
        "is_intrinsic_resistance": 0,
    }


# ── Basic contract ────────────────────────────────────────────────────────────

def test_basic_ordered_pair(generator: PairGenerator) -> None:
    """A single upstream + downstream row produces exactly one directed pair."""
    pairs = generator.build(
        pd.DataFrame([_episode("A1", "CIPRO", "RESISTANT")]),
        pd.DataFrame([_downstream("A1", "MEROPENEM")]),
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["upstream_antibiotic"] == "CIPRO"
    assert pairs.iloc[0]["downstream_antibiotic"] == "MEROPENEM"
    assert pairs.iloc[0]["upstream_susceptibility"] == "RESISTANT"
    assert int(pairs.iloc[0]["downstream_tested"]) == 0


def test_self_pair_excluded(generator: PairGenerator) -> None:
    """upstream == downstream must never appear in the output."""
    pairs = generator.build(
        pd.DataFrame([_episode("A1", "CIPRO", "RESISTANT")]),
        pd.DataFrame([
            _downstream("A1", "CIPRO"),    # same drug — should be excluded
            _downstream("A1", "MEROPENEM"),
        ]),
    )
    antibiotics = set(zip(pairs["upstream_antibiotic"], pairs["downstream_antibiotic"]))
    assert ("CIPRO", "CIPRO") not in antibiotics
    assert ("CIPRO", "MEROPENEM") in antibiotics


def test_empty_inputs_return_empty(generator: PairGenerator) -> None:
    pairs = generator.build(pd.DataFrame(), pd.DataFrame())
    assert pairs.empty


# ── Uniqueness invariant ──────────────────────────────────────────────────────

def test_output_keys_are_unique(generator: PairGenerator) -> None:
    """Generated pair keys must be unique — one row per episode/up/down triple."""
    upstream_rows = [
        _episode("A1", "CIPRO", "RESISTANT"),
        _episode("A1", "CIPRO", "RESISTANT"),  # exact duplicate upstream record
    ]
    downstream_rows = [_downstream("A1", "MEROPENEM")]
    pairs = generator.build(
        pd.DataFrame(upstream_rows),
        pd.DataFrame(downstream_rows),
    )
    key_cols = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded",
                "order_time_jittered", "organism", "source_site",
                "upstream_antibiotic", "downstream_antibiotic"]
    existing_keys = [c for c in key_cols if c in pairs.columns]
    assert not pairs.duplicated(subset=existing_keys).any(), (
        "Output contains duplicate episode/upstream/downstream keys"
    )


# ── Same-result duplicate collapse ───────────────────────────────────────────

def test_same_result_duplicate_collapsed_to_one_row(generator: PairGenerator) -> None:
    """Two upstream records with the same susceptibility should become one pair row."""
    upstream_rows = [
        _episode("A1", "CIPRO", "RESISTANT"),
        _episode("A1", "CIPRO", "RESISTANT"),  # second identical culture record
    ]
    downstream_rows = [_downstream("A1", "MEROPENEM", tested=1)]
    pairs = generator.build(
        pd.DataFrame(upstream_rows),
        pd.DataFrame(downstream_rows),
    )
    cipro_mero = pairs[
        (pairs["upstream_antibiotic"] == "CIPRO")
        & (pairs["downstream_antibiotic"] == "MEROPENEM")
    ]
    assert len(cipro_mero) == 1, (
        f"Expected 1 row after same-result collapse, got {len(cipro_mero)}"
    )
    assert cipro_mero.iloc[0]["upstream_susceptibility"] == "RESISTANT"


def test_same_result_susceptible_collapsed(generator: PairGenerator) -> None:
    """Same-result collapse works for SUSCEPTIBLE too (not just RESISTANT)."""
    upstream_rows = [
        _episode("A1", "CIPRO", "SUSCEPTIBLE"),
        _episode("A1", "CIPRO", "SUSCEPTIBLE"),
    ]
    downstream_rows = [_downstream("A1", "MEROPENEM")]
    pairs = generator.build(
        pd.DataFrame(upstream_rows),
        pd.DataFrame(downstream_rows),
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["upstream_susceptibility"] == "SUSCEPTIBLE"


# ── Discordant duplicate exclusion ───────────────────────────────────────────

def test_discordant_duplicates_excluded_entirely(generator: PairGenerator) -> None:
    """An episode with both RESISTANT and SUSCEPTIBLE upstream records must be excluded."""
    upstream_rows = [
        _episode("A1", "CIPRO", "RESISTANT"),
        _episode("A1", "CIPRO", "SUSCEPTIBLE"),  # discordant second culture
    ]
    downstream_rows = [_downstream("A1", "MEROPENEM", tested=1)]
    pairs = generator.build(
        pd.DataFrame(upstream_rows),
        pd.DataFrame(downstream_rows),
    )
    cipro_mero = pairs[
        (pairs["upstream_antibiotic"] == "CIPRO")
        & (pairs["downstream_antibiotic"] == "MEROPENEM")
    ]
    assert len(cipro_mero) == 0, (
        "Discordant episode pair should be excluded, not retained"
    )


def test_discordant_exclusion_does_not_affect_other_episodes(generator: PairGenerator) -> None:
    """Discordant exclusion for one episode must not remove clean rows from other episodes."""
    upstream_rows = [
        # Episode A1: discordant — should be excluded
        _episode("A1", "CIPRO", "RESISTANT"),
        _episode("A1", "CIPRO", "SUSCEPTIBLE"),
        # Episode A2: clean — should be retained
        _episode("A2", "CIPRO", "RESISTANT"),
    ]
    downstream_rows = [
        _downstream("A1", "MEROPENEM"),
        _downstream("A2", "MEROPENEM"),
    ]
    # Make A2 have different identifiers so it's a separate episode
    upstream_df = pd.DataFrame(upstream_rows)
    downstream_df = pd.DataFrame(downstream_rows)
    # Distinguish A2 rows from A1 by anon_id (already set above)
    pairs = generator.build(upstream_df, downstream_df)

    a1_rows = pairs[pairs["anon_id"] == "A1"]
    a2_rows = pairs[pairs["anon_id"] == "A2"]
    assert len(a1_rows) == 0, "Discordant episode A1 should be fully excluded"
    assert len(a2_rows) == 1, "Clean episode A2 should be retained"


def test_concordant_and_discordant_mixed_cohort(generator: PairGenerator) -> None:
    """Cohort with both concordant and discordant duplicates: only concordant kept."""
    upstream_rows = [
        # Concordant duplicate — should collapse to 1
        _episode("A1", "CIPRO", "RESISTANT"),
        _episode("A1", "CIPRO", "RESISTANT"),
        # Discordant — should be excluded (0 rows)
        _episode("A2", "CIPRO", "RESISTANT"),
        _episode("A2", "CIPRO", "SUSCEPTIBLE"),
        # Unique — should pass through unchanged (1 row)
        _episode("A3", "CIPRO", "SUSCEPTIBLE"),
    ]
    downstream_rows = [
        _downstream("A1", "MEROPENEM"),
        _downstream("A2", "MEROPENEM"),
        _downstream("A3", "MEROPENEM"),
    ]
    pairs = generator.build(pd.DataFrame(upstream_rows), pd.DataFrame(downstream_rows))

    assert len(pairs[pairs["anon_id"] == "A1"]) == 1, "Concordant duplicate should collapse to 1"
    assert len(pairs[pairs["anon_id"] == "A2"]) == 0, "Discordant should be excluded"
    assert len(pairs[pairs["anon_id"] == "A3"]) == 1, "Unique row should pass through"
