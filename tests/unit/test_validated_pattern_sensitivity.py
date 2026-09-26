import pandas as pd
import pytest

from amr_cascade_platform.cascade.workflows import validated_pattern_sensitivity as vps


def _edges(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "upstream_antibiotic": [f"UP{i:03d}" for i in range(n)],
            "downstream_antibiotic": ["DOWN"] * n,
            "escalation_ratio": [2.0] * n,
        }
    )


def _keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["upstream_antibiotic"], frame["downstream_antibiotic"]))


@pytest.mark.parametrize("n_edges, n_shards", [(37, 16), (541, 16), (5, 16), (16, 16), (0, 4)])
def test_shards_cover_every_pattern_exactly_once(n_edges: int, n_shards: int) -> None:
    edges = _edges(n_edges)

    covered = [key for index in range(n_shards) for key in _keys(vps.shard_slice(edges, index, n_shards))]

    assert sorted(covered) == sorted(_keys(edges))


def test_shard_assignment_does_not_depend_on_input_row_order() -> None:
    edges = _edges(40)

    assert vps.shard_slice(edges, 2, 7).equals(vps.shard_slice(edges.iloc[::-1], 2, 7))


def test_merge_accepts_an_exact_partition_and_refuses_gaps_or_overlaps() -> None:
    edges = _edges(6)
    expected = vps.expected_keys(edges)

    assert len(vps.merge_shard_frames([edges.iloc[:3], edges.iloc[3:]], expected)) == 6
    with pytest.raises(ValueError, match="1 missing"):
        vps.merge_shard_frames([edges.iloc[:5]], expected)
    with pytest.raises(ValueError, match="more than one shard"):
        vps.merge_shard_frames([edges, edges.iloc[:1]], expected)


def test_merge_of_all_empty_shards_is_empty_when_nothing_was_selected() -> None:
    empty = _edges(0)

    assert vps.merge_shard_frames([empty, empty], set()).empty


def test_each_analysis_runs_on_the_population_the_supplement_describes() -> None:
    validation = pd.DataFrame(
        {
            "upstream_antibiotic": ["B", "A", "C", "D"],
            "downstream_antibiotic": ["X", "X", "X", "X"],
            "observed_escalation_ratio": [2.0, 3.0, 0.5, 1.2],
            "validation_status": ["robust", "supported", "robust", "mixed"],
        }
    )

    era = vps.select_edges(validation, vps.ANALYSES["era-stratified"].statuses)
    patient = vps.select_edges(validation, vps.ANALYSES["patient-cluster"].statuses)

    assert list(era["upstream_antibiotic"]) == ["A", "B", "C"]
    assert list(patient["upstream_antibiotic"]) == ["B", "C"]
    assert "escalation_ratio" in patient.columns and "observed_escalation_ratio" not in patient.columns
