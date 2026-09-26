import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amr_cascade_platform.data.replica import ast_chain_model as chain
from amr_cascade_platform.data.replica.raw_formats import (
    detect_time_format,
    generate_identifiers,
    numeric_spec,
    render_numbers,
    render_times,
)
from amr_cascade_platform.data.replica.raw_profiler import ProfileOptions, RawProfiler
from amr_cascade_platform.data.replica.replica_generator import GenerationOptions, ReplicaGenerator


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ("2009-11-27 02:52:00+00:00", "space_offset"),
        ("2017-08-14 22:40:00 UTC", "space_utc"),
        ("2020-09-11T06:42:00.000Z", "t_millis_z"),
        ("2008-01-16 18:30:00", "space_naive"),
    ],
)
def test_time_formats_round_trip(value, key):
    assert detect_time_format(["Null", value]) == key
    stamp = pd.Series(pd.to_datetime([value], utc=True, format="mixed").tz_localize(None))
    assert render_times(stamp, key).iloc[0] == value


def test_numeric_spellings_are_reproduced():
    rng = np.random.default_rng(0)
    assert numeric_spec(pd.Series(["12", "Null", "-3"]))["kind"] == "int"
    assert numeric_spec(pd.Series(["Glaucoma", "Asthma"]))["kind"] == "text"
    spec = numeric_spec(pd.Series(["1949.0", "12.0", "Null"]))
    rendered = render_numbers(np.array([1949.0, np.nan]), spec, rng, "Null")
    assert rendered.tolist() == ["1949.0", "Null"]
    float32 = render_numbers(np.array([98.4]), {"kind": "float", "decimals": 1, "float32_share": 1.0}, rng, "")
    assert float32.tolist() == ["98.4000015258789"]
    assert render_numbers(np.array([np.nan]), {"kind": "int"}, rng, "").tolist() == [""]


def test_identifiers_keep_the_raw_shape_but_are_marked_synthetic():
    rng = np.random.default_rng(1)
    used: set[str] = set()
    letters = generate_identifiers(500, [["LLDDDDDDD", 1.0]], rng, used=used)
    digits = generate_identifiers(500, [["DDDDDDDD", 1.0]], rng)
    assert len(set(letters)) == 500 and used == set(letters)
    assert all(len(v) == 9 and v[0] == "Z" and v[1].isalpha() and v[2:].isdigit() for v in letters)
    assert all(len(v) == 8 and v[0] == "9" and v.isdigit() for v in digits)


def _cascade_episodes(n: int, rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Drug A always reported; drug B mostly reported after A is resistant; B only from 2015."""
    years = rng.choice([2012, 2014, 2016, 2018], size=n)
    a_resistant = rng.random(n) < 0.3
    states = np.zeros((n, 2), dtype=np.int8)
    states[:, 0] = np.where(a_resistant, chain.STATE_RESISTANT, chain.STATE_SUSCEPTIBLE)
    b_reported = (years >= 2015) & (rng.random(n) < np.where(a_resistant, 0.8, 0.1))
    states[b_reported, 1] = np.where(rng.random(int(b_reported.sum())) < 0.2, chain.STATE_RESISTANT, chain.STATE_SUSCEPTIBLE)
    context = pd.DataFrame({"year": years, "culture": "URINE", "mode": "Inpatient"})
    return states, context


def test_chain_model_reproduces_result_conditioned_reporting_and_availability():
    rng = np.random.default_rng(2)
    states, context = _cascade_episodes(20_000, rng)
    model = chain.fit_chain(states, ["A", "B"], context, {}, min_cell=11)
    assert [drug["label"] for drug in model["drugs"]] == ["A", "B"]
    assert model["drugs"][1]["available_years"] == [2016, 2018]
    sampled = chain.sample_chain(model, context, np.random.default_rng(3))
    a_r = sampled[:, 0] == chain.STATE_RESISTANT
    b = sampled[:, 1] > 0
    late = context["year"].to_numpy() >= 2015
    assert not b[~late].any()
    assert b[late & a_r].mean() == pytest.approx(0.8, abs=0.04)
    assert b[late & ~a_r].mean() == pytest.approx(0.1, abs=0.03)


def test_episode_states_collapse_duplicates_to_the_most_resistant_result():
    rows = pd.DataFrame(
        {
            "episode": ["e1", "e1", "e1", "e2", "e3"],
            "antibiotic": ["A", "A", "B", "A", "Null"],
            "susceptibility": ["Susceptible", "Resistant", "Inconclusive", "Susceptible", "Null"],
        }
    )
    episodes, drugs, states, others = chain.episode_states(rows, ["episode"])
    lookup = dict(zip(episodes["episode"], range(len(episodes)), strict=True))
    assert drugs == ["A", "B"]
    assert states[lookup["e1"]].tolist() == [chain.STATE_RESISTANT, chain.STATE_OTHER]
    assert states[lookup["e2"]].tolist() == [chain.STATE_SUSCEPTIBLE, chain.STATE_NOT_REPORTED]
    assert states[lookup["e3"]].tolist() == [0, 0]
    assert others == {"B": {"Inconclusive": 1}}


def _write_raw_site(root: Path, rng: np.random.Generator) -> None:
    """A small raw site with the real layout and spellings (UTSW-like)."""
    site = root / "raw" / "site_x"
    site.mkdir(parents=True)
    orders = []
    for index in range(600):
        orders.append(
            {
                "anon_id": f"AB{1000 + index // 2:07d}",
                "pat_enc_csn_id_coded": f"{5000000 + index // 2}",
                "order_proc_id_coded": f"{8000000 + index}",
                "order_time_jittered": f"{2014 + index % 6}-0{1 + index % 9}-1{index % 10} 0{index % 10}:15:00",
            }
        )
    rows = []
    for index, order in enumerate(orders):
        positive = index % 3 != 0
        base = {**order, "ordering_mode": ["Inpatient", "Outpatient"][index % 2], "culture_description": ["URINE", "BLOOD"][index % 2], "was_positive": "1" if positive else "0"}
        if not positive:
            rows.append({**base, "organism": "Null", "antibiotic": "Null", "susceptibility": "Null"})
            continue
        resistant = rng.random() < 0.3
        rows.append({**base, "organism": "ESCHERICHIA COLI", "antibiotic": "Ceftriaxone", "susceptibility": "Resistant" if resistant else "Susceptible"})
        if rng.random() < (0.8 if resistant else 0.1):
            rows.append({**base, "organism": "ESCHERICHIA COLI", "antibiotic": "Meropenem", "susceptibility": "Susceptible"})
    pd.DataFrame(rows).to_csv(site / "cohort.csv", index=False)
    pd.DataFrame(
        [{**{k: o[k] for k in ("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded")}, "age": "65-74", "gender": ["0", "1"][i % 2]} for i, o in enumerate(orders)]
    ).to_csv(site / "demographics.csv", index=False)
    pd.DataFrame(
        [
            {**o, "comorbidity_component": ["Asthma", "Hypertension", "Glaucoma"][j], "comorbidity_component_start_days_culture": str(10 * j), "comorbidity_component_end_days_culture": "Null"}
            for o in orders
            for j in range(3)
        ]
    ).to_csv(site / "comorbidity.csv", index=False)
    reference = root / "reference"
    reference.mkdir()
    (reference / "canonical_table_map.yaml").write_text(
        "sites:\n  site_x:\n    cohort: cohort.csv\n    demographics: demographics.csv\n    comorbidity: comorbidity.csv\n",
        encoding="utf-8",
    )


def test_profile_then_generate_reproduces_layout_without_copying_records(tmp_path):
    rng = np.random.default_rng(4)
    _write_raw_site(tmp_path, rng)
    table_map = {"site_x": {"cohort": "cohort.csv", "demographics": "demographics.csv", "comorbidity": "comorbidity.csv"}}
    profile = RawProfiler(tmp_path / "raw", table_map, ProfileOptions(min_cell=5, chain_min_episodes=50, sample_rows=10_000)).profile()
    json.dumps(profile)  # the profile must be plain JSON
    output = tmp_path / "replica"
    manifest = ReplicaGenerator(profile, GenerationOptions(scale=1.0, seed=5)).write(
        output_root=output, reference_source=tmp_path / "reference", force=False
    )
    real = pd.read_csv(tmp_path / "raw" / "site_x" / "cohort.csv", dtype=str, keep_default_na=False)
    synthetic = pd.read_csv(output / "raw" / "site_x" / "cohort.csv", dtype=str, keep_default_na=False)
    assert list(synthetic.columns) == list(real.columns)
    assert detect_time_format(synthetic["order_time_jittered"]) == "space_naive"
    assert set(synthetic["anon_id"]).isdisjoint(real["anon_id"])
    assert synthetic["anon_id"].str.startswith("Z").all()
    assert set(synthetic["antibiotic"]) <= {"Ceftriaxone", "Meropenem", "Null"}
    # Meropenem stays result-conditioned on ceftriaxone.
    episodes = synthetic.loc[synthetic["organism"].ne("Null")]
    wide = episodes.pivot_table(index="order_proc_id_coded", columns="antibiotic", values="susceptibility", aggfunc="first")
    meropenem = wide["Meropenem"].notna()
    ceftriaxone_r = wide["Ceftriaxone"].eq("Resistant")
    assert meropenem[ceftriaxone_r].mean() > 3 * meropenem[~ceftriaxone_r].mean()
    # Every covariate row joins to a synthetic culture order.
    demographics = pd.read_csv(output / "raw" / "site_x" / "demographics.csv", dtype=str, keep_default_na=False)
    assert set(demographics["order_proc_id_coded"]) <= set(synthetic["order_proc_id_coded"])
    comorbidity = pd.read_csv(output / "raw" / "site_x" / "comorbidity.csv", dtype=str, keep_default_na=False)
    assert (comorbidity["comorbidity_component_end_days_culture"] == "Null").all()
    assert (output / "reference" / "canonical_table_map.yaml").exists()
    assert manifest["row_counts"]["site_x"]["cohort"] == len(synthetic)
    with pytest.raises(FileExistsError):
        ReplicaGenerator(profile, GenerationOptions(scale=1.0)).write(output_root=output, reference_source=tmp_path / "reference", force=False)


def test_generator_refuses_to_write_over_the_real_raw_data(tmp_path):
    (tmp_path / "data" / "reference").mkdir(parents=True)
    with pytest.raises(ValueError, match="real raw data"):
        ReplicaGenerator({"sites": {}}, GenerationOptions()).write(
            output_root=tmp_path / "data", reference_source=tmp_path / "data" / "reference", force=True
        )
