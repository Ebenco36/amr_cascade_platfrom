"""Build a small, site-stratified E. coli gold sample that runs fully on a laptop.

Semi-joins culture_episodes / eligible_pairs / drug_pair_episodes down to a random
subset of episodes (stratified by source_site so all three sites stay represented,
which the cascade validation analyzer's site-replication checks need). Uses DuckDB
so the 151M-row drug_pair_episodes.parquet is filtered out-of-core rather than
loaded into pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

GOLD_ROOT = Path(__file__).resolve().parents[1] / "data" / "gold" / "combined" / "organisms"
SOURCE_DIR = GOLD_ROOT / "escherichia_coli"
EPISODE_KEY_COLUMNS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "order_time_jittered",
    "organism",
    "source_site",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-name", default="escherichia_coli_sample")
    args = parser.parse_args()

    out_dir = GOLD_ROOT / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    seed_fraction = (args.seed % 2**31) / 2**31
    con.execute("SELECT setseed(?)", [seed_fraction])

    culture_path = SOURCE_DIR / "culture_episodes.parquet"
    eligible_path = SOURCE_DIR / "eligible_pairs.parquet"
    pairs_path = SOURCE_DIR / "drug_pair_episodes.parquet"

    total = con.execute(f"SELECT count(*) FROM read_parquet('{culture_path}')").fetchone()[0]
    fraction = min(1.0, args.n_episodes / total)
    print(f"Sampling ~{args.n_episodes} of {total} episodes (fraction={fraction:.4f}), stratified by source_site")

    join_cols = " AND ".join(f"e.{c} = k.{c}" for c in EPISODE_KEY_COLUMNS)
    key_select = ", ".join(EPISODE_KEY_COLUMNS)

    con.execute(
        f"""
        CREATE TEMP TABLE sample_keys AS
        SELECT {key_select} FROM (
            SELECT *, row_number() OVER (PARTITION BY source_site ORDER BY random()) AS rn,
                   count(*) OVER (PARTITION BY source_site) AS site_n
            FROM read_parquet('{culture_path}')
        )
        WHERE rn <= greatest(1, CAST(site_n * {fraction} AS INTEGER))
        """
    )
    sampled_n = con.execute("SELECT count(*) FROM sample_keys").fetchone()[0]
    print(f"Sampled {sampled_n} episodes")

    con.execute(
        f"""
        COPY (
            SELECT e.* FROM read_parquet('{culture_path}') e
            JOIN sample_keys k ON {join_cols}
        ) TO '{out_dir / "culture_episodes.parquet"}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT e.* FROM read_parquet('{eligible_path}') e
            JOIN sample_keys k ON {join_cols}
        ) TO '{out_dir / "eligible_pairs.parquet"}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT e.* FROM read_parquet('{pairs_path}') e
            JOIN sample_keys k ON {join_cols}
        ) TO '{out_dir / "drug_pair_episodes.parquet"}' (FORMAT PARQUET)
        """
    )

    for name in ("culture_episodes.parquet", "eligible_pairs.parquet", "drug_pair_episodes.parquet"):
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out_dir / name}')").fetchone()[0]
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()
