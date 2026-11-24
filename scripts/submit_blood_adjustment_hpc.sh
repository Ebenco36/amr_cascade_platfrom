#!/usr/bin/env bash
# Covariate-adjusted models for the patterns validated in blood cultures (paper 1, Supplementary Note "Blood-culture analysis").
# Fits the primary analysis's pair-specific adjusted logistic models to the 188 blood-validated pairs, reusing the blood
# validation run made on the Mac (copied to this checkout, see below), with the final run's harmonised covariate tables.

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $arg (use --dry-run)" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
BLOOD_RUN="data/blood_check/blood_run"
OUT="data/blood_check/blood_adjusted"
mkdir -p logs

# The job must use exactly the data of the final run (archive 20261101_164213): check before submitting.
"$PY" - "$BLOOD_RUN" <<'PYEOF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
problems = []
blood_run = Path(sys.argv[1])
for name in ("validation_results.parquet", "retained_edges.parquet"):
    if not (blood_run / name).exists():
        problems.append(f"{blood_run / name} missing: copy the blood run from the Mac first")
pairs = Path("data/gold/combined/organisms/escherichia_coli/drug_pair_episodes.parquet")
if not pairs.exists() or pairs.stat().st_size != 115967231 or pq.ParquetFile(pairs).metadata.num_rows != 111247965:
    problems.append(f"{pairs} is not the final run's file (expected 115,967,231 bytes and 111,247,965 rows)")
expected = {"armd": 118901224, "armd_ecuh": 13566752, "armd_utsw": 13794973}  # harmonization_run.json of the final run
for site, rows in expected.items():
    f = Path("data/harmonized/site_aligned") / site / "comorbidity.parquet"
    if not f.exists() or pq.ParquetFile(f).metadata.num_rows != rows:
        problems.append(f"{f} is not the final run's table (expected {rows:,} rows)")
if problems:
    print("Not submitted:\n  " + "\n  ".join(problems), file=sys.stderr); sys.exit(1)
print("Checks passed: final-run gold pairs and harmonised comorbidity tables; blood run present.")
PYEOF

dirty="$(git status --porcelain -- src scripts configs pyproject.toml requirements-hpc.txt)"
if [[ -n "$dirty" && $DRY_RUN -eq 0 ]]; then
  echo "Commit the code first (uncommitted changes in src/scripts/configs). Nothing submitted." >&2
  exit 1
fi
EXPECTED_COMMIT="$(git rev-parse HEAD)"
WRAP="cd '$ROOT' && { test \"\$(git rev-parse HEAD)\" = '$EXPECTED_COMMIT' || { echo 'ERROR: checkout commit changed after submission; expected $EXPECTED_COMMIT.' >&2; exit 1; }; } && unset PYTHONHOME PYTHONPATH && export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR='$ROOT/.cache/matplotlib' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 && '$PY'"
COMMAND="scripts/run_blood_culture_validation.py --env hpc --run-root . --adjust --validation-dir $BLOOD_RUN --out-dir $OUT"
ARGS=(--parsable --cpus-per-task=8 --partition=main --job-name=blood_adjustment --mem=128G --time=04:00:00 --output="logs/blood_adjustment_%j.out")
if [[ $DRY_RUN -eq 1 ]]; then
  echo "sbatch ${ARGS[*]} --wrap \"$WRAP $COMMAND\""
else
  job=$(sbatch "${ARGS[@]}" --wrap "$WRAP $COMMAND")
  echo "Submitted blood_adjustment as job $job (log: logs/blood_adjustment_${job}.out)"
fi
