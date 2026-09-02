#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_supplementary_prediction_hpc.sh [options]

Queue scripts/run_supplementary_prediction.py as a single SLURM job, using the
same environment setup, logging convention, and job-naming pattern as every
stage submitted by scripts/submit_pipeline_dag_hpc.sh.

This does NOT rebuild features or retrain models — it only reads artefacts
that already exist on disk:
  data/artifacts/modeling/downstream_testing/combined/organisms/<organism>/site__all_models/{metrics,threshold_metrics}.parquet
  data/artifacts/modeling/downstream_testing/combined/organisms/<organism>/site__all_models/full/{model}_predictions.parquet
  data/features/combined/organisms/<organism>/model_ready_pair_features.parquet
(plus one LR refit on the training split, for coefficients only). If any of
these are missing, run the feature/training stages first via
submit_pipeline_dag_hpc.sh --run-features --run-training.

Options:
  --organism NAME       Default: ESCHERICHIA COLI
  --partition NAME      Default: main
  --mem SIZE            Default: 128G   (env: SUPP_PRED_MEM)
  --time HH:MM:SS       Default: 04:00:00 (env: SUPP_PRED_TIME)
  --cpus N              Default: 8      (env: CPUS_PER_TASK)
  --dependency JOBID    Optional: chain after another SLURM job (e.g. a
                         training job submitted in the same session)
  --python-bin PATH     Default: <project>/.venv/bin/python
  --dry-run             Print the sbatch command without submitting
  --help                Show this message

Examples:
  # Inputs already on disk (the common case) — just queue it:
  bash scripts/submit_supplementary_prediction_hpc.sh

  # See the exact sbatch command first:
  bash scripts/submit_supplementary_prediction_hpc.sh --dry-run

  # Chain after a training job submitted in the same tmux session:
  bash scripts/submit_supplementary_prediction_hpc.sh --dependency 123456
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

ORGANISM="ESCHERICHIA COLI"
PARTITION="main"
MEM="${SUPP_PRED_MEM:-128G}"
TIME_LIMIT="${SUPP_PRED_TIME:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
DEPENDENCY=""
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DRY_RUN=0

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --organism) [[ $# -ge 2 ]] || die "--organism requires a value"; ORGANISM="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || die "--partition requires a value"; PARTITION="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "--mem requires a value"; MEM="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "--time requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "--cpus requires a value"; CPUS_PER_TASK="$2"; shift 2 ;;
    --dependency) [[ $# -ge 2 ]] || die "--dependency requires a value"; DEPENDENCY="$2"; shift 2 ;;
    --python-bin) [[ $# -ge 2 ]] || die "--python-bin requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

organism_slug="$(slugify "${ORGANISM}")"
job_name="amr_${organism_slug}_supp_pred"

command="\"\${AMR_CASCADE_PYTHON}\" scripts/run_supplementary_prediction.py --organism '${ORGANISM}'"

args=(
  --parsable
  --partition="${PARTITION}"
  --job-name="${job_name}"
  --output="${LOG_DIR}/${job_name}_%j.out"
  --error="${LOG_DIR}/${job_name}_%j.err"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --kill-on-invalid-dep=yes
)
[[ -n "${DEPENDENCY}" ]] && args+=(--dependency="afterok:${DEPENDENCY}")

# Same wrapper convention as submit_job() in submit_pipeline_dag_hpc.sh: verify
# the venv Python before running, pin BLAS/OpenMP thread counts to the job's
# CPU allocation, and give matplotlib a writable config dir.
args+=(--wrap "cd '${PROJECT_ROOT}' && export PYTHONNOUSERSITE=1 && export MPLCONFIGDIR='${PROJECT_ROOT}/.cache/matplotlib' && mkdir -p \"\${MPLCONFIGDIR}\" && export AMR_CASCADE_PYTHON='${PYTHON_BIN}' && \"\${AMR_CASCADE_PYTHON}\" scripts/check_python_runtime.py || { echo \"ERROR: venv Python not working on \$(hostname): \${AMR_CASCADE_PYTHON}\" >&2; exit 127; } && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS='${CPUS_PER_TASK}' && export OPENBLAS_NUM_THREADS='${CPUS_PER_TASK}' && export MKL_NUM_THREADS='${CPUS_PER_TASK}' && ${command}")

cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'sbatch'
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

job_id="$(sbatch "${args[@]}")"
echo "Submitted ${job_name}: job ${job_id}"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/${job_name}_${job_id}.out"
echo
echo "On completion, outputs land in:"
echo "  outputs/figures/combined/organisms/${organism_slug}/supp_pred_fig{1..6}_*.png"
echo "  outputs/tables/combined/organisms/${organism_slug}/supp_pred_table{1..3}_*.csv"
