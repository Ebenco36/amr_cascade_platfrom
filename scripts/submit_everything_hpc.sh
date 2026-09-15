#!/usr/bin/env bash
# One entry point for the full submission wave: primary run, all three
# availability-sensitivity runs, and the era-stratified follow-up -- the
# follow-up is gated on the primary run's own cascade-merge job via SLURM's
# --dependency=afterok, not a wall-clock wait, so this script finishes as
# soon as everything is queued (seconds) rather than blocking until the
# jobs actually complete. Run with no options for the default full wave.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_everything_hpc.sh [options]

With no options, submits all four pieces of the full wave:
  1. Primary run (--fix-stale --run-publication-package)
  2. Availability sensitivity: support >= 5
  3. Availability sensitivity: 1-year era
  4. Availability sensitivity: joint (both at once)
  then queues the era-stratified follow-up, held on job 1's cascade-merge
  job via SLURM --dependency=afterok (no wall-clock wait).

Cohort/hardware pass-through (forwarded to every real submission below):
  --sites SITE1,SITE2          Same as submit_pipeline_dag_hpc.sh's own flag.
  --organisms O1,O2            Same. A single organism is also used for the
                                era-stratified follow-up's --organism; with
                                more than one, the follow-up is skipped with
                                a warning (ambiguous which one to check).
  --partition NAME
  --cascade-partition NAME     Also used as the era-stratified job's partition.
  --python-bin PATH
  --force-rerun-existing

Main-run-only pass-through:
  --delete-venv                Only applied once, on the primary run -- the
                                venv is shared, not per sensitivity.
  --eskape-targets T1,T2

Turn individual pieces on/off:
  --skip-main                  Don't submit the primary publication-package run.
  --skip-support-sensitivity   Don't submit the support>=5 sensitivity.
  --skip-annual-sensitivity    Don't submit the 1-year-era sensitivity.
  --skip-joint-sensitivity     Don't submit the joint sensitivity.
  --skip-era-stratified        Don't queue the era-stratified follow-up.
  --main-only                  Shorthand: skip all three sensitivities and
                                the era-stratified follow-up.
  --sensitivities-only         Shorthand: skip the primary run only.
  --era-stratified-after JOBID Use this job ID as the era-stratified
                                follow-up's dependency instead of the one
                                auto-detected from the primary run's output
                                (needed if --skip-main is set and the primary
                                run's cascade already finished earlier).

Era-stratified follow-up budget (not pre-calibrated -- adjust if needed):
  --era-stratified-mem SIZE    Default: 128G
  --era-stratified-time HH:MM:SS  Default: 08:00:00

  --dry-run                    Forward --dry-run to every real submission
                                below, and preview the final sbatch command
                                instead of actually calling sbatch.
  --help                       Show this message.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

SITES=""
ORGANISMS=""
PARTITION=""
CASCADE_PARTITION=""
PYTHON_BIN=""
FORCE_RERUN_EXISTING=0
DELETE_VENV=0
ESKAPE_TARGETS=""
SKIP_MAIN=0
SKIP_SUPPORT_SENSITIVITY=0
SKIP_ANNUAL_SENSITIVITY=0
SKIP_JOINT_SENSITIVITY=0
SKIP_ERA_STRATIFIED=0
ERA_STRATIFIED_AFTER=""
ERA_STRATIFIED_MEM="128G"
ERA_STRATIFIED_TIME="08:00:00"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites) [[ $# -ge 2 ]] || { echo "--sites requires a value" >&2; exit 2; }; SITES="$2"; shift 2 ;;
    --organisms) [[ $# -ge 2 ]] || { echo "--organisms requires a value" >&2; exit 2; }; ORGANISMS="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || { echo "--partition requires a value" >&2; exit 2; }; PARTITION="$2"; shift 2 ;;
    --cascade-partition) [[ $# -ge 2 ]] || { echo "--cascade-partition requires a value" >&2; exit 2; }; CASCADE_PARTITION="$2"; shift 2 ;;
    --python-bin) [[ $# -ge 2 ]] || { echo "--python-bin requires a value" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    --force-rerun-existing) FORCE_RERUN_EXISTING=1; shift ;;
    --delete-venv) DELETE_VENV=1; shift ;;
    --eskape-targets) [[ $# -ge 2 ]] || { echo "--eskape-targets requires a value" >&2; exit 2; }; ESKAPE_TARGETS="$2"; shift 2 ;;
    --skip-main) SKIP_MAIN=1; shift ;;
    --skip-support-sensitivity) SKIP_SUPPORT_SENSITIVITY=1; shift ;;
    --skip-annual-sensitivity) SKIP_ANNUAL_SENSITIVITY=1; shift ;;
    --skip-joint-sensitivity) SKIP_JOINT_SENSITIVITY=1; shift ;;
    --skip-era-stratified) SKIP_ERA_STRATIFIED=1; shift ;;
    --main-only) SKIP_SUPPORT_SENSITIVITY=1; SKIP_ANNUAL_SENSITIVITY=1; SKIP_JOINT_SENSITIVITY=1; SKIP_ERA_STRATIFIED=1; shift ;;
    --sensitivities-only) SKIP_MAIN=1; shift ;;
    --era-stratified-after) [[ $# -ge 2 ]] || { echo "--era-stratified-after requires a value" >&2; exit 2; }; ERA_STRATIFIED_AFTER="$2"; shift 2 ;;
    --era-stratified-mem) [[ $# -ge 2 ]] || { echo "--era-stratified-mem requires a value" >&2; exit 2; }; ERA_STRATIFIED_MEM="$2"; shift 2 ;;
    --era-stratified-time) [[ $# -ge 2 ]] || { echo "--era-stratified-time requires a value" >&2; exit 2; }; ERA_STRATIFIED_TIME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Shared pass-through flags, built once and forwarded identically to every
# real submit_pipeline_dag_hpc.sh call below (sensitivity runs included --
# they should target the same cohort/hardware as the primary run).
common_args=()
[[ -n "${SITES}" ]] && common_args+=(--sites "${SITES}")
[[ -n "${ORGANISMS}" ]] && common_args+=(--organisms "${ORGANISMS}")
[[ -n "${PARTITION}" ]] && common_args+=(--partition "${PARTITION}")
[[ -n "${CASCADE_PARTITION}" ]] && common_args+=(--cascade-partition "${CASCADE_PARTITION}")
[[ -n "${PYTHON_BIN}" ]] && common_args+=(--python-bin "${PYTHON_BIN}")
[[ "${FORCE_RERUN_EXISTING}" -eq 1 ]] && common_args+=(--force-rerun-existing)
[[ "${DRY_RUN}" -eq 1 ]] && common_args+=(--dry-run)

main_args=("${common_args[@]}")
[[ "${DELETE_VENV}" -eq 1 ]] && main_args+=(--delete-venv)
[[ -n "${ESKAPE_TARGETS}" ]] && main_args+=(--eskape-targets "${ESKAPE_TARGETS}")

merge_job_id=""

if [[ "${SKIP_MAIN}" -eq 0 ]]; then
  echo "=== [1/4] Primary run: --fix-stale --run-publication-package ==="
  main_output="$(bash scripts/submit_pipeline_dag_hpc.sh --fix-stale --run-publication-package "${main_args[@]}")"
  printf '%s\n' "${main_output}"
  # merge_cascade_validation_shards.py is what actually writes
  # validation_results.parquet (run_cascade_analysis.py, the job after it,
  # only reads/detects it) -- gate on this job specifically so the
  # era-stratified follow-up starts as soon as its real input exists.
  merge_job_id="$(printf '%s\n' "${main_output}" | grep -E "^Cascade validation merge " | tail -1 | awk -F': ' '{print $2}' || true)"
else
  echo "=== [1/4] Primary run: skipped (--skip-main) ==="
fi

setup_sensitivity_symlinks() {
  local data_root="$1"
  if [[ ! -d "${data_root}" ]]; then
    mkdir -p "${data_root}"
    for d in raw sample bronze silver harmonized metadata reference interim features; do
      ln -s "${PROJECT_ROOT}/data/${d}" "${data_root}/${d}"
    done
  fi
  mkdir -p "${data_root}/gold" "${data_root}/artifacts"
}

echo ""
if [[ "${SKIP_SUPPORT_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [2/4] Availability sensitivity: support >= 5 ==="
  setup_sensitivity_symlinks "data_availability_sensitivity"
  bash scripts/submit_pipeline_dag_hpc.sh \
    --env hpc_availability_sensitivity --data-root data_availability_sensitivity \
    --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity \
    --skip-prevalence --skip-reporting "${common_args[@]}"
else
  echo "=== [2/4] Availability sensitivity: support >= 5 -- skipped ==="
fi

echo ""
if [[ "${SKIP_ANNUAL_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [3/4] Availability sensitivity: 1-year era ==="
  setup_sensitivity_symlinks "data_availability_annual_sensitivity"
  bash scripts/submit_pipeline_dag_hpc.sh \
    --env hpc_availability_annual_sensitivity --data-root data_availability_annual_sensitivity \
    --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity \
    --skip-prevalence --skip-reporting "${common_args[@]}"
else
  echo "=== [3/4] Availability sensitivity: 1-year era -- skipped ==="
fi

echo ""
if [[ "${SKIP_JOINT_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [4/4] Availability sensitivity: joint (support >= 5 within 1-year era) ==="
  setup_sensitivity_symlinks "data_availability_joint_sensitivity"
  bash scripts/submit_pipeline_dag_hpc.sh \
    --env hpc_availability_joint_sensitivity --data-root data_availability_joint_sensitivity \
    --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity \
    --skip-prevalence --skip-reporting "${common_args[@]}"
else
  echo "=== [4/4] Availability sensitivity: joint -- skipped ==="
fi

echo ""
if [[ "${SKIP_ERA_STRATIFIED}" -eq 1 ]]; then
  echo "Era-stratified follow-up: skipped (--skip-era-stratified)."
  echo ""
  echo "All submissions queued. Check status with: squeue -u \$USER"
  exit 0
fi

dependency_job="${ERA_STRATIFIED_AFTER:-${merge_job_id}}"

era_organism="ESCHERICHIA COLI"
if [[ -n "${ORGANISMS}" ]]; then
  if [[ "${ORGANISMS}" == *,* ]]; then
    echo "Era-stratified follow-up: skipped -- --organisms lists more than one"
    echo "organism (${ORGANISMS}), and the follow-up needs exactly one. Submit it"
    echo "yourself per organism with --era-stratified-after JOBID."
    echo ""
    echo "All submissions queued. Check status with: squeue -u \$USER"
    exit 0
  fi
  era_organism="${ORGANISMS}"
fi

era_partition="${CASCADE_PARTITION:-long}"
era_wrap="cd '${PROJECT_ROOT}' && .venv/bin/python scripts/run_era_stratified_permutation_sensitivity.py --env hpc --gold-scope combined --organism '${era_organism}'"

if [[ -z "${dependency_job}" ]]; then
  echo "WARNING: no cascade-merge job ID available (primary run was skipped or its"
  echo "output could not be parsed, and no --era-stratified-after was given)."
  echo "Queuing the era-stratified follow-up WITHOUT a dependency -- it will start"
  echo "immediately and fail fast if validation_results.parquet does not exist yet."
  dependency_args=()
else
  dependency_args=(--dependency "afterok:${dependency_job}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "=== [dry-run] Would queue era-stratified follow-up ==="
  echo "  sbatch --parsable --partition=${era_partition} --job-name=amr_era_stratified_sensitivity \\"
  echo "    --output=${PROJECT_ROOT}/logs/era_stratified_sensitivity_%j.out \\"
  echo "    --error=${PROJECT_ROOT}/logs/era_stratified_sensitivity_%j.err \\"
  echo "    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=${ERA_STRATIFIED_MEM} --time=${ERA_STRATIFIED_TIME} \\"
  [[ -n "${dependency_job}" ]] && echo "    --dependency=afterok:${dependency_job} \\"
  echo "    --wrap='${era_wrap}'"
else
  echo "=== Queuing era-stratified follow-up${dependency_job:+, held until job ${dependency_job} succeeds} ==="
  era_job_id="$(
    sbatch --parsable --partition="${era_partition}" --job-name=amr_era_stratified_sensitivity \
      --output="${PROJECT_ROOT}/logs/era_stratified_sensitivity_%j.out" \
      --error="${PROJECT_ROOT}/logs/era_stratified_sensitivity_%j.err" \
      --nodes=1 --ntasks=1 --cpus-per-task=8 --mem="${ERA_STRATIFIED_MEM}" --time="${ERA_STRATIFIED_TIME}" \
      "${dependency_args[@]}" \
      --wrap="${era_wrap}"
  )"
  echo "Era-stratified job queued: ${era_job_id}"
fi

echo ""
echo "All submissions queued. Check status with: squeue -u \$USER"
