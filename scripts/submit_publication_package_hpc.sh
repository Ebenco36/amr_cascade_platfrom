#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_publication_package_hpc.sh [options]

Submit the full HPC publication package:
  1. rebuild the end-to-end ARMD pipeline,
  2. run feature build and downstream-testing training,
  3. run ESKAPE-family validation,
  4. regenerate all publication tables and all registered figure families,
  5. run scientific audit and publication-readiness audit.

Defaults are intentionally strict for final production:
  - main partition for standard pipeline stages
  - long partition for cascade/features/training/ESKAPE stages
  - all figure formats: html, png, svg, pdf
  - force rerun of existing outputs

Options:
  --sites SITE1,SITE2          Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2            Default: ESCHERICHIA COLI
  --partition NAME             Default: main
  --cascade-partition NAME     Default: long
  --python-bin PATH            Passed through to submit_pipeline_dag_hpc.sh
  --delete-venv                Rebuild the Python venv before submitting jobs
  --resume-existing            Reuse completed outputs instead of forcing rerun
  --dry-run                    Print sbatch commands without submitting
  --wait                       Monitor jobs after submission
  --monitor-interval N         Default: 60
  --help                       Show this message
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SITES="armd,armd_ecuh,armd_utsw"
ORGANISMS="ESCHERICHIA COLI"
PARTITION="main"
CASCADE_PARTITION_VALUE="long"
PYTHON_BIN=""
FORCE_FLAG="--force-rerun-existing"
DELETE_VENV=0
DRY_RUN=0
WAIT_FOR_JOBS=0
MONITOR_INTERVAL=60

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites) [[ $# -ge 2 ]] || die "--sites requires a value"; SITES="$2"; shift 2 ;;
    --organisms) [[ $# -ge 2 ]] || die "--organisms requires a value"; ORGANISMS="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || die "--partition requires a value"; PARTITION="$2"; shift 2 ;;
    --cascade-partition) [[ $# -ge 2 ]] || die "--cascade-partition requires a value"; CASCADE_PARTITION_VALUE="$2"; shift 2 ;;
    --python-bin) [[ $# -ge 2 ]] || die "--python-bin requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --delete-venv) DELETE_VENV=1; shift ;;
    --resume-existing) FORCE_FLAG=""; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --wait) WAIT_FOR_JOBS=1; shift ;;
    --monitor-interval)
      [[ $# -ge 2 ]] || die "--monitor-interval requires a value"
      MONITOR_INTERVAL="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

cmd=(
  bash "${PROJECT_ROOT}/scripts/submit_pipeline_dag_hpc.sh"
  --partition "${PARTITION}"
  --cascade-partition "${CASCADE_PARTITION_VALUE}"
  --sites "${SITES}"
  --organisms "${ORGANISMS}"
  --run-publication-package
)

[[ -n "${FORCE_FLAG}" ]] && cmd+=("${FORCE_FLAG}")
[[ -n "${PYTHON_BIN}" ]] && cmd+=(--python-bin "${PYTHON_BIN}")
[[ "${DELETE_VENV}" -eq 1 ]] && cmd+=(--delete-venv)
[[ "${DRY_RUN}" -eq 1 ]] && cmd+=(--dry-run)
if [[ "${WAIT_FOR_JOBS}" -eq 1 ]]; then
  cmd+=(--wait --monitor-interval "${MONITOR_INTERVAL}")
fi

cd "${PROJECT_ROOT}"
printf 'Submitting publication package with cascade partition: %s\n' "${CASCADE_PARTITION_VALUE}" >&2
printf 'Command:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2
exec "${cmd[@]}"
