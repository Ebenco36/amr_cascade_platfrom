#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_publication_package_mac.sh [options]

Run the local Mac publication package: full DAG, scientific audit, and
paper-facing artifact readiness audit.

Options:
  --env ENV                         Runtime environment. Default: mac
  --sites SITE1,SITE2               Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2                 Default: ESCHERICHIA COLI
  --python-bin PATH                 Default: ./.venv/bin/python
  --estimators A,B                  Default: logistic_regression
  --resume-existing                 Reuse completed DAG artifacts. Default.
  --force-rerun-existing            Rerun enabled DAG stages.
  --skip-pipeline                   Run audits only.
  --skip-training                   Do not require/run training.
  --min-primary-validated-edges N   Default: 1
  --min-table-count N               Default: 10
  --min-figure-count N              Minimum per PDF/PNG/SVG/HTML. Default: 1
  --help                            Show this message.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="mac"
SITES="armd,armd_ecuh,armd_utsw"
ORGANISMS="ESCHERICHIA COLI"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
ESTIMATORS="logistic_regression"
REUSE_FLAG="--resume-existing"
RUN_PIPELINE=1
RUN_TRAINING=1
MIN_PRIMARY_VALIDATED_EDGES=1
MIN_TABLE_COUNT=10
MIN_FIGURE_COUNT=1

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

split_csv() {
  local raw="$1"
  local old_ifs="${IFS}"
  local item
  IFS=','
  SPLIT_VALUES=()
  for item in ${raw}; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "${item}" ]] && SPLIT_VALUES+=("${item}")
  done
  IFS="${old_ifs}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) [[ $# -ge 2 ]] || die "--env requires a value"; ENV_NAME="$2"; shift 2 ;;
    --sites) [[ $# -ge 2 ]] || die "--sites requires a value"; SITES="$2"; shift 2 ;;
    --organisms) [[ $# -ge 2 ]] || die "--organisms requires a value"; ORGANISMS="$2"; shift 2 ;;
    --python-bin) [[ $# -ge 2 ]] || die "--python-bin requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --estimators) [[ $# -ge 2 ]] || die "--estimators requires a value"; ESTIMATORS="$2"; shift 2 ;;
    --resume-existing) REUSE_FLAG="--resume-existing"; shift ;;
    --force-rerun-existing) REUSE_FLAG="--force-rerun-existing"; shift ;;
    --skip-pipeline) RUN_PIPELINE=0; shift ;;
    --skip-training) RUN_TRAINING=0; shift ;;
    --min-primary-validated-edges)
      [[ $# -ge 2 ]] || die "--min-primary-validated-edges requires a value"
      MIN_PRIMARY_VALIDATED_EDGES="$2"
      shift 2
      ;;
    --min-table-count)
      [[ $# -ge 2 ]] || die "--min-table-count requires a value"
      MIN_TABLE_COUNT="$2"
      shift 2
      ;;
    --min-figure-count)
      [[ $# -ge 2 ]] || die "--min-figure-count requires a value"
      MIN_FIGURE_COUNT="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -x "${PYTHON_BIN}" ]] || die "Python executable is not executable: ${PYTHON_BIN}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RUN_PIPELINE}" -eq 1 ]]; then
  pipeline_cmd=(
    bash scripts/run_pipeline_mac_dag.sh
    --env "${ENV_NAME}"
    --skip-sampling
    "${REUSE_FLAG}"
    --sites "${SITES}"
    --organisms "${ORGANISMS}"
    --run-site-cascade
    --run-features
    --report-formats html,png,pdf,svg
    --python-bin "${PYTHON_BIN}"
  )
  if [[ "${RUN_TRAINING}" -eq 1 ]]; then
    pipeline_cmd+=(--run-training --estimators "${ESTIMATORS}")
  fi
  "${pipeline_cmd[@]}"
fi

split_csv "${ORGANISMS}"
for organism in "${SPLIT_VALUES[@]}"; do
  "${PYTHON_BIN}" scripts/run_scientific_audit.py --env "${ENV_NAME}" --scope combined --organism "${organism}"

  audit_cmd=(
    "${PYTHON_BIN}" scripts/run_publication_readiness_audit.py
    --env "${ENV_NAME}"
    --scope combined
    --organism "${organism}"
    --min-primary-validated-edges "${MIN_PRIMARY_VALIDATED_EDGES}"
    --min-table-count "${MIN_TABLE_COUNT}"
    --min-pdf-figures "${MIN_FIGURE_COUNT}"
    --min-png-figures "${MIN_FIGURE_COUNT}"
    --min-svg-figures "${MIN_FIGURE_COUNT}"
    --min-html-figures "${MIN_FIGURE_COUNT}"
  )
  [[ "${RUN_TRAINING}" -eq 1 ]] && audit_cmd+=(--require-training)
  "${audit_cmd[@]}"
done
