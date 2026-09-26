#!/usr/bin/env bash
# Run the whole cluster submission wave locally, on the synthetic replica of
# data/raw, in dependency order, stopping at the first failure. Stages the
# cluster runs as independent jobs (site and combined validation shards, the
# availability sensitivities, the follow-ups) run concurrently here too.
# It mirrors scripts/submit_everything_hpc.sh: the primary publication-package
# run (ingestion through site cascades, features, training, every figure and
# both audits), the three availability-denominator sensitivities, and the
# follow-ups (site-by-era permutation, patient-cluster bootstrap, ridge refit,
# supplementary prediction, supplementary tables); then it dry-runs the
# manuscript package and writes an inventory of every table produced.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_replica_end_to_end.sh [options]

Build the replica first:
  python scripts/create_raw_replica_dataset.py all --scale 0.03

Options:
  --env NAME              Environment config. Default: test_replica
  --organism NAME         Default: ESCHERICHIA COLI
  --shards N              Validation shards, combined scope and sensitivities. Default: 4
  --site-shards N         Validation shards per site. Default: 2
  --follow-up-shards N    Shards for the site-by-era and patient-cluster follow-ups. Default: 2
  --skip-sensitivities    Do not run the three availability-denominator sensitivities.
  --from STEP             Resume at STEP, keeping earlier outputs. Steps, in order:
                          ingestion preprocessing harmonization comorbidity cascades
                          prevalence_features training report audits follow_ups
                          supplementary_tables package_dry_run inventory
                          (cascades = site and combined gold + validation + cascade;
                          follow_ups = availability sensitivities, site-by-era
                          permutation, patient-cluster bootstrap, ridge refit and
                          supplementary prediction)
  --python PATH           Default: ./.venv/bin/python
  --dry-run               Print the commands without running them.
  --help                  Show this message.

A run without --from first deletes the environment's derived layers (bronze,
silver, harmonized, interim, gold, features, artifacts, metadata, outputs), so
nothing from an earlier run is reused. raw/, reference/ and profile/ are kept.
Logs: <data_root>/logs/e2e/<step>.log
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="test_replica"
ORGANISM="ESCHERICHIA COLI"
SITES=(armd armd_ecuh armd_utsw)
SHARDS=4
SITE_SHARDS=2
FOLLOW_UP_SHARDS=2
RUN_SENSITIVITIES=1
FROM_STEP=""
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DRY_RUN=0

STEPS=(ingestion preprocessing harmonization comorbidity cascades prevalence_features training report
       audits follow_ups supplementary_tables package_dry_run inventory)

need_value() { [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) need_value "$@"; ENV_NAME="$2"; shift 2 ;;
    --organism) need_value "$@"; ORGANISM="$2"; shift 2 ;;
    --shards) need_value "$@"; SHARDS="$2"; shift 2 ;;
    --site-shards) need_value "$@"; SITE_SHARDS="$2"; shift 2 ;;
    --follow-up-shards) need_value "$@"; FOLLOW_UP_SHARDS="$2"; shift 2 ;;
    --skip-sensitivities) RUN_SENSITIVITIES=0; shift ;;
    --from) need_value "$@"; FROM_STEP="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "${SHARDS}" "${SITE_SHARDS}" "${FOLLOW_UP_SHARDS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Shard counts must be positive integers" >&2; exit 2; }
done
if [[ -n "${FROM_STEP}" ]] && ! printf '%s\n' "${STEPS[@]}" | grep -qx "${FROM_STEP}"; then
  echo "Unknown step for --from: ${FROM_STEP}" >&2
  exit 2
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }

DATA_ROOT="$("${PYTHON_BIN}" - "${PROJECT_ROOT}" "${ENV_NAME}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from amr_cascade_platform.core.config.config_loader import ConfigLoader
print(ConfigLoader(Path(sys.argv[1])).load(sys.argv[2]).environment.data_root)
PY
)"
DATA_ROOT_ABS="${PROJECT_ROOT}/${DATA_ROOT}"
if [[ "$(cd "${DATA_ROOT_ABS}" 2>/dev/null && pwd -P)" == "$(cd "${PROJECT_ROOT}/data" && pwd -P)" ]]; then
  echo "Refusing to run: ${ENV_NAME} points at the real data root (${DATA_ROOT})." >&2
  exit 2
fi
for site in "${SITES[@]}"; do
  compgen -G "${DATA_ROOT_ABS}/raw/${site}/*cohort.csv" > /dev/null || {
    echo "No replica raw files under ${DATA_ROOT_ABS}/raw/${site}; run scripts/create_raw_replica_dataset.py first." >&2
    exit 2
  }
done

LOG_DIR="${DATA_ROOT_ABS}/logs/e2e"
mkdir -p "${LOG_DIR}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "${MPLCONFIGDIR}"

RUN_STARTED=$(date +%s)
started_step=0
[[ -z "${FROM_STEP}" ]] && started_step=1
should_run() {
  local step="$1"
  if [[ "${started_step}" -eq 0 && "${step}" == "${FROM_STEP}" ]]; then
    started_step=1
  fi
  [[ "${started_step}" -eq 1 ]]
}

# run NAME COMMAND... : run one command, log it, stop the whole run on failure.
run() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %s:' "${name}"; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  local start end status
  start=$(date +%s)
  set +e
  "$@" > "${log}" 2>&1
  status=$?
  set -e
  end=$(date +%s)
  if [[ "${status}" -ne 0 ]]; then
    printf '%-44s FAILED (exit %s, %ss) -- last lines of %s:\n' "${name}" "${status}" "$((end - start))" "${log}"
    tail -25 "${log}" | sed 's/^/    /'
    return "${status}"
  fi
  printf '%-44s ok (%ss)\n' "${name}" "$((end - start))"
}

# in_parallel FUNCTION ARG... : FUNCTION ARG for each ARG at once.
in_parallel() {
  local function_name="$1"; shift
  local pids=() item status=0
  for item in "$@"; do
    "${function_name}" "${item}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

# concurrently "CMD ARGS" "CMD ARGS" ... : run whole commands (functions) at once.
concurrently() {
  local pids=() command status=0
  for command in "$@"; do
    eval "${command}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

PY=("${PYTHON_BIN}")
O="${ORGANISM}"

validate_scope() {
  # validate_scope ENV SCOPE SITE SHARDS LABEL: gold, sharded validation, merge, cascade.
  local env="$1" scope="$2" site="$3" shards="$4" label="$5"
  local site_args=()
  [[ "${scope}" == "site" ]] && site_args=(--site "${site}")
  local gold_scope_args=(--source-scope "${scope}")
  run "${label}_gold" "${PY[@]}" scripts/run_gold.py --env "${env}" "${gold_scope_args[@]}" ${site_args[@]+"${site_args[@]}"} --organism "${O}"
  local i pids=() status=0
  for ((i = 0; i < shards; i++)); do
    run "${label}_validation_shard_${i}" "${PY[@]}" scripts/run_cascade_validation_shard.py --env "${env}" --gold-scope "${scope}" ${site_args[@]+"${site_args[@]}"} --organism "${O}" --shard-index "${i}" --shard-total "${shards}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [[ "${status}" -eq 0 ]] || return 1
  run "${label}_validation_merge" "${PY[@]}" scripts/merge_cascade_validation_shards.py --env "${env}" --gold-scope "${scope}" ${site_args[@]+"${site_args[@]}"} --organism "${O}" --shard-total "${shards}"
  run "${label}_cascade" "${PY[@]}" scripts/run_cascade_analysis.py --env "${env}" --gold-scope "${scope}" ${site_args[@]+"${site_args[@]}"} --organism "${O}"
}

ingest_site() { run "ingestion_$1" "${PY[@]}" scripts/run_ingestion.py --env "${ENV_NAME}" --site "$1" --source-layer raw; }
preprocess_site() { run "preprocessing_$1" "${PY[@]}" scripts/run_preprocessing.py --env "${ENV_NAME}" --site "$1"; }
comorbidity_site() { run "comorbidity_$1" "${PY[@]}" scripts/pre_aggregate_comorbidities.py --env "${ENV_NAME}" --site "$1" --min-coverage 0.8; }
site_cascade() { validate_scope "${ENV_NAME}" site "$1" "${SITE_SHARDS}" "site_$1"; }

echo "Replica end-to-end run | env ${ENV_NAME} | data root ${DATA_ROOT} | organism ${O}"
echo "Logs: ${LOG_DIR}"
if [[ -z "${FROM_STEP}" && "${DRY_RUN}" -eq 0 ]]; then
  for layer in bronze silver harmonized interim gold features artifacts metadata outputs; do
    rm -rf "${DATA_ROOT_ABS:?}/${layer}"
  done
  echo "Cleared derived layers under ${DATA_ROOT}"
fi
run check_python_runtime "${PY[@]}" scripts/check_python_runtime.py

if should_run ingestion; then in_parallel ingest_site "${SITES[@]}"; fi
if should_run preprocessing; then in_parallel preprocess_site "${SITES[@]}"; fi
if should_run harmonization; then run harmonization "${PY[@]}" scripts/run_harmonization.py --env "${ENV_NAME}"; fi
if should_run comorbidity; then in_parallel comorbidity_site "${SITES[@]}"; fi
combined_cascade() { validate_scope "${ENV_NAME}" combined "" "${SHARDS}" combined; }
all_site_cascades() { in_parallel site_cascade "${SITES[@]}"; }
if should_run cascades; then concurrently all_site_cascades combined_cascade; fi

prevalence() {
  run prevalence "${PY[@]}" scripts/run_prevalence_analysis.py --env "${ENV_NAME}" --scope combined --organism "${O}"     --figure-format html --figure-format png --figure-format svg --figure-format pdf
}
features() { run features "${PY[@]}" scripts/run_feature_build.py --env "${ENV_NAME}" --scope combined --organism "${O}"; }
if should_run prevalence_features; then concurrently prevalence features; fi
if should_run training; then run training "${PY[@]}" scripts/run_training.py --env "${ENV_NAME}" --scope combined --organism "${O}"; fi
if should_run report; then
  # Same figure list as submit_pipeline_dag_hpc.sh --run-all-figures
  # --include-site-comparison-figures, with --fail-on-missing-figures.
  figures=(cascade_directional cascade_evidence_scatter cascade_consequence_summary operational_availability_suite observation_coverage descriptive_summaries
           prevalence_shift_forest prevalence_shift_curves mnar_tipping_point upstream_contribution_forest
           downstream_trigger_forest threshold_sensitivity model_metrics_comparison model_pr_curve model_roc_curve
           model_calibration model_threshold_analysis validation_diagnostics er_landscape
           mnar_prevalence_shift_distribution validation_funnel panel_bundling temporal_stability consort_diagram
           dataset_characterization site_vs_combined_summary cross_site_concordance)
  figure_args=()
  for figure in "${figures[@]}"; do figure_args+=(--figure "${figure}"); done
  run report "${PY[@]}" scripts/run_reporting.py --env "${ENV_NAME}" --scope combined --organism "${O}"     --figure-format html --figure-format png --figure-format svg --figure-format pdf --fail-on-missing-figures "${figure_args[@]}"
fi
readiness_audit() {
  run readiness_audit "${PY[@]}" scripts/run_publication_readiness_audit.py --env "${ENV_NAME}" --scope combined --organism "${O}" --require-training
}
scientific_audit() { run scientific_audit "${PY[@]}" scripts/run_scientific_audit.py --env "${ENV_NAME}" --scope combined --organism "${O}"; }
if should_run audits; then concurrently readiness_audit scientific_audit; fi

sensitivity() {
  local variant="$1"
  local variant_env="${ENV_NAME}_${variant}"
  local variant_root="${PROJECT_ROOT}/${ENV_NAME}_${variant}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    # Same layout as setup_sensitivity_symlinks in submit_everything_hpc.sh:
    # upstream layers shared, gold and artifacts rebuilt under the variant.
    rm -rf "${variant_root:?}"
    mkdir -p "${variant_root}/gold" "${variant_root}/artifacts"
    local d
    for d in raw sample bronze silver harmonized metadata reference interim features; do
      ln -s "${DATA_ROOT_ABS}/${d}" "${variant_root}/${d}"
    done
  fi
  validate_scope "${variant_env}" combined "" "${SHARDS}" "${variant}"
}
follow_up() {
  local analysis="$1" label="$2" i pids=() status=0
  for ((i = 0; i < FOLLOW_UP_SHARDS; i++)); do
    run "${label}_shard_${i}" "${PY[@]}" scripts/run_validated_pattern_sensitivity.py --analysis "${analysis}" --env "${ENV_NAME}"       --gold-scope combined --organism "${O}" --shard-total "${FOLLOW_UP_SHARDS}" --shard-index "${i}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [[ "${status}" -eq 0 ]] || return 1
  if [[ "${FOLLOW_UP_SHARDS}" -gt 1 ]]; then
    run "${label}_merge" "${PY[@]}" scripts/run_validated_pattern_sensitivity.py --analysis "${analysis}" --env "${ENV_NAME}"       --gold-scope combined --organism "${O}" --shard-total "${FOLLOW_UP_SHARDS}" --merge-shards
  fi
}
sensitivities() {
  [[ "${RUN_SENSITIVITIES}" -eq 1 ]] || return 0
  in_parallel sensitivity availability_sensitivity availability_annual_sensitivity availability_joint_sensitivity
}
ridge() {
  run ridge_penalty "${PY[@]}" scripts/run_ridge_penalty_sensitivity.py --env "${ENV_NAME}" --gold-scope combined --organism "${O}" --penalties 1,10,100
}
prediction() { run supplementary_prediction "${PY[@]}" scripts/run_supplementary_prediction.py --env "${ENV_NAME}" --organism "${O}"; }
if should_run follow_ups; then
  concurrently sensitivities "follow_up era-stratified era_stratified" "follow_up patient-cluster patient_cluster" ridge prediction
fi
if should_run supplementary_tables; then
  run supplementary_tables "${PY[@]}" scripts/build_supplementary_tables.py --env "${ENV_NAME}" --organism "${O}"
fi
PACKAGE_STATUS=0
if should_run package_dry_run; then
  # Reported, not fatal: the inventory below still runs so every produced
  # table is listed; the run's exit status carries the packaging result.
  run package_dry_run "${PY[@]}" scripts/build_manuscript_package.py --env "${ENV_NAME}" --organism "${O}" --dry-run || PACKAGE_STATUS=$?
fi
if should_run inventory; then
  run inventory "${PY[@]}" - "${PROJECT_ROOT}" "${ENV_NAME}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

project_root, environment = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(project_root / "src"))
from amr_cascade_platform.core.config.config_loader import ConfigLoader

settings = ConfigLoader(project_root).load(environment)
tables_root = project_root / settings.reporting.tables_dir
rows = []
for path in sorted(tables_root.rglob("*.csv")):
    try:
        frame = pd.read_csv(path, low_memory=False)
        rows.append({"table": str(path.relative_to(tables_root)), "rows": len(frame), "columns": frame.shape[1]})
    except pd.errors.EmptyDataError:
        rows.append({"table": str(path.relative_to(tables_root)), "rows": 0, "columns": 0})
inventory = pd.DataFrame(rows)
out = tables_root / "TABLE_INVENTORY.csv"
inventory.to_csv(out, index=False)
empty = inventory.loc[inventory["rows"].eq(0), "table"].tolist()
print(f"{len(inventory)} tables under {tables_root} ({len(empty)} with no rows) -> {out}")
for name in empty:
    print("  no rows:", name)
PY
  [[ "${DRY_RUN}" -eq 1 ]] || sed -n 1,200p "${LOG_DIR}/inventory.log"
fi
echo "Finished in $(( $(date +%s) - RUN_STARTED ))s. Logs: ${LOG_DIR}"
if [[ "${PACKAGE_STATUS}" -ne 0 ]]; then
  echo "Manuscript package dry run failed (exit ${PACKAGE_STATUS}); see ${LOG_DIR}/package_dry_run.log"
  exit "${PACKAGE_STATUS}"
fi
