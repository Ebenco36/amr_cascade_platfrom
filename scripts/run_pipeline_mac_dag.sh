#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline_mac_dag.sh [options]

Run the Mac sample pipeline as local subprocess waves. This mirrors the HPC DAG
submitter without SLURM: independent site stages run in parallel, dependency
barriers are enforced locally, completed outputs are skipped by default, and
each subprocess writes its own log file.

Default mode is Mac-safe:
  - sample-first execution, default sample fraction 0.001
  - site sampling, ingestion, and preprocessing run concurrently
  - harmonization runs after all selected site preprocessing finishes
  - combined organism gold, cascade, prevalence, and reporting run in order
  - site cascade, pooled sensitivity, feature build, and training are skipped
    unless explicitly enabled

Options:
  --sites SITE1,SITE2             Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2               Default: ESCHERICHIA COLI
  --env ENV                       Runtime config environment. Default: mac
  --sample-fraction FLOAT         Default: 0.001
  --python-bin PATH               Default: ./.venv/bin/python, python3
  --report-formats F1,F2          Default: html,png,pdf,svg
  --resume-existing               Reuse completed artefacts when upstream did not rerun. Default.
  --force-rerun-existing          Rerun enabled stages even when outputs exist.
  --skip-preflight
  --preflight-only
  --skip-sampling
  --skip-ingestion
  --skip-preprocessing
  --skip-harmonization
  --skip-gold
  --skip-cascade
  --skip-prevalence
  --skip-reporting
  --run-site-cascade              Also run per-site organism gold/cascade.
  --run-features                  Build combined model-ready features.
  --skip-features                 Keep feature build disabled. Accepted for compatibility.
  --run-training                  Train combined downstream-testing models.
  --skip-training                 Keep training disabled. Accepted for compatibility.
  --skip-pooled-sensitivity       Accepted for compatibility; pooled runs are disabled here.
  --skip-sensitivity              Accepted for compatibility; sensitivity runs are disabled here.
  --estimators A,B                Restrict training to specific estimators.
  --dry-run                       Print subprocess commands without running.
  --help                          Show this message.

Examples:
  bash scripts/run_pipeline_mac_dag.sh --dry-run
  bash scripts/run_pipeline_mac_dag.sh --sample-fraction 0.001
  bash scripts/run_pipeline_mac_dag.sh --force-rerun-existing --sample-fraction 0.001
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SITES=(armd armd_ecuh armd_utsw)
ORGANISMS=("ESCHERICHIA COLI")
ENV_NAME="mac"
SAMPLE_FRACTION="0.001"
ESKAPE_SAMPLE_MIN_ROWS="25"
PYTHON_BIN="${PYTHON_BIN:-}"
REPORT_FORMATS=(html png pdf svg)
ESTIMATORS=()

REUSE_EXISTING=1
DRY_RUN=0

RUN_PREFLIGHT=1
PREFLIGHT_ONLY=0
RUN_SAMPLING=1
RUN_INGESTION=1
RUN_PREPROCESSING=1
RUN_HARMONIZATION=1
RUN_GOLD=1
RUN_CASCADE=1
RUN_PREVALENCE=1
RUN_REPORTING=1
RUN_SITE_CASCADE=0
RUN_FEATURES=0
RUN_TRAINING=0

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
  exit 1
}

split_csv_to_array() {
  local raw="$1"
  local array_name="$2"
  local old_ifs="${IFS}"
  local item
  IFS=','
  eval "${array_name}=()"
  for item in ${raw}; do
    item="$(trim "${item}")"
    [[ -n "${item}" ]] && eval "${array_name}+=(\"\${item}\")"
  done
  IFS="${old_ifs}"
}

join_csv() {
  local IFS=","
  printf '%s' "$*"
}

trim() {
  printf '%s' "$1" | sed -E 's/^ +//; s/ +$//'
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

resolve_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
    return
  fi
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    return
  fi
  for candidate in python3.12 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      return
    fi
  done
  die "Unable to locate python3.12, python3, or python."
}

load_runtime_layout() {
  local raw
  raw="$("${PYTHON_BIN}" -c '
from pathlib import Path
import sys
from amr_cascade_platform.core.config.config_loader import ConfigLoader

project_root = Path(sys.argv[1]).resolve()
environment_name = sys.argv[2]
settings = ConfigLoader(project_root).load(environment_name)
data_root = project_root / settings.environment.data_root

values = [
    str(data_root),
    str(data_root / "artifacts" / settings.cascade.outputs.result_dir),
    settings.cascade.outputs.summary_filename,
    str(data_root / "artifacts" / settings.prevalence.output_dir),
    str(data_root / "features"),
    str(data_root / "artifacts" / settings.modeling.output_dir / settings.modeling.task_name),
    str(project_root / settings.reporting.reports_dir),
]
print("|".join(values))
' "${PROJECT_ROOT}" "${ENV_NAME}")"
  IFS='|' read -r DATA_ROOT_ABS CASCADE_ARTIFACT_ROOT CASCADE_SUMMARY_FILENAME PREVALENCE_ARTIFACT_ROOT FEATURE_ROOT MODELING_TASK_ROOT REPORTS_ROOT <<< "${raw}"
}

scope_dir_path() {
  local root="$1"
  local scope="$2"
  local site="${3:-}"
  local organism="${4:-}"
  local base
  if [[ "${scope}" == "site" && -n "${site}" ]]; then
    base="${root}/${site}"
  else
    base="${root}/combined"
  fi
  if [[ -n "${organism}" ]]; then
    base="${base}/organisms/$(slugify "${organism}")"
  fi
  printf '%s\n' "${base}"
}

sample_ready() {
  local site="$1"
  local dir="${DATA_ROOT_ABS}/sample/${site}"
  compgen -G "${dir}/*cohort.csv" >/dev/null &&
    compgen -G "${dir}/*microbial_resistance.csv" >/dev/null
}

bronze_ready() {
  local site="$1"
  local dir="${DATA_ROOT_ABS}/bronze/${site}"
  [[ ( -e "${dir}/cohort" || -f "${dir}/cohort.parquet" ) &&
     ( -e "${dir}/microbial_resistance" || -f "${dir}/microbial_resistance.parquet" ) ]]
}

silver_ready() {
  local site="$1"
  [[ -f "${DATA_ROOT_ABS}/silver/${site}/cohort.parquet" &&
     -f "${DATA_ROOT_ABS}/silver/${site}/microbial_resistance.parquet" ]]
}

harmonized_ready() {
  local raw_site site
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    [[ -f "${DATA_ROOT_ABS}/harmonized/site_aligned/${site}/cohort.parquet" &&
       -f "${DATA_ROOT_ABS}/harmonized/site_aligned/${site}/microbial_resistance.parquet" ]] || return 1
  done
  [[ -f "${DATA_ROOT_ABS}/harmonized/combined/cohort.parquet" &&
     -f "${DATA_ROOT_ABS}/harmonized/combined/microbial_resistance.parquet" ]]
}

gold_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${DATA_ROOT_ABS}/gold" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/culture_episodes.parquet" &&
     -f "${dir}/culture_drug_episodes.parquet" &&
     -f "${dir}/drug_pair_episodes.parquet" &&
     -f "${dir}/testing_matrix.parquet" &&
     -f "${dir}/eligible_pairs.parquet" ]]
}

cascade_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${CASCADE_ARTIFACT_ROOT}" "${scope}" "${site}" "${organism}")"
  # network_edges/network_nodes/network_graph.json are written by NetworkExporter
  # as the last step of the cascade analysis run, after retained_edges/edge_report/
  # the summary file. A crash inside network export (e.g. the categorical-dtype bug
  # fixed in network_exporter.py) leaves those three earlier files complete while the
  # network ones are missing or stale -- checking only the earlier files let a failed
  # run look "ready" on the next invocation and silently skip re-running it.
  [[ -f "${dir}/retained_edges.parquet" &&
     -f "${dir}/edge_report.parquet" &&
     -f "${dir}/${CASCADE_SUMMARY_FILENAME}" &&
     -f "${dir}/network_edges.parquet" &&
     -f "${dir}/network_nodes.parquet" &&
     -f "${dir}/network_graph.json" &&
     "${dir}/network_graph.json" -nt "${dir}/retained_edges.parquet" ]]
}

prevalence_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PREVALENCE_ARTIFACT_ROOT}" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/prevalence_shift.parquet" &&
     -f "${dir}/prevalence_mnar_sensitivity_curves.parquet" &&
     -f "${dir}/prevalence_shift_summary.json" ]]
}

feature_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${FEATURE_ROOT}" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/model_ready_pair_features.parquet" &&
     -f "${dir}/feature_build_summary.json" ]]
}

training_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${MODELING_TASK_ROOT}" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/metrics.parquet" &&
     -f "${dir}/threshold_metrics.parquet" &&
     -f "${dir}/selected_thresholds.parquet" &&
     -f "${dir}/modeling_summary.json" &&
     -f "${dir}/logistic_regression_predictions.parquet" &&
     -f "${dir}/random_forest_predictions.parquet" &&
     -f "${dir}/xgboost_predictions.parquet" ]]
}

reporting_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${REPORTS_ROOT}" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/report_manifest.json" ]]
}

should_run() {
  local upstream_dirty="$1"
  shift
  if [[ "${REUSE_EXISTING}" -eq 0 || "${upstream_dirty}" -eq 1 ]]; then
    return 0
  fi
  "$@" && return 1
  return 0
}

wave_pids=()
wave_labels=()
wave_logs=()

run_task() {
  local label="$1"
  shift
  local slug
  local task_log
  slug="$(slugify "${label}")"
  task_log="${LOG_DIR}/mac_dag_${slug}_$(date +%Y%m%d_%H%M%S).log"

  log "START ${label}"
  log "LOG: ${task_log}"
  log "CMD: $*"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  (
    cd "${PROJECT_ROOT}"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
    mkdir -p "${MPLCONFIGDIR}"
    "$@"
  ) >"${task_log}" 2>&1 &

  wave_pids+=("$!")
  wave_labels+=("${label}")
  wave_logs+=("${task_log}")
}

wait_wave() {
  local failed=0
  local idx pid label task_log
  for idx in "${!wave_pids[@]}"; do
    pid="${wave_pids[$idx]}"
    label="${wave_labels[$idx]}"
    task_log="${wave_logs[$idx]}"
    if wait "${pid}"; then
      log "DONE ${label}"
    else
      log "FAILED ${label}; tailing log ${task_log}" >&2
      tail -n 80 "${task_log}" >&2 || true
      failed=1
    fi
  done
  wave_pids=()
  wave_labels=()
  wave_logs=()
  [[ "${failed}" -eq 0 ]] || die "One or more subprocesses failed."
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --sites)
        [[ $# -ge 2 ]] || die "--sites requires a value"
        split_csv_to_array "$2" SITES
        shift 2
        ;;
      --organisms)
        [[ $# -ge 2 ]] || die "--organisms requires a value"
        split_csv_to_array "$2" ORGANISMS
        shift 2
        ;;
      --env)
        [[ $# -ge 2 ]] || die "--env requires a value"
        ENV_NAME="$2"
        shift 2
        ;;
      --sample-fraction)
        [[ $# -ge 2 ]] || die "--sample-fraction requires a value"
        SAMPLE_FRACTION="$2"
        shift 2
        ;;
      --python-bin)
        [[ $# -ge 2 ]] || die "--python-bin requires a value"
        PYTHON_BIN="$2"
        shift 2
        ;;
      --report-formats)
        [[ $# -ge 2 ]] || die "--report-formats requires a value"
        split_csv_to_array "$2" REPORT_FORMATS
        shift 2
        ;;
      --estimators)
        [[ $# -ge 2 ]] || die "--estimators requires a value"
        split_csv_to_array "$2" ESTIMATORS
        shift 2
        ;;
      --resume-existing) REUSE_EXISTING=1; shift ;;
      --force-rerun-existing) REUSE_EXISTING=0; shift ;;
      --skip-preflight) RUN_PREFLIGHT=0; shift ;;
      --preflight-only) PREFLIGHT_ONLY=1; shift ;;
      --skip-sampling) RUN_SAMPLING=0; shift ;;
      --skip-ingestion) RUN_INGESTION=0; shift ;;
      --skip-preprocessing) RUN_PREPROCESSING=0; shift ;;
      --skip-harmonization) RUN_HARMONIZATION=0; shift ;;
      --skip-gold) RUN_GOLD=0; shift ;;
      --skip-cascade) RUN_CASCADE=0; shift ;;
      --skip-prevalence) RUN_PREVALENCE=0; shift ;;
      --skip-reporting) RUN_REPORTING=0; shift ;;
      --run-site-cascade) RUN_SITE_CASCADE=1; shift ;;
      --run-features) RUN_FEATURES=1; shift ;;
      --skip-features) RUN_FEATURES=0; shift ;;
      --run-training) RUN_TRAINING=1; shift ;;
      --skip-training) RUN_TRAINING=0; shift ;;
      --skip-pooled-sensitivity) shift ;;
      --skip-sensitivity) shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      --help|-h) usage; exit 0 ;;
      *) die "Unknown option: $1" ;;
    esac
  done
}

parse_args "$@"
resolve_python
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
load_runtime_layout

log "Starting Mac DAG sample pipeline"
log "Environment: ${ENV_NAME}"
log "Sites: $(join_csv "${SITES[@]}")"
log "Organisms: $(join_csv "${ORGANISMS[@]}")"
log "Sample fraction: ${SAMPLE_FRACTION}"
log "Resume existing: ${REUSE_EXISTING}"
log "Site cascade: ${RUN_SITE_CASCADE}"
log "Features: ${RUN_FEATURES}"
log "Training: ${RUN_TRAINING}"
log "Python: ${PYTHON_BIN}"

if [[ "${RUN_PREFLIGHT}" -eq 1 ]]; then
  preflight_cmd=("${PYTHON_BIN}" scripts/run_preflight.py --env "${ENV_NAME}" --sites "$(join_csv "${SITES[@]}")" --organisms "$(join_csv "${ORGANISMS[@]}")" --report-formats "$(join_csv "${REPORT_FORMATS[@]}")")
  preflight_cmd+=(--skip-pooled-sensitivity --skip-sensitivity)
  [[ "${RUN_SAMPLING}" -eq 0 ]] && preflight_cmd+=(--skip-sampling)
  [[ "${RUN_INGESTION}" -eq 0 ]] && preflight_cmd+=(--skip-ingestion)
  [[ "${RUN_PREPROCESSING}" -eq 0 ]] && preflight_cmd+=(--skip-preprocessing)
  [[ "${RUN_HARMONIZATION}" -eq 0 ]] && preflight_cmd+=(--skip-harmonization)
  [[ "${RUN_GOLD}" -eq 0 ]] && preflight_cmd+=(--skip-gold)
  [[ "${RUN_CASCADE}" -eq 0 ]] && preflight_cmd+=(--skip-cascade)
  [[ "${RUN_PREVALENCE}" -eq 0 ]] && preflight_cmd+=(--skip-cascade)
  [[ "${RUN_SITE_CASCADE}" -eq 0 ]] && preflight_cmd+=(--skip-site-cascade)
  [[ "${RUN_FEATURES}" -eq 0 ]] && preflight_cmd+=(--skip-features)
  [[ "${RUN_TRAINING}" -eq 0 ]] && preflight_cmd+=(--skip-training)
  [[ "${RUN_REPORTING}" -eq 0 ]] && preflight_cmd+=(--skip-reporting)
  run_task "preflight" "${preflight_cmd[@]}"
  wait_wave
fi

if [[ "${PREFLIGHT_ONLY}" -eq 1 ]]; then
  log "Preflight-only mode completed successfully."
  exit 0
fi

sampling_dirty=0
if [[ "${RUN_SAMPLING}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run 0 sample_ready "${site}"; then
      run_task "sample ${site}" "${PYTHON_BIN}" scripts/create_sample_dataset.py --env "${ENV_NAME}" --site "${site}" --fraction "${SAMPLE_FRACTION}" --ensure-eskape-coverage --eskape-min-rows-per-target "${ESKAPE_SAMPLE_MIN_ROWS}"
      sampling_dirty=1
    else
      log "SKIP sample ${site}: ready"
    fi
  done
  wait_wave
fi

ingestion_dirty=0
if [[ "${RUN_INGESTION}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run "${sampling_dirty}" bronze_ready "${site}"; then
      run_task "ingest ${site}" "${PYTHON_BIN}" scripts/run_ingestion.py --env "${ENV_NAME}" --site "${site}"
      ingestion_dirty=1
    else
      log "SKIP ingestion ${site}: ready"
    fi
  done
  wait_wave
fi

preprocess_dirty=0
if [[ "${RUN_PREPROCESSING}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run "${ingestion_dirty}" silver_ready "${site}"; then
      run_task "preprocess ${site}" "${PYTHON_BIN}" scripts/run_preprocessing.py --env "${ENV_NAME}" --site "${site}"
      preprocess_dirty=1
    else
      log "SKIP preprocessing ${site}: ready"
    fi
  done
  wait_wave
fi

harmonize_dirty=0
if [[ "${RUN_HARMONIZATION}" -eq 1 ]]; then
  if should_run "${preprocess_dirty}" harmonized_ready; then
    harmonize_cmd=("${PYTHON_BIN}" scripts/run_harmonization.py --env "${ENV_NAME}")
    for raw_site in "${SITES[@]}"; do
      site="$(trim "${raw_site}")"
      [[ -n "${site}" ]] && harmonize_cmd+=(--site "${site}")
    done
    run_task "harmonize" "${harmonize_cmd[@]}"
    harmonize_dirty=1
    wait_wave
  else
    log "SKIP harmonization: ready"
  fi
fi

organism_dirty=0
for raw_organism in "${ORGANISMS[@]}"; do
  organism="$(trim "${raw_organism}")"
  [[ -n "${organism}" ]] || continue

  if [[ "${RUN_SITE_CASCADE}" -eq 1 ]]; then
    site_gold_dirty=0
    if [[ "${RUN_GOLD}" -eq 1 ]]; then
      for raw_site in "${SITES[@]}"; do
        site="$(trim "${raw_site}")"
        [[ -n "${site}" ]] || continue
        if should_run "${harmonize_dirty}" gold_ready "site" "${site}" "${organism}"; then
          run_task "site gold ${site} ${organism}" "${PYTHON_BIN}" scripts/run_gold.py --env "${ENV_NAME}" --source-scope site --site "${site}" --organism "${organism}"
          site_gold_dirty=1
        else
          log "SKIP site gold ${site} | ${organism}: ready"
        fi
      done
      wait_wave
    fi

    if [[ "${RUN_CASCADE}" -eq 1 ]]; then
      for raw_site in "${SITES[@]}"; do
        site="$(trim "${raw_site}")"
        [[ -n "${site}" ]] || continue
        if should_run "${site_gold_dirty}" cascade_ready "site" "${site}" "${organism}"; then
          site_cascade_cmd=("${PYTHON_BIN}" scripts/run_cascade_analysis.py --env "${ENV_NAME}" --gold-scope site --site "${site}" --organism "${organism}")
          [[ "${REUSE_EXISTING}" -eq 0 ]] && site_cascade_cmd+=(--ignore-precomputed-validation)
          run_task "site cascade ${site} ${organism}" "${site_cascade_cmd[@]}"
          organism_dirty=1
        else
          log "SKIP site cascade ${site} | ${organism}: ready"
        fi
      done
      wait_wave
    fi
  fi

  combined_gold_dirty=0
  if [[ "${RUN_GOLD}" -eq 1 ]]; then
    if should_run "${harmonize_dirty}" gold_ready "combined" "" "${organism}"; then
      run_task "combined gold ${organism}" "${PYTHON_BIN}" scripts/run_gold.py --env "${ENV_NAME}" --source-scope combined --organism "${organism}"
      combined_gold_dirty=1
      organism_dirty=1
      wait_wave
    else
      log "SKIP combined gold ${organism}: ready"
    fi
  fi

  combined_cascade_dirty=0
  if [[ "${RUN_CASCADE}" -eq 1 ]]; then
    if should_run "${combined_gold_dirty}" cascade_ready "combined" "" "${organism}"; then
      combined_cascade_cmd=("${PYTHON_BIN}" scripts/run_cascade_analysis.py --env "${ENV_NAME}" --gold-scope combined --organism "${organism}")
      [[ "${REUSE_EXISTING}" -eq 0 ]] && combined_cascade_cmd+=(--ignore-precomputed-validation)
      run_task "combined cascade ${organism}" "${combined_cascade_cmd[@]}"
      combined_cascade_dirty=1
      organism_dirty=1
      wait_wave
    else
      log "SKIP combined cascade ${organism}: ready"
    fi
  fi

  prevalence_dirty=0
  if [[ "${RUN_PREVALENCE}" -eq 1 ]]; then
    if should_run "${combined_cascade_dirty}" prevalence_ready "combined" "" "${organism}"; then
      prevalence_cmd=("${PYTHON_BIN}" scripts/run_prevalence_analysis.py --env "${ENV_NAME}" --scope combined --organism "${organism}")
      for fmt in "${REPORT_FORMATS[@]}"; do
        [[ -n "${fmt}" ]] && prevalence_cmd+=(--figure-format "${fmt}")
      done
      run_task "prevalence ${organism}" "${prevalence_cmd[@]}"
      prevalence_dirty=1
      organism_dirty=1
      wait_wave
    else
      log "SKIP prevalence ${organism}: ready"
    fi
  fi

  feature_dirty=0
  if [[ "${RUN_FEATURES}" -eq 1 ]]; then
    if should_run "${combined_gold_dirty}" feature_ready "combined" "" "${organism}"; then
      run_task "features ${organism}" "${PYTHON_BIN}" scripts/run_feature_build.py --env "${ENV_NAME}" --scope combined --organism "${organism}"
      feature_dirty=1
      wait_wave
    else
      log "SKIP features ${organism}: ready"
    fi
  fi

  training_dirty=0
  if [[ "${RUN_TRAINING}" -eq 1 ]]; then
    if should_run "${feature_dirty}" training_ready "combined" "" "${organism}"; then
      train_cmd=("${PYTHON_BIN}" scripts/run_training.py --env "${ENV_NAME}" --scope combined --organism "${organism}")
      for estimator in "${ESTIMATORS[@]}"; do
        [[ -n "${estimator}" ]] && train_cmd+=(--estimator "${estimator}")
      done
      run_task "training ${organism}" "${train_cmd[@]}"
      training_dirty=1
      wait_wave
    else
      log "SKIP training ${organism}: ready"
    fi
  fi

  if [[ "${RUN_REPORTING}" -eq 1 ]]; then
    report_upstream_dirty=0
    [[ "${prevalence_dirty}" -eq 1 || "${training_dirty}" -eq 1 ]] && report_upstream_dirty=1
    if should_run "${report_upstream_dirty}" reporting_ready "combined" "" "${organism}"; then
      report_cmd=("${PYTHON_BIN}" scripts/run_reporting.py --env "${ENV_NAME}" --scope combined --organism "${organism}")
      for fmt in "${REPORT_FORMATS[@]}"; do
        [[ -n "${fmt}" ]] && report_cmd+=(--figure-format "${fmt}")
      done
      run_task "report ${organism}" "${report_cmd[@]}"
      organism_dirty=1
      wait_wave
    else
      log "SKIP report ${organism}: ready"
    fi
  fi
done

log "Mac DAG sample pipeline completed successfully. Any subprocess logs are in ${LOG_DIR}/mac_dag_*.log."
