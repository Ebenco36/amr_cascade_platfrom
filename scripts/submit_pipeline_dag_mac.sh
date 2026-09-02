#!/usr/bin/env bash
# submit_pipeline_dag_mac.sh
#
# Non-blocking local DAG runner. Mirrors submit_pipeline_dag_hpc.sh exactly:
# all tasks are spawned immediately, dependency chains are enforced internally,
# independent stages run in parallel, and the submitter returns at once.
#
# IMPORTANT — run directly in a terminal or tmux, NOT via bash in background
# without --wait.  Each spawned task is a background subprocess; if the shell
# exits they are orphaned.
#   tmux new -s mac-pipeline
#   bash scripts/submit_pipeline_dag_mac.sh --force-rerun-existing
#   # Ctrl-B D to detach; tmux attach -t mac-pipeline to reattach

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_pipeline_dag_mac.sh [options]

Non-blocking local DAG runner. All tasks are spawned immediately with
dependency chains encoded internally. Independent stages (e.g. per-site
ingestion) run in parallel without blocking each other.

Options:
  --sites SITE1,SITE2          Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2            Default: ESCHERICHIA COLI
  --sample-fraction FLOAT      Default: 0.001
  --python-bin PATH            Default: .venv/bin/python, python3
  --report-formats F1,F2       Default: html,png
  --skip-sampling
  --skip-ingestion
  --skip-preprocessing
  --skip-harmonization
  --skip-gold
  --skip-cascade
  --skip-prevalence
  --skip-reporting
  --run-site-cascade           Also run per-site organism gold/cascade jobs
  --run-features               Build combined model-ready features
  --run-training               Train combined downstream-testing models
  --run-eskape                 Run ESKAPE-family validation (one job per target)
  --eskape-targets T1,T2       Default: Enterococcus,Staphylococcus,Klebsiella,
                               Acinetobacter,Pseudomonas,Enterobacter
  --force-rerun-existing       Spawn enabled stages even when outputs exist
  --dry-run                    Print task graph without executing
  --wait                       Block until all tasks finish and print status table
  --monitor-interval N         Poll interval in seconds when --wait is active. Default: 10
  --status STATUS_DIR          Skip submission; report state of a prior run
  --help
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
STATUS_DIR="${LOG_DIR}/.dag_${RUN_ID}"

SITES=(armd armd_ecuh armd_utsw)
ORGANISMS=("ESCHERICHIA COLI")
ESKAPE_TARGETS=(Enterococcus Staphylococcus Klebsiella Acinetobacter Pseudomonas Enterobacter)
SAMPLE_FRACTION="0.001"
PYTHON_BIN="${PYTHON_BIN:-}"
REPORT_FORMATS=(html png)
DRY_RUN=0
WAIT_FOR_TASKS=0
MONITOR_INTERVAL=10
REUSE_EXISTING=1
RESUME_STATUS_DIR=""
ENV_NAME="mac"
SOURCE_LAYER=""   # auto-detected from ENV_NAME after arg parse

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
RUN_ESKAPE=0

submitted_tasks=()

# ── helpers ──────────────────────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { printf '[%s] %s\n' "$(timestamp)" "$*"; }
trim() { printf '%s' "$1" | sed -E 's/^ +//; s/ +$//'; }

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

split_csv() {
  local raw="$1"
  local -n _out="$2"
  IFS=',' read -r -a _out <<< "${raw}"
}

join_colon() { local IFS=':'; printf '%s' "$*"; }

resolve_python() {
  [[ -n "${PYTHON_BIN}" ]] && return
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"; return
  fi
  for c in python3.12 python3 python; do
    command -v "${c}" >/dev/null 2>&1 && { PYTHON_BIN="${c}"; return; }
  done
  log "ERROR: no Python found" >&2; exit 1
}

load_runtime_layout() {
  local raw
  raw="$("${PYTHON_BIN}" -c '
from pathlib import Path
import sys
from amr_cascade_platform.core.config.config_loader import ConfigLoader
project_root = Path(sys.argv[1]).resolve()
s = ConfigLoader(project_root).load(sys.argv[2])
d = project_root / s.environment.data_root
print("|".join([
  str(d),
  str(d / "artifacts" / s.cascade.outputs.result_dir),
  s.cascade.outputs.summary_filename,
  str(d / "artifacts" / s.prevalence.output_dir),
  str(d / "features"),
  str(d / "artifacts" / s.modeling.output_dir / s.modeling.task_name),
  str(project_root / s.reporting.reports_dir),
]))
' "${PROJECT_ROOT}" "${ENV_NAME}")"
  IFS='|' read -r \
    DATA_ROOT_ABS CASCADE_ARTIFACT_ROOT CASCADE_SUMMARY_FILENAME \
    PREVALENCE_ARTIFACT_ROOT FEATURE_ROOT MODELING_TASK_ROOT REPORTS_ROOT \
    <<< "${raw}"
}

scope_dir_path() {
  local root="$1" scope="$2" site="${3:-}" organism="${4:-}" base
  if [[ "${scope}" == "site" && -n "${site}" ]]; then
    base="${root}/${site}"
  else
    base="${root}/combined"
  fi
  [[ -n "${organism}" ]] && base="${base}/organisms/$(slugify "${organism}")"
  printf '%s\n' "${base}"
}

# ── output-ready predicates ───────────────────────────────────────────────────

sample_ready()     { local d="${DATA_ROOT_ABS}/sample/$1"; compgen -G "${d}/*cohort.csv" >/dev/null && compgen -G "${d}/*microbial_resistance.csv" >/dev/null; }
bronze_ready()     { local d="${DATA_ROOT_ABS}/bronze/$1"; [[ -e "${d}/cohort" || -f "${d}/cohort.parquet" ]] && [[ -e "${d}/microbial_resistance" || -f "${d}/microbial_resistance.parquet" ]]; }
silver_ready()     { [[ -f "${DATA_ROOT_ABS}/silver/$1/cohort.parquet" && -f "${DATA_ROOT_ABS}/silver/$1/microbial_resistance.parquet" ]]; }

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
  local dir; dir="$(scope_dir_path "${DATA_ROOT_ABS}/gold" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/culture_episodes.parquet" && -f "${dir}/culture_drug_episodes.parquet" &&
     -f "${dir}/drug_pair_episodes.parquet" && -f "${dir}/testing_matrix.parquet" &&
     -f "${dir}/eligible_pairs.parquet" ]]
}

cascade_ready() {
  local dir; dir="$(scope_dir_path "${CASCADE_ARTIFACT_ROOT}" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/retained_edges.parquet" && -f "${dir}/edge_report.parquet" &&
     -f "${dir}/network_nodes.parquet" &&
     -f "${dir}/${CASCADE_SUMMARY_FILENAME}" ]]
}

prevalence_ready() {
  local dir; dir="$(scope_dir_path "${PREVALENCE_ARTIFACT_ROOT}" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/prevalence_shift.parquet" && -f "${dir}/prevalence_mnar_sensitivity_curves.parquet" &&
     -f "${dir}/prevalence_shift_summary.json" ]]
}

feature_ready() {
  local dir; dir="$(scope_dir_path "${FEATURE_ROOT}" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/model_ready_pair_features.parquet" && -f "${dir}/feature_build_summary.json" ]]
}

training_ready() {
  local dir; dir="$(scope_dir_path "${MODELING_TASK_ROOT}" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/metrics.parquet" && -f "${dir}/threshold_metrics.parquet" &&
     -f "${dir}/selected_thresholds.parquet" && -f "${dir}/modeling_summary.json" ]]
}

reporting_ready() {
  local dir; dir="$(scope_dir_path "${REPORTS_ROOT}" "$1" "${2:-}" "${3:-}")"
  [[ -f "${dir}/report_manifest.json" ]]
}

eskape_ready() {
  local scope="$1" site="${2:-}" target="$3" scope_dir target_slug
  [[ "${scope}" == "combined" ]] && scope_dir="combined" || scope_dir="${site}"
  target_slug="$(slugify "${target}")"
  [[ -f "${PROJECT_ROOT}/data/artifacts/eskape_validation/armd/${scope_dir}/organisms/${target_slug}/eskape_existence_summary.csv" ]]
}

should_submit() {
  local deps="$1"; shift
  [[ "${REUSE_EXISTING}" -eq 0 || -n "${deps}" ]] && return 0
  "$@" && return 1; return 0
}

# ── core: spawn_task ──────────────────────────────────────────────────────────
#
# spawn_task NAME DEPS_COLON COMMAND...
#
# Spawns COMMAND as a background process. The process polls STATUS_DIR for each
# dep name to reach COMPLETED before executing. On dep failure the task is
# marked SKIPPED. On success it is marked COMPLETED; on error FAILED.
#
# Dry-run: prints the task graph and marks every task COMPLETED immediately so
# dep chains can be traced without running anything.

spawn_task() {
  local name="$1"
  local deps_raw="$2"
  shift 2
  local command=("$@")
  local task_log="${LOG_DIR}/mac_dag_${name}_${RUN_ID}.log"
  local status_file="${STATUS_DIR}/${name}.status"
  local deps=()
  [[ -n "${deps_raw}" ]] && IFS=':' read -r -a deps <<< "${deps_raw}"

  submitted_tasks+=("${name}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "SPAWN ${name}"
    log "  deps : ${deps_raw:-none}"
    log "  cmd  : ${command[*]}"
    echo "COMPLETED" > "${status_file}"
    return 0
  fi

  (
    echo "PENDING" > "${status_file}"

    # Wait for every named dependency.
    local dep elapsed=0 status
    for dep in "${deps[@]}"; do
      [[ -n "${dep}" ]] || continue
      while true; do
        if [[ -f "${STATUS_DIR}/${dep}.status" ]]; then
          status="$(cat "${STATUS_DIR}/${dep}.status")"
          [[ "${status}" == "COMPLETED" ]] && break
          if [[ "${status}" == "FAILED" || "${status}" == "SKIPPED" ]]; then
            log "SKIP ${name}: upstream '${dep}' ${status}" >> "${task_log}" 2>&1
            echo "SKIPPED" > "${status_file}"
            exit 0
          fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ "${elapsed}" -ge 86400 ]]; then
          log "TIMEOUT ${name}: waited 24 h for '${dep}'" >> "${task_log}" 2>&1
          echo "FAILED" > "${status_file}"
          exit 1
        fi
      done
    done

    echo "RUNNING" > "${status_file}"

    {
      log "START ${name}"
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

      if "${command[@]}"; then
        echo "COMPLETED" > "${status_file}"
        log "DONE ${name}"
      else
        echo "FAILED" > "${status_file}"
        log "FAILED ${name} — see ${task_log}" >&2
      fi
    } >> "${task_log}" 2>&1

  ) &
}

# ── monitoring ────────────────────────────────────────────────────────────────

wait_for_tasks() {
  local tasks=("$@")
  [[ "${#tasks[@]}" -eq 0 ]] && return 0

  log "Watching ${#tasks[@]} task(s). Polling every ${MONITOR_INTERVAL}s."
  log "Live logs: ${LOG_DIR}/mac_dag_*_${RUN_ID}.log"
  log "Press Ctrl-C to stop watching; tasks continue in the background."
  echo

  declare -A prev_state
  local t; for t in "${tasks[@]}"; do prev_state["${t}"]="PENDING"; done

  local poll=0
  while true; do
    poll=$((poll + 1))
    local active=0
    for t in "${tasks[@]}"; do
      local prev="${prev_state[${t}]}" curr="${prev_state[${t}]}"
      [[ -f "${STATUS_DIR}/${t}.status" ]] && curr="$(cat "${STATUS_DIR}/${t}.status")"
      if [[ "${prev}" != "${curr}" ]]; then
        case "${curr}" in
          RUNNING)   log "STARTED   ${t}" ;;
          COMPLETED) log "FINISHED  ${t}" ;;
          FAILED)    log "FAILED    ${t}" ;;
          SKIPPED)   log "SKIPPED   ${t}" ;;
          *)         log "STATE     ${t}: ${prev} → ${curr}" ;;
        esac
        prev_state["${t}"]="${curr}"
      fi
      case "${curr}" in
        PENDING|RUNNING) active=$((active + 1)) ;;
      esac
    done

    [[ "${active}" -eq 0 ]] && { log "ALL DONE."; break; }
    [[ $((poll % 6)) -eq 0 ]] && log "SUMMARY  ${active} of ${#tasks[@]} task(s) still active"
    sleep "${MONITOR_INTERVAL}"
  done
}

report_final_states() {
  local tasks=("$@")
  echo
  printf '%-45s  %s\n' "TASK" "STATUS"
  printf '%-45s  %s\n' "$(printf '%0.s-' {1..45})" "------"
  local t status failed=0
  for t in "${tasks[@]}"; do
    status="$(cat "${STATUS_DIR}/${t}.status" 2>/dev/null || echo UNKNOWN)"
    printf '%-45s  %s\n' "${t}" "${status}"
    [[ "${status}" != "COMPLETED" && "${status}" != "SKIPPED" ]] && failed=1
  done
  echo
  [[ "${failed}" -eq 0 ]] && log "All tasks completed." || { log "One or more tasks failed."; return 1; }
}

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites)              split_csv "$2" SITES;            shift 2 ;;
    --organisms)          split_csv "$2" ORGANISMS;        shift 2 ;;
    --sample-fraction)    SAMPLE_FRACTION="$2";            shift 2 ;;
    --python-bin)         PYTHON_BIN="$2";                 shift 2 ;;
    --report-formats)     split_csv "$2" REPORT_FORMATS;   shift 2 ;;
    --eskape-targets)     split_csv "$2" ESKAPE_TARGETS;   shift 2 ;;
    --skip-sampling)      RUN_SAMPLING=0;        shift ;;
    --skip-ingestion)     RUN_INGESTION=0;       shift ;;
    --skip-preprocessing) RUN_PREPROCESSING=0;  shift ;;
    --skip-harmonization) RUN_HARMONIZATION=0;  shift ;;
    --skip-gold)          RUN_GOLD=0;            shift ;;
    --skip-cascade)       RUN_CASCADE=0;         shift ;;
    --skip-prevalence)    RUN_PREVALENCE=0;      shift ;;
    --skip-reporting)     RUN_REPORTING=0;       shift ;;
    --run-site-cascade)   RUN_SITE_CASCADE=1;    shift ;;
    --run-features)       RUN_FEATURES=1;        shift ;;
    --run-training)       RUN_TRAINING=1;        shift ;;
    --run-eskape)         RUN_ESKAPE=1;          shift ;;
    --force-rerun-existing) REUSE_EXISTING=0;   shift ;;
    --dry-run)            DRY_RUN=1;             shift ;;
    --wait)               WAIT_FOR_TASKS=1;      shift ;;
    --monitor-interval)   MONITOR_INTERVAL="$2"; shift 2 ;;
    --env)                ENV_NAME="$2";           shift 2 ;;
    --status)             RESUME_STATUS_DIR="$2"; shift 2 ;;
    --help|-h)            usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RUN_SITE_CASCADE}" -eq 1 && "${RUN_CASCADE}" -eq 0 ]]; then
  echo "ERROR: --run-site-cascade requires cascade analysis, but --skip-cascade was also supplied." >&2
  echo "Remove --skip-cascade to regenerate site-cascade comparison artifacts, or omit --run-site-cascade for a combined-only output refresh." >&2
  exit 2
fi

# ── re-attach to prior run ────────────────────────────────────────────────────

if [[ -n "${RESUME_STATUS_DIR}" ]]; then
  [[ -d "${RESUME_STATUS_DIR}" ]] || { echo "Status dir not found: ${RESUME_STATUS_DIR}" >&2; exit 2; }
  STATUS_DIR="${RESUME_STATUS_DIR}"
  mapfile -t submitted_tasks < <(
    find "${STATUS_DIR}" -name '*.status' -exec basename {} .status \; | sort
  )
  log "Re-attaching to ${#submitted_tasks[@]} task(s) in ${STATUS_DIR}"
  if report_final_states "${submitted_tasks[@]}"; then exit 0; else exit 1; fi
fi

# ── setup ─────────────────────────────────────────────────────────────────────

# Auto-detect source layer: mac uses sample data; anything else (mac_full) uses raw.
[[ -z "${SOURCE_LAYER}" ]] && { [[ "${ENV_NAME}" == "mac" ]] && SOURCE_LAYER="sample" || SOURCE_LAYER="raw"; }
# When reading raw data, sampling is not needed.
[[ "${SOURCE_LAYER}" == "raw" ]] && RUN_SAMPLING=0

resolve_python
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
load_runtime_layout
mkdir -p "${STATUS_DIR}"

log "Submitting Mac pipeline DAG"
log "Status dir : ${STATUS_DIR}"
log "Env        : ${ENV_NAME}"
log "Source     : ${SOURCE_LAYER}"
log "Sites      : ${SITES[*]}"
log "Organisms  : ${ORGANISMS[*]}"
log "Fraction   : ${SAMPLE_FRACTION} (ignored when source=raw)"
log "Python     : ${PYTHON_BIN}"
log "Reuse      : ${REUSE_EXISTING}"
log "Features   : ${RUN_FEATURES} | Training: ${RUN_TRAINING} | ESKAPE: ${RUN_ESKAPE}"
echo

# ── sampling ──────────────────────────────────────────────────────────────────

sample_deps=""
if [[ "${RUN_SAMPLING}" -eq 1 ]]; then
  site_sample_names=()
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    name="sample_${site_slug}"
    if should_submit "" sample_ready "${site}"; then
      spawn_task "${name}" "" \
        "${PYTHON_BIN}" scripts/create_sample_dataset.py \
          --env "${ENV_NAME}" --site "${site}" \
          --fraction "${SAMPLE_FRACTION}" \
          --ensure-eskape-coverage
      site_sample_names+=("${name}")
    else
      log "Ready; not spawning ${name}"
    fi
  done
  sample_deps="$(join_colon "${site_sample_names[@]}")"
fi

# ── ingestion ─────────────────────────────────────────────────────────────────

ingest_deps=""
if [[ "${RUN_INGESTION}" -eq 1 ]]; then
  site_ingest_names=()
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    name="ingest_${site_slug}"
    if should_submit "${sample_deps}" bronze_ready "${site}"; then
      spawn_task "${name}" "${sample_deps}" \
        "${PYTHON_BIN}" scripts/run_ingestion.py --env "${ENV_NAME}" --source-layer "${SOURCE_LAYER}" --site "${site}"
      site_ingest_names+=("${name}")
    else
      log "Ready; not spawning ${name}"
    fi
  done
  ingest_deps="$(join_colon "${site_ingest_names[@]}")"
fi

# ── preprocessing ─────────────────────────────────────────────────────────────

preprocess_deps=""
if [[ "${RUN_PREPROCESSING}" -eq 1 ]]; then
  site_preprocess_names=()
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    name="preprocess_${site_slug}"
    if should_submit "${ingest_deps}" silver_ready "${site}"; then
      spawn_task "${name}" "${ingest_deps}" \
        "${PYTHON_BIN}" scripts/run_preprocessing.py --env "${ENV_NAME}" --site "${site}"
      site_preprocess_names+=("${name}")
    else
      log "Ready; not spawning ${name}"
    fi
  done
  preprocess_deps="$(join_colon "${site_preprocess_names[@]}")"
fi

# ── harmonization ─────────────────────────────────────────────────────────────

harmonize_dep=""
if [[ "${RUN_HARMONIZATION}" -eq 1 ]]; then
  if should_submit "${preprocess_deps}" harmonized_ready; then
    harmonize_cmd=("${PYTHON_BIN}" scripts/run_harmonization.py --env "${ENV_NAME}")
    for raw_site in "${SITES[@]}"; do
      site="$(trim "${raw_site}")"
      [[ -n "${site}" ]] && harmonize_cmd+=(--site "${site}")
    done
    spawn_task "harmonize" "${preprocess_deps}" "${harmonize_cmd[@]}"
    harmonize_dep="harmonize"
  else
    log "Ready; not spawning harmonize"
  fi
fi

# ── per-organism DAG ──────────────────────────────────────────────────────────

for raw_organism in "${ORGANISMS[@]}"; do
  organism="$(trim "${raw_organism}")"
  [[ -n "${organism}" ]] || continue
  organism_slug="$(slugify "${organism}")"
  org_pfx="${organism_slug}"

  # optional site-level cascade
  site_cascade_deps=""
  if [[ "${RUN_SITE_CASCADE}" -eq 1 ]]; then
    site_cascade_names=()
    for raw_site in "${SITES[@]}"; do
      site="$(trim "${raw_site}")"
      [[ -n "${site}" ]] || continue
      site_slug="$(slugify "${site}")"

      site_gold_name="${org_pfx}_site_gold_${site_slug}"
      if [[ "${RUN_GOLD}" -eq 1 ]] && should_submit "${harmonize_dep}" gold_ready "site" "${site}" "${organism}"; then
        spawn_task "${site_gold_name}" "${harmonize_dep}" \
          "${PYTHON_BIN}" scripts/run_gold.py --env "${ENV_NAME}" \
            --source-scope site --site "${site}" --organism "${organism}"
      fi

      if [[ "${RUN_CASCADE}" -eq 1 ]]; then
        site_cascade_name="${org_pfx}_site_cascade_${site_slug}"
        site_gold_dep="${harmonize_dep}"
        if [[ "${RUN_GOLD}" -eq 1 ]]; then
          site_gold_dep="${site_gold_name}"
        fi
        if should_submit "${site_gold_dep}" cascade_ready "site" "${site}" "${organism}"; then
          spawn_task "${site_cascade_name}" "${site_gold_dep}" \
            "${PYTHON_BIN}" scripts/run_cascade_analysis.py --env "${ENV_NAME}" \
              --gold-scope site --site "${site}" --organism "${organism}"
          site_cascade_names+=("${site_cascade_name}")
        fi
      fi
    done
    site_cascade_deps="$(join_colon "${site_cascade_names[@]}")"
  fi

  # combined gold
  gold_dep="${harmonize_dep}"
  if [[ "${RUN_GOLD}" -eq 1 ]]; then
    gold_name="${org_pfx}_gold"
    if should_submit "${harmonize_dep}" gold_ready "combined" "" "${organism}"; then
      spawn_task "${gold_name}" "${harmonize_dep}" \
        "${PYTHON_BIN}" scripts/run_gold.py --env "${ENV_NAME}" \
          --source-scope combined --organism "${organism}"
      gold_dep="${gold_name}"
    else
      log "Ready; not spawning ${gold_name}"
    fi
  fi

  # combined cascade
  cascade_dep="${gold_dep}"
  if [[ "${RUN_CASCADE}" -eq 1 ]]; then
    cascade_name="${org_pfx}_cascade"
    if should_submit "${gold_dep}" cascade_ready "combined" "" "${organism}"; then
      spawn_task "${cascade_name}" "${gold_dep}" \
        "${PYTHON_BIN}" scripts/run_cascade_analysis.py --env "${ENV_NAME}" \
          --gold-scope combined --organism "${organism}"
      cascade_dep="${cascade_name}"
    else
      log "Ready; not spawning ${cascade_name}"
    fi
  fi

  # prevalence
  prevalence_dep="${cascade_dep}"
  if [[ "${RUN_PREVALENCE}" -eq 1 ]]; then
    prevalence_name="${org_pfx}_prevalence"
    if should_submit "${cascade_dep}" prevalence_ready "combined" "" "${organism}"; then
      prevalence_cmd=("${PYTHON_BIN}" scripts/run_prevalence_analysis.py --env "${ENV_NAME}"
        --scope combined --organism "${organism}")
      for fmt in "${REPORT_FORMATS[@]}"; do
        [[ -n "${fmt}" ]] && prevalence_cmd+=(--figure-format "${fmt}")
      done
      spawn_task "${prevalence_name}" "${cascade_dep}" "${prevalence_cmd[@]}"
      prevalence_dep="${prevalence_name}"
    else
      log "Ready; not spawning ${prevalence_name}"
    fi
  fi

  # features
  feature_dep="${gold_dep}"
  if [[ "${RUN_FEATURES}" -eq 1 ]]; then
    feature_name="${org_pfx}_features"
    # Feature matrices include training-site cascade features when
    # modeling.include_cascade_features=true. Depend on site cascade tasks when
    # they are part of this DAG so local tests match production semantics.
    feature_inputs=("${gold_dep}")
    [[ -n "${site_cascade_deps}" ]] && feature_inputs+=("${site_cascade_deps}")
    feature_input_deps="$(join_colon "${feature_inputs[@]}")"
    if should_submit "${feature_input_deps}" feature_ready "combined" "" "${organism}"; then
      spawn_task "${feature_name}" "${feature_input_deps}" \
        "${PYTHON_BIN}" scripts/run_feature_build.py --env "${ENV_NAME}" \
          --scope combined --organism "${organism}"
      feature_dep="${feature_name}"
    else
      log "Ready; not spawning ${feature_name}"
    fi
  fi

  # training
  training_dep=""
  if [[ "${RUN_TRAINING}" -eq 1 ]]; then
    training_name="${org_pfx}_training"
    if should_submit "${feature_dep}" training_ready "combined" "" "${organism}"; then
      spawn_task "${training_name}" "${feature_dep}" \
        "${PYTHON_BIN}" scripts/run_training.py --env "${ENV_NAME}" \
          --scope combined --organism "${organism}"
      training_dep="${training_name}"
    else
      log "Ready; not spawning ${training_name}"
    fi
  fi

  # report — waits for prevalence + training (if training enabled)
  if [[ "${RUN_REPORTING}" -eq 1 ]]; then
    report_name="${org_pfx}_report"
    report_deps_arr=("${prevalence_dep}")
    [[ -n "${training_dep}" ]] && report_deps_arr+=("${training_dep}")
    report_deps="$(join_colon "${report_deps_arr[@]}")"
    if should_submit "${report_deps}" reporting_ready "combined" "" "${organism}"; then
      report_cmd=("${PYTHON_BIN}" scripts/run_reporting.py --env "${ENV_NAME}"
        --scope combined --organism "${organism}")
      for fmt in "${REPORT_FORMATS[@]}"; do
        [[ -n "${fmt}" ]] && report_cmd+=(--figure-format "${fmt}")
      done
      spawn_task "${report_name}" "${report_deps}" "${report_cmd[@]}"
    else
      log "Ready; not spawning ${report_name}"
    fi
  fi
done

# ── ESKAPE validation ─────────────────────────────────────────────────────────

if [[ "${RUN_ESKAPE}" -eq 1 ]]; then
  eskape_root="${PROJECT_ROOT}/data/artifacts/eskape_validation/armd"
  eskape_tables_root="${PROJECT_ROOT}/outputs/tables/eskape"

  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    site_target_names=()

    for raw_target in "${ESKAPE_TARGETS[@]}"; do
      target="$(trim "${raw_target}")"
      [[ -n "${target}" ]] || continue
      target_slug="$(slugify "${target}")"
      name="eskape_site_${site_slug}_${target_slug}"
      per_target_dir="${eskape_tables_root}/_per_target/site/${site_slug}/${target_slug}"
      if should_submit "${harmonize_dep}" eskape_ready "site" "${site}" "${target}"; then
        spawn_task "${name}" "${harmonize_dep}" \
          "${PYTHON_BIN}" scripts/run_eskape_cascade_validation.py --env "${ENV_NAME}" \
            --armd-source-scope site --armd-site "${site}" \
            --targets "${target}" \
            --armd-output-root "${eskape_root}" \
            --summary-dir "${per_target_dir}"
        site_target_names+=("${name}")
      else
        log "Ready; not spawning ${name}"
      fi
    done

    if [[ "${#site_target_names[@]}" -gt 0 ]]; then
      merge_name="eskape_site_${site_slug}_merge"
      merge_deps="$(join_colon "${site_target_names[@]}")"
      spawn_task "${merge_name}" "${merge_deps}" \
        "${PYTHON_BIN}" scripts/merge_eskape_summaries.py \
          --input-glob "outputs/tables/eskape/_per_target/site/${site_slug}/*/eskape_cascade_comparison.csv" \
          --output-csv "outputs/tables/eskape/site_${site}/eskape_cascade_comparison.csv" \
          --output-json "outputs/tables/eskape/site_${site}/eskape_cascade_comparison.json"
    fi
  done

  combined_target_names=()
  for raw_target in "${ESKAPE_TARGETS[@]}"; do
    target="$(trim "${raw_target}")"
    [[ -n "${target}" ]] || continue
    target_slug="$(slugify "${target}")"
    name="eskape_combined_${target_slug}"
    per_target_dir="${eskape_tables_root}/_per_target/combined/${target_slug}"
    if should_submit "${harmonize_dep}" eskape_ready "combined" "" "${target}"; then
      spawn_task "${name}" "${harmonize_dep}" \
        "${PYTHON_BIN}" scripts/run_eskape_cascade_validation.py --env "${ENV_NAME}" \
          --armd-source-scope combined \
          --targets "${target}" \
          --armd-output-root "${eskape_root}" \
          --summary-dir "${per_target_dir}"
      combined_target_names+=("${name}")
    else
      log "Ready; not spawning ${name}"
    fi
  done

  if [[ "${#combined_target_names[@]}" -gt 0 ]]; then
    merge_deps="$(join_colon "${combined_target_names[@]}")"
    spawn_task "eskape_combined_merge" "${merge_deps}" \
      "${PYTHON_BIN}" scripts/merge_eskape_summaries.py \
        --input-glob "outputs/tables/eskape/_per_target/combined/*/eskape_cascade_comparison.csv" \
        --output-csv "outputs/tables/eskape/combined_armd/eskape_cascade_comparison.csv" \
        --output-json "outputs/tables/eskape/combined_armd/eskape_cascade_comparison.json"
  fi
fi

# ── summary ───────────────────────────────────────────────────────────────────

echo
if [[ "${#submitted_tasks[@]}" -eq 0 ]]; then
  log "No tasks spawned — all enabled stages are already complete."
  exit 0
fi

log "Spawned ${#submitted_tasks[@]} task(s)."
log "Status dir : ${STATUS_DIR}"
log "Re-attach  : bash $(basename "${BASH_SOURCE[0]}") --status ${STATUS_DIR}"
log "Live logs  : tail -f ${LOG_DIR}/mac_dag_*_${RUN_ID}.log"
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "Dry run complete — no processes started."
  exit 0
fi

# Save task list for --status re-attach
task_list_file="${STATUS_DIR}/tasks.list"
printf '%s\n' "${submitted_tasks[@]}" > "${task_list_file}"

if [[ "${WAIT_FOR_TASKS}" -eq 1 ]]; then
  wait_for_tasks "${submitted_tasks[@]}"
  if report_final_states "${submitted_tasks[@]}"; then exit 0; else exit 1; fi
fi
