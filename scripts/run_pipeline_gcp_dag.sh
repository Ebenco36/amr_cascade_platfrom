#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline_gcp_dag.sh [options]

Run the full AMR Cascade Platform production pipeline on a single machine
(GCP VM or local) without SLURM. Uses full raw data (--env hpc,
--source-layer raw). This is the production equivalent of
run_pipeline_mac_dag.sh which runs on sample data only.

Default mode runs the core manuscript pipeline:
  - per-site ingestion/preprocessing run concurrently
  - harmonization runs after all site preprocessing
  - combined organism gold → cascade → prevalence → report run in order
  - site cascade, features, training, and supplementary stages are
    skipped unless explicitly enabled

Options:
  --sites SITE1,SITE2          Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2            Default: ESCHERICHIA COLI
  --python-bin PATH            Default: ./.venv/bin/python, ./venv/bin/python, python3
  --max-parallel N             Max concurrent subprocesses per wave. Default: 3
  --skip-ingestion
  --skip-preprocessing
  --skip-harmonization
  --skip-comorbidity
  --skip-gold
  --skip-cascade
  --skip-prevalence
  --skip-reporting
  --run-site-cascade           Also run site-level organism gold/cascade.
  --run-features               Build combined model-ready features.
  --run-training               Train combined downstream-testing models.
  --run-supp                   Run supplementary prediction figures/tables.
  --run-eskape                 Run ESKAPE-family cascade validation.
  --run-sensitivity            Run cascade sensitivity analysis.
  --run-audit                  Run scientific audit exports.
  --run-manuscript             Enable features, training, and all supplementary
                               stages for a complete production + manuscript run.
  --force-rerun-existing       Run enabled stages even when outputs exist.
  --dry-run                    Print commands without executing.
  --help                       Show this message.

Full production command:
  bash scripts/run_pipeline_gcp_dag.sh \
    --organisms "ESCHERICHIA COLI" \
    --run-manuscript
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITES=(armd armd_ecuh armd_utsw)
ORGANISMS=("ESCHERICHIA COLI")
PYTHON_BIN="${PYTHON_BIN:-}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
DRY_RUN=0
REUSE_EXISTING=1

RUN_INGESTION=1
RUN_PREPROCESSING=1
RUN_HARMONIZATION=1
RUN_COMORBIDITY=1
RUN_GOLD=1
RUN_CASCADE=1
RUN_PREVALENCE=1
RUN_REPORTING=1
RUN_SITE_CASCADE=0
RUN_FEATURES=0
RUN_TRAINING=0
RUN_SUPP=0
RUN_ESKAPE=0
RUN_SENSITIVITY=0
RUN_AUDIT=0

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log()       { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die()       { printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2; exit 1; }

split_csv() {
  local raw="$1" array_name="$2" item old_ifs
  old_ifs="${IFS}"; IFS=','
  eval "${array_name}=()"
  for item in ${raw}; do
    item="$(trim "${item}")"
    [[ -n "${item}" ]] && eval "${array_name}+=(\"\${item}\")"
  done
  IFS="${old_ifs}"
}

trim()    { printf '%s' "$1" | sed -E 's/^ +//; s/ +$//'; }
join_csv() { local IFS=","; printf '%s' "$*"; }
slugify() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites)
      [[ $# -ge 2 ]] || die "--sites requires a value"
      split_csv "$2" SITES; shift 2 ;;
    --organisms)
      [[ $# -ge 2 ]] || die "--organisms requires a value"
      split_csv "$2" ORGANISMS; shift 2 ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      PYTHON_BIN="$2"; shift 2 ;;
    --max-parallel)
      [[ $# -ge 2 ]] || die "--max-parallel requires a value"
      MAX_PARALLEL="$2"; shift 2 ;;
    --skip-ingestion)     RUN_INGESTION=0;     shift ;;
    --skip-preprocessing) RUN_PREPROCESSING=0; shift ;;
    --skip-harmonization) RUN_HARMONIZATION=0; shift ;;
    --skip-comorbidity)   RUN_COMORBIDITY=0;   shift ;;
    --skip-gold)          RUN_GOLD=0;          shift ;;
    --skip-cascade)       RUN_CASCADE=0;       shift ;;
    --skip-prevalence)    RUN_PREVALENCE=0;    shift ;;
    --skip-reporting)     RUN_REPORTING=0;     shift ;;
    --run-site-cascade)   RUN_SITE_CASCADE=1;  shift ;;
    --run-features)       RUN_FEATURES=1;      shift ;;
    --run-training)       RUN_TRAINING=1;      shift ;;
    --run-supp)           RUN_SUPP=1;          shift ;;
    --run-eskape)         RUN_ESKAPE=1;        shift ;;
    --run-sensitivity)    RUN_SENSITIVITY=1;   shift ;;
    --run-audit)          RUN_AUDIT=1;         shift ;;
    --run-manuscript)
      RUN_FEATURES=1; RUN_TRAINING=1
      RUN_SUPP=1; RUN_ESKAPE=1; RUN_SENSITIVITY=1; RUN_AUDIT=1
      shift ;;
    --force-rerun-existing) REUSE_EXISTING=0; shift ;;
    --dry-run)              DRY_RUN=1;        shift ;;
    --help|-h)              usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
else
  PYTHON="python3"
fi

command -v "${PYTHON}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON}"
[[ "${MAX_PARALLEL}" =~ ^[0-9]+$ && "${MAX_PARALLEL}" -ge 1 ]] || die "--max-parallel must be a positive integer"

# ── Output-readiness checks ───────────────────────────────────────────────────

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

bronze_ready() {
  local site="$1"
  [[ -e "${PROJECT_ROOT}/data/bronze/${site}/cohort" &&
     -e "${PROJECT_ROOT}/data/bronze/${site}/microbial_resistance" ]]
}

silver_ready() {
  local site="$1"
  [[ -f "${PROJECT_ROOT}/data/silver/${site}/cohort.parquet" &&
     -f "${PROJECT_ROOT}/data/silver/${site}/microbial_resistance.parquet" ]]
}

harmonized_ready() {
  local raw_site site
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    [[ -f "${PROJECT_ROOT}/data/harmonized/site_aligned/${site}/cohort.parquet" &&
       -f "${PROJECT_ROOT}/data/harmonized/site_aligned/${site}/microbial_resistance.parquet" ]] || return 1
  done
  [[ -f "${PROJECT_ROOT}/data/harmonized/combined/cohort.parquet" &&
     -f "${PROJECT_ROOT}/data/harmonized/combined/microbial_resistance.parquet" ]]
}

comorbidity_ready() {
  local site="$1"
  [[ -f "${PROJECT_ROOT}/data/interim/${site}/feature_matrices/comorbidity_aggregated.parquet" ]]
}

gold_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/gold" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/culture_episodes.parquet" &&
     -f "${dir}/culture_drug_episodes.parquet" &&
     -f "${dir}/drug_pair_episodes.parquet" &&
     -f "${dir}/testing_matrix.parquet" &&
     -f "${dir}/eligible_pairs.parquet" ]]
}

cascade_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/cascade" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/retained_edges.parquet" &&
     -f "${dir}/edge_report.parquet" &&
     -f "${dir}/cascade_summary.json" ]]
}

prevalence_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/prevalence_shift" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/prevalence_shift.parquet" &&
     -f "${dir}/prevalence_mnar_sensitivity_curves.parquet" &&
     -f "${dir}/prevalence_shift_summary.json" ]]
}

feature_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/features" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/model_ready_pair_features.parquet" &&
     -f "${dir}/feature_build_summary.json" ]]
}

training_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/modeling/downstream_testing" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/metrics.parquet" &&
     -f "${dir}/threshold_metrics.parquet" &&
     -f "${dir}/selected_thresholds.parquet" &&
     -f "${dir}/modeling_summary.json" ]]
}

reporting_ready() {
  local scope="$1" site="${2:-}" organism="${3:-}" dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/outputs/reports" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/report_manifest.json" ]]
}

supp_prediction_ready() {
  local organism="$1"
  local dir="${PROJECT_ROOT}/outputs/figures/combined/organisms/$(slugify "${organism}")"
  [[ -f "${dir}/supp_pred_fig1_roc_curves.png" &&
     -f "${dir}/supp_pred_fig6_lr_coefficients.png" ]]
}

should_run() {
  local upstream_dirty="$1"; shift
  if [[ "${REUSE_EXISTING}" -eq 0 || "${upstream_dirty}" -eq 1 ]]; then
    return 0
  fi
  "$@" && return 1
  return 0
}

# ── Wave runner ───────────────────────────────────────────────────────────────

wave_pids=()
wave_labels=()
wave_logs=()

run_task() {
  local label="$1"; shift
  local task_log
  task_log="${LOG_DIR}/gcp_dag_$(slugify "${label}")_$(date +%Y%m%d_%H%M%S).log"

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
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
    "$@"
  ) >"${task_log}" 2>&1 &

  wave_pids+=("$!")
  wave_labels+=("${label}")
  wave_logs+=("${task_log}")

  # Drain the current wave when the concurrency ceiling is reached so the
  # next task starts clean. The outer wait_wave after each stage loop is
  # then a safe no-op when this path fires.
  if [[ "${#wave_pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    wait_wave
  fi
}

wait_wave() {
  local failed=0 idx pid label task_log
  for idx in "${!wave_pids[@]}"; do
    pid="${wave_pids[$idx]}"
    label="${wave_labels[$idx]}"
    task_log="${wave_logs[$idx]}"
    if wait "${pid}"; then
      log "DONE ${label}"
    else
      log "FAILED ${label}; tailing ${task_log}" >&2
      tail -n 100 "${task_log}" >&2 || true
      failed=1
    fi
  done
  wave_pids=(); wave_labels=(); wave_logs=()
  [[ "${failed}" -eq 0 ]] || die "One or more subprocesses failed."
}

# ── Banner ────────────────────────────────────────────────────────────────────

log "=== AMR Cascade Platform — production run ==="
log "Project root   : ${PROJECT_ROOT}"
log "Python         : ${PYTHON}"
log "Sites          : $(join_csv "${SITES[@]}")"
log "Organisms      : $(join_csv "${ORGANISMS[@]}")"
log "Max parallel   : ${MAX_PARALLEL}"
log "Site cascade   : ${RUN_SITE_CASCADE}"
log "Features       : ${RUN_FEATURES}"
log "Training       : ${RUN_TRAINING}"
log "Supp prediction: ${RUN_SUPP}"
log "ESKAPE         : ${RUN_ESKAPE}"
log "Sensitivity    : ${RUN_SENSITIVITY}"
log "Audit          : ${RUN_AUDIT}"
log "Reuse existing : ${REUSE_EXISTING}"

# ── Stage 1: Ingestion (per-site, concurrent) ─────────────────────────────────
ingestion_dirty=0
if [[ "${RUN_INGESTION}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run 0 bronze_ready "${site}"; then
      run_task "ingest ${site}" \
        "${PYTHON}" scripts/run_ingestion.py --env hpc --site "${site}" --source-layer raw
      ingestion_dirty=1
    else
      log "SKIP ingestion ${site}: ready"
    fi
  done
  wait_wave
fi

# ── Stage 2: Preprocessing (per-site, concurrent) ────────────────────────────
preprocess_dirty=0
if [[ "${RUN_PREPROCESSING}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run "${ingestion_dirty}" silver_ready "${site}"; then
      run_task "preprocess ${site}" \
        "${PYTHON}" scripts/run_preprocessing.py --env hpc --site "${site}"
      preprocess_dirty=1
    else
      log "SKIP preprocessing ${site}: ready"
    fi
  done
  wait_wave
fi

# ── Stage 3: Harmonization ────────────────────────────────────────────────────
harmonize_dirty=0
if [[ "${RUN_HARMONIZATION}" -eq 1 ]]; then
  if should_run "${preprocess_dirty}" harmonized_ready; then
    harmonize_cmd=("${PYTHON}" scripts/run_harmonization.py --env hpc)
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

# ── Stage 4: Comorbidity pre-aggregation (per-site; only when features enabled) ─
comorbidity_dirty=0
if [[ "${RUN_COMORBIDITY}" -eq 1 && "${RUN_FEATURES}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    if should_run "${harmonize_dirty}" comorbidity_ready "${site}"; then
      run_task "comorbidity ${site}" \
        "${PYTHON}" scripts/pre_aggregate_comorbidities.py \
          --env hpc --site "${site}" --min-coverage 0.8
      comorbidity_dirty=1
    else
      log "SKIP comorbidity ${site}: ready"
    fi
  done
  wait_wave
fi

# ── Stages 5-11: Per-organism pipeline ───────────────────────────────────────
for raw_organism in "${ORGANISMS[@]}"; do
  organism="$(trim "${raw_organism}")"
  [[ -n "${organism}" ]] || continue

  # Stage 5a: Site-level gold + cascade (optional) ---------------------------
  if [[ "${RUN_SITE_CASCADE}" -eq 1 ]]; then
    site_gold_dirty=0
    if [[ "${RUN_GOLD}" -eq 1 ]]; then
      for raw_site in "${SITES[@]}"; do
        site="$(trim "${raw_site}")"
        [[ -n "${site}" ]] || continue
        if should_run "${harmonize_dirty}" gold_ready "site" "${site}" "${organism}"; then
          run_task "site gold ${site} ${organism}" \
            "${PYTHON}" scripts/run_gold.py \
              --env hpc --source-scope site --site "${site}" --organism "${organism}"
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
          run_task "site cascade ${site} ${organism}" \
            "${PYTHON}" scripts/run_cascade_analysis.py \
              --env hpc --gold-scope site --site "${site}" --organism "${organism}"
        else
          log "SKIP site cascade ${site} | ${organism}: ready"
        fi
      done
      wait_wave
    fi
  fi

  # Stage 5b: Combined gold --------------------------------------------------
  combined_gold_dirty=0
  if [[ "${RUN_GOLD}" -eq 1 ]]; then
    if should_run "${harmonize_dirty}" gold_ready "combined" "" "${organism}"; then
      run_task "combined gold ${organism}" \
        "${PYTHON}" scripts/run_gold.py \
          --env hpc --source-scope combined --organism "${organism}"
      combined_gold_dirty=1
      wait_wave
    else
      log "SKIP combined gold ${organism}: ready"
    fi
  fi

  # Stage 6: Combined cascade ------------------------------------------------
  combined_cascade_dirty=0
  if [[ "${RUN_CASCADE}" -eq 1 ]]; then
    if should_run "${combined_gold_dirty}" cascade_ready "combined" "" "${organism}"; then
      run_task "combined cascade ${organism}" \
        "${PYTHON}" scripts/run_cascade_analysis.py \
          --env hpc --gold-scope combined --organism "${organism}"
      combined_cascade_dirty=1
      wait_wave
    else
      log "SKIP combined cascade ${organism}: ready"
    fi
  fi

  # Stage 7: Prevalence shift ------------------------------------------------
  prevalence_dirty=0
  if [[ "${RUN_PREVALENCE}" -eq 1 ]]; then
    if should_run "${combined_cascade_dirty}" prevalence_ready "combined" "" "${organism}"; then
      run_task "prevalence ${organism}" \
        "${PYTHON}" scripts/run_prevalence_analysis.py \
          --env hpc --scope combined --organism "${organism}" \
          --figure-format html --figure-format png --figure-format svg --figure-format pdf
      prevalence_dirty=1
      wait_wave
    else
      log "SKIP prevalence ${organism}: ready"
    fi
  fi

  # Stage 8: Feature build ---------------------------------------------------
  feature_dirty=0
  if [[ "${RUN_FEATURES}" -eq 1 ]]; then
    upstream_dirty=0
    [[ "${combined_gold_dirty}" -eq 1 || "${comorbidity_dirty}" -eq 1 ]] && upstream_dirty=1
    if should_run "${upstream_dirty}" feature_ready "combined" "" "${organism}"; then
      run_task "features ${organism}" \
        "${PYTHON}" scripts/run_feature_build.py \
          --env hpc --scope combined --organism "${organism}"
      feature_dirty=1
      wait_wave
    else
      log "SKIP features ${organism}: ready"
    fi
  fi

  # Stage 9: Training --------------------------------------------------------
  training_dirty=0
  if [[ "${RUN_TRAINING}" -eq 1 ]]; then
    if should_run "${feature_dirty}" training_ready "combined" "" "${organism}"; then
      run_task "training ${organism}" \
        "${PYTHON}" scripts/run_training.py \
          --env hpc --scope combined --organism "${organism}"
      training_dirty=1
      wait_wave
    else
      log "SKIP training ${organism}: ready"
    fi
  fi

  # Stage 10: Reporting ------------------------------------------------------
  if [[ "${RUN_REPORTING}" -eq 1 ]]; then
    report_dirty=0
    [[ "${prevalence_dirty}" -eq 1 || "${training_dirty}" -eq 1 ]] && report_dirty=1
    if should_run "${report_dirty}" reporting_ready "combined" "" "${organism}"; then
      run_task "report ${organism}" \
        "${PYTHON}" scripts/run_reporting.py \
          --env hpc --scope combined --organism "${organism}" \
          --figure-format html --figure-format png --figure-format svg --figure-format pdf
      wait_wave
    else
      log "SKIP report ${organism}: ready"
    fi
  fi

  # Stage 11: Supplementary prediction figures/tables ─────────────────────────
  if [[ "${RUN_SUPP}" -eq 1 ]]; then
    if should_run "${training_dirty}" supp_prediction_ready "${organism}"; then
      run_task "supplementary prediction ${organism}" \
        "${PYTHON}" scripts/run_supplementary_prediction.py \
          --organism "${organism}"
      wait_wave
    else
      log "SKIP supplementary prediction ${organism}: ready"
    fi
  fi

done

# ── Stage 12: ESKAPE cascade validation (cross-organism) ─────────────────────
if [[ "${RUN_ESKAPE}" -eq 1 ]]; then
  run_task "eskape validation" \
    "${PYTHON}" scripts/run_eskape_cascade_validation.py \
      --env hpc
  wait_wave
fi

# ── Stage 13: Cascade sensitivity analysis ───────────────────────────────────
if [[ "${RUN_SENSITIVITY}" -eq 1 ]]; then
  run_task "cascade sensitivity" \
    "${PYTHON}" scripts/run_cascade_sensitivity.py \
      --env hpc
  wait_wave
fi

# ── Stage 14: Scientific audit ────────────────────────────────────────────────
if [[ "${RUN_AUDIT}" -eq 1 ]]; then
  run_task "scientific audit" \
    "${PYTHON}" scripts/run_scientific_audit.py \
      --env hpc
  wait_wave
fi

log "=== Production pipeline completed successfully ==="
