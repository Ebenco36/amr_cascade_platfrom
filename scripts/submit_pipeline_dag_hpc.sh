#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_pipeline_dag_hpc.sh [options]

Submit the AMR Cascade Platform as a SLURM dependency graph. Independent site
and organism tasks are submitted separately, dependencies are enforced with
afterok, and a failed stage does not waste every downstream stage.

Default mode is manuscript-safe:
  - per-site ingestion/preprocessing can run concurrently
  - harmonization runs after all site preprocessing succeeds
  - per-site comorbidity pre-aggregation runs only when feature build is enabled
  - combined organism gold/cascade/prevalence/report run as dependency chains
  - site-level cascade, pooled all-organism sensitivity, feature build, and
    training are skipped unless explicitly enabled

Options:
  --sites SITE1,SITE2          Default: armd,armd_ecuh,armd_utsw
  --organisms O1,O2            Default: ESCHERICHIA COLI
  --partition NAME             Default: main
  --python-bin PATH            Python 3.12 to use for the venv and all SLURM jobs.
                               Default: auto-detect (prefers system Python at
                               /usr/bin/python3.12 over conda Python, because the
                               conda stdlib may not be mounted on compute nodes).
                               Example: --python-bin /usr/bin/python3.12
                               If the venv is missing/broken the script rebuilds it.
  --skip-ingestion
  --skip-preprocessing
  --skip-harmonization
  --skip-comorbidity
  --skip-gold
  --skip-cascade
  --skip-prevalence
  --skip-reporting
  --run-site-cascade           Also run site-level organism gold/cascade jobs.
  --run-features               Build combined model-ready features.
  --run-training               Train combined downstream-testing models.
  --run-eskape                 Run ESKAPE-family cascade validation (site + combined
                                scope) as one independent job per target, rather than
                                one sequential job looping over all targets. Each
                                target gets its own memory/time budget and can run in
                                parallel with the others; a re-submission skips
                                targets whose output already exists, matching every
                                other stage here.
  --run-all-figures            Export every registered reporting figure family,
                               including opt-in dataset characterization figures.
  --include-site-comparison-figures
                               Include site-vs-combined and cross-site concordance
                               figures during reporting using existing site-cascade
                               artifacts. This does not rerun site cascade; use
                               --run-site-cascade without --skip-cascade when those
                               artifacts need to be regenerated.
  --run-audit                  Run scientific traceability/contract audit after
                               manuscript-facing outputs are generated.
  --run-readiness-audit        Run publication-readiness audit after each organism's
                               report outputs are generated.
  --run-publication-package    Shortcut for --run-site-cascade --run-features
                               --run-training --run-eskape --run-all-figures
                               --run-audit --run-readiness-audit.
  --eskape-targets T1,T2        Default: Enterococcus,Staphylococcus,Klebsiella,
                                Acinetobacter,Pseudomonas,Enterobacter
  --delete-venv                Delete the existing .venv before bootstrapping,
                               forcing a clean reinstall of all Python packages.
                               Use this when switching Python versions or when
                               packages are corrupted.
  --force-rerun-existing       Submit enabled stages even when outputs exist.
  --dry-run                    Print sbatch commands without submitting.
  --wait                       After submission, block until all jobs finish
                               and print a final per-job status table. Exits
                               non-zero if any job did not complete cleanly.
  --monitor-interval N         Seconds between status polls when --wait is
                               active. Default: 60.
  --status JOBFILE             Skip submission and instead report the current
                               state of jobs listed in JOBFILE (one job ID
                               per line; lines starting with # ignored).
  --help                       Show this message.

Memory/time environment overrides (defaults below include a safety margin
to avoid TIMEOUT cuts; lower them only if you are confident a stage will
finish faster on your data):
  CPUS_PER_TASK       default 8
  INGEST_MEM          default 128G       INGEST_TIME          default 12:00:00
  PREPROCESS_MEM      default 256G       PREPROCESS_TIME      default 16:00:00
  HARMONIZE_MEM       default 384G       HARMONIZE_TIME       default 20:00:00
  COMORBIDITY_MEM     default 256G       COMORBIDITY_TIME     default 16:00:00
  GOLD_MEM            default 256G       GOLD_TIME            default 16:00:00
  CASCADE_MEM         default 512G       CASCADE_TIME         default 24:00:00
  CASCADE_SHARD_NUM   default 41         (edges per shard ≈ ceil(total/41), ~40 for E. coli)
  CASCADE_SHARD_MEM   default 512G       CASCADE_SHARD_TIME   default 36:00:00
  CASCADE_MERGE_MEM   default 64G        CASCADE_MERGE_TIME   default 04:00:00
                      (shard jobs replay the pre-validation pipeline then validate ~40 edges
                      each; ~24h per shard. Merge job applies global BH-FDR across all
                      shards. Main cascade job loads the merged parquet and skips validation.)
  CASCADE_SITE_SHARD_NUM  default 22     (used only with --run-site-cascade; sized for the
                      largest observed single-site edge count, ~850-900 edges, at ~40/shard.
                      Reuses CASCADE_SHARD_MEM/TIME and CASCADE_MERGE_MEM/TIME -- per-site
                      shards are the same scale of work as combined-scope shards.)
  --cascade-partition NAME     Partition for cascade, features, training, and ESKAPE
                               jobs. Default: long (has Python 3.12 and sufficient
                               wall-time). Must have Python 3.12 — jobs use the same
                               .venv and the interpreter must exist on its nodes.
  CASCADE_PARTITION   env-var override for the above (same semantics)
  PREVALENCE_MEM      default 256G       PREVALENCE_TIME      default 16:00:00
  FEATURE_MEM         default 512G       FEATURE_TIME         default 12:00:00
  TRAINING_MEM        default 384G       TRAINING_TIME        default 12:00:00
  REPORT_MEM          default 384G       REPORT_TIME          default 08:00:00
  ESKAPE_MEM          default 512G       ESKAPE_TIME          default 48:00:00
                      (per ESKAPE target; each target job filters cohort, builds its
                      own gold layer, and runs full cascade validation -- comparable
                      weight to CASCADE_MEM/CASCADE_TIME, not a lighter stage.)

IMPORTANT — run this script directly in a tmux/screen session, NOT via sbatch.
The submission itself is fast (seconds). The individual jobs run under SLURM
with their own time limits and continue even if you disconnect.
  tmux new -s pipeline
  bash scripts/submit_pipeline_dag_hpc.sh --run-features --run-training
  # Ctrl-B D to detach; tmux attach -t pipeline to reattach
If you accidentally sbatch this script it will be killed by the partition's
wall-clock limit before all jobs finish submitting.
EOF
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

SITES=(armd armd_ecuh armd_utsw)
ORGANISMS=("ESCHERICHIA COLI")
ESKAPE_TARGETS=(Enterococcus Staphylococcus Klebsiella Acinetobacter Pseudomonas Enterobacter)
PARTITION="main"
PYTHON_BIN="${PYTHON_BIN:-}"
# Never inherit VENV_SITE from the caller. A stale path would point at packages
# for a different Python version or a different venv rebuild.
VENV_SITE=""
DELETE_VENV=0
DRY_RUN=0
WAIT_FOR_JOBS=0
MONITOR_INTERVAL=60
STATUS_FILE=""

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
RUN_ESKAPE=0
RUN_ALL_FIGURES=0
RUN_SITE_COMPARISON_FIGURES=0
RUN_AUDIT=0
RUN_READINESS_AUDIT=0
REUSE_EXISTING=1

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${raw}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites)
      [[ $# -ge 2 ]] || { echo "--sites requires a value" >&2; exit 2; }
      split_csv "$2" SITES
      shift 2
      ;;
    --organisms)
      [[ $# -ge 2 ]] || { echo "--organisms requires a value" >&2; exit 2; }
      split_csv "$2" ORGANISMS
      shift 2
      ;;
    --partition)
      [[ $# -ge 2 ]] || { echo "--partition requires a value" >&2; exit 2; }
      PARTITION="$2"
      shift 2
      ;;
    --cascade-partition)
      [[ $# -ge 2 ]] || { echo "--cascade-partition requires a value" >&2; exit 2; }
      CASCADE_PARTITION="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || { echo "--python-bin requires a value" >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-ingestion) RUN_INGESTION=0; shift ;;
    --skip-preprocessing) RUN_PREPROCESSING=0; shift ;;
    --skip-harmonization) RUN_HARMONIZATION=0; shift ;;
    --skip-comorbidity) RUN_COMORBIDITY=0; shift ;;
    --skip-gold) RUN_GOLD=0; shift ;;
    --skip-cascade) RUN_CASCADE=0; shift ;;
    --skip-prevalence) RUN_PREVALENCE=0; shift ;;
    --skip-reporting) RUN_REPORTING=0; shift ;;
    --run-site-cascade) RUN_SITE_CASCADE=1; shift ;;
    --run-features) RUN_FEATURES=1; shift ;;
    --run-training) RUN_TRAINING=1; shift ;;
    --run-eskape) RUN_ESKAPE=1; shift ;;
    --run-all-figures) RUN_ALL_FIGURES=1; shift ;;
    --include-site-comparison-figures) RUN_SITE_COMPARISON_FIGURES=1; shift ;;
    --run-audit) RUN_AUDIT=1; shift ;;
    --run-readiness-audit) RUN_READINESS_AUDIT=1; shift ;;
    --run-publication-package)
      RUN_SITE_CASCADE=1
      RUN_FEATURES=1
      RUN_TRAINING=1
      RUN_ESKAPE=1
      RUN_ALL_FIGURES=1
      RUN_SITE_COMPARISON_FIGURES=1
      RUN_AUDIT=1
      RUN_READINESS_AUDIT=1
      shift
      ;;
    --eskape-targets)
      [[ $# -ge 2 ]] || { echo "--eskape-targets requires a value" >&2; exit 2; }
      split_csv "$2" ESKAPE_TARGETS
      shift 2
      ;;
    --delete-venv) DELETE_VENV=1; shift ;;
    --force-rerun-existing) REUSE_EXISTING=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --wait) WAIT_FOR_JOBS=1; shift ;;
    --monitor-interval)
      [[ $# -ge 2 ]] || { echo "--monitor-interval requires a value" >&2; exit 2; }
      MONITOR_INTERVAL="$2"
      shift 2
      ;;
    --status)
      [[ $# -ge 2 ]] || { echo "--status requires a job-id file" >&2; exit 2; }
      STATUS_FILE="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RUN_SITE_CASCADE}" -eq 1 && "${RUN_CASCADE}" -eq 0 ]]; then
  echo "ERROR: --run-site-cascade requires cascade analysis, but --skip-cascade was also supplied." >&2
  echo "Remove --skip-cascade to regenerate site-cascade comparison artifacts, or omit --run-site-cascade for a combined-only output refresh." >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "ERROR: no python3 or python found in PATH." >&2
  exit 1
fi

if [[ -z "${VENV_SITE}" && -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  # Build path from the running Python's actual version_info — sysconfig can
  # return a stale cross-version path when the venv was previously created by
  # a different Python minor version.
  VENV_SITE="$(PYTHONNOUSERSITE=1 "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import sys, os; print(os.path.join(sys.prefix, "lib",
     f"python{sys.version_info.major}.{sys.version_info.minor}",
     "site-packages"))' 2>/dev/null || true)"
fi

python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

python_version_string() {
  "$1" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || printf 'unknown'
}

find_base_python() {
  local project_python="${PROJECT_ROOT}/.venv/bin/python"
  local candidates=()
  if [[ -n "${PYTHON_BIN}" && "${PYTHON_BIN}" != "${project_python}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  # Prefer PATH-resolved python3.12 first — on HPC clusters, Python is usually
  # made available via the module system or a shared PATH entry that works on
  # every node type (login, main, long, …).  Hardcoded absolute paths come last
  # because the same path may not exist on every partition.
  candidates+=(python3.12 python3 python
               /usr/bin/python3.12 /usr/local/bin/python3.12
               /opt/miniconda/bin/python3.12 /opt/miniconda3/bin/python3.12)

  local candidate resolved
  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" = /* ]]; then
      [[ -x "${candidate}" ]] || continue
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
      [[ -n "${resolved}" ]] || continue
    fi
    if python_is_supported "${resolved}"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  return 1
}

# Bootstrap the Python runtime before submitting any jobs. This is intentionally
# strict: if the runtime cannot be created, populated, and imported from the
# exact Python path that SLURM will run, the script exits before submitting any
# jobs. Submitting jobs against a missing/incomplete Python environment only
# produces a fast wall of failed jobs.

bootstrap_python_runtime() {
  local project_python="${PROJECT_ROOT}/.venv/bin/python"
  local requirements_file="${PROJECT_ROOT}/requirements-hpc.txt"
  # Validate with a clean environment: no PYTHONPATH, no user site-packages.
  # This mirrors what compute nodes see inside each SLURM job wrapper.
  local runtime_ok='import pandas, pyarrow, scipy, sklearn, amr_cascade_platform'

  if command -v git >/dev/null 2>&1 && git -C "${PROJECT_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "${PROJECT_ROOT}" fetch --quiet 2>/dev/null || true
    local behind
    behind="$(git -C "${PROJECT_ROOT}" rev-list --count HEAD..@{u} 2>/dev/null || echo 0)"
    if [[ "${behind}" -gt 0 ]]; then
      echo "WARNING: local branch is ${behind} commit(s) behind upstream." >&2
      echo "         Run 'git pull' in ${PROJECT_ROOT} before submitting." >&2
    fi
  fi

  if [[ ! -f "${requirements_file}" ]]; then
    echo "ERROR: requirements-hpc.txt not found at ${requirements_file}." >&2
    return 1
  fi

  # Fast path: accept an existing venv only if it passes the same clean import
  # check submitted jobs run. Do not rely on PROJECT_ROOT/src in PYTHONPATH here:
  # jobs do not set it before check_python_runtime.py.
  if [[ -x "${project_python}" ]]; then
    if python_is_supported "${project_python}"; then
      local _site
      _site="$(PYTHONNOUSERSITE=1 "${project_python}" -c \
        'import sys, os; print(os.path.join(sys.prefix, "lib",
         f"python{sys.version_info.major}.{sys.version_info.minor}",
         "site-packages"))' 2>/dev/null || true)"
      if [[ -n "${_site}" ]] && PYTHONNOUSERSITE=1 "${project_python}" -c "${runtime_ok}" 2>/dev/null; then
        VENV_SITE="${_site}"
        PYTHON="$(readlink -f "${project_python}" 2>/dev/null || printf '%s' "${project_python}")"
        echo "Runtime ready (existing env): $("${project_python}" --version 2>&1)" >&2
        echo "Venv Python resolves to: ${PYTHON}" >&2
        echo "Package site-packages: ${VENV_SITE}" >&2
        return 0
      fi
      echo "Existing venv failed clean import validation; rebuilding." >&2
    else
      echo "Existing venv uses unsupported Python $(python_version_string "${project_python}"); rebuilding with Python >=3.12." >&2
    fi
  fi

  # Find a base Python (not the .venv itself — it is broken or missing).
  local base_python=""
  base_python="$(find_base_python || true)"
  if [[ -z "${base_python}" ]]; then
    echo "ERROR: no Python >=3.12 found in PATH." >&2
    echo "       Tried: /usr/bin/python3.12, python3.12, python3, python." >&2
    echo "       Current python3: $(python3 --version 2>&1 || true)" >&2
    echo "       Pass --python-bin /usr/bin/python3.12 or the full path to the system Python 3.12." >&2
    return 1
  fi

  echo "Creating venv at ${PROJECT_ROOT}/.venv using $("${base_python}" --version 2>&1) ..." >&2
  rm -rf "${PROJECT_ROOT}/.venv"
  # No --copies: /scratch is mounted noexec so copied ELF binaries cannot run.
  # Without --copies, .venv/bin/python is a symlink to base_python which lives
  # on an exec filesystem and resolves correctly on every compute node.
  "${base_python}" -m venv "${PROJECT_ROOT}/.venv" || {
    echo "ERROR: could not create virtualenv using ${base_python}." >&2
    return 1
  }
  local venv_python="${project_python}"
  if [[ ! -x "${venv_python}" ]]; then
    echo "ERROR: virtualenv was created but ${venv_python} is not executable." >&2
    ls -la "${PROJECT_ROOT}/.venv/bin" >&2 || true
    return 1
  fi
  echo "Venv Python resolves to: $(readlink -f "${venv_python}" 2>/dev/null || printf '%s' "${venv_python}")" >&2

  echo "Upgrading pip ..." >&2
  PYTHONNOUSERSITE=1 "${venv_python}" -m pip install --upgrade pip 2>&1 | tail -2 >&2 || true

  echo "Installing requirements from ${requirements_file} ..." >&2
  PYTHONNOUSERSITE=1 "${venv_python}" -m pip install -r "${requirements_file}" || {
    echo "" >&2
    echo "ERROR: pip install failed — likely no internet on this node." >&2
    echo "Run the script with 'bash' (not sbatch) from the login node." >&2
    return 1
  }

  echo "Installing project package ..." >&2
  PYTHONNOUSERSITE=1 "${venv_python}" -m pip install -e "${PROJECT_ROOT}" || {
    echo "ERROR: could not install project package." >&2
    return 1
  }

  # Use sys.version_info to construct the site-packages path — sysconfig.get_path
  # can return a stale path when a venv was previously created by a different
  # Python minor version.
  VENV_SITE="$(PYTHONNOUSERSITE=1 "${venv_python}" -c \
    'import sys, os; print(os.path.join(sys.prefix, "lib",
     f"python{sys.version_info.major}.{sys.version_info.minor}",
     "site-packages"))')"
  echo "Python version in venv: $("${venv_python}" --version 2>&1)" >&2
  echo "Site-packages: ${VENV_SITE}" >&2

  echo "Validating imports (clean environment) ..." >&2
  if PYTHONNOUSERSITE=1 "${venv_python}" -c "${runtime_ok}"; then
    PYTHON="$(readlink -f "${venv_python}" 2>/dev/null || printf '%s' "${venv_python}")"
    echo "Runtime ready: $("${venv_python}" --version 2>&1)" >&2
    return 0
  fi
  echo "ERROR: package import validation failed after install." >&2
  return 1
}

if [[ "${DELETE_VENV}" -eq 1 && -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "Deleting existing venv at ${PROJECT_ROOT}/.venv ..." >&2
  rm -rf "${PROJECT_ROOT}/.venv"
  echo "Venv deleted." >&2
fi

if [[ -z "${STATUS_FILE}" && "${DRY_RUN}" -eq 0 ]]; then
  bootstrap_python_runtime || exit 1
fi

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
INGEST_MEM="${INGEST_MEM:-128G}"
PREPROCESS_MEM="${PREPROCESS_MEM:-256G}"
HARMONIZE_MEM="${HARMONIZE_MEM:-384G}"
COMORBIDITY_MEM="${COMORBIDITY_MEM:-256G}"
GOLD_MEM="${GOLD_MEM:-256G}"
CASCADE_MEM="${CASCADE_MEM:-512G}"
# Shard jobs re-run the pre-validation steps (cotesting filter, ER, regression,
# retained edges) and then validate their slice of ~40 edges at B=1000.
# Each shard takes ~4h (pre-validation replay) + ~20h (validation) ≈ 24h.
# Use 36h to be safe; the merge job is lightweight (~64G, 4h).
CASCADE_SHARD_NUM="${CASCADE_SHARD_NUM:-41}"
CASCADE_SHARD_MEM="${CASCADE_SHARD_MEM:-512G}"
CASCADE_SHARD_TIME="${CASCADE_SHARD_TIME:-36:00:00}"
CASCADE_MERGE_MEM="${CASCADE_MERGE_MEM:-64G}"
CASCADE_MERGE_TIME="${CASCADE_MERGE_TIME:-04:00:00}"
CASCADE_SITE_SHARD_NUM="${CASCADE_SITE_SHARD_NUM:-22}"
PREVALENCE_MEM="${PREVALENCE_MEM:-256G}"
FEATURE_MEM="${FEATURE_MEM:-512G}"
TRAINING_MEM="${TRAINING_MEM:-384G}"
REPORT_MEM="${REPORT_MEM:-384G}"
ESKAPE_MEM="${ESKAPE_MEM:-512G}"
AUDIT_MEM="${AUDIT_MEM:-64G}"
READINESS_AUDIT_MEM="${READINESS_AUDIT_MEM:-32G}"

INGEST_TIME="${INGEST_TIME:-12:00:00}"
PREPROCESS_TIME="${PREPROCESS_TIME:-16:00:00}"
HARMONIZE_TIME="${HARMONIZE_TIME:-20:00:00}"
COMORBIDITY_TIME="${COMORBIDITY_TIME:-16:00:00}"
GOLD_TIME="${GOLD_TIME:-16:00:00}"
CASCADE_TIME="${CASCADE_TIME:-24:00:00}"
# long partition has Python 3.12 and sufficient wall-time for cascade/features/training/ESKAPE.
# Override with --cascade-partition or CASCADE_PARTITION=<name> if needed.
CASCADE_PARTITION="${CASCADE_PARTITION:-long}"
PREVALENCE_TIME="${PREVALENCE_TIME:-16:00:00}"
FEATURE_TIME="${FEATURE_TIME:-12:00:00}"
TRAINING_TIME="${TRAINING_TIME:-12:00:00}"
REPORT_TIME="${REPORT_TIME:-08:00:00}"
ESKAPE_TIME="${ESKAPE_TIME:-48:00:00}"
AUDIT_TIME="${AUDIT_TIME:-02:00:00}"
READINESS_AUDIT_TIME="${READINESS_AUDIT_TIME:-01:00:00}"

REPORT_FIGURE_ARGS=""
if [[ "${RUN_ALL_FIGURES}" -eq 1 ]]; then
  REPORT_FIGURE_ARGS="--fail-on-missing-figures --figure primary_cascade_forest --figure validated_primary_cascade_forest --figure aware_transition_heatmap --figure prevalence_shift_forest --figure prevalence_shift_curves --figure mnar_tipping_point --figure upstream_contribution_forest --figure downstream_trigger_forest --figure threshold_sensitivity --figure cascade_sankey --figure cascade_chord --figure cascade_network --figure model_metrics_comparison --figure model_pr_curve --figure model_roc_curve --figure model_calibration --figure model_threshold_analysis --figure validation_diagnostics --figure er_landscape --figure mnar_prevalence_shift_distribution --figure validation_funnel --figure panel_bundling --figure temporal_stability --figure consort_diagram --figure dataset_characterization"
  if [[ "${RUN_SITE_CASCADE}" -eq 1 || "${RUN_SITE_COMPARISON_FIGURES}" -eq 1 ]]; then
    REPORT_FIGURE_ARGS="${REPORT_FIGURE_ARGS} --figure site_vs_combined_summary --figure cross_site_concordance"
  fi
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

trim() {
  printf '%s' "$1" | sed -E 's/^ +//; s/ +$//'
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
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

bronze_ready() {
  local site="$1"
  [[ -e "${PROJECT_ROOT}/data/bronze/${site}/cohort" &&
     -e "${PROJECT_ROOT}/data/bronze/${site}/microbial_resistance" ]]
}

silver_ready() {
  local site="$1"
  [[ -e "${PROJECT_ROOT}/data/silver/${site}/cohort.parquet" &&
     -e "${PROJECT_ROOT}/data/silver/${site}/microbial_resistance.parquet" ]]
}

harmonized_ready() {
  local site
  for site in "${SITES[@]}"; do
    site="$(trim "${site}")"
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
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/gold" "${scope}" "${site}" "${organism}")"
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
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/cascade" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/retained_edges.parquet" &&
     -f "${dir}/edge_report.parquet" &&
     -f "${dir}/network_nodes.parquet" &&
     -f "${dir}/cascade_summary.json" ]]
}

# Returns true when the shard merge step has written validation_results.parquet.
# This is the handoff point between the shard jobs and the main cascade job.
validation_merge_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/cascade" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/validation_results.parquet" ]]
}

# Returns true when a specific shard parquet has already been written.
shard_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local shard_idx="$4"
  local shard_total="$5"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/cascade" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/validation_shards/shard_$(printf '%04d' "${shard_idx}")_of_$(printf '%04d' "${shard_total}").parquet" ]]
}

prevalence_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/prevalence_shift" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/prevalence_shift.parquet" &&
     -f "${dir}/prevalence_mnar_sensitivity_curves.parquet" &&
     -f "${dir}/prevalence_mnar_tipping_points.parquet" &&
     -f "${dir}/prevalence_shift_summary.json" ]]
}

feature_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/features" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/model_ready_pair_features.parquet" &&
     -f "${dir}/feature_build_summary.json" ]]
}

training_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/data/artifacts/modeling/downstream_testing" "${scope}" "${site}" "${organism}")"
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
  dir="$(scope_dir_path "${PROJECT_ROOT}/outputs/reports" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/report_manifest.json" ]]
}

scientific_audit_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/outputs/reports" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/cascade_traceability_audit.csv" &&
     -f "${dir}/pipeline_contract_audit.csv" &&
     -f "${dir}/scientific_qa_summary.json" ]]
}

publication_readiness_ready() {
  local scope="$1"
  local site="${2:-}"
  local organism="${3:-}"
  local dir
  dir="$(scope_dir_path "${PROJECT_ROOT}/outputs/reports" "${scope}" "${site}" "${organism}")"
  [[ -f "${dir}/audits/publication_readiness_audit.json" &&
     -f "${dir}/audits/publication_readiness_audit.csv" &&
     -f "${dir}/audits/publication_readiness_audit.md" ]]
}

eskape_ready() {
  local scope="$1"
  local site="${2:-}"
  local target="$3"
  local scope_dir
  if [[ "${scope}" == "combined" ]]; then
    scope_dir="combined"
  else
    scope_dir="${site}"
  fi
  local target_slug
  target_slug="$(slugify "${target}")"
  [[ -f "${PROJECT_ROOT}/data/artifacts/eskape_validation/armd/${scope_dir}/organisms/${target_slug}/eskape_existence_summary.csv" ]]
}

eskape_merge_ready() {
  local scope="$1"
  local site="${2:-}"
  local dir
  if [[ "${scope}" == "combined" ]]; then
    dir="${PROJECT_ROOT}/outputs/tables/eskape/combined_armd"
  else
    dir="${PROJECT_ROOT}/outputs/tables/eskape/site_${site}"
  fi
  [[ -f "${dir}/eskape_cascade_comparison.csv" ]]
}

should_submit() {
  local deps="$1"
  shift
  if [[ "${REUSE_EXISTING}" -eq 0 ]]; then
    return 0
  fi
  if [[ -n "${deps}" ]]; then
    return 0
  fi
  "$@" && return 1
  return 0
}

skip_ready() {
  local label="$1"
  echo "Ready; not submitting ${label}"
}

join_colon() {
  local IFS=':'
  printf '%s' "$*"
}

dependency_arg() {
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  local ids=()
  local id
  for id in "$@"; do
    [[ -n "${id}" ]] && ids+=("${id}")
  done
  [[ "${#ids[@]}" -eq 0 ]] && return 0
  printf '%s' "--dependency=afterok:$(join_colon "${ids[@]}")"
}

# bootstrap_python_runtime() {
#   local project_python="${PROJECT_ROOT}/.venv/bin/python"
#   local requirements_file="${PROJECT_ROOT}/requirements-hpc.txt"

#   # Warn if there are unpulled upstream commits so the user knows to git pull
#   # before submitting.  Non-fatal — some HPC scratch dirs are not git repos.
#   if command -v git >/dev/null 2>&1 && git -C "${PROJECT_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
#     git -C "${PROJECT_ROOT}" fetch --quiet 2>/dev/null || true
#     local behind
#     behind="$(git -C "${PROJECT_ROOT}" rev-list --count HEAD..@{u} 2>/dev/null || echo 0)"
#     if [[ "${behind}" -gt 0 ]]; then
#       echo "WARNING: local branch is ${behind} commit(s) behind upstream." >&2
#       echo "         Run 'git pull' in ${PROJECT_ROOT} before submitting to pick up memory fixes." >&2
#     fi
#   fi

#   if [[ ! -f "${requirements_file}" ]]; then
#     echo "WARNING: requirements-hpc.txt not found at ${requirements_file}; skipping venv bootstrap." >&2
#     return 0
#   fi

#   if [[ ! -x "${project_python}" ]]; then
#     echo "Creating repo-local virtualenv at ${PROJECT_ROOT}/.venv ..." >&2
#     "${PYTHON}" -m venv "${PROJECT_ROOT}/.venv"
#   fi

#   PYTHON="${project_python}"
#   echo "Using repo-local virtualenv: ${PYTHON}" >&2

#   local ok
#   ok="$("${PYTHON}" -c 'import pandas, pyarrow, scipy, sklearn; print("ok")' 2>/dev/null || echo "missing")"
#   if [[ "${ok}" != "ok" ]]; then
#     echo "Installing HPC dependencies into ${PROJECT_ROOT}/.venv ..." >&2
#     "${PYTHON}" -m pip install --upgrade pip -q
#     "${PYTHON}" -m pip install -r "${requirements_file}" -q
#   fi

#   # Verify the project package itself is importable.
#   "${PYTHON}" -c 'import amr_cascade_platform' 2>/dev/null || {
#     echo "Installing project package (editable) ..." >&2
#     "${PYTHON}" -m pip install -e "${PROJECT_ROOT}" -q
#   }

#   echo "Python runtime ready: $("${PYTHON}" --version 2>&1)" >&2
# }

submit_job() {
  local name="$1"
  local time_limit="$2"
  local mem="$3"
  local deps_raw="$4"
  shift 4
  local command="$*"
  local deps=()
  if [[ -n "${deps_raw}" ]]; then
    IFS=':' read -r -a deps <<< "${deps_raw}"
  fi
  local args=(
    --parsable
    --partition="${PARTITION}"
    --job-name="${name}"
    --output="${LOG_DIR}/${name}_%j.out"
    --error="${LOG_DIR}/${name}_%j.err"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${mem}"
    --time="${time_limit}"
    --kill-on-invalid-dep=yes
  )
  local dep_arg
  dep_arg="$(dependency_arg "${deps[@]}")"
  [[ -n "${dep_arg}" ]] && args+=("${dep_arg}")
  # Jobs run the venv Python directly.  The venv is the source of truth for
  # both the interpreter and the packages — no PATH detection needed at runtime.
  # If the venv symlink is broken on a node (meaning the base Python used to
  # create it doesn't exist there), the fix is to delete the venv and rerun
  # with --python-bin pointing to a python3.12 that exists on all node types.
  args+=(--wrap "cd '${PROJECT_ROOT}' && export PYTHONNOUSERSITE=1 && export MPLCONFIGDIR='${PROJECT_ROOT}/.cache/matplotlib' && mkdir -p \"\${MPLCONFIGDIR}\" && export AMR_CASCADE_PYTHON='${PROJECT_ROOT}/.venv/bin/python' && \"\${AMR_CASCADE_PYTHON}\" scripts/check_python_runtime.py || { echo \"ERROR: venv Python not working on \$(hostname): \${AMR_CASCADE_PYTHON}\" >&2; echo \"Delete the venv and rerun with --python-bin pointing to a python3.12 that exists on all nodes.\" >&2; exit 127; } && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS='${CPUS_PER_TASK}' && export OPENBLAS_NUM_THREADS='${CPUS_PER_TASK}' && export MKL_NUM_THREADS='${CPUS_PER_TASK}' && ${command}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'sbatch' >&2
    printf ' %q' "${args[@]}" >&2
    printf '\n' >&2
    printf 'DRYRUN_%s\n' "${name}"
    return 0
  fi
  sbatch "${args[@]}"
}

submitted_jobs=()
site_terminal_jobs=()
comorbidity_jobs=()

wait_for_jobs() {
  local job_ids=("$@")
  [[ "${#job_ids[@]}" -eq 0 ]] && return 0
  local job_csv
  job_csv="$(IFS=,; printf '%s' "${job_ids[*]}")"

  # Per-poll event log file (one per --wait invocation).
  local event_log="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').events.log"
  echo
  echo "Watching ${#job_ids[@]} job(s). Polling every ${MONITOR_INTERVAL}s."
  echo "Live stage events are also written to: ${event_log}"
  echo "Press Ctrl-C to stop watching; jobs continue to run in SLURM."
  echo

  # Helper: emit one event line to both stdout and the events log file.
  emit() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "${ts}" "$*" | tee -a "${event_log}"
  }

  # Build job-id -> job-name map once (best effort; sacct may lag at submit).
  declare -A jobname
  if command -v sacct >/dev/null 2>&1; then
    local jid jn
    while IFS='|' read -r jid jn; do
      [[ -z "${jid}" || "${jid}" =~ \.(batch|extern|[0-9]+)$ ]] && continue
      jobname["${jid}"]="${jn}"
    done < <(sacct -j "${job_csv}" --format=JobID,JobName --parsable2 --noheader 2>/dev/null)
  fi
  # Fill any gaps with the raw job ID so emit() always has a label.
  local id
  for id in "${job_ids[@]}"; do
    [[ -n "${jobname[${id}]:-}" ]] || jobname["${id}"]="${id}"
  done

  declare -A prev_state
  for id in "${job_ids[@]}"; do
    prev_state["${id}"]="PENDING"
  done

  emit "WATCHING ${#job_ids[@]} job(s): ${job_csv}"

  local poll_count=0
  while true; do
    poll_count=$((poll_count + 1))
    declare -A curr_state
    # Query current state for every job in one batch call.
    if command -v sacct >/dev/null 2>&1; then
      while IFS='|' read -r jid st; do
        [[ -z "${jid}" || "${jid}" =~ \.(batch|extern|[0-9]+)$ ]] && continue
        curr_state["${jid}"]="${st}"
      done < <(sacct -j "${job_csv}" --format=JobID,State --parsable2 --noheader 2>/dev/null)
    fi

    # Detect state transitions for each tracked job and emit events.
    local active=0
    for id in "${job_ids[@]}"; do
      local prev="${prev_state[${id}]}"
      local curr="${curr_state[${id}]:-${prev}}"
      local name="${jobname[${id}]}"
      if [[ "${prev}" != "${curr}" ]]; then
        case "${curr}" in
          RUNNING)
            emit "STARTED   ${name}   (jobid=${id})"
            ;;
          COMPLETED)
            local elapsed
            elapsed="$(sacct -j "${id}" --format=Elapsed --noheader -P 2>/dev/null \
              | awk 'NR==1 && $0 != "" {print; exit}')"
            emit "FINISHED  ${name}   (jobid=${id}, elapsed ${elapsed:-unknown})"
            ;;
          FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|BOOT_FAIL|OUT_OF_MEMORY)
            emit "FAILED    ${name}   (jobid=${id}, state=${curr})"
            ;;
          *)
            emit "STATE     ${name}   (jobid=${id}) ${prev} -> ${curr}"
            ;;
        esac
        prev_state["${id}"]="${curr}"
      fi
      case "${curr}" in
        PENDING|RUNNING|CONFIGURING|SUSPENDED|REQUEUED) active=$((active + 1)) ;;
      esac
    done

    if [[ "${active}" -eq 0 ]]; then
      emit "ALL DONE  every job has left the active queue."
      break
    fi

    # Periodic summary line so the user knows the watcher is alive even
    # when nothing has transitioned. Default: every 10 polls (~10 minutes
    # at the default poll interval).
    if [[ "${poll_count}" -eq 1 || $((poll_count % 10)) -eq 0 ]]; then
      emit "SUMMARY   ${active} of ${#job_ids[@]} job(s) still active (poll #${poll_count})"
    fi

    sleep "${MONITOR_INTERVAL}"
  done
}

report_final_states() {
  local job_ids=("$@")
  [[ "${#job_ids[@]}" -eq 0 ]] && return 0
  local job_csv
  job_csv="$(IFS=,; printf '%s' "${job_ids[*]}")"
  echo
  echo "Pipeline stage timeline (per-job submit/start/end times and final state):"
  if ! command -v sacct >/dev/null 2>&1; then
    echo "(sacct not on PATH; run manually: sacct -j ${job_csv})"
    return 0
  fi
  # Build a timeline log file path with the current timestamp so each
  # status report saves to its own file (avoids clobbering on re-attach).
  local timeline_file="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').timeline.log"
  # Strip .batch/.extern step rows; keep only top-level job rows.
  # Print to stdout and tee to the timeline log so a dropped session still
  # leaves the timing record on disk.
  {
    printf '# Pipeline timeline generated %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '# Job IDs: %s\n' "${job_csv}"
    sacct -j "${job_csv}" --format=JobID,JobName%30,Submit,Start,End,Elapsed,State --noheader \
      | awk '$1 !~ /\.(batch|extern|[0-9]+)$/ {print}'
  } | tee "${timeline_file}"
  echo
  echo "Timeline saved to: ${timeline_file}"
  echo
  local bad
  bad="$(sacct -j "${job_csv}" --format=JobID,State --parsable2 --noheader \
    | awk -F'|' '$1 !~ /\.(batch|extern|[0-9]+)$/ && $2 != "COMPLETED" && $2 != "RUNNING" && $2 != "PENDING" {print $1 "|" $2}')"
  if [[ -n "${bad}" ]]; then
    echo "WARNING: one or more jobs did not complete cleanly:"
    printf '  %s\n' ${bad}
    return 1
  fi
  echo "All jobs completed successfully."
  return 0
}

load_status_file() {
  local file="$1"
  [[ -f "${file}" ]] || { echo "Status file not found: ${file}" >&2; exit 2; }
  local id
  while IFS= read -r line || [[ -n "${line}" ]]; do
    id="$(printf '%s' "${line}" | sed -E 's/[[:space:]]+//g')"
    [[ -z "${id}" || "${id}" == \#* ]] && continue
    submitted_jobs+=("${id}")
  done < "${file}"
}

submit_or_skip() {
  local label="$1"
  local ready_func="$2"
  local deps="$3"
  local job_name="$4"
  local time_limit="$5"
  local mem="$6"
  shift 6
  if should_submit "${deps}" ${ready_func}; then
    submit_job "${job_name}" "${time_limit}" "${mem}" "${deps}" "$@"
  else
    skip_ready "${label}" >&2
    printf '\n'
  fi
}

if [[ -n "${STATUS_FILE}" ]]; then
  echo "Reading job IDs from ${STATUS_FILE}..."
  load_status_file "${STATUS_FILE}"
  echo "Loaded ${#submitted_jobs[@]} job ID(s)."
  if report_final_states "${submitted_jobs[@]}"; then
    exit 0
  else
    exit 1
  fi
fi

echo "Submitting AMR pipeline DAG"
echo "Project root: ${PROJECT_ROOT}"
echo "Venv Python (jobs use): ${PROJECT_ROOT}/.venv/bin/python -> $(readlink -f "${PROJECT_ROOT}/.venv/bin/python" 2>/dev/null || echo "unresolved")"
echo "Runtime Python: ${PYTHON}"
echo "Sites: ${SITES[*]}"
echo "Organisms: ${ORGANISMS[*]}"
echo "Partition: ${PARTITION}"
echo "Site cascade: ${RUN_SITE_CASCADE}"
echo "Features: ${RUN_FEATURES}"
echo "Training: ${RUN_TRAINING}"
echo "ESKAPE: ${RUN_ESKAPE}"
echo "All report figures: ${RUN_ALL_FIGURES}"
echo "Site-comparison figures: ${RUN_SITE_COMPARISON_FIGURES}"
echo "Scientific audit: ${RUN_AUDIT}"
echo "Publication readiness audit: ${RUN_READINESS_AUDIT}"
echo "Reuse existing outputs: ${REUSE_EXISTING}"
echo "Wait for completion: ${WAIT_FOR_JOBS}"

publication_terminal_jobs=()

for raw_site in "${SITES[@]}"; do
  site="$(trim "${raw_site}")"
  [[ -n "${site}" ]] || continue
  site_slug="$(slugify "${site}")"
  deps=""

  if [[ "${RUN_INGESTION}" -eq 1 ]]; then
    if should_submit "" bronze_ready "${site}"; then
      ingest_job="$(
        submit_job "amr_${site_slug}_ingest" "${INGEST_TIME}" "${INGEST_MEM}" "" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_ingestion.py --env hpc --site '${site}' --source-layer raw"
      )"
      echo "Ingestion ${site}: ${ingest_job}"
      submitted_jobs+=("${ingest_job}")
      deps="${ingest_job}"
    else
      skip_ready "ingestion ${site}"
    fi
  fi

  if [[ "${RUN_PREPROCESSING}" -eq 1 ]]; then
    if should_submit "${deps}" silver_ready "${site}"; then
      preprocess_job="$(
        submit_job "amr_${site_slug}_preprocess" "${PREPROCESS_TIME}" "${PREPROCESS_MEM}" "${deps}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_preprocessing.py --env hpc --site '${site}'"
      )"
      echo "Preprocessing ${site}: ${preprocess_job}"
      submitted_jobs+=("${preprocess_job}")
      deps="${preprocess_job}"
    else
      skip_ready "preprocessing ${site}"
    fi
  fi

  [[ -n "${deps}" ]] && site_terminal_jobs+=("${deps}")
done

harmonize_deps="$(join_colon "${site_terminal_jobs[@]}")"
if [[ "${RUN_HARMONIZATION}" -eq 1 ]]; then
  harmonize_command="\"\${AMR_CASCADE_PYTHON}\" scripts/run_harmonization.py --env hpc"
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] && harmonize_command+=" --site '${site}'"
  done
  if should_submit "${harmonize_deps}" harmonized_ready; then
    harmonize_job="$(
      submit_job "amr_harmonize" "${HARMONIZE_TIME}" "${HARMONIZE_MEM}" "${harmonize_deps}" "${harmonize_command}"
    )"
    echo "Harmonization: ${harmonize_job}"
    submitted_jobs+=("${harmonize_job}")
    common_dep="${harmonize_job}"
  else
    skip_ready "harmonization"
    common_dep=""
  fi
else
  common_dep="${harmonize_deps}"
fi

if [[ "${RUN_COMORBIDITY}" -eq 1 && "${RUN_FEATURES}" -eq 1 ]]; then
  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    if should_submit "${common_dep}" comorbidity_ready "${site}"; then
      comorbidity_job="$(
        submit_job "amr_${site_slug}_comorbidity" "${COMORBIDITY_TIME}" "${COMORBIDITY_MEM}" "${common_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/pre_aggregate_comorbidities.py --env hpc --site '${site}' --min-coverage 0.8"
      )"
      echo "Comorbidity ${site}: ${comorbidity_job}"
      submitted_jobs+=("${comorbidity_job}")
      comorbidity_jobs+=("${comorbidity_job}")
    else
      skip_ready "comorbidity ${site}"
    fi
  done
fi

for raw_organism in "${ORGANISMS[@]}"; do
  organism="$(trim "${raw_organism}")"
  [[ -n "${organism}" ]] || continue
  organism_slug="$(slugify "${organism}")"
  organism_prefix="amr_${organism_slug}"

  site_cascade_jobs=()
  if [[ "${RUN_SITE_CASCADE}" -eq 1 ]]; then
    for raw_site in "${SITES[@]}"; do
      site="$(trim "${raw_site}")"
      [[ -n "${site}" ]] || continue
      site_slug="$(slugify "${site}")"
      site_dep="${common_dep}"
      if [[ "${RUN_GOLD}" -eq 1 ]]; then
        if should_submit "${site_dep}" gold_ready "site" "${site}" "${organism}"; then
          site_gold_job="$(
            submit_job "${organism_prefix}_${site_slug}_gold" "${GOLD_TIME}" "${GOLD_MEM}" "${site_dep}" \
              "\"\${AMR_CASCADE_PYTHON}\" scripts/run_gold.py --env hpc --source-scope site --site '${site}' --organism '${organism}'"
          )"
          echo "Site gold ${site} | ${organism}: ${site_gold_job}"
          submitted_jobs+=("${site_gold_job}")
          site_dep="${site_gold_job}"
        else
          skip_ready "site gold ${site} | ${organism}"
        fi
      fi
      if [[ "${RUN_CASCADE}" -eq 1 ]]; then
        if should_submit "${site_dep}" cascade_ready "site" "${site}" "${organism}"; then
          # Unsharded site-scope validation was observed to need ~85-125h serially
          # (854/815/724 retained edges across the three known sites) against a 24h
          # CASCADE_TIME allocation -- it was killed by SLURM's time limit every time.
          # Mirror the combined-scope shard/merge split: run_cascade_validation_shard.py
          # and merge_cascade_validation_shards.py already accept --gold-scope site
          # --site, so this reuses the same scripts, just scoped per site.
          site_shard_total="${CASCADE_SITE_SHARD_NUM}"
          site_shard_jobs=()
          for site_shard_idx in $(seq 0 $((site_shard_total - 1))); do
            if should_submit "${site_dep}" shard_ready "site" "${site}" "${organism}" "${site_shard_idx}" "${site_shard_total}"; then
              site_shard_job="$(
                PARTITION="${CASCADE_PARTITION}"
                submit_job "${organism_prefix}_${site_slug}_cval_s${site_shard_idx}" "${CASCADE_SHARD_TIME}" "${CASCADE_SHARD_MEM}" "${site_dep}" \
                  "\"\${AMR_CASCADE_PYTHON}\" scripts/run_cascade_validation_shard.py --env hpc --gold-scope site --site '${site}' --organism '${organism}' --shard-index ${site_shard_idx} --shard-total ${site_shard_total}"
              )"
              echo "  Site cascade shard ${site_shard_idx}/${site_shard_total} ${site} | ${organism}: ${site_shard_job}"
              submitted_jobs+=("${site_shard_job}")
              site_shard_jobs+=("${site_shard_job}")
            else
              skip_ready "site cascade shard ${site_shard_idx}/${site_shard_total} ${site} | ${organism}"
            fi
          done

          site_shard_dep="$(join_colon "${site_shard_jobs[@]}")"
          if should_submit "${site_shard_dep}" validation_merge_ready "site" "${site}" "${organism}"; then
            site_merge_job="$(
              PARTITION="${CASCADE_PARTITION}"
              submit_job "${organism_prefix}_${site_slug}_cval_merge" "${CASCADE_MERGE_TIME}" "${CASCADE_MERGE_MEM}" "${site_shard_dep}" \
                "\"\${AMR_CASCADE_PYTHON}\" scripts/merge_cascade_validation_shards.py --env hpc --gold-scope site --site '${site}' --organism '${organism}' --shard-total ${site_shard_total}"
            )"
            echo "Site cascade validation merge ${site} | ${organism}: ${site_merge_job}"
            submitted_jobs+=("${site_merge_job}")
            site_dep="${site_merge_job}"
          else
            skip_ready "site cascade validation merge ${site} | ${organism}"
          fi

          site_cascade_job="$(
            PARTITION="${CASCADE_PARTITION}"
            submit_job "${organism_prefix}_${site_slug}_cascade" "${CASCADE_TIME}" "${CASCADE_MEM}" "${site_dep}" \
              "\"\${AMR_CASCADE_PYTHON}\" scripts/run_cascade_analysis.py --env hpc --gold-scope site --site '${site}' --organism '${organism}'"
          )"
          echo "Site cascade ${site} | ${organism}: ${site_cascade_job}"
          submitted_jobs+=("${site_cascade_job}")
          site_cascade_jobs+=("${site_cascade_job}")
        else
          skip_ready "site cascade ${site} | ${organism}"
        fi
      fi
    done
  fi

  combined_dep="${common_dep}"
  if [[ "${RUN_GOLD}" -eq 1 ]]; then
    if should_submit "${combined_dep}" gold_ready "combined" "" "${organism}"; then
      combined_gold_job="$(
        submit_job "${organism_prefix}_gold" "${GOLD_TIME}" "${GOLD_MEM}" "${combined_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_gold.py --env hpc --source-scope combined --organism '${organism}'"
      )"
      echo "Combined gold ${organism}: ${combined_gold_job}"
      submitted_jobs+=("${combined_gold_job}")
      combined_dep="${combined_gold_job}"
    else
      skip_ready "combined gold ${organism}"
    fi
  fi

  cascade_dep="${combined_dep}"
  if [[ "${RUN_CASCADE}" -eq 1 ]]; then
    if should_submit "${cascade_dep}" cascade_ready "combined" "" "${organism}"; then
      # The full validation loop at B=1000 requires ~800h sequentially for E. coli
      # (1,637 retained edges).  Split across CASCADE_SHARD_NUM parallel shard jobs,
      # each validating ~40 edges in ~24h.  A merge job then applies global BH-FDR
      # across all shards, and the main cascade job loads the merged result instead
      # of re-running validation.
      shard_total="${CASCADE_SHARD_NUM}"
      shard_jobs=()
      for shard_idx in $(seq 0 $((shard_total - 1))); do
        if should_submit "${cascade_dep}" shard_ready "combined" "" "${organism}" "${shard_idx}" "${shard_total}"; then
          shard_job="$(
            PARTITION="${CASCADE_PARTITION}"
            submit_job "${organism_prefix}_cval_s${shard_idx}" "${CASCADE_SHARD_TIME}" "${CASCADE_SHARD_MEM}" "${cascade_dep}" \
              "\"\${AMR_CASCADE_PYTHON}\" scripts/run_cascade_validation_shard.py --env hpc --gold-scope combined --organism '${organism}' --shard-index ${shard_idx} --shard-total ${shard_total}"
          )"
          echo "  Cascade shard ${shard_idx}/${shard_total} ${organism}: ${shard_job}"
          submitted_jobs+=("${shard_job}")
          shard_jobs+=("${shard_job}")
        else
          skip_ready "cascade shard ${shard_idx}/${shard_total} ${organism}"
        fi
      done

      shard_dep="$(join_colon "${shard_jobs[@]}")"
      if should_submit "${shard_dep}" validation_merge_ready "combined" "" "${organism}"; then
        merge_job="$(
          PARTITION="${CASCADE_PARTITION}"
          submit_job "${organism_prefix}_cval_merge" "${CASCADE_MERGE_TIME}" "${CASCADE_MERGE_MEM}" "${shard_dep}" \
            "\"\${AMR_CASCADE_PYTHON}\" scripts/merge_cascade_validation_shards.py --env hpc --gold-scope combined --organism '${organism}' --shard-total ${shard_total}"
        )"
        echo "Cascade validation merge ${organism}: ${merge_job}"
        submitted_jobs+=("${merge_job}")
        cascade_dep="${merge_job}"
      else
        skip_ready "cascade validation merge ${organism}"
      fi

      combined_cascade_job="$(
        PARTITION="${CASCADE_PARTITION}"
        submit_job "${organism_prefix}_cascade" "${CASCADE_TIME}" "${CASCADE_MEM}" "${cascade_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_cascade_analysis.py --env hpc --gold-scope combined --organism '${organism}'"
      )"
      echo "Combined cascade ${organism}: ${combined_cascade_job}"
      submitted_jobs+=("${combined_cascade_job}")
      cascade_dep="${combined_cascade_job}"
    else
      skip_ready "combined cascade ${organism}"
    fi
  fi

  prevalence_dep="${cascade_dep}"
  if [[ "${RUN_PREVALENCE}" -eq 1 ]]; then
    if should_submit "${prevalence_dep}" prevalence_ready "combined" "" "${organism}"; then
      prevalence_job="$(
        submit_job "${organism_prefix}_prev" "${PREVALENCE_TIME}" "${PREVALENCE_MEM}" "${prevalence_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_prevalence_analysis.py --env hpc --scope combined --organism '${organism}' --figure-format html --figure-format png --figure-format svg --figure-format pdf"
      )"
      echo "Prevalence ${organism}: ${prevalence_job}"
      submitted_jobs+=("${prevalence_job}")
      prevalence_dep="${prevalence_job}"
    else
      skip_ready "prevalence ${organism}"
    fi
  fi

  feature_dep="${combined_dep}"
  if [[ "${RUN_FEATURES}" -eq 1 ]]; then
    # Feature matrices include training-site cascade features when
    # modeling.include_cascade_features=true. Depend on site cascade outputs when
    # they are part of this DAG so features cannot train on stale edge reports.
    feature_inputs=("${feature_dep}" "${site_cascade_jobs[@]}" "${comorbidity_jobs[@]}")
    feature_input_deps="$(join_colon "${feature_inputs[@]}")"
    if should_submit "${feature_input_deps}" feature_ready "combined" "" "${organism}"; then
      feature_job="$(
        submit_job "${organism_prefix}_features" "${FEATURE_TIME}" "${FEATURE_MEM}" "${feature_input_deps}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_feature_build.py --env hpc --scope combined --organism '${organism}'"
      )"
      echo "Features ${organism}: ${feature_job}"
      submitted_jobs+=("${feature_job}")
      feature_dep="${feature_job}"
    else
      skip_ready "features ${organism}"
    fi
  fi

  training_dep=""
  if [[ "${RUN_TRAINING}" -eq 1 ]]; then
    if should_submit "${feature_dep}" training_ready "combined" "" "${organism}"; then
      training_job="$(
        submit_job "${organism_prefix}_training" "${TRAINING_TIME}" "${TRAINING_MEM}" "${feature_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_training.py --env hpc --scope combined --organism '${organism}'"
      )"
      echo "Training ${organism}: ${training_job}"
      submitted_jobs+=("${training_job}")
      training_dep="${training_job}"
    else
      skip_ready "training ${organism}"
    fi
  fi

  if [[ "${RUN_REPORTING}" -eq 1 ]]; then
    report_inputs=("${prevalence_dep}")
    [[ -n "${training_dep}" ]] && report_inputs+=("${training_dep}")
    report_deps="$(join_colon "${report_inputs[@]}")"
    organism_publication_dep="${report_deps}"
    if should_submit "${report_deps}" reporting_ready "combined" "" "${organism}"; then
      report_job="$(
        submit_job "${organism_prefix}_report" "${REPORT_TIME}" "${REPORT_MEM}" "${report_deps}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_reporting.py --env hpc --scope combined --organism '${organism}' --figure-format html --figure-format png --figure-format svg --figure-format pdf ${REPORT_FIGURE_ARGS}"
      )"
      echo "Report ${organism}: ${report_job}"
      submitted_jobs+=("${report_job}")
      organism_publication_dep="${report_job}"
    else
      skip_ready "report ${organism}"
    fi
    if [[ "${RUN_READINESS_AUDIT}" -eq 1 ]]; then
      readiness_args=""
      [[ "${RUN_TRAINING}" -eq 1 || "${RUN_ALL_FIGURES}" -eq 1 ]] && readiness_args="--require-training"
      if should_submit "${organism_publication_dep}" publication_readiness_ready "combined" "" "${organism}"; then
        readiness_job="$(
          submit_job "${organism_prefix}_readiness_audit" "${READINESS_AUDIT_TIME}" "${READINESS_AUDIT_MEM}" "${organism_publication_dep}" \
            "\"\${AMR_CASCADE_PYTHON}\" scripts/run_publication_readiness_audit.py --env hpc --scope combined --organism '${organism}' ${readiness_args}"
        )"
        echo "Publication readiness audit ${organism}: ${readiness_job}"
        submitted_jobs+=("${readiness_job}")
        publication_terminal_jobs+=("${readiness_job}")
      else
        skip_ready "publication readiness audit ${organism}"
      fi
    elif [[ -n "${organism_publication_dep}" ]]; then
      publication_terminal_jobs+=("${organism_publication_dep}")
    fi
  fi
done

if [[ "${RUN_ESKAPE}" -eq 1 ]]; then
  echo
  echo "Submitting ESKAPE-family validation: ${#SITES[@]} site(s) x ${#ESKAPE_TARGETS[@]} target(s) (site scope), plus ${#ESKAPE_TARGETS[@]} target(s) (combined scope)."
  echo "Each (scope, target) pair is an independent job -- no target waits on another."

  eskape_root="${PROJECT_ROOT}/data/artifacts/eskape_validation/armd"
  eskape_tables_root="${PROJECT_ROOT}/outputs/tables/eskape"

  for raw_site in "${SITES[@]}"; do
    site="$(trim "${raw_site}")"
    [[ -n "${site}" ]] || continue
    site_slug="$(slugify "${site}")"
    site_target_jobs=()

    for raw_target in "${ESKAPE_TARGETS[@]}"; do
      target="$(trim "${raw_target}")"
      [[ -n "${target}" ]] || continue
      target_slug="$(slugify "${target}")"
      per_target_dir="${eskape_tables_root}/_per_target/site/${site_slug}/${target_slug}"

      if should_submit "${common_dep}" eskape_ready "site" "${site}" "${target}"; then
        eskape_job="$(
          PARTITION="${CASCADE_PARTITION}"
          submit_job "amr_eskape_${site_slug}_${target_slug}" "${ESKAPE_TIME}" "${ESKAPE_MEM}" "${common_dep}" \
            "\"\${AMR_CASCADE_PYTHON}\" scripts/run_eskape_cascade_validation.py --env hpc --armd-source-scope site --armd-site '${site}' --targets '${target}' --armd-output-root '${eskape_root}' --summary-dir '${per_target_dir}'"
        )"
        echo "ESKAPE ${site} | ${target}: ${eskape_job}"
        submitted_jobs+=("${eskape_job}")
        site_target_jobs+=("${eskape_job}")
      else
        skip_ready "ESKAPE ${site} | ${target}"
      fi
    done

    site_merge_deps="$(join_colon "${site_target_jobs[@]}")"
    if should_submit "${site_merge_deps}" eskape_merge_ready "site" "${site}"; then
      eskape_site_merge_job="$(
        submit_job "amr_eskape_${site_slug}_merge" "00:30:00" "16G" "${site_merge_deps}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/merge_eskape_summaries.py --input-glob 'outputs/tables/eskape/_per_target/site/${site_slug}/*/eskape_cascade_comparison.csv' --output-csv 'outputs/tables/eskape/site_${site}/eskape_cascade_comparison.csv' --output-json 'outputs/tables/eskape/site_${site}/eskape_cascade_comparison.json'"
      )"
      echo "ESKAPE merge ${site}: ${eskape_site_merge_job}"
      submitted_jobs+=("${eskape_site_merge_job}")
      publication_terminal_jobs+=("${eskape_site_merge_job}")
    else
      skip_ready "ESKAPE merge ${site}"
    fi
  done

  combined_target_jobs=()
  for raw_target in "${ESKAPE_TARGETS[@]}"; do
    target="$(trim "${raw_target}")"
    [[ -n "${target}" ]] || continue
    target_slug="$(slugify "${target}")"
    per_target_dir="${eskape_tables_root}/_per_target/combined/${target_slug}"

    if should_submit "${common_dep}" eskape_ready "combined" "" "${target}"; then
      eskape_job="$(
        PARTITION="${CASCADE_PARTITION}"
        submit_job "amr_eskape_combined_${target_slug}" "${ESKAPE_TIME}" "${ESKAPE_MEM}" "${common_dep}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_eskape_cascade_validation.py --env hpc --armd-source-scope combined --targets '${target}' --armd-output-root '${eskape_root}' --summary-dir '${per_target_dir}'"
      )"
      echo "ESKAPE combined | ${target}: ${eskape_job}"
      submitted_jobs+=("${eskape_job}")
      combined_target_jobs+=("${eskape_job}")
    else
      skip_ready "ESKAPE combined | ${target}"
    fi
  done

  combined_merge_deps="$(join_colon "${combined_target_jobs[@]}")"
  if should_submit "${combined_merge_deps}" eskape_merge_ready "combined"; then
    eskape_combined_merge_job="$(
      submit_job "amr_eskape_combined_merge" "00:30:00" "16G" "${combined_merge_deps}" \
        "\"\${AMR_CASCADE_PYTHON}\" scripts/merge_eskape_summaries.py --input-glob 'outputs/tables/eskape/_per_target/combined/*/eskape_cascade_comparison.csv' --output-csv 'outputs/tables/eskape/combined_armd/eskape_cascade_comparison.csv' --output-json 'outputs/tables/eskape/combined_armd/eskape_cascade_comparison.json'"
    )"
    echo "ESKAPE merge combined: ${eskape_combined_merge_job}"
    submitted_jobs+=("${eskape_combined_merge_job}")
    publication_terminal_jobs+=("${eskape_combined_merge_job}")
  else
    skip_ready "ESKAPE merge combined"
  fi
fi

if [[ "${RUN_AUDIT}" -eq 1 ]]; then
  audit_deps="$(join_colon "${publication_terminal_jobs[@]}")"
  for organism in "${ORGANISMS[@]}"; do
    organism_slug="$(slugify "${organism}")"
    if should_submit "${audit_deps}" scientific_audit_ready "combined" "" "${organism}"; then
      audit_job="$(
        submit_job "amr_scientific_audit_${organism_slug}" "${AUDIT_TIME}" "${AUDIT_MEM}" "${audit_deps}" \
          "\"\${AMR_CASCADE_PYTHON}\" scripts/run_scientific_audit.py --env hpc --scope combined --organism '${organism}'"
      )"
      echo "Scientific audit | ${organism}: ${audit_job}"
      submitted_jobs+=("${audit_job}")
    else
      skip_ready "scientific audit | ${organism}"
    fi
  done
fi

if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
  echo
  echo "Submitted ${#submitted_jobs[@]} job(s)."
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    # Persist the job ID list so the user can re-attach with --status later.
    job_file="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').jobs"
    {
      printf '# AMR Cascade Platform pipeline submission\n'
      printf '# Submitted: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
      printf '# Sites: %s\n' "${SITES[*]}"
      printf '# Organisms: %s\n' "${ORGANISMS[*]}"
      printf '%s\n' "${submitted_jobs[@]}"
    } > "${job_file}"
    echo "Job IDs saved to: ${job_file}"
    echo
    echo "Track live status:"
    job_csv="$(IFS=,; printf '%s' "${submitted_jobs[*]}")"
    echo "  squeue -j ${job_csv}"
    echo "Re-attach to status anytime:"
    echo "  bash $(basename "${BASH_SOURCE[0]}") --status ${job_file}"
  fi
else
  echo
  echo "No jobs submitted; all enabled stages are already complete."
fi

# Optional blocking wait + final state report.
if [[ "${WAIT_FOR_JOBS}" -eq 1 && "${DRY_RUN}" -eq 0 && "${#submitted_jobs[@]}" -gt 0 ]]; then
  wait_for_jobs "${submitted_jobs[@]}"
  if report_final_states "${submitted_jobs[@]}"; then
    exit 0
  else
    exit 1
  fi
fi
