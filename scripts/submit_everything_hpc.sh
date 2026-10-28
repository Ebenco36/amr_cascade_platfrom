#!/usr/bin/env bash
# One entry point for the full submission wave: the primary run, the three
# availability-sensitivity runs, and the follow-up jobs the supplements need.
# Every follow-up is held on the job that produces its input via SLURM's
# --dependency=afterok, not a wall-clock wait, so this script finishes as soon
# as everything is queued (seconds) rather than blocking until the jobs run.
# Run with no options for the two-manuscript wave, not all exploratory outputs.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_everything_hpc.sh [options]

Publication scope:
  --publication manuscripts  DEFAULT: support both current E. coli manuscripts and
                             supplements, sharing the primary cascade run. Includes
                             Paper 2 bounds, trigger/SMD diagnostics, MNAR and antibiograms.
                             Omits SHAP, ESKAPE and uncited exploratory figure suites.
  --publication all          Opt into every platform analysis/figure group.
  --publication paper1       Keep the analyses cited by paper1_detection_validation;
                             omit Paper 2 prevalence/MNAR and unrelated figure groups.
                             Availability variants stop at validation merge. Default
                             reuse is checked against saved provenance before queuing.
                             A final named-asset audit follows supplementary export.
  --hide-figure-titles       Remove figure-level headings in report and prediction exports.
  --fresh                   With a manuscript profile, explicitly move old gold/cascade outputs
                             aside and rebuild them. NOT recommended for recovery.
  --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity
  --skip-gold               Primary-run-only skips for already completed input layers.

With no options, first installs Python packages, then submits:
  0. Python packages: creates .venv if missing and installs every package in
     requirements-hpc.txt (which includes requirements.txt) whenever the
     requirement files changed since the last install; otherwise nothing to do.
     Runs once, on this login node, before anything is queued.
  1. Shared primary run (site/combined cascade, features, training, threshold
     comparison, Paper 2 prevalence and only the figures cited by both papers)
  2. Availability sensitivity: support >= 5
  3. Availability sensitivity: 1-year era
  4. Availability sensitivity: joint (both at once)
  then the follow-ups, each held on the primary-run job it needs:
     - site-by-era permutation, validated patterns    (after cascade validation merge)
     - patient-cluster bootstrap, robust patterns     (after cascade validation merge)
     - ridge-penalty refit, validated patterns        (after cascade validation merge)
     - supplementary prediction tables and figures    (after training, the report and
                                                       its readiness audit)
     - masking experiment, Paper 2                    (after the combined cascade;
                                                       not with --publication paper1)
     - year-by-year course of validated patterns      (after the report and its
                                                       readiness audit)
     - supplementary tables and quoted numbers        (after the report job, the site
                                                       cascades, all follow-ups above, and
                                                       the three sensitivity runs' validation)
  Manuscript profiles do not move completed data aside by default. Incompatible
  validation provenance stops submission; inspect it before choosing --fresh.
  --publication all preserves the historical --fix-stale behaviour.

Cohort/hardware pass-through (forwarded to every real submission below):
  --sites SITE1,SITE2          Same as submit_pipeline_dag_hpc.sh's own flag.
  --organisms O1,O2            Same. The follow-ups need exactly one organism;
                                with more than one they are skipped with a note.
  --partition NAME             Also used for the prediction and table jobs.
  --cascade-partition NAME     Also used for the shard and ridge follow-ups.
  --python-bin PATH            Base Python 3.12 used to create .venv. Every job,
                                follow-ups included, runs .venv/bin/python.
  --force-rerun-existing

Python packages (step 0):
  --delete-venv                Rebuild .venv from scratch (it is shared by every
                                run, so it is rebuilt once).
  --skip-install               Use .venv as it is: only check that it imports.

Main-run-only pass-through:
  --with-eskape                Also run the ESKAPE-family validation. Off by
                                default: neither paper uses it.
  --eskape-targets T1,T2       Targets for --with-eskape.

Turn individual pieces on/off:
  --skip-main                  Don't submit the primary publication-package run.
  --skip-support-sensitivity   Don't submit the support>=5 sensitivity.
  --skip-annual-sensitivity    Don't submit the 1-year-era sensitivity.
  --skip-joint-sensitivity     Don't submit the joint sensitivity.
  --skip-era-stratified        Don't queue the site-by-era permutation.
  --skip-patient-cluster       Don't queue the patient-cluster bootstrap.
  --skip-ridge-sensitivity     Don't queue the ridge-penalty refit.
  --skip-supplementary-prediction  Don't queue the prediction tables/figures.
  --skip-masking               Don't queue Paper 2's masking experiment.
  --skip-temporal              Don't queue the year-by-year course of the validated patterns.
  --skip-supplementary-tables  Don't queue the supplementary-table job.
  --main-only                  Shorthand: skip all three sensitivities and
                                every follow-up.
  --sensitivities-only         Shorthand: skip the primary run only.

When --skip-main is used, give the primary run's job IDs yourself (a finished
job needs no dependency; omit its ID):
  --validation-merge-job JOBID Cascade validation merge (alias: --era-stratified-after).
  --training-job JOBID         Training job, for the prediction follow-up.
  --report-job JOBID           Report job, for the prediction and supplementary-table jobs.
  --readiness-audit-job JOBID  Publication readiness audit job, for the prediction job.

Follow-up sizing:
  --era-stratified-shards N    Default: 48 (1 = a single job, no merge)
  --era-stratified-mem SIZE    Per shard. Default: 512G (loads the full pair table)
  --era-stratified-time HH:MM:SS  Per shard. Default: 24:00:00
  --patient-cluster-shards N   Default: 16 (1 = a single job, no merge)
  --patient-cluster-mem SIZE   Per shard. Default: 512G
  --patient-cluster-time HH:MM:SS  Per shard. Default: 24:00:00
  --ridge-mem SIZE             Default: 512G
  --ridge-time HH:MM:SS        Default: 24:00:00

  --allow-dirty-checkout       Forward to every submission: allow src/, scripts/ or
                               configs/ to differ from the checked-out commit.
  --dry-run                    Forward --dry-run to every real submission
                                below, and print each follow-up sbatch command
                                instead of calling sbatch.
  --help                       Show this message.

After everything finishes, bring the outputs to the machine that holds the
paper folders and run scripts/build_manuscript_package.py there (the paper
folders are not in git, so packaging cannot run on the cluster).
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
SKIP_INSTALL=0
ESKAPE_TARGETS=""
WITH_ESKAPE=0
SKIP_MAIN=0
SKIP_SUPPORT_SENSITIVITY=0
SKIP_ANNUAL_SENSITIVITY=0
SKIP_JOINT_SENSITIVITY=0
SKIP_ERA_STRATIFIED=0
SKIP_PATIENT_CLUSTER=0
SKIP_RIDGE=0
SKIP_PREDICTION=0
SKIP_SUPPLEMENTARY_TABLES=0
SKIP_MASKING=0
SKIP_TEMPORAL=0
VALIDATION_MERGE_JOB=""
TRAINING_JOB=""
REPORT_JOB=""
READINESS_JOB=""
ERA_STRATIFIED_MEM="512G"
ERA_STRATIFIED_TIME="24:00:00"
# 48 rather than 16: at B = 1000 the site-by-era permutation cost ~144 s per pattern on
# replica-sized data; with ~33x the rows per pattern and ~540 validated patterns on the
# real data, 16 shards put each shard near or past its 24 h limit.
ERA_STRATIFIED_SHARDS=48
PATIENT_CLUSTER_MEM="512G"
PATIENT_CLUSTER_TIME="24:00:00"
PATIENT_CLUSTER_SHARDS=16
RIDGE_MEM="512G"
RIDGE_TIME="24:00:00"
PREDICTION_MEM="128G"
# 4 h ran out on 2026-10-17 (217 million prediction rows); the curves are now thinned for drawing.
PREDICTION_TIME="12:00:00"
TABLES_MEM="64G"
TABLES_TIME="02:00:00"
MASKING_MEM="256G"
MASKING_TIME="24:00:00"
# Reads the pair table in batches: peak 4.2 GB on the full E. coli data (28 October 2026).
TEMPORAL_MEM="64G"
TEMPORAL_TIME="02:00:00"
DRY_RUN=0
ALLOW_DIRTY_CHECKOUT=0
PUBLICATION=manuscripts
HIDE_FIGURE_TITLES=0
FRESH=0
PRIMARY_SKIP_ARGS=()

need_value() { [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites) need_value "$@"; SITES="$2"; shift 2 ;;
    --publication) need_value "$@"; PUBLICATION="$2"; shift 2 ;;
    --hide-figure-titles) HIDE_FIGURE_TITLES=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --skip-ingestion|--skip-preprocessing|--skip-harmonization|--skip-comorbidity|--skip-gold)
      PRIMARY_SKIP_ARGS+=("$1"); shift ;;
    --organisms) need_value "$@"; ORGANISMS="$2"; shift 2 ;;
    --partition) need_value "$@"; PARTITION="$2"; shift 2 ;;
    --cascade-partition) need_value "$@"; CASCADE_PARTITION="$2"; shift 2 ;;
    --python-bin) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --force-rerun-existing) FORCE_RERUN_EXISTING=1; shift ;;
    --delete-venv) DELETE_VENV=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --with-eskape) WITH_ESKAPE=1; shift ;;
    --eskape-targets) need_value "$@"; ESKAPE_TARGETS="$2"; shift 2 ;;
    --skip-main) SKIP_MAIN=1; shift ;;
    --skip-support-sensitivity) SKIP_SUPPORT_SENSITIVITY=1; shift ;;
    --skip-annual-sensitivity) SKIP_ANNUAL_SENSITIVITY=1; shift ;;
    --skip-joint-sensitivity) SKIP_JOINT_SENSITIVITY=1; shift ;;
    --skip-era-stratified) SKIP_ERA_STRATIFIED=1; shift ;;
    --skip-patient-cluster) SKIP_PATIENT_CLUSTER=1; shift ;;
    --skip-ridge-sensitivity) SKIP_RIDGE=1; shift ;;
    --skip-supplementary-prediction) SKIP_PREDICTION=1; shift ;;
    --skip-supplementary-tables) SKIP_SUPPLEMENTARY_TABLES=1; shift ;;
    --skip-masking) SKIP_MASKING=1; shift ;;
    --skip-temporal) SKIP_TEMPORAL=1; shift ;;
    --main-only)
      SKIP_SUPPORT_SENSITIVITY=1; SKIP_ANNUAL_SENSITIVITY=1; SKIP_JOINT_SENSITIVITY=1
      SKIP_ERA_STRATIFIED=1; SKIP_PATIENT_CLUSTER=1; SKIP_RIDGE=1; SKIP_PREDICTION=1; SKIP_SUPPLEMENTARY_TABLES=1
      SKIP_MASKING=1; SKIP_TEMPORAL=1
      shift ;;
    --sensitivities-only) SKIP_MAIN=1; shift ;;
    --validation-merge-job|--era-stratified-after) need_value "$@"; VALIDATION_MERGE_JOB="$2"; shift 2 ;;
    --training-job) need_value "$@"; TRAINING_JOB="$2"; shift 2 ;;
    --report-job) need_value "$@"; REPORT_JOB="$2"; shift 2 ;;
    --readiness-audit-job) need_value "$@"; READINESS_JOB="$2"; shift 2 ;;
    --era-stratified-mem) need_value "$@"; ERA_STRATIFIED_MEM="$2"; shift 2 ;;
    --era-stratified-time) need_value "$@"; ERA_STRATIFIED_TIME="$2"; shift 2 ;;
    --era-stratified-shards) need_value "$@"; ERA_STRATIFIED_SHARDS="$2"; shift 2 ;;
    --patient-cluster-mem) need_value "$@"; PATIENT_CLUSTER_MEM="$2"; shift 2 ;;
    --patient-cluster-time) need_value "$@"; PATIENT_CLUSTER_TIME="$2"; shift 2 ;;
    --patient-cluster-shards) need_value "$@"; PATIENT_CLUSTER_SHARDS="$2"; shift 2 ;;
    --ridge-mem) need_value "$@"; RIDGE_MEM="$2"; shift 2 ;;
    --ridge-time) need_value "$@"; RIDGE_TIME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-dirty-checkout) ALLOW_DIRTY_CHECKOUT=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for shards in "${ERA_STRATIFIED_SHARDS}" "${PATIENT_CLUSTER_SHARDS}"; do
  [[ "${shards}" =~ ^[1-9][0-9]*$ ]] || { echo "Shard counts must be positive integers" >&2; exit 2; }
done
[[ "${PUBLICATION}" == all || "${PUBLICATION}" == paper1 || "${PUBLICATION}" == manuscripts ]] || { echo "--publication must be all, paper1 or manuscripts" >&2; exit 2; }
if [[ "${PUBLICATION}" != all && "${WITH_ESKAPE}" -eq 1 ]]; then
  echo "ESKAPE is outside both current manuscripts; use --publication all for that extension." >&2
  exit 2
fi
if [[ "${PUBLICATION}" != all && -n "${ORGANISMS}" && "${ORGANISMS}" != "ESCHERICHIA COLI" ]]; then
  echo "The current manuscripts cover ESCHERICHIA COLI only; use --publication all for extensions." >&2
  exit 2
fi
if [[ "${PUBLICATION}" != all && -n "${SITES}" ]]; then
  site_set="$(printf '%s' "${SITES}" | tr ',' '\n' | sort | paste -sd, -)"
  [[ "${site_set}" == armd,armd_ecuh,armd_utsw ]] || { echo "Both manuscripts require armd,armd_ecuh,armd_utsw." >&2; exit 2; }
fi

# Shared pass-through flags, built once and forwarded identically to every
# real submit_pipeline_dag_hpc.sh call below (sensitivity runs included --
# they should target the same cohort/hardware as the primary run).
common_args=()
common_args+=(--publication "${PUBLICATION}")
presentation_args=""
if [[ "${HIDE_FIGURE_TITLES}" -eq 1 ]]; then
  common_args+=(--hide-figure-titles)
  presentation_args="--hide-figure-titles"
fi
[[ -n "${SITES}" ]] && common_args+=(--sites "${SITES}")
[[ -n "${ORGANISMS}" ]] && common_args+=(--organisms "${ORGANISMS}")
[[ -n "${PARTITION}" ]] && common_args+=(--partition "${PARTITION}")
[[ -n "${CASCADE_PARTITION}" ]] && common_args+=(--cascade-partition "${CASCADE_PARTITION}")
[[ -n "${PYTHON_BIN}" ]] && common_args+=(--python-bin "${PYTHON_BIN}")
[[ "${FORCE_RERUN_EXISTING}" -eq 1 ]] && common_args+=(--force-rerun-existing)
[[ "${DRY_RUN}" -eq 1 ]] && common_args+=(--dry-run)
[[ "${ALLOW_DIRTY_CHECKOUT}" -eq 1 ]] && common_args+=(--allow-dirty-checkout)
[[ "${SKIP_INSTALL}" -eq 1 ]] && common_args+=(--skip-install)

main_args=(${common_args[@]+"${common_args[@]}"})
main_args+=(${PRIMARY_SKIP_ARGS[@]+"${PRIMARY_SKIP_ARGS[@]}"})
if [[ -n "${ESKAPE_TARGETS}" && "${WITH_ESKAPE}" -eq 0 ]]; then
  echo "--eskape-targets has no effect without --with-eskape" >&2
  exit 2
fi
[[ -n "${ESKAPE_TARGETS}" ]] && main_args+=(--eskape-targets "${ESKAPE_TARGETS}")

# --run-publication-package minus ESKAPE, which only runs with --with-eskape. Keep the two
# in step: the threshold-strategy comparison (validation holdout vs patient-grouped OOF)
# is reported in the Supplement, so it is part of every full run.
package_args=(--run-site-cascade --include-site-comparison-figures --run-features --run-training --run-threshold-comparison --run-all-figures --run-audit --run-readiness-audit)
stale_args=(--fix-stale)
if [[ "${PUBLICATION}" != all ]]; then
  package_args=(--run-site-cascade --run-features --run-training --run-threshold-comparison --run-all-figures)
  # Recovery must not silently move completed gold/validation artifacts aside.
  stale_args=()
  [[ "${FRESH}" -eq 1 ]] && stale_args=(--fix-stale)
fi
[[ "${WITH_ESKAPE}" -eq 1 ]] && package_args+=(--run-eskape)

# Job IDs printed by submit_pipeline_dag_hpc.sh, e.g. "Report ESCHERICHIA COLI: 123".
job_id_after() {
  printf '%s\n' "$1" | grep -E "^$2" | tail -1 | awk -F': ' '{print $2}' || true
}

# Step 0, once for the whole wave and before anything is queued: every job
# (primary, sensitivities, follow-ups) runs the shared .venv. The runs below then
# find its packages up to date and install nothing.
setup_args=(--setup-only)
[[ -n "${PYTHON_BIN}" ]] && setup_args+=(--python-bin "${PYTHON_BIN}")
[[ "${DELETE_VENV}" -eq 1 ]] && setup_args+=(--delete-venv)
[[ "${SKIP_INSTALL}" -eq 1 ]] && setup_args+=(--skip-install)
[[ "${DRY_RUN}" -eq 1 ]] && setup_args+=(--dry-run)
echo "=== [0/4] Python packages: requirements-hpc.txt -> .venv ==="
bash scripts/submit_pipeline_dag_hpc.sh "${setup_args[@]}"
echo ""

# Check all saved results before submitting any part of the wave, not after
# hundreds of jobs have already been queued. A fresh run rebuilds these layers.
if [[ "${PUBLICATION}" != all && "${FRESH}" -eq 0 ]]; then
  reuse_args=(--sites "${SITES:-armd,armd_ecuh,armd_utsw}" --organisms "${ORGANISMS:-ESCHERICHIA COLI}")
  check_reuse() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '[dry-run] .venv/bin/python scripts/check_validation_reuse.py'; printf ' %q' "$@" "${reuse_args[@]}"; printf '\n'
    else
      "${PROJECT_ROOT}/.venv/bin/python" scripts/check_validation_reuse.py "$@" "${reuse_args[@]}"
    fi
  }
  check_reuse --env hpc --include-sites
  [[ "${SKIP_SUPPORT_SENSITIVITY}" -eq 1 ]] || check_reuse --env hpc_availability_sensitivity
  [[ "${SKIP_ANNUAL_SENSITIVITY}" -eq 1 ]] || check_reuse --env hpc_availability_annual_sensitivity
  [[ "${SKIP_JOINT_SENSITIVITY}" -eq 1 ]] || check_reuse --env hpc_availability_joint_sensitivity
fi

site_cascade_jobs=""
combined_cascade_job=""
harmonization_job=""
if [[ "${SKIP_MAIN}" -eq 0 ]]; then
  echo "=== [1/4] Primary run: ${PUBLICATION} ${package_args[*]} ==="
  main_output="$(bash scripts/submit_pipeline_dag_hpc.sh ${stale_args[@]+"${stale_args[@]}"} "${package_args[@]}" ${main_args[@]+"${main_args[@]}"})"
  printf '%s\n' "${main_output}"
  # merge_cascade_validation_shards.py writes validation_results.parquet, the
  # input of the three validated-pattern follow-ups.
  VALIDATION_MERGE_JOB="$(job_id_after "${main_output}" "Cascade validation merge ")"
  TRAINING_JOB="$(job_id_after "${main_output}" "Training ")"
  REPORT_JOB="$(job_id_after "${main_output}" "Report ")"
  READINESS_JOB="$(job_id_after "${main_output}" "Publication readiness audit ")"
  combined_cascade_job="$(job_id_after "${main_output}" "Combined cascade ")"
  harmonization_job="$(job_id_after "${main_output}" "Harmonization:")"
  site_cascade_jobs="$(printf '%s\n' "${main_output}" | grep -E "^Site cascade [^|]+\| " | grep -v "^Site cascade validation merge" | awk -F': ' '{print $2}' | paste -sd: - || true)"
else
  echo "=== [1/4] Primary run: skipped (--skip-main) ==="
fi

setup_sensitivity_symlinks() {
  local data_root="$1"
  if [[ ! -d "${data_root}" ]]; then
    mkdir -p "${data_root}"
    for d in raw sample bronze silver harmonized reference interim features; do
      ln -s "${PROJECT_ROOT}/data/${d}" "${data_root}/${d}"
    done
  fi
  # Its own metadata directory, not a link to data/metadata: the variant's gold build
  # writes gold_build_*.json there, which through a link overwrote the primary run's
  # record (read by the manuscript and supplementary builders). Nothing in a
  # sensitivity run reads metadata, so an empty directory is all it needs.
  [[ -L "${data_root}/metadata" ]] && rm "${data_root}/metadata"
  mkdir -p "${data_root}/metadata" "${data_root}/gold" "${data_root}/artifacts"
}

# submit_sensitivity DATA_ROOT ENV: queue one availability-sensitivity run and record the job
# that writes its validation_results.parquet, which the supplementary-table job compares.
sensitivity_jobs=()
submit_sensitivity() {
  local data_root="$1" environment="$2" output
  [[ "${DRY_RUN}" -eq 1 ]] || setup_sensitivity_symlinks "${data_root}"
  local sensitivity_args=()
  [[ "${PUBLICATION}" == all ]] || sensitivity_args+=(--validation-only)
  # Variants read the primary harmonized tables through symlinks. Do not race
  # those writes during a raw-to-publication run.
  [[ -z "${harmonization_job}" ]] || sensitivity_args+=(--after-job "${harmonization_job}")
  output="$(bash scripts/submit_pipeline_dag_hpc.sh \
    --env "${environment}" --data-root "${data_root}" \
    --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity \
    --skip-prevalence --skip-reporting ${stale_args[@]+"${stale_args[@]}"} ${sensitivity_args[@]+"${sensitivity_args[@]}"} ${common_args[@]+"${common_args[@]}"})"
  printf '%s\n' "${output}"
  sensitivity_jobs+=("$(job_id_after "${output}" "Cascade validation merge ")")
}

echo ""
if [[ "${SKIP_SUPPORT_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [2/4] Availability sensitivity: support >= 5 ==="
  submit_sensitivity data_availability_sensitivity hpc_availability_sensitivity
else
  echo "=== [2/4] Availability sensitivity: support >= 5 -- skipped ==="
fi

echo ""
if [[ "${SKIP_ANNUAL_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [3/4] Availability sensitivity: 1-year era ==="
  submit_sensitivity data_availability_annual_sensitivity hpc_availability_annual_sensitivity
else
  echo "=== [3/4] Availability sensitivity: 1-year era -- skipped ==="
fi

echo ""
if [[ "${SKIP_JOINT_SENSITIVITY}" -eq 0 ]]; then
  echo "=== [4/4] Availability sensitivity: joint (support >= 5 within 1-year era) ==="
  submit_sensitivity data_availability_joint_sensitivity hpc_availability_joint_sensitivity
else
  echo "=== [4/4] Availability sensitivity: joint -- skipped ==="
fi

echo ""
if [[ $((SKIP_ERA_STRATIFIED & SKIP_PATIENT_CLUSTER & SKIP_RIDGE & SKIP_PREDICTION & SKIP_SUPPLEMENTARY_TABLES & SKIP_MASKING & SKIP_TEMPORAL)) -eq 1 ]]; then
  echo "Follow-ups: all skipped."
  echo ""
  echo "All submissions queued. Check status with: squeue -u \$USER"
  exit 0
fi

organism="ESCHERICHIA COLI"
if [[ -n "${ORGANISMS}" ]]; then
  if [[ "${ORGANISMS}" == *,* ]]; then
    echo "Follow-ups: skipped -- --organisms lists more than one organism (${ORGANISMS})"
    echo "and the follow-ups need exactly one. Rerun with --skip-main, one organism,"
    echo "and the primary run's job IDs."
    echo ""
    echo "All submissions queued. Check status with: squeue -u \$USER"
    exit 0
  fi
  organism="${ORGANISMS}"
fi

# The venv step 0 maintains, as in submit_pipeline_dag_hpc.sh's own jobs;
# --python-bin only chooses the Python that creates it.
python_bin="${PROJECT_ROOT}/.venv/bin/python"
# The code identity the follow-up jobs must run (see submit_pipeline_dag_hpc.sh, which
# already refused a dirty checkout unless --allow-dirty-checkout was given).
CODE_IDENTITY=""
if [[ -x "${python_bin}" ]]; then
  CODE_IDENTITY="$(cd "${PROJECT_ROOT}" && PYTHONNOUSERSITE=1 "${python_bin}" -c \
    'import sys; sys.path.insert(0, "src"); from amr_cascade_platform.core.utils.provenance import code_identity; print(code_identity("."))' \
    2>/dev/null || true)"
fi
cpus="${CPUS_PER_TASK:-8}"
cascade_partition="${CASCADE_PARTITION:-long}"
main_partition="${PARTITION:-main}"

# Same job environment as submit_pipeline_dag_hpc.sh's submit_job.
wrap_command() {
  printf '%s' "cd '${PROJECT_ROOT}' && export PYTHONNOUSERSITE=1 && export MPLCONFIGDIR='${PROJECT_ROOT}/.cache/matplotlib' && mkdir -p \"\${MPLCONFIGDIR}\" && export AMR_CASCADE_PYTHON='${python_bin}' && export AMR_CASCADE_EXPECTED_CODE='${CODE_IDENTITY}' && \"\${AMR_CASCADE_PYTHON}\" scripts/check_python_runtime.py || { echo \"ERROR: start-up check failed on \$(hostname) (see above): a broken venv (\${AMR_CASCADE_PYTHON}) or a checkout changed since submission.\" >&2; exit 127; } && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS='${cpus}' && export OPENBLAS_NUM_THREADS='${cpus}' && export MKL_NUM_THREADS='${cpus}' && $1"
}

# follow_up NAME PARTITION MEM TIME DEPENDENCIES(colon-separated, may be empty) COMMAND
# Prints the job ID; under --dry-run prints the sbatch command to stderr and a placeholder ID.
follow_up() {
  local name="$1" partition="$2" mem="$3" time_limit="$4" deps="$5" command="$6"
  local args=(
    --parsable --partition="${partition}" --job-name="${name}"
    --output="${PROJECT_ROOT}/logs/${name}_%j.out" --error="${PROJECT_ROOT}/logs/${name}_%j.err"
    --nodes=1 --ntasks=1 --cpus-per-task="${cpus}" --mem="${mem}" --time="${time_limit}"
    --kill-on-invalid-dep=yes
  )
  [[ -n "${deps}" ]] && args+=(--dependency="afterok:${deps}")
  # As in submit_pipeline_dag_hpc.sh: nodes to keep every job off.
  [[ -n "${AMR_SBATCH_EXCLUDE:-}" ]] && args+=(--exclude="${AMR_SBATCH_EXCLUDE}")
  args+=(--wrap "$(wrap_command "${command}")")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  [dry-run] sbatch' >&2
    printf ' %q' "${args[@]}" >&2
    printf '\n' >&2
    echo "dryrun_${name}"
    return 0
  fi
  sbatch "${args[@]}"
}

join_ids() { local IFS=:; echo "$*" | sed -E 's/:+/:/g; s/^://; s/:$//'; }

# sharded_follow_up LABEL ANALYSIS SHARDS MEM TIME -> prints the ID of the job that
# finishes the analysis (the merge job, or the single job when SHARDS is 1).
sharded_follow_up() {
  local label="$1" analysis="$2" shards="$3" mem="$4" time_limit="$5"
  local base="\"\${AMR_CASCADE_PYTHON}\" scripts/run_validated_pattern_sensitivity.py --analysis ${analysis} --env hpc --gold-scope combined --organism '${organism}' --shard-total ${shards}"
  if [[ "${DRY_RUN}" -eq 0 && -z "${VALIDATION_MERGE_JOB}" && "${FORCE_RERUN_EXISTING}" -eq 0 ]] &&
    "${python_bin}" scripts/run_validated_pattern_sensitivity.py --analysis "${analysis}" --env hpc --gold-scope combined --organism "${organism}" --check-ready; then
    echo "  ${label}: reusing provenance-matched completed result; no shards queued." >&2
    return 0
  fi
  local ids=()
  for ((i = 0; i < shards; i++)); do
    ids+=("$(follow_up "amr_${label}_s${i}" "${cascade_partition}" "${mem}" "${time_limit}" "${VALIDATION_MERGE_JOB}" "${base} --shard-index ${i}")")
  done
  echo "  ${label} shard jobs: ${ids[*]}" >&2
  if [[ "${shards}" -gt 1 ]]; then
    follow_up "amr_${label}_merge" "${main_partition}" "16G" "00:30:00" "$(join_ids "${ids[@]}")" "${base} --merge-shards"
  else
    echo "${ids[0]}"
  fi
}

if [[ -z "${VALIDATION_MERGE_JOB}" && $((SKIP_ERA_STRATIFIED & SKIP_PATIENT_CLUSTER & SKIP_RIDGE)) -eq 0 ]]; then
  echo "NOTE: no cascade-validation-merge job ID (primary run skipped and no"
  echo "--validation-merge-job given). The validated-pattern follow-ups start"
  echo "immediately and fail fast if validation_results.parquet does not exist."
fi

echo "=== Queuing follow-ups for ${organism} ==="
finished_jobs=()
if [[ "${SKIP_ERA_STRATIFIED}" -eq 0 ]]; then
  era_job="$(sharded_follow_up era_stratified era-stratified "${ERA_STRATIFIED_SHARDS}" "${ERA_STRATIFIED_MEM}" "${ERA_STRATIFIED_TIME}")"
  echo "Site-by-era permutation (validated patterns): ${era_job}"
  finished_jobs+=("${era_job}")
fi
if [[ "${SKIP_PATIENT_CLUSTER}" -eq 0 ]]; then
  patient_job="$(sharded_follow_up patient_cluster patient-cluster "${PATIENT_CLUSTER_SHARDS}" "${PATIENT_CLUSTER_MEM}" "${PATIENT_CLUSTER_TIME}")"
  echo "Patient-cluster bootstrap (robust patterns): ${patient_job}"
  finished_jobs+=("${patient_job}")
fi
if [[ "${SKIP_RIDGE}" -eq 0 ]]; then
  if [[ "${DRY_RUN}" -eq 0 && -z "${VALIDATION_MERGE_JOB}" && "${FORCE_RERUN_EXISTING}" -eq 0 ]] &&
    "${python_bin}" scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined --organism "${organism}" --penalties 1,10,100 --check-ready; then
    echo "Ridge sensitivity: reusing provenance-matched completed result; no refit queued."
  else
    ridge_job="$(follow_up amr_ridge_penalty_sensitivity "${cascade_partition}" "${RIDGE_MEM}" "${RIDGE_TIME}" "${VALIDATION_MERGE_JOB}" \
      "\"\${AMR_CASCADE_PYTHON}\" scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined --organism '${organism}' --penalties 1,10,100")"
    echo "Ridge-penalty refit (validated patterns): ${ridge_job}"
    finished_jobs+=("${ridge_job}")
  fi
fi
if [[ "${SKIP_PREDICTION}" -eq 0 ]]; then
  # After the report and its readiness audit, as in the local runner: the prediction
  # writes supp_pred_* into the report's figure directory, the report archives every
  # earlier top-level figure there, and the audit fails on files outside its manifest.
  # Held on training alone, a prediction that finished first lost its figures.
  prediction_deps="$(join_ids "${TRAINING_JOB}" "${REPORT_JOB}" "${READINESS_JOB}")"
  prediction_job="$(follow_up amr_supplementary_prediction "${main_partition}" "${PREDICTION_MEM}" "${PREDICTION_TIME}" "${prediction_deps}" \
    "\"\${AMR_CASCADE_PYTHON}\" scripts/run_supplementary_prediction.py --env hpc --organism '${organism}' ${presentation_args}")"
  echo "Supplementary prediction tables and figures: ${prediction_job}"
  finished_jobs+=("${prediction_job}")
fi
if [[ "${SKIP_MASKING}" -eq 0 && "${PUBLICATION}" != paper1 ]]; then
  # Paper 2's masking experiment, on the full data; reads the combined cascade's edge report.
  organism_slug="$(printf '%s' "${organism}" | tr '[:upper:] ' '[:lower:]_')"
  masking_job="$(follow_up amr_masking_experiment "${main_partition}" "${MASKING_MEM}" "${MASKING_TIME}" "${combined_cascade_job}" \
    "\"\${AMR_CASCADE_PYTHON}\" scripts/run_masking_experiment.py --env hpc --organism '${organism}' --output-dir outputs/tables/combined/organisms/${organism_slug}/masking_experiment")"
  echo "Masking experiment (Paper 2): ${masking_job}"
  finished_jobs+=("${masking_job}")
fi
if [[ "${SKIP_TEMPORAL}" -eq 0 ]]; then
  # Year-by-year course of the validated patterns. After the report and its readiness audit for the
  # same reason as the prediction: it draws a figure into the report's figure directory.
  temporal_deps="$(join_ids "${combined_cascade_job}" "${REPORT_JOB}" "${READINESS_JOB}")"
  temporal_job="$(follow_up amr_temporal_regimes "${main_partition}" "${TEMPORAL_MEM}" "${TEMPORAL_TIME}" "${temporal_deps}" \
    "\"\${AMR_CASCADE_PYTHON}\" scripts/run_temporal_regimes.py --env hpc --organism '${organism}'")"
  echo "Year-by-year course of the validated patterns: ${temporal_job}"
  finished_jobs+=("${temporal_job}")
fi
if [[ "${SKIP_SUPPLEMENTARY_TABLES}" -eq 0 ]]; then
  # Also held on the three sensitivity runs' validation merges: the Supplement's availability-sensitivity table compares them with the primary run.
  table_deps="$(join_ids "${REPORT_JOB}" "${combined_cascade_job}" "${site_cascade_jobs}" ${finished_jobs[@]+"${finished_jobs[@]}"} ${sensitivity_jobs[@]+"${sensitivity_jobs[@]}"})"
  tables_job="$(follow_up amr_supplementary_tables "${main_partition}" "${TABLES_MEM}" "${TABLES_TIME}" "${table_deps}" \
    "\"\${AMR_CASCADE_PYTHON}\" scripts/build_supplementary_tables.py --env hpc --organism '${organism}' --publication '${PUBLICATION}'")"
  echo "Supplementary tables and quoted numbers: ${tables_job}${table_deps:+ (after ${table_deps})}"
  if [[ "${PUBLICATION}" != all ]]; then
    audit_job="$(follow_up "amr_${PUBLICATION}_artifact_audit" "${main_partition}" "8G" "00:30:00" "${tables_job}" \
      "\"\${AMR_CASCADE_PYTHON}\" scripts/audit_paper1_outputs.py --env hpc --organism '${organism}' --publication '${PUBLICATION}'")"
    echo "Manuscript artifact audit (${PUBLICATION}): ${audit_job}"
  fi
fi

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete; no HPC jobs were submitted."
else
  echo "All submissions queued. Check status with: squeue -u \$USER"
fi
echo "When everything has finished, export the outputs and run"
echo "scripts/build_manuscript_package.py where the paper folders are."
