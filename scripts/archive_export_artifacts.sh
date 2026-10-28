#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/archive_export_artifacts.sh [options]

Export everything the manuscripts need from a finished run in one verified archive. The script is
non-destructive: it does not delete, move, or modify pipeline outputs.

What it does, in order:
  1. Checks that the run is finished and current, and refuses to export otherwise:
     - no amr_/fix_ jobs queued or running (when squeue exists);
     - every ESKAPE target finished (_SUCCESS.json; --eskape-expected, default 24);
     - after the covariate re-run (scripts/submit_covariate_fix_hpc.sh; logs/covariate_fix_submitted.txt
       exists): its key outputs and every manuscript figure are newer than its submission, the SHAP
       figures exist and its audit passed (logs/fix_audit_*.out);
     - after a full run (scripts/submit_everything_hpc.sh; no such file): the manuscripts audit
       passed (logs/amr_*_artifact_audit_*.out);
     - Table 1 carries the corrected comorbidity rows;
     - the masking experiment ran on the full data.
  2. Writes a run record into logs/: commit, uncommitted code, validation-reuse check, job
     accounting and installed packages.
  3. Archives the default contents below plus any requested extras.
  4. Verifies the archive: checksum, and every key file present in it.

Default archive contents (paths that do not exist are skipped with a warning):
  data/artifacts  data/gold  data/features  data/metadata  data/reference  outputs  logs
  data_availability_sensitivity/artifacts
  data_availability_annual_sensitivity/artifacts
  data_availability_joint_sensitivity/artifacts

Options:
  --output-dir DIR           Directory for archives. Default: export_archives
  --name NAME                Archive basename prefix. Default: amr_cascade_hpc_outputs
  --allow-incomplete         Archive even if a check in step 1 fails (the failures are written
                             into the run record). Use for test or partial exports.
  --eskape-expected N        Finished ESKAPE targets required. Default: 24
  --include-manuscripts      Also include paper1_detection_validation and
                             paper2_prevalence_surveillance.
  --include-test-outputs     Also include test_dataset/outputs when present
  --include-intermediate     Also include data/bronze, data/silver, data/harmonized and
                             data/interim when present
  --include-raw              Also include data/raw. This can be very large.
  --include-all-data         Include the full data directory instead of the default data paths.
                             This can be huge.
  --include-path PATH        Include an additional path, relative to the project root.
                             Repeatable; a missing path is skipped with a warning.
  --include-logs, --include-reference
                             Accepted for compatibility; logs and data/reference are now
                             included by default.
  --keep-stale-figures       Keep */_stale_figure_archive/* in the archive.
                             Default: exclude stale archived figures.
  --keep-ds-store            Keep .DS_Store files. Default: exclude them.
  --dry-run                  Run the checks and print what would be archived, without
                             creating files.
  --help                     Show this message.

On the cluster, from the checkout's root, once every job has finished:
  bash scripts/archive_export_artifacts.sh --dry-run
  bash scripts/archive_export_artifacts.sh

Mac test export:
  bash scripts/archive_export_artifacts.sh --allow-incomplete --include-test-outputs
EOF
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "$(timestamp)" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
  exit 1
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="export_archives"
ARCHIVE_NAME="amr_cascade_hpc_outputs"
ALLOW_INCOMPLETE=0
ESKAPE_EXPECTED=24
INCLUDE_TEST_OUTPUTS=0
INCLUDE_MANUSCRIPTS=0
INCLUDE_INTERMEDIATE=0
INCLUDE_RAW=0
INCLUDE_ALL_DATA=0
EXTRA_PATHS=()
KEEP_STALE_FIGURES=0
KEEP_DS_STORE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    --allow-incomplete)
      ALLOW_INCOMPLETE=1
      shift
      ;;
    --eskape-expected)
      [[ $# -ge 2 ]] || die "--eskape-expected requires a value"
      ESKAPE_EXPECTED="$2"
      shift 2
      ;;
    --include-test-outputs)
      INCLUDE_TEST_OUTPUTS=1
      shift
      ;;
    --include-manuscripts)
      INCLUDE_MANUSCRIPTS=1
      shift
      ;;
    --include-reference|--include-logs)
      shift
      ;;
    --include-intermediate)
      INCLUDE_INTERMEDIATE=1
      shift
      ;;
    --include-raw)
      INCLUDE_RAW=1
      shift
      ;;
    --include-all-data)
      INCLUDE_ALL_DATA=1
      shift
      ;;
    --include-path)
      [[ $# -ge 2 ]] || die "--include-path requires a value"
      EXTRA_PATHS+=("$2")
      shift 2
      ;;
    --keep-stale-figures)
      KEEP_STALE_FIGURES=1
      shift
      ;;
    --keep-ds-store)
      KEEP_DS_STORE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

cd "${PROJECT_ROOT}"

PY=".venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"
E="data/artifacts/cascade/combined/organisms/escherichia_coli"
P="data/artifacts/prevalence_shift/combined/organisms/escherichia_coli"
T="outputs/tables/combined/organisms/escherichia_coli"
MARKER="logs/covariate_fix_submitted.txt"
RECORD="logs/export_record.txt"

# ---------------------------------------------------------------------------- 1. checks
problems=()
fail() { problems+=("$1"); printf '  FAIL  %s\n' "$1"; }
pass() { printf '  ok    %s\n' "$1"; }

log "Checking that the run is finished and current"

if command -v squeue >/dev/null 2>&1; then
  active="$(squeue -h -u "${USER}" -o '%j' 2>/dev/null | grep -E '^(amr_|fix_)' | sort | uniq -c | tr -s ' ' | tr '\n' ';' || true)"
  if [[ -n "${active}" ]]; then fail "jobs still queued or running:${active}"; else pass "no amr_/fix_ jobs queued or running"; fi
fi

eskape="$(find data/artifacts/eskape_validation -name _SUCCESS.json 2>/dev/null | wc -l | tr -d ' ' || true)"
if [[ "${eskape}" -ge "${ESKAPE_EXPECTED}" ]]; then pass "ESKAPE: ${eskape} finished targets"
else fail "ESKAPE: ${eskape} of ${ESKAPE_EXPECTED} targets finished"; fi

if [[ -f "${MARKER}" ]]; then
  pass "covariate re-run submitted $(head -1 "${MARKER}")"
  for f in "${E}/adjusted_results.parquet" "${E}/edge_report.parquet" "${P}/prevalence_shift.parquet" \
           "${P}/prevalence_mnar_sensitivity_curves.parquet" "${T}/table_v_cohort_characteristics.csv" \
           "${T}/supplementary/latex/generated_numbers.tex" "${T}/supplementary/supplementary_numbers.json" \
           "${T}/masking_experiment/masking_results.csv" "${T}/temporal_regimes/temporal_regime_summary.csv" \
           "data/artifacts/modeling/downstream_testing/combined/organisms/escherichia_coli/site__all_models/full/logistic_regression_coefficients.parquet" \
           "data/features/combined/organisms/escherichia_coli/model_ready_pair_features.parquet" \
           "outputs/figures/combined/organisms/escherichia_coli/dataset_summary/dataset_missing_data_profile.pdf"; do
    if [[ ! -f "${f}" ]]; then fail "missing: ${f}"
    elif [[ -z "$(find "${f}" -newer "${MARKER}" 2>/dev/null)" ]]; then fail "older than the covariate re-run: ${f}"
    else pass "${f}"; fi
  done
else
  warn "no ${MARKER}: checking a full run (scripts/submit_everything_hpc.sh), not the covariate re-run"
fi
[[ -f "${E}/validation_results.parquet" ]] && pass "${E}/validation_results.parquet" || fail "missing: ${E}/validation_results.parquet"

# Every figure the two manuscripts cite (the publication profiles' list; Fig. 1 is hand-made and not
# here) must have been redrawn by the covariate re-run, so that all come from the current plotting code.
F="outputs/figures/combined/organisms/escherichia_coli"
if [[ -f "${MARKER}" ]]; then
  stems="$(.venv/bin/python -c 'import sys; sys.path.insert(0, "src")
from amr_cascade_platform.reporting.manuscript_profile import MANUSCRIPT_REPORT_FIGURES
from amr_cascade_platform.reporting.paper1_profile import PAPER1_PREDICTION_FIGURES, PAPER1_TEMPORAL_FIGURES
print(" ".join(MANUSCRIPT_REPORT_FIGURES + PAPER1_PREDICTION_FIGURES + PAPER1_TEMPORAL_FIGURES))' 2>/dev/null || true)"
  if [[ -z "${stems}" ]]; then fail "could not read the manuscript figure list from the publication profiles"
  else
    stale_figs=(); missing_figs=(); n_figs=0
    for stem in ${stems}; do
      n_figs=$((n_figs + 1))
      if [[ ! -f "${F}/${stem}.pdf" ]]; then missing_figs+=("${stem}")
      elif [[ -z "$(find "${F}/${stem}.pdf" -newer "${MARKER}" 2>/dev/null)" ]]; then stale_figs+=("${stem}"); fi
    done
    if [[ ${#missing_figs[@]} -gt 0 ]]; then fail "manuscript figures missing: ${missing_figs[*]}"; fi
    if [[ ${#stale_figs[@]} -gt 0 ]]; then fail "manuscript figures not redrawn by the covariate re-run: ${stale_figs[*]}"; fi
    [[ ${#missing_figs[@]} -eq 0 && ${#stale_figs[@]} -eq 0 ]] && pass "all ${n_figs} manuscript figures redrawn after the covariate re-run"
  fi
fi

if [[ -n "$(find outputs/shap_explainability -type f 2>/dev/null | head -1)" ]]; then
  pass "SHAP figures: $(find outputs/shap_explainability -type f | wc -l | tr -d ' ') files"
elif [[ -f "${MARKER}" ]]; then fail "outputs/shap_explainability is missing or empty"
else warn "no SHAP figures (a full manuscripts run does not draw them; neither paper cites them)"; fi

if grep -q "Condition with unrecorded start" "${T}/table_v_cohort_characteristics.csv" 2>/dev/null; then
  pass "Table 1 has the corrected comorbidity rows"
else fail "Table 1 lacks the 'Condition with unrecorded start' row (report not re-run with the corrected code?)"; fi

if [[ -f "${T}/masking_experiment/masking_run.json" ]]; then
  masking="$("${PY}" -c "import json,sys; r=json.load(open(sys.argv[1])); print('ok' if not r.get('data_root_override') and (r.get('scenarios') or 0)>0 else 'bad', r.get('scenarios'))" "${T}/masking_experiment/masking_run.json" 2>&1 || echo "bad unreadable")"
  if [[ "${masking}" == ok* ]]; then pass "masking experiment: ${masking#ok } scenarios on the full data"
  else fail "masking_run.json: ${masking}"; fi
else fail "missing: ${T}/masking_experiment/masking_run.json"; fi

if [[ -f "${MARKER}" ]]; then audit_log="$(ls -t logs/fix_audit_*.out 2>/dev/null | head -1 || true)"
else audit_log="$(ls -t logs/amr_*_artifact_audit_*.out 2>/dev/null | head -1 || true)"; fi
if [[ -n "${audit_log}" ]] && grep -q "artifact audit: PASS" "${audit_log}"; then pass "audit PASS (${audit_log})"
else fail "manuscripts audit did not pass (${audit_log:-no audit log in logs/})"; fi

if [[ ${#problems[@]} -gt 0 ]]; then
  if [[ "${ALLOW_INCOMPLETE}" -eq 1 ]]; then
    warn "${#problems[@]} check(s) failed; archiving anyway (--allow-incomplete). They are listed in ${RECORD}."
  else
    die "Not exported: ${#problems[@]} check(s) failed (listed above). Fix them, or use --allow-incomplete for a partial export."
  fi
fi

# ---------------------------------------------------------------------------- 2. run record
if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p logs
  {
    echo "Export record, $(timestamp) on $(hostname)"
    echo "git HEAD: $(git rev-parse --verify -q HEAD 2>/dev/null || echo unknown)"
    echo "uncommitted code changes (src, scripts, configs):"
    git status --porcelain -- src scripts configs 2>/dev/null || true
    echo
    echo "checks failed: ${#problems[@]}"
    if [[ ${#problems[@]} -gt 0 ]]; then printf '  %s\n' "${problems[@]}"; fi
    echo
    echo "validation reuse check:"
    "${PY}" scripts/check_validation_reuse.py --env hpc 2>&1 || true
    if command -v sacct >/dev/null 2>&1 && [[ -f "${MARKER}" ]]; then
      echo
      echo "job accounting (covariate re-run):"
      sacct -u "${USER}" -S "$(head -1 "${MARKER}" | cut -c1-10)" -X -o JobID,JobName%18,State,Elapsed,MaxRSS \
        --name=fix_reclean,fix_harmonize,fix_site_armd,fix_site_ecuh,fix_site_utsw,fix_dataset_summary,fix_features,fix_training,fix_threshold,fix_cascade,fix_ridge,fix_prev,fix_masking,fix_report,fix_shap_figures,fix_pred_figures,fix_temporal,fix_tables,fix_audit 2>&1 || true
    fi
  } > "${RECORD}"
  "${PY}" -m pip freeze > logs/export_pip_freeze.txt 2>/dev/null || true
  log "Run record: ${RECORD}"
fi

# ---------------------------------------------------------------------------- 3. archive
includes=()
if [[ "${INCLUDE_ALL_DATA}" -eq 1 ]]; then
  includes+=(data)
else
  includes+=(data/artifacts data/gold data/features data/metadata data/reference)
  [[ "${INCLUDE_INTERMEDIATE}" -eq 1 ]] && includes+=(data/bronze data/silver data/harmonized data/interim)
  [[ "${INCLUDE_RAW}" -eq 1 ]] && includes+=(data/raw)
fi
includes+=(outputs logs
           data_availability_sensitivity/artifacts
           data_availability_annual_sensitivity/artifacts
           data_availability_joint_sensitivity/artifacts)
[[ "${INCLUDE_MANUSCRIPTS}" -eq 1 ]] && includes+=(paper1_detection_validation paper2_prevalence_surveillance)
[[ "${INCLUDE_TEST_OUTPUTS}" -eq 1 ]] && includes+=(test_dataset/outputs)
if [[ "${#EXTRA_PATHS[@]}" -gt 0 ]]; then
  includes+=("${EXTRA_PATHS[@]}")
fi

existing=()
for path in "${includes[@]}"; do
  if [[ -e "${path}" ]]; then
    existing+=("${path}")
  else
    warn "Skipping missing path: ${path}"
  fi
done

[[ "${#existing[@]}" -gt 0 ]] || die "No requested archive paths exist"

excludes=()
[[ "${KEEP_DS_STORE}" -eq 0 ]] && excludes+=(--exclude=.DS_Store)
[[ "${KEEP_STALE_FIGURES}" -eq 0 ]] && excludes+=(--exclude='*/_stale_figure_archive/*')

stamp="$(date +%Y%m%d_%H%M%S)"
archive_dir="${OUTPUT_DIR}"
archive_path="${archive_dir}/${ARCHIVE_NAME}_${stamp}.tar.gz"
manifest_path="${archive_path%.tar.gz}.manifest.txt"
checksum_path="${archive_path}.sha256"

log "Project root: ${PROJECT_ROOT}"
log "Archive paths:"
for path in "${existing[@]}"; do
  size="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"
  printf '  - %s (%s)\n' "${path}" "${size:-unknown}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "Dry run only. Checks done; no archive created."
  exit 0
fi

mkdir -p "${archive_dir}"

log "Creating archive: ${archive_path}"
tar "${excludes[@]}" -czf "${archive_path}" "${existing[@]}"

log "Writing manifest: ${manifest_path}"
tar -tzf "${archive_path}" > "${manifest_path}"

log "Writing checksum: ${checksum_path}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${archive_path}" > "${checksum_path}"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${archive_path}" > "${checksum_path}"
else
  warn "No sha256sum or shasum command found; checksum was not written"
  rm -f "${checksum_path}"
fi

# ---------------------------------------------------------------------------- 4. verification
log "Verifying the archive"
if [[ -f "${checksum_path}" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -c "${checksum_path}"; else shasum -a 256 -c "${checksum_path}"; fi
fi
missing=0
check_in_archive() {
  if grep -qxF "$1" "${manifest_path}"; then printf '  in archive  %s\n' "$1"
  else printf '  MISSING     %s\n' "$1" >&2; missing=1; fi
}
for f in "${E}/validation_results.parquet" "${E}/adjusted_results.parquet" "${E}/edge_report.parquet" \
         "${P}/prevalence_shift.parquet" "${T}/supplementary/latex/generated_numbers.tex" \
         "${T}/table_v_cohort_characteristics.csv" "${T}/masking_experiment/masking_results.csv" \
         "${T}/masking_experiment/masking_run.json" "${T}/temporal_regimes/temporal_regime_years.csv" "data/gold/combined/organisms/escherichia_coli/culture_episodes.parquet" \
         "${RECORD}"; do
  check_in_archive "${f}"
done
markers="$(grep -c '^data/artifacts/eskape_validation/.*_SUCCESS.json$' "${manifest_path}" || true)"
if [[ "${markers}" -ge "${ESKAPE_EXPECTED}" ]]; then printf '  in archive  ESKAPE results (%s completion markers)\n' "${markers}"
else printf '  MISSING     ESKAPE results (%s of %s completion markers)\n' "${markers}" "${ESKAPE_EXPECTED}" >&2; missing=1; fi
if grep -q '^outputs/shap_explainability/.' "${manifest_path}"; then printf '  in archive  SHAP figures\n'
else printf '  MISSING     SHAP figures\n' >&2; missing=1; fi
if [[ "${missing}" -eq 1 && "${ALLOW_INCOMPLETE}" -eq 0 ]]; then
  die "The archive lacks key files; do not use it."
fi

file_count="$(wc -l < "${manifest_path}" | tr -d ' ')"
archive_size="$(du -sh "${archive_path}" | awk '{print $1}')"

log "Archive complete"
printf 'Archive:  %s\n' "${archive_path}"
printf 'Size:     %s\n' "${archive_size}"
printf 'Files:    %s\n' "${file_count}"
printf 'Manifest: %s\n' "${manifest_path}"
if [[ -f "${checksum_path}" ]]; then
  printf 'Checksum: %s\n' "${checksum_path}"
fi
printf 'Copy all three files into the project folder on the Mac.\n'
