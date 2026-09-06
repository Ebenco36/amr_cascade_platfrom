#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/archive_export_artifacts.sh [options]

Create an export-ready archive of AMR Cascade Platform artefacts. The script is
non-destructive: it does not delete, move, or modify pipeline outputs.

Default archive contents:
  data/artifacts
  data/gold
  data/features
  outputs

Options:
  --output-dir DIR           Directory for archives. Default: export_archives
  --name NAME                Archive basename prefix. Default: amr_cascade_export
  --include-manuscripts      Include split manuscript source directories:
                             paper1_detection_validation and
                             paper2_prevalence_surveillance.
  --include-test-outputs     Include test_dataset/outputs when present
  --include-reference        Include data/reference when present
  --include-intermediate     Include data/bronze, data/silver, data/harmonized,
                             data/interim, and data/metadata when present
  --include-raw              Include data/raw. This can be very large.
  --include-all-data         Include the full data directory. This can be huge.
                             Overrides the default data/artifacts/gold/features
                             selection and ignores other data include flags.
  --include-logs             Include the top-level logs/ directory (SLURM job
                             .out/.err files, pipeline event and timeline logs,
                             and the job-ID file). Not part of the default set
                             because logs are run-specific, not pipeline
                             artefacts. Composable with any data mode above,
                             including --include-all-data.
  --include-path PATH        Include an additional path, relative to the
                             project root. Repeatable: pass --include-path
                             more than once to add several arbitrary paths
                             without needing a dedicated flag for each.
                             Existence is checked the same way as every other
                             include; a missing path is skipped with a warning,
                             not a failure. Composable with any data mode above.
  --keep-stale-figures       Keep */_stale_figure_archive/* in the archive.
                             Default: exclude stale archived figures.
  --keep-ds-store            Keep .DS_Store files. Default: exclude them.
  --dry-run                  Print what would be archived, without creating files.
  --help                     Show this message.

Examples:
  bash scripts/archive_export_artifacts.sh
  bash scripts/archive_export_artifacts.sh --include-test-outputs
  bash scripts/archive_export_artifacts.sh --include-reference --include-intermediate
  bash scripts/archive_export_artifacts.sh --include-raw --name amr_cascade_full_data
  bash scripts/archive_export_artifacts.sh --include-logs --include-intermediate
  bash scripts/archive_export_artifacts.sh --include-path configs --include-path notebooks

Recommended HPC usage after production finishes:
  cd /scratch/projekte/FG37_ARS_ZKI_PH5/amr_cascade_platfrom
  bash scripts/archive_export_artifacts.sh --include-logs --include-intermediate --name amr_cascade_hpc_outputs

Recommended Mac usage:
  cd /Users/awotoroebenezer/Desktop/amr_cascade_platform
  bash scripts/archive_export_artifacts.sh --include-test-outputs
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
ARCHIVE_NAME="amr_cascade_export"
INCLUDE_TEST_OUTPUTS=0
INCLUDE_MANUSCRIPTS=0
INCLUDE_REFERENCE=0
INCLUDE_INTERMEDIATE=0
INCLUDE_RAW=0
INCLUDE_ALL_DATA=0
INCLUDE_LOGS=0
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
    --include-test-outputs)
      INCLUDE_TEST_OUTPUTS=1
      shift
      ;;
    --include-manuscripts)
      INCLUDE_MANUSCRIPTS=1
      shift
      ;;
    --include-reference)
      INCLUDE_REFERENCE=1
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
    --include-logs)
      INCLUDE_LOGS=1
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

includes=()
if [[ "${INCLUDE_ALL_DATA}" -eq 1 ]]; then
  includes+=(data)
else
  includes+=(data/artifacts data/gold data/features outputs)
  [[ "${INCLUDE_MANUSCRIPTS}" -eq 1 ]] && includes+=(paper1_detection_validation paper2_prevalence_surveillance)
  [[ "${INCLUDE_TEST_OUTPUTS}" -eq 1 ]] && includes+=(test_dataset/outputs)
  [[ "${INCLUDE_REFERENCE}" -eq 1 ]] && includes+=(data/reference)
  if [[ "${INCLUDE_INTERMEDIATE}" -eq 1 ]]; then
    includes+=(data/bronze data/silver data/harmonized data/interim data/metadata)
  fi
  [[ "${INCLUDE_RAW}" -eq 1 ]] && includes+=(data/raw)
fi
[[ "${INCLUDE_LOGS}" -eq 1 ]] && includes+=(logs)
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
  log "Dry run only. No archive created."
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
