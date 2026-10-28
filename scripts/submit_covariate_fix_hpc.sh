#!/usr/bin/env bash
# Re-run the stages that use episode covariates, after the covariate correction of 25 October 2026
# (comorbidity rows without a named condition are missing data; undated conditions are flagged;
# the ADI state rank has its own availability flag). See RUNBOOK "Covariate correction re-run".
#
# Since 28 October it also carries two upstream fixes:
#   - UT Southwestern comorbidity: silver cleaning kept one row per (culture, condition) before the timing
#     rules (TableCleaner._DEDUP_AFTER_TIMING_TABLES); re-cleans that table and re-harmonises comorbidity.
#   - Sex in the prediction benchmark: the 0/1 source codes now get their own indicators
#     (DemographicsFeatureBuilder); rebuilds the features and re-trains the three models.
# Re-runs: comorbidity re-clean and harmonisation; the dataset summary; adjusted models at all three sites;
# adjusted models + SHAP (combined cascade, reusing the
# saved validation), ridge-penalty sensitivity, prevalence shift, masking experiment, report (every report
# figure), SHAP figures; features, training and the threshold comparison; the prediction figures and
# tables (after training and the report), the year-by-year course of the validated patterns (after the
# report), supplementary tables and the manuscripts audit. Does not re-run
# validation, sensitivity shards, gold or ESKAPE. Each job waits for the jobs it needs.
#
#   bash scripts/submit_covariate_fix_hpc.sh --dry-run   # print the sbatch commands only
#   bash scripts/submit_covariate_fix_hpc.sh             # submit
#   bash scripts/submit_covariate_fix_hpc.sh --resume    # after a failure: re-run only what did not complete
#
# --resume re-submits each stage whose newest job since the first submission did not end COMPLETED or
# started before a stage it needs had finished (out of order), and every stage that waits for one of
# them; the other completed stages are kept and not waited for. It keeps
# logs/covariate_fix_submitted.txt (the export checks outputs against it) and moves the old logs of
# re-run stages to logs/superseded/. Combine with --dry-run to see the decisions first.
set -euo pipefail

DRY_RUN=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --resume) RESUME=1 ;;
    *) echo "Unknown option: $arg (use --dry-run and/or --resume)" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
ORG="--organism 'ESCHERICHIA COLI'"
mkdir -p logs

# Stop before submitting anything if the saved validation would not be reused.
if ! "$PY" scripts/check_validation_reuse.py --env hpc --include-sites --require-results; then
  echo "Validation reuse check did not pass: nothing submitted." >&2
  exit 1
fi

# Jobs share this checkout. A dirty tree is not represented by the commit in the marker.
dirty="$(git status --porcelain -- src scripts configs pyproject.toml requirements-hpc.txt)"
if [[ -n "$dirty" ]]; then
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "Commit the reviewed code before submission/resume; otherwise completed and queued jobs can use different code. Nothing submitted." >&2
    exit 1
  fi
  echo "DRY RUN ONLY: working-tree changes must be committed before submission." >&2
fi
if command -v squeue >/dev/null 2>&1; then
  active="$(squeue -h -u "$USER" -o '%j')"
  active="$(printf '%s\n' "$active" | grep -E '^(fix_|amr_)' || true)"
  [[ -z "$active" ]] || { echo "Pipeline jobs are still queued/running; do not overwrite their shared inputs: $active" >&2; exit 1; }
fi

MARKER="logs/covariate_fix_submitted.txt"
if [[ $RESUME -eq 1 ]]; then
  [[ -f "$MARKER" ]] || { echo "--resume: no $MARKER, so the chain was never submitted here; run without --resume." >&2; exit 1; }
  active="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep '^fix_' | sort | uniq -c | tr -s ' ' || true)"
  [[ -z "$active" ]] || { echo "--resume: fix_ jobs are still queued or running:$active. Wait for them, or cancel stuck ones, then resume." >&2; exit 1; }
  first_commit="$(sed -n 2p "$MARKER")"
  # Re-run stages must run the same code as the stages that completed.
  if ! git diff --quiet "$first_commit" HEAD -- src scripts configs pyproject.toml ':(exclude)scripts/submit_covariate_fix_hpc.sh'; then
    echo "--resume: the code differs from the first submission ($first_commit); resuming would mix code versions. Nothing submitted." >&2
    exit 1
  fi
  SINCE="$(head -1 "$MARKER" | tr ' ' T)"
  echo "Resuming the chain first submitted $(head -1 "$MARKER") at ${first_commit:0:7}:" >&2
fi

last_job() {  # FIELD of the newest job with this name since the first submission ("" if none)
  sacct -X -n -P -u "$USER" -S "$SINCE" --name="$1" -o "JobIDRaw,$2" 2>/dev/null | sort -t'|' -k1,1n | tail -1 | cut -d'|' -f2 || true
}

# Refuse queued jobs if tracked source changed after submission. No validation sidecars are altered.
EXPECTED_COMMIT="$(git rev-parse HEAD)"
WRAP="cd '$ROOT' && test \"\$(git rev-parse HEAD)\" = '$EXPECTED_COMMIT' && test -z \"\$(git status --porcelain -- src scripts configs pyproject.toml requirements-hpc.txt)\" && unset PYTHONHOME PYTHONPATH && export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR='$ROOT/.cache/matplotlib' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 && '$PY'"

submit() {  # submit NAME PARTITION MEM TIME DEPENDENCY COMMAND [UPSTREAM_STAGE_NAMES]
  local name=$1 partition=$2 mem=$3 time=$4 dependency=$5 command=$6 upstream=${7:-}
  local args=(--parsable --cpus-per-task=8 --partition="$partition" --job-name="$name" --mem="$mem" --time="$time"
              --output="logs/${name}_%j.out")
  # Kept stages contribute an empty id: drop it, so a job waits only for jobs submitted now.
  dependency="$(echo "$dependency" | tr -s ':' | sed 's/^://; s/:$//')"
  if [[ $RESUME -eq 1 ]]; then
    if [[ -z "$dependency" ]]; then
      local state stale="" started; state="$(last_job "$name" State)"
      if [[ "$state" == COMPLETED ]]; then
        started="$(last_job "$name" Start)"
        for u in $upstream; do  # ISO timestamps compare as text
          [[ "$(last_job "$u" End)" > "$started" ]] && stale="$u"
        done
        if [[ -z "$stale" ]]; then
          echo "  kept   $name (completed)" >&2
          echo ""
          return
        fi
        echo "  re-run $name (completed, but started before $stale had finished)" >&2
      else
        echo "  re-run $name (last job: ${state:-none})" >&2
      fi
    else
      echo "  re-run $name (waits for a re-run stage)" >&2
    fi
    if [[ $DRY_RUN -eq 0 ]]; then
      mkdir -p logs/superseded
      mv logs/"${name}"_*.out logs/superseded/ 2>/dev/null || true
    fi
  fi
  [[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "sbatch ${args[*]} --wrap \"$WRAP $command\"" >&2
    echo "DRYRUN_$name"
  else
    sbatch "${args[@]}" --wrap "$WRAP $command"
  fi
}

# Submission time: archive_export_artifacts.sh checks that the outputs it packs are newer than this.
[[ $DRY_RUN -eq 0 && $RESUME -eq 0 ]] && { date '+%Y-%m-%d %H:%M:%S'; git rev-parse HEAD; } > "$MARKER"

# Comorbidity: re-clean UT Southwestern's table only, then re-harmonise comorbidity for all three sites
# (the combined table is built from all three; the other tables and the gold layer are not touched).
reclean=$(submit fix_reclean main 256G 16:00:00 "" "scripts/run_preprocessing.py --env hpc --site armd_utsw --table comorbidity")
harm=$(submit fix_harmonize main 384G 20:00:00 "$reclean" "scripts/run_harmonization.py --env hpc --site armd --site armd_ecuh --site armd_utsw --table comorbidity" "fix_reclean")
site_armd=$(submit fix_site_armd long 512G 24:00:00 "$harm" "scripts/run_cascade_analysis.py --env hpc --gold-scope site --site armd $ORG --skip-shap" "fix_harmonize")
site_ecuh=$(submit fix_site_ecuh long 512G 24:00:00 "$harm" "scripts/run_cascade_analysis.py --env hpc --gold-scope site --site armd_ecuh $ORG --skip-shap" "fix_harmonize")
site_utsw=$(submit fix_site_utsw long 512G 24:00:00 "$harm" "scripts/run_cascade_analysis.py --env hpc --gold-scope site --site armd_utsw $ORG --skip-shap" "fix_harmonize")
casc=$(submit fix_cascade long 512G 24:00:00 "$harm" "scripts/run_cascade_analysis.py --env hpc --gold-scope combined $ORG" "fix_harmonize")
# Dataset summary (cohort, eligibility, covariate availability), redrawn from the corrected covariates;
# titles kept, because these figures are read on their own, not inside a manuscript.
dsum=$(submit fix_dataset_summary main 384G 08:00:00 "$harm" "scripts/run_dataset_summary.py --env hpc --scope combined $ORG --figure-format png --figure-format pdf --figure-format html --figure-format svg" "fix_harmonize")
# Pair features also read the training site's adjusted edge report; do not use its old ORs.
feat=$(submit fix_features main 512G 12:00:00 "$harm:$site_armd:$site_ecuh:$site_utsw" "scripts/run_feature_build.py --env hpc --scope combined $ORG" "fix_harmonize fix_site_armd fix_site_ecuh fix_site_utsw")
train=$(submit fix_training main 1024G 20:00:00 "$feat" "scripts/run_training.py --env hpc --scope combined $ORG" "fix_features")
thr=$(submit fix_threshold main 1024G 20:00:00 "$feat" "scripts/run_threshold_strategy_comparison.py --env hpc --scope combined $ORG --split-strategy site" "fix_features")
ridge=$(submit fix_ridge long 512G 24:00:00 "$casc" "scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined $ORG --penalties 1,10,100" "fix_cascade")
prev=$(submit fix_prev main 256G 16:00:00 "$casc" "scripts/run_prevalence_analysis.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf" "fix_cascade")
mask=$(submit fix_masking main 256G 24:00:00 "$casc" "scripts/run_masking_experiment.py --env hpc $ORG --output-dir outputs/tables/combined/organisms/escherichia_coli/masking_experiment" "fix_cascade")
rep=$(submit fix_report main 384G 08:00:00 "$prev:$train:$thr:$site_armd:$site_ecuh:$site_utsw" "scripts/check_publication_renderer.py && '$PY' scripts/run_reporting.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf --publication manuscripts --fail-on-missing-figures --hide-figure-titles" "fix_prev fix_training fix_threshold fix_site_armd fix_site_ecuh fix_site_utsw")
# 64G ran out of memory on 27 October (peak about 67 GB).
shap=$(submit fix_shap_figures main 256G 04:00:00 "$casc" "scripts/export_shap_figures.py --env hpc $ORG --hide-figure-titles" "fix_cascade")
# Draws the six prediction figures and three prediction tables from the re-trained models' predictions.
# It waits for the report: the report moves every figure already in the folder to _stale_figure_archive/
# before drawing (on 27 October it moved these six), and the audit requires them to be newer than the report.
pred=$(submit fix_pred_figures main 64G 02:00:00 "$rep:$train:$thr" "scripts/run_supplementary_prediction.py --env hpc $ORG --hide-figure-titles" "fix_report fix_training fix_threshold")
# Year-by-year course of the validated patterns (tables, numbers and one figure); after the report for the
# same reason as the prediction figures. Reads the gold pair table in batches (peak 4.2 GB, under a minute on the Mac).
temp=$(submit fix_temporal main 64G 02:00:00 "$rep" "scripts/run_temporal_regimes.py --env hpc $ORG" "fix_report")
tabs=$(submit fix_tables main 64G 02:00:00 "$rep:$ridge:$shap:$pred:$mask:$temp" "scripts/build_supplementary_tables.py --env hpc $ORG --publication manuscripts" "fix_report fix_ridge fix_shap_figures fix_pred_figures fix_masking fix_temporal")
audit=$(submit fix_audit main 8G 00:30:00 "$tabs:$dsum" "scripts/audit_paper1_outputs.py --env hpc $ORG --publication manuscripts --refresh-marker '$ROOT/$MARKER'" "fix_tables fix_dataset_summary")

echo "Submitted (empty = kept): re-clean $reclean, harmonise $harm, site cascades $site_armd/$site_ecuh/$site_utsw, dataset summary $dsum, features $feat, training $train, threshold $thr, cascade $casc, ridge $ridge, prevalence $prev, masking $mask, report $rep, SHAP figures $shap, prediction figures $pred, year-by-year $temp, tables $tabs, audit $audit"
echo "Watch with: squeue -u \$USER    Logs: logs/fix_*_<jobid>.out"
