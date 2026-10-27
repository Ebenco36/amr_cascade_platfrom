#!/usr/bin/env bash
# Re-run the stages that use episode covariates, after the covariate correction of 25 October 2026
# (comorbidity rows without a named condition are missing data; undated conditions are flagged;
# the ADI state rank has its own availability flag). See RUNBOOK "Covariate correction re-run".
#
# Re-runs: adjusted models + SHAP (combined cascade, reusing the saved validation), ridge-penalty
# sensitivity, prevalence shift, masking experiment, report (every report figure), SHAP figures, the
# prediction figures (redrawn from saved predictions), supplementary tables and the manuscripts audit. Does not re-run validation, sensitivity shards, features, training or
# prediction. Each job waits for the jobs it needs.
#
#   bash scripts/submit_covariate_fix_hpc.sh --dry-run   # print the sbatch commands only
#   bash scripts/submit_covariate_fix_hpc.sh             # submit
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
ORG="--organism 'ESCHERICHIA COLI'"
mkdir -p logs

# Stop before submitting anything if the saved validation would not be reused.
if ! "$PY" scripts/check_validation_reuse.py --env hpc; then
  echo "Validation reuse check did not pass: nothing submitted." >&2
  exit 1
fi

WRAP="cd '$ROOT' && export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR='$ROOT/.cache/matplotlib' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 && '$PY'"

submit() {  # submit NAME PARTITION MEM TIME DEPENDENCY COMMAND
  local name=$1 partition=$2 mem=$3 time=$4 dependency=$5 command=$6
  local args=(--parsable --cpus-per-task=8 --partition="$partition" --job-name="$name" --mem="$mem" --time="$time"
              --output="logs/${name}_%j.out")
  [[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "sbatch ${args[*]} --wrap \"$WRAP $command\"" >&2
    echo "DRYRUN_$name"
  else
    sbatch "${args[@]}" --wrap "$WRAP $command"
  fi
}

# Submission time: archive_export_artifacts.sh checks that the outputs it packs are newer than this.
[[ $DRY_RUN -eq 0 ]] && { date '+%Y-%m-%d %H:%M:%S'; git rev-parse HEAD; } > logs/covariate_fix_submitted.txt

casc=$(submit fix_cascade long 512G 24:00:00 "" "scripts/run_cascade_analysis.py --env hpc --gold-scope combined $ORG")
ridge=$(submit fix_ridge long 512G 24:00:00 "$casc" "scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined $ORG --penalties 1,10,100")
prev=$(submit fix_prev main 256G 16:00:00 "$casc" "scripts/run_prevalence_analysis.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf")
mask=$(submit fix_masking main 256G 24:00:00 "$casc" "scripts/run_masking_experiment.py --env hpc $ORG --output-dir outputs/tables/combined/organisms/escherichia_coli/masking_experiment")
rep=$(submit fix_report main 384G 08:00:00 "$prev" "scripts/check_publication_renderer.py && '$PY' scripts/run_reporting.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf --publication manuscripts --fail-on-missing-figures --hide-figure-titles")
shap=$(submit fix_shap_figures main 64G 04:00:00 "$casc" "scripts/export_shap_figures.py --env hpc $ORG --hide-figure-titles")
# Redraws the six prediction figures and three prediction tables from the saved predictions (no
# retraining), so that every manuscript figure comes from the current plotting code.
pred=$(submit fix_pred_figures main 64G 02:00:00 "" "scripts/run_supplementary_prediction.py --env hpc $ORG --hide-figure-titles")
tabs=$(submit fix_tables main 64G 02:00:00 "$rep:$ridge:$shap:$pred" "scripts/build_supplementary_tables.py --env hpc $ORG --publication manuscripts")
audit=$(submit fix_audit main 8G 00:30:00 "$tabs" "scripts/audit_paper1_outputs.py --env hpc $ORG --publication manuscripts")

echo "Submitted: cascade $casc, ridge $ridge, prevalence $prev, masking $mask, report $rep, SHAP figures $shap, prediction figures $pred, tables $tabs, audit $audit"
echo "Watch with: squeue -u \$USER    Logs: logs/fix_*_<jobid>.out"
