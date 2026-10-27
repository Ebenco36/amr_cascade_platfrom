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
#   bash scripts/submit_covariate_fix_hpc.sh --resume    # after a failure: re-run only what did not complete
#
# --resume re-submits each stage whose newest job since the first submission did not end COMPLETED,
# and every stage that waits for one of them; completed stages are kept and not waited for. It keeps
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
if ! "$PY" scripts/check_validation_reuse.py --env hpc; then
  echo "Validation reuse check did not pass: nothing submitted." >&2
  exit 1
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

last_state() {  # State of the newest job with this name since the first submission ("" if none)
  sacct -X -n -P -u "$USER" -S "$SINCE" --name="$1" -o JobIDRaw,State 2>/dev/null | sort -t'|' -k1,1n | tail -1 | cut -d'|' -f2 || true
}

WRAP="cd '$ROOT' && export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR='$ROOT/.cache/matplotlib' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 && '$PY'"

submit() {  # submit NAME PARTITION MEM TIME DEPENDENCY COMMAND
  local name=$1 partition=$2 mem=$3 time=$4 dependency=$5 command=$6
  local args=(--parsable --cpus-per-task=8 --partition="$partition" --job-name="$name" --mem="$mem" --time="$time"
              --output="logs/${name}_%j.out")
  # Kept stages contribute an empty id: drop it, so a job waits only for jobs submitted now.
  dependency="$(echo "$dependency" | tr -s ':' | sed 's/^://; s/:$//')"
  if [[ $RESUME -eq 1 ]]; then
    if [[ -z "$dependency" ]]; then
      local state; state="$(last_state "$name")"
      if [[ "$state" == COMPLETED ]]; then
        echo "  kept   $name (completed)" >&2
        echo ""
        return
      fi
      echo "  re-run $name (last job: ${state:-none})" >&2
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

casc=$(submit fix_cascade long 512G 24:00:00 "" "scripts/run_cascade_analysis.py --env hpc --gold-scope combined $ORG")
ridge=$(submit fix_ridge long 512G 24:00:00 "$casc" "scripts/run_ridge_penalty_sensitivity.py --env hpc --gold-scope combined $ORG --penalties 1,10,100")
prev=$(submit fix_prev main 256G 16:00:00 "$casc" "scripts/run_prevalence_analysis.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf")
mask=$(submit fix_masking main 256G 24:00:00 "$casc" "scripts/run_masking_experiment.py --env hpc $ORG --output-dir outputs/tables/combined/organisms/escherichia_coli/masking_experiment")
rep=$(submit fix_report main 384G 08:00:00 "$prev" "scripts/check_publication_renderer.py && '$PY' scripts/run_reporting.py --env hpc --scope combined $ORG --figure-format html --figure-format png --figure-format svg --figure-format pdf --publication manuscripts --fail-on-missing-figures --hide-figure-titles")
# 64G ran out of memory on 27 October (peak about 67 GB).
shap=$(submit fix_shap_figures main 256G 04:00:00 "$casc" "scripts/export_shap_figures.py --env hpc $ORG --hide-figure-titles")
# Redraws the six prediction figures and three prediction tables from the saved predictions (no
# retraining), so that every manuscript figure comes from the current plotting code.
pred=$(submit fix_pred_figures main 64G 02:00:00 "" "scripts/run_supplementary_prediction.py --env hpc $ORG --hide-figure-titles")
tabs=$(submit fix_tables main 64G 02:00:00 "$rep:$ridge:$shap:$pred" "scripts/build_supplementary_tables.py --env hpc $ORG --publication manuscripts")
audit=$(submit fix_audit main 8G 00:30:00 "$tabs" "scripts/audit_paper1_outputs.py --env hpc $ORG --publication manuscripts")

echo "Submitted (empty = kept): cascade $casc, ridge $ridge, prevalence $prev, masking $mask, report $rep, SHAP figures $shap, prediction figures $pred, tables $tabs, audit $audit"
echo "Watch with: squeue -u \$USER    Logs: logs/fix_*_<jobid>.out"
