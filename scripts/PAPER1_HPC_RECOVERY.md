# Paper 1: scoped execution and HPC recovery

This profile follows `paper1_detection_validation` and its supplement. It is
not a new scientific method and does not lower any validation threshold.

## What runs

`submit_everything_hpc.sh --publication paper1` retains the three site cascades,
combined cascade, adjusted associations, prediction benchmark, threshold-strategy
comparison, three availability variants, site-by-era permutation, patient bootstrap,
and ridge sensitivity. Availability variants stop at the validation merge; their
supplementary comparisons do not require a second adjusted-model/SHAP run.

It omits Paper 2 prevalence/MNAR analysis, the broad dataset-summary job, optional
SHAP computation, and figure families not cited in Paper 1. Adjusted models still
run. SHAP's omission is recorded in the cascade completion marker. Using
`--publication all` explicitly opts into the previous full-platform behaviour.
The wrapper now defaults to `--publication manuscripts`, covering both papers;
see [MANUSCRIPTS_HPC.md](MANUSCRIPTS_HPC.md) for the combined run.

The reporting contract is 17 generated cascade/cohort figures plus 6 prediction
figures, each in PDF, PNG, SVG, and HTML. The manually maintained
`figure_pipeline_overview.pdf` is checked separately during manuscript packaging.
Empty directional subsets are labelled explicitly; a nonempty file alone is not
evidence of a nonempty scientific result.

## Do not overwrite a running checkout or environment

The October logs show both startup failures and earlier modeling out-of-memory
failures. A successful login-node import does not prove that the same interpreter
works on compute nodes. Do not delete/recreate `.venv` while queued jobs reference
it. Do not change the checkout used by queued/running jobs. Inspect SLURM status
first; this guide does not cancel any jobs.

```bash
squeue -u "$USER"
```

After installing these changes in a stable checkout, run these small checks on
the actual cluster, from the project root (no data analysis is performed):

```bash
srun --partition main --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:10:00 \
  bash -c 'PYTHONNOUSERSITE=1 .venv/bin/python scripts/check_python_runtime.py && .venv/bin/python scripts/check_publication_renderer.py'

srun --partition long --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:10:00 \
  bash -c 'PYTHONNOUSERSITE=1 .venv/bin/python scripts/check_python_runtime.py'
```

These check the allocated nodes, not every node in either partition. If the Python
startup failure recurs, record the hostname, executable symlink target, and
`pyvenv.cfg`; the existing logs cannot distinguish an unavailable base interpreter
from a removed environment or a mount problem.

## Validation reuse is checked, never forced

Before queuing any Paper 1 wave, the wrapper compares saved primary/site/variant
validation provenance against the current code, config, and gold input identity.
Missing results can be computed. Incompatible saved results stop submission.

```bash
.venv/bin/python scripts/check_validation_reuse.py --env hpc --include-sites
```

The existing provenance scheme hashes the whole checkout, so even reporting-only
changes can invalidate its identity. This is deliberately not bypassed here.
Finish an existing validation wave using its recorded frozen checkout, or plan an
explicit rebuild; a compatibility migration requires a separately reviewed code
and input comparison. Do not edit sidecars or use `--allow-unrecorded-validation`
to disguise an incompatible run.

## Inspect the proposed wave first

For completed primary ingestion, harmonization and gold layers:

```bash
CASCADE_PARTITION=long bash scripts/submit_everything_hpc.sh \
  --publication paper1 --partition main --cascade-partition long \
  --sites armd,armd_ecuh,armd_utsw --organisms "ESCHERICHIA COLI" \
  --skip-ingestion --skip-preprocessing --skip-harmonization \
  --skip-comorbidity --skip-gold --skip-install --dry-run
```

Remove `--dry-run` only after the compute-node runtime checks and validation reuse
checks pass. The wrapper performs the reuse checks for each enabled sensitivity
environment as well. Do not add `--force-rerun-existing` or `--fresh` to a recovery
command casually: the former forces recomputation; the latter moves old gold and
cascade directories aside and rebuilds them. `--fresh` is an intentional restart,
not a fix for incompatible provenance. Omit layer-skip flags for a genuinely new
cohort whose inputs are not yet built.

## Figures and audit

### Optional title-free publication exports

Add `--hide-figure-titles` to the reporting command or the HPC/Mac wrapper.
It removes global titles and subtitles in HTML, PDF, SVG, and PNG. Axis titles,
legends, panel/drug identifiers, reference lines, and empty-result messages stay.
Network heading watermarks are removed too. Fonts and the nominal canvas are not
reduced; title space is reclaimed while retaining room for legends and labels.
Leave the option off to restore the titled presentation.

The wrapper forwards this option to reporting, dataset summaries (when requested),
and supplementary prediction exports. A changed presentation option invalidates
the reporting cache, not the statistical estimates. The existing validation
provenance checks described above still apply to pipeline submission.

To regenerate only Paper 1 figures/tables from existing compatible artifacts,
without refitting models or repeating validation:

```bash
.venv/bin/python scripts/run_reporting.py \
  --env hpc --scope combined --organism "ESCHERICHIA COLI" \
  --publication paper1 --hide-figure-titles \
  --figure-format html --figure-format png --figure-format svg --figure-format pdf \
  --fail-on-missing-figures

.venv/bin/python scripts/run_supplementary_prediction.py \
  --env hpc --organism "ESCHERICHIA COLI" --hide-figure-titles
```

Run these inside a suitable compute allocation on HPC. They reuse saved results
and replace the corresponding figure exports; they do not create missing training
artifacts. The statistical content formerly explained in subtitles (cohort,
direction, display filtering, uncertainty and scale definitions) must still appear
in the manuscript captions. Title removal is a presentation change, not a
scientific completeness check. Manually maintained pipeline illustrations are
not modified by this option.

### Queue a figures-and-tables-only refresh

From a stable project checkout, after previous jobs writing these outputs have
finished, the following queues one report job. It does not submit ingestion,
cascade validation, feature building, training, or sensitivity refits. All their
required saved inputs must already exist. The memory request retains the DAG's
production reporting allocation; it is not a measured minimum.

```bash
mkdir -p logs
sbatch --partition=main --job-name=paper1_figures --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=384G --time=08:00:00 \
  --output=logs/paper1_figures_%j.log <<'SLURM'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" scripts/check_python_runtime.py
"$PYTHON" scripts/check_publication_renderer.py
"$PYTHON" scripts/run_reporting.py --env hpc --scope combined \
  --organism "ESCHERICHIA COLI" --publication paper1 --hide-figure-titles \
  --figure-format html --figure-format png --figure-format svg --figure-format pdf \
  --fail-on-missing-figures
"$PYTHON" scripts/run_supplementary_prediction.py --env hpc \
  --organism "ESCHERICHIA COLI" --hide-figure-titles
"$PYTHON" scripts/build_supplementary_tables.py --env hpc \
  --organism "ESCHERICHIA COLI" --publication paper1
"$PYTHON" scripts/audit_paper1_outputs.py --env hpc --organism "ESCHERICHIA COLI"
SLURM
```

The report exporter moves previous top-level figures in the organism directory
to `_stale_figure_archive` before writing the current Paper 1 set. Thus figures
outside this profile are not regenerated by this command. Do not delete outputs
first. The job fails if the saved LR coefficient artifact or a supplementary
analysis input is missing; it does not silently refit or allow missing tables.

Threshold panels now share an explicit 0.0--1.0 probability axis and a single
axis label. Diamonds show saved decision thresholds, not newly selected cutoffs.
Their note identifies validation-holdout versus grouped-OOF selection from the
saved metadata. Empty directional matrices have an explicit message without
collapsed axes. These changes affect presentation, not statistical estimates.

Paper 1 reporting refuses the generic Plotly-to-Matplotlib approximation. That
fallback can collapse subplot axes, omit Sankey flows and misplace annotations.
Native PDF/PNG/SVG export is tested inside the report job; verified plot-specific
fallbacks remain available where supplied. HTML alone does not certify static
publication figures.

The supplementary coefficient figure now reads coefficients saved from the actual
fitted, MaxAbs-scaled LR model. It no longer fits a different sampled model during
reporting. Coefficients and exponentiated coefficients must be described on that
scaled design, not as raw-unit or causal effects. Older training outputs without
this artifact fail clearly rather than silently recreating different results.

Feature-dataset export now scans bounded Arrow batches, avoiding a full-cohort
pandas copy before training. This removes one demonstrated memory hazard; it is
not proof that full-scale RF, XGBoost, or grouped-OOF training will fit in an
arbitrary allocation. Keep the existing high-memory allocations and inspect
`sacct` MaxRSS after the first completed training job.

The final `paper1_artifact_audit.json` checks the named exports, basic file integrity,
report profile, and supplementary missing-input report. It is an artifact audit,
not a claim of full statistical or visual certification. Inspect the production
PDFs after the run, especially dense Sankeys, networks and per-drug labels.
On the machine containing the manuscript folders, use
`scripts/build_manuscript_package.py --publication paper1 --env hpc --run-root PATH --dry-run`
to inspect the package plan, then remove `--dry-run` to package it. `PATH` is the
extracted run directory containing `data/` and `outputs/`. Paper 2 is untouched.
Neither statistical estimates nor validated-edge selection were changed by this
profile. The Sankey captions distinguish display-filtered from all validated
edges; the ER-versus-OR plot explicitly warns that their magnitudes are different
scales.
