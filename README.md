# AMR Cascade Platform

A denominator-aware framework for detecting and validating directional patterns
in antimicrobial susceptibility testing (AST) observation.

The workflow constructs eligible episode-drug opportunities, estimates escalation
ratios, and evaluates patterns using permutation tests, bootstrap stability, and
site and temporal replication. Adjusted associations, supplementary prediction,
and an ESKAPE extension provide additional analyses.

**Interpretation:** observed AST visibility does not establish testing chronology,
causal ordering behaviour, or whether an absent result was never tested or withheld.

## Setup and Data

Requires **Python 3.12 or newer**. Production HPC execution uses **SLURM**.
Run commands from the repository root.

For local development:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/amr-cascade --help
```

For HPC, the submission scripts create/check `.venv` and install
`requirements-hpc.txt`. Use `--skip-install` only for an already prepared environment.
The base Python installation, including its standard library, must be accessible
on every execution node; a virtual environment alone is not a portable Python runtime.

Place source CSVs under `data/raw/<site>/`. The study site identifiers are
`armd` (Stanford), `armd_ecuh` (ECU Health), and `armd_utsw` (UT Southwestern).
Input schemas and mappings are defined in `configs/base/schema.yaml` and the
other base configuration files. Source datasets are separate from the code;
retain their original provenance and usage terms.

Source archives listed in the study's data-availability statement:
[Stanford](https://doi.org/10.5061/dryad.jq2bvq8kp),
[ECU Health](https://doi.org/10.5061/dryad.7sqv9s55x), and
[UT Southwestern](https://doi.org/10.5061/dryad.0rxwdbsd5).

Configuration uses `configs/base/*.yaml` plus overrides in
`configs/environments/<environment>.yaml`. Important files:

| File | Purpose |
|---|---|
| `configs/base/cascade.yaml` | Pair screening, estimation and validation |
| `configs/base/features.yaml` | Clinical and cascade-derived features |
| `configs/base/modeling.yaml` | Predictive models and threshold selection |
| `configs/base/reporting.yaml` | Tables and figure exports |
| `data/reference/intrinsic_resistance.csv` | Biological eligibility reference |
| `data/antibiotic_classification_complete.csv` | Drug classifications and display labels |

## Production Run on HPC

Use `submit_everything_hpc.sh --publication paper1` for the detection-and-validation
workflow, including supplementary prediction, sensitivity analyses and artifact audit.
This profile excludes prevalence/MNAR analysis and unrelated exploratory figures.
**Inspect the dry run before submitting:**

```bash
CASCADE_PARTITION=long \
bash scripts/submit_everything_hpc.sh \
  --publication paper1 \
  --partition main \
  --cascade-partition long \
  --sites armd,armd_ecuh,armd_utsw \
  --organisms "ESCHERICHIA COLI" \
  --hide-figure-titles \
  --dry-run
```

Remove `--dry-run` to submit. Partition names are cluster-specific. This command
uses `main` and `long` for the study cluster; inspect the printed resource requests
and dependencies before running it elsewhere.

- Add `--skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity --skip-gold`
  only when those primary input layers are complete and compatible.
- Add `--with-eskape` when running the cross-organism extension. It is **not enabled
  by default** and is needed to reproduce the ESKAPE results.
- Use `scripts/submit_pipeline_dag_hpc.sh` for individual-stage control. Its ESKAPE
  flag is `--run-eskape`, not the wrapper's `--with-eskape`.

Use each script's `--help` for supported flags and resource overrides.

## Recovery and Reuse

Completed outputs are reused according to each stage's readiness checks;
manuscript submissions also check validation provenance. Reuse does not establish
that outputs remain scientifically compatible after changing inputs or methods.

1. Read the failed job's `.err` and `.out` files in `logs/`, and confirm its state
   with SLURM accounting. Distinguish `TIMEOUT`, `OUT_OF_MEMORY` and dependency cancellation.
2. Dry-run a targeted recovery and inspect which jobs would actually be submitted.
3. Resubmit only the failed analysis and dependent outputs that need regeneration.

**Do not add `--force-rerun-existing` to routine recovery:** it disables reuse for
enabled stages. `--fresh` rebuilds input/analysis layers and is not a resume option.
Do not edit provenance records to bypass incompatibility checks.

Keep `src/`, `scripts/` and `configs/` unchanged while jobs are queued or running.
ESKAPE completion is tracked per target and scope; an unfinished target restarts
its analysis rather than resuming individual permutation iterations.

## Local Checks

Use a small sample for Mac development, not the full production cohort:

```bash
bash scripts/run_pipeline_mac_dag.sh \
  --env mac \
  --sample-fraction 0.001 \
  --sites armd,armd_ecuh,armd_utsw \
  --organisms "ESCHERICHIA COLI" \
  --skip-prevalence \
  --dry-run
```

Remove `--dry-run` after checking the paths and available resources. Sampling uses
the source data; it is not synthetic-data generation. Sampling itself can still
read large files, and independent site stages can run concurrently. The `mac`
environment shares the `data/` root: do not use it to overwrite production artifacts.
Use a separate checkout or a dedicated test environment when preserving a run.

The current `mac` environment uses 50 permutation/bootstrap replicates; `hpc` uses
1,000. Small-sample runs check execution, not the strength of production findings.

For unit tests:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/unit -q
```

## Outputs and Export

Paths are environment-dependent. With the production defaults:

| Location | Contents |
|---|---|
| `data/gold/` | Episode, observation, eligibility and ordered-pair tables |
| `data/artifacts/cascade/` | Estimates, validation labels and adjusted results |
| `data/artifacts/eskape_validation/` | ESKAPE results by scope, target and organism |
| `data/features/`, `data/artifacts/modeling/` | Prediction inputs and model outputs |
| `outputs/tables/`, `outputs/figures/` | Exported tables and figures |
| `outputs/reports/`, `logs/` | Manifests, diagnostics and execution records |

Typical organism outputs use `combined/organisms/escherichia_coli/` beneath these
roots. Static figures must remain readable without HTML hover text; inspect the
PDF/PNG/SVG exports as well as the interactive figures.

Reporting regenerates outputs from saved inputs; it does not replace missing
upstream analysis or model-evaluation artifacts. Preserve both `data/` artifacts
and `outputs/` for reproducible exports.

After jobs finish, preview an export:

```bash
bash scripts/archive_export_artifacts.sh \
  --include-logs \
  --name amr_cascade_hpc_outputs \
  --dry-run
```

Remove `--dry-run` to create the archive, manifest and checksum in `export_archives/`.
The default includes `data/artifacts`, `data/gold`, `data/features` and `outputs`.
`--include-path` adds to these defaults; it does not restrict the export.
Archiving is non-destructive and does not certify analysis completion.

## Repository Guide

- `src/amr_cascade_platform/`: analysis, modeling, reporting and visualization code.
- `scripts/`: stage runners, orchestration, audits and export tools.
- `configs/`: schemas, defaults and environment overrides.
- `tests/`: automated checks.

The repository does not currently include a formal license or a finalized citation
file. Do not treat the README as a published citation or an open-source license grant.
