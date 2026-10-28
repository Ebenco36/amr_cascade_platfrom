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

## Reproducing the Two Papers on HPC

The default profile (`--publication manuscripts`) produces what both *E. coli* papers cite.
**Inspect the dry run before submitting:**

```bash
CASCADE_PARTITION=long \
bash scripts/submit_everything_hpc.sh \
  --partition main \
  --cascade-partition long \
  --sites armd,armd_ecuh,armd_utsw \
  --organisms "ESCHERICHIA COLI" \
  --hide-figure-titles \
  --dry-run
```

Remove `--dry-run` to submit. Partition names are cluster-specific; inspect the printed
resource requests and dependencies before running elsewhere. The script queues the primary
run, the three availability-sensitivity runs and the follow-ups each paper needs: site-by-era
permutation, patient-cluster bootstrap, ridge-penalty refit, supplementary prediction, the
masking experiment (paper 2), the year-by-year course of the validated patterns (paper 1),
the supplementary tables with every quoted number, and a final audit of the named outputs.

- Add `--skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity --skip-gold`
  only when those primary input layers are complete and compatible.
- The ESKAPE extension (paper 1) runs after the primary run, through the stage-level submitter:

  ```bash
  bash scripts/submit_pipeline_dag_hpc.sh --run-eskape \
    --skip-ingestion --skip-preprocessing --skip-harmonization --skip-comorbidity --skip-gold \
    --skip-cascade --skip-prevalence --skip-reporting --skip-dataset-summary --skip-install --dry-run
  ```

  Do not use `submit_everything_hpc.sh --with-eskape` for this: it requires `--publication all`,
  which moves the finished gold and cascade outputs aside and rebuilds them.
- `scripts/run_pipeline_gcp_dag.sh` runs the core pipeline on one machine without SLURM.

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
bash scripts/archive_export_artifacts.sh --dry-run
```

Remove `--dry-run` to create the archive, manifest and checksum in `export_archives/`.
The script first checks that the run is finished and current (no jobs running, all ESKAPE
targets finished, masking on the full data, the manuscripts audit passed and, after the
28 October 2026 covariate re-run, its outputs newer than its submission) and refuses to export
otherwise (`--allow-incomplete` overrides). The default
includes `data/artifacts`, `data/gold`, `data/features`, `data/metadata`, `data/reference`,
`outputs`, `logs` and the three availability-sensitivity runs' artifacts; it writes a run record
into `logs/` and verifies the archive's checksum and key files. `--include-path` adds to these
defaults. Archiving is non-destructive.

The paper folders are not in this repository. Where they are, extract the archive into a
git-ignored folder and copy every figure and table the manuscripts cite into place with
`.venv/bin/python scripts/build_manuscript_package.py --env hpc --publication manuscripts --run-root <extracted folder>`
(add `--dry-run` first).

## Repository Guide

- `src/amr_cascade_platform/`: analysis, modeling, reporting and visualization code.
- `scripts/`: stage runners, orchestration, audits and export tools.
- `configs/`: schemas, defaults and environment overrides.
- `tests/`: automated checks.

## License

Released under the MIT License (see `LICENSE`). The source datasets are not part of this
repository and keep their own terms. A citation file will accompany the archived release.
