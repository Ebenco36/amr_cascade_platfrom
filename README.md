# AMR Cascade Platform

*Public repository: `amr_cascade_platfrom`.*

**A reproducible, denominator-aware framework for analysing antimicrobial susceptibility testing (AST) as a selective observation process — not a neutral resistance table.**

The platform implements the full analytical pipeline described in:

> *Selective AST Testing, Diagnostic Cascade Structure, and Bias-Aware Resistance Surveillance.*

It detects and validates directional AST observation patterns, quantifies their surveillance consequences, and reports prevalence sensitivity summaries under explicit denominator and resistance-assumption choices.

---

## Quick Start

If your harmonized layer is already built and you just want to run the manuscript pipeline for *E. coli*:

```bash
tmux new -s amr 'bash scripts/submit_pipeline_dag_hpc.sh \
  --organisms "ESCHERICHIA COLI" \
  --skip-ingestion --skip-preprocessing --skip-harmonization \
  --run-features --run-training \
  --force-rerun-existing --wait'
```

If you are starting from raw data with nothing built yet:

```bash
tmux new -s amr 'bash scripts/submit_pipeline_dag_hpc.sh \
  --organisms "ESCHERICHIA COLI" \
  --run-features --run-training \
  --force-rerun-existing --wait'
```

See **[`mini_command.md`](mini_command.md)** for the full command reference (11 recipes covering cold-start, dry-run, single-stage re-run, ESKAPE generalisation, re-attach to a running submission, cancel, and more).

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Reference Data Bundled with the Repository](#reference-data-bundled-with-the-repository)
- [Pipeline Stages](#pipeline-stages)
- [Per-Script Reference](#per-script-reference)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Routine AST data are not a complete census of clinically meaningful organism–drug opportunities. They arise through selective testing, reporting, and inference. The platform models that observation pathway explicitly, separating:

1. Biologically uninterpretable organism–drug pairs (intrinsic resistance or no clinical breakpoint)
2. Operationally unavailable site–organism–era opportunities
3. Operationally available but unobserved opportunities
4. Directly observed AST results

It then estimates directional observation patterns within the operationally eligible denominator, validates them with permutation, bootstrap, and cross-site replication checks, and reports denominator-aware prevalence sensitivity summaries.

Key capabilities:

- **Denominator construction**: layered biological and operational eligibility filtering
- **Cascade detection**: identifies directional patterns in which the result of one antibiotic test influences whether a second is tested
- **Anti-artefact validation**: permutation tests, bootstrap sign stability, and site replication filters
- **Prevalence-shift estimation**: model-free Manski-style bounds, cascade-trigger diagnostics, observed enrichment summaries, and MNAR odds-tilt sensitivity curves
- **AWaRe classification**: WHO-aligned Access / Watch / Reserve annotation for every validated pattern
- **Predictive modelling**: logistic regression, random forest, and XGBoost baselines for downstream-testing prediction (supplementary)
- **Publication artefacts**: forest plots, chord diagrams, Sankey flows, AWaRe heatmaps, calibration and threshold-analysis figures

---

## Installation

Requires **Python ≥ 3.12**.

```bash
# Clone
git clone https://github.com/Ebenco36/amr_cascade_platfrom.git
cd amr_cascade_platfrom

# Install runtime dependencies
pip install -r requirements.txt
pip install -e .

# Development install (adds pytest, linters)
pip install -r requirements-dev.txt
```

HPC install (uses platform-specific wheel pins):

```bash
pip install -r requirements-hpc.txt
```

Verify installation:

```bash
amr-cascade --help
python -m pytest tests/ -q
```

---

## Data Requirements

### Bundled input dataset (recommended)

The full input dataset for the three published sites is available as a single zip archive:

> <https://drive.google.com/file/d/1WExT5dRYE4F-OV3SwxuXMrsOeAVz9UNd/view?usp=sharing>

After download, unzip into `data/raw/` so the per-site subdirectories appear at `data/raw/armd/`, `data/raw/armd_ecuh/`, and `data/raw/armd_utsw/`. The canonical Dryad DOIs (listed in the manuscript's Data Availability section) are the underlying source; the bundled zip is provided as a convenience.

### Per-site input CSVs (under `data/raw/<site>/`)

Each configured site requires the following files. Site identifiers used in the published analysis are `armd` (Stanford), `armd_ecuh` (ECU Health), and `armd_utsw` (UT Southwestern). Site IDs are configurable in `configs/base/ingestion.yaml`.

| File | Description |
|---|---|
| `microbiology_cultures_cohort.csv` | Culture episode identifiers, timestamps, source-site |
| `microbiology_cultures_microbial_resistance.csv` | Raw AST results (organism, drug, susceptibility, order time) |
| `microbiology_cultures_demographics.csv` | Patient demographics |
| `microbiology_cultures_comorbidity.csv` | Comorbidity indicators (raw long form) |
| `microbiology_cultures_labs.csv` | Laboratory values |
| `microbiology_cultures_vitals.csv` | Vital signs |
| `microbiology_cultures_ward_info.csv` | Ward and admission context |
| `microbiology_cultures_prior_med.csv` | Prior antibiotic exposure |
| `microbiology_cultures_prior_procedures.csv` | Prior procedures |
| `microbiology_cultures_nursing_home_visits.csv` | Nursing home exposure |
| `microbiology_cultures_antibiotic_class_exposure.csv` | Antibiotic class-level exposure |
| `microbiology_cultures_antibiotic_subtype_exposure.csv` | Antibiotic subtype-level exposure |
| `microbiology_cultures_adi_scores.csv` | Area deprivation index |
| `microbiology_cultures_implied_susceptibility.csv` | Laboratory-rule implied susceptibility output |
| `implied_susceptibility_rules.csv` | Laboratory-rule definitions |

### Data layers produced by the pipeline

| Layer | Path | Built by |
|---|---|---|
| **Bronze** | `data/bronze/<site>/` | `run_ingestion.py` |
| **Silver** | `data/silver/<site>/` | `run_preprocessing.py` |
| **Harmonized** | `data/harmonized/site_aligned/<site>/` and `data/harmonized/combined/` | `run_harmonization.py` |
| **Comorbidity features** | `data/interim/<site>/feature_matrices/` | `pre_aggregate_comorbidities.py` |
| **Gold** | `data/gold/combined/organisms/<organism>/` (and per-site) | `run_gold.py` |
| **Cascade artefacts** | `data/artifacts/cascade/...` | `run_cascade_analysis.py` |
| **Prevalence-shift artefacts** | `data/artifacts/prevalence_shift/...` | `run_prevalence_analysis.py` |
| **Features** | `data/features/...` | `run_feature_build.py` |
| **Models** | `data/artifacts/modeling/downstream_testing/...` | `run_training.py` |
| **Reports** | `outputs/reports/...` | `run_reporting.py` |

---

## Reference Data Bundled with the Repository

The repository ships two curated reference tables developed for this work, plus one auxiliary copy. They are documented here so external users can verify provenance.

| File | Purpose | Source provenance |
|---|---|---|
| `data/reference/intrinsic_resistance.csv` | Biological eligibility lookup ($E^{bio}_{oa}$). Encodes which organism–drug pairs are intrinsically resistant or analytically non-interpretable and must be excluded from the eligible denominator. | Grounded in the AMR for R reference-data infrastructure and the EUCAST *Expected Resistant Phenotypes* knowledge base. |
| `data/antibiotic_classification_complete.csv` | Maps platform antibiotic names to WHO AWaRe categories (Access / Watch / Reserve), CLSI display abbreviations, broad antibiotic class (β-lactam, fluoroquinolone, etc.), and structural class. Used by reporting and AWaRe-transition analyses. | Harmonised from WHO *AWaRe classification of antibiotics for evaluation and monitoring of use, 2023*. |
| `data/antibiotic_classification_complete_original.csv` | Pre-harmonisation backup of the AWaRe table. Retained for auditability; not used at runtime. | Internal snapshot. |
| `data/AWaRe Classification 2023.csv` | Raw WHO source export used during harmonisation. | WHO 2023 AWaRe release. |

These references are guideline-versioned; the dependence on them is documented as a limitation in the manuscript.

---

## Pipeline Stages

The pipeline is implemented as a directed acyclic graph (DAG) of SLURM jobs on HPC, or a sequential bash DAG locally on Mac. Each stage has its own memory, time, and parallelism characteristics.

```
                   per-site (parallel)
                  ┌─────────────────┐
ingest ─→ preprocess ─→ harmonization ─→ comorbidity (per site, parallel)
                                          │
                                          ▼
                              gold (per organism, parallel)
                                          │
                                ┌─────────┴─────────┐
                                ▼                   ▼
                          cascade           features ─→ training
                                │                       │
                                ▼                       │
                          prevalence ──────┐            │
                                           ▼            ▼
                                          report (depends on prev + training)
```

A failure in any box auto-cancels every downstream box (via SLURM `--kill-on-invalid-dep=yes`). Parallel boxes at the same depth run concurrently. See `mini_command.md` for per-stage memory and time budgets.

---

## Per-Script Reference

Every script under `scripts/` is documented here so external readers can audit provenance.

### Pipeline runners (used by the DAG)

| Script | Stage | Purpose |
|---|---|---|
| `run_ingestion.py` | 1 | Raw CSV → bronze parquet, per site |
| `run_preprocessing.py` | 2 | Bronze → silver, type coercion, dedup, validation |
| `run_harmonization.py` | 3 | Silver → harmonized; drug name resolution; cross-site alignment |
| `pre_aggregate_comorbidities.py` | 4 | Harmonized → comorbidity feature matrix (top-coverage flags) |
| `run_gold.py` | 5 | Harmonized → gold; episode key construction, eligibility filtering |
| `run_cascade_analysis.py` | 6 | Cascade detection, escalation ratio, permutation, bootstrap, replication |
| `run_prevalence_analysis.py` | 7 | Denominator-aware prevalence bounds, cascade-trigger diagnostics, observed enrichment, and MNAR sensitivity curves |
| `run_feature_build.py` | 8 | Model-ready episode–drug feature matrix |
| `run_training.py` | 9 | LR, RF, XGBoost downstream-testing model fit + cross-site evaluation |
| `run_reporting.py` | 10 | All manuscript tables and figures from analysis-ready artefacts |

### DAG orchestration scripts

| Script | Purpose |
|---|---|
| `scripts/submit_pipeline_dag_hpc.sh` | **Recommended HPC entry point.** Submits every stage as a separate SLURM job with stage-appropriate memory, time, and dependency edges. Supports `--wait`, `--status`, `--dry-run`. |
| `scripts/run_pipeline_mac_dag.sh` | Mac-local equivalent of the DAG submitter. Runs stages sequentially with backgrounded site-level parallelism. |
| `scripts/run_pipeline_gcp_dag.sh` | GCP VM equivalent of the DAG submitter. Runs the full pipeline on a single machine without SLURM. |

ESKAPE-family generalisation runs use `submit_pipeline_dag_hpc.sh --run-eskape` (one independent, resumable SLURM job per site/organism target, replacing the former single-job `run_eskape_hpc.sh` wrapper).

### Diagnostic and auxiliary scripts

| Script | Purpose |
|---|---|
| `run_preflight.py` | Pre-submission environment and data-availability checks. |
| `run_cascade_sensitivity.py` | Support-threshold sensitivity grid for validated patterns. |
| `run_eskape_cascade_validation.py` | Cross-organism ESKAPE validation runner. |
| `run_supplementary_prediction.py` | Generates the supplementary LR / RF / XGBoost comparison figures. |
| `run_scientific_audit.py` | Cross-checks output manifests and detects stale artefacts. |
| `scripts/legacy/generate_projection_audit_figure.py` | Archived helper retained only for reproducing older exploratory figures; not part of the active manuscript pipeline. |
| `run_sampling.py` | Sampling utilities for sub-cohort exploration. |
| `create_sample_dataset.py` | Generates a small synthetic dataset for testing. |
| `create_keynote_presentation.py` | Builds an internal Keynote deck (not part of the manuscript pipeline). |

---

## Running the Pipeline

### Recommended: single-command DAG (HPC)

```bash
bash scripts/submit_pipeline_dag_hpc.sh --wait
```

Default options run the full *E. coli* pipeline assuming raw data is present. See `mini_command.md` for variations (cold start, partial re-run, fire-and-forget, re-attach, ESKAPE, cancellation).

### Mac (local DAG)

```bash
bash scripts/run_pipeline_mac_dag.sh
```

This runs each stage sequentially with per-site parallelism where safe. Use for development and small-data verification. Mac runs use reduced replicate budgets (`permutation_iterations=50`) for speed.

### Advanced: manual per-stage invocation

For debugging or isolated stage re-runs, every pipeline stage can be invoked directly:

```bash
python scripts/run_ingestion.py --env hpc --site armd
python scripts/run_preprocessing.py --env hpc --site armd
python scripts/run_harmonization.py --env hpc
python scripts/run_gold.py --env hpc --source-scope combined --organism "ESCHERICHIA COLI"
python scripts/run_cascade_analysis.py --env hpc --gold-scope combined --organism "ESCHERICHIA COLI"
python scripts/run_prevalence_analysis.py --env hpc --scope combined --organism "ESCHERICHIA COLI"
python scripts/run_feature_build.py --env hpc --scope combined --organism "ESCHERICHIA COLI"
python scripts/run_training.py --env hpc --scope combined --organism "ESCHERICHIA COLI"
python scripts/run_reporting.py --env hpc --scope combined --organism "ESCHERICHIA COLI"
```

Use the DAG submitter unless you have a specific reason to run manually.

---

## Configuration

Configuration is layered. `configs/base/*.yaml` provides defaults; environment files override per target (`configs/environments/mac.yaml`, `configs/environments/hpc.yaml`).

Key validation budgets:

| Setting | Base default | Mac override | HPC override (production) |
|---|---|---|---|
| `permutation_iterations` | 250 | 50 | **1000** |
| `bootstrap_iterations` | 250 | 50 | **1000** |

The published results use the HPC values. Set the environment with `--env hpc` (or `--env mac`) on every script invocation, or rely on the DAG submitter to set it automatically.

Other key cascade settings (`configs/base/cascade.yaml`):

```yaml
cascade:
  min_total_support: 25                 # minimum episode rows per candidate
  min_result_support: 5                 # minimum rows in each upstream branch
  cotesting_probability_threshold: 0.95 # panel-bundling screen
  retained_min_escalation_ratio: 1.5    # primary retention filter
  continuity_correction: 0.5            # Laplace smoothing
  validation:
    minimum_replicated_sites: 2
    minimum_site_direction_agreement_rate: 0.67
    permutation_p_value_threshold: 0.05
    bootstrap_sign_stability_threshold: 0.80
```

---

## Outputs

All artefacts are written under `outputs/`:

```
outputs/
  tables/combined/organisms/escherichia_coli/
    table_a_data_quality_flow.csv          # Row-flow audit (Supp Table S2)
    table_b_eligibility.csv                # Eligibility denominator summary
    table_c_primary_cascade.csv            # Full cascade results
    table_l_validated_primary_cascade.csv  # Validated patterns only
    table_m_aware_transition_summary.csv   # AWaRe transitions
    table_k_prevalence_shift.csv           # Prevalence-shift summary
    table_supplementary_antibiotic_abbreviations.csv

  figures/combined/organisms/escherichia_coli/
    figure_primary_cascade_forest.png
    figure_validated_primary_cascade_forest.png
    figure_aware_transition_heatmap.png
    figure_prevalence_shift_forest.png
    figure_prevalence_shift_curves.png
    figure_supp_projection_audit.png
    supp_pred_fig1_roc_curves.png
    supp_pred_fig2_pr_curves.png
    supp_pred_fig3_calibration.png
    supp_pred_fig4_cross_split_roc.png
    supp_pred_fig5_threshold_analysis.png
    supp_pred_fig6_lr_coefficients.png

  reports/combined/organisms/escherichia_coli/
    report_manifest.json
```

Figures are produced as PNG by default. Pass `--figure-format pdf --figure-format svg` to the reporting and prevalence scripts (or run the DAG, which already does this) for publication-grade outputs.

---

## Tests

```bash
python -m pytest tests/ -q                                   # all tests
python -m pytest tests/ -v                                   # verbose
python -m pytest tests/unit/test_cascade_validator.py -v     # one file
```

The current suite contains 85 unit tests covering cascade detection and validation, AWaRe resolution, gold dataset construction, prevalence-shift bounds, feature engineering, model training, reporting builders, and figure generation.

---

## Troubleshooting

Common issues, in decreasing order of frequency:

| Symptom | Likely cause | Fix |
|---|---|---|
| A SLURM job sits in `PENDING` forever | Partition busy | `squeue -u $USER`; try a different `--partition`. |
| All downstream jobs are `CANCELLED` | An upstream job FAILED and `--kill-on-invalid-dep=yes` auto-cancelled the chain | Read the upstream `logs/<jobname>_<jobid>.err`, fix the root cause, resubmit with `--force-rerun-existing`. |
| Cascade job hits `TIMEOUT` | 2 days insufficient for permutation budget | `CASCADE_TIME=3-00:00:00 bash scripts/submit_pipeline_dag_hpc.sh ... --force-rerun-existing --wait` |
| `--wait` session dies (SSH drop, accidental Ctrl-C) | Terminal lost; jobs still running | `bash scripts/submit_pipeline_dag_hpc.sh --status logs/pipeline_<TS>.jobs` |
| `sacct` shows partial states immediately after job ends | Accounting database lag | Wait 1–5 minutes, re-run `--status`. |
| Ingestion fails with "no such file" | Raw CSVs not present | Confirm `data/raw/<site>/microbiology_cultures_*.csv` exists for each configured site. |
| Cascade adjusted ORs come back astronomical | Ridge penalty too small (legacy bug, fixed) | Verify `_RIDGE_PENALTY = 10.0` in `downstream_testing_regression.py`. |

For the full recovery and re-attach playbook, see `mini_command.md` § *When things go wrong*.

---

## Project Structure

```
amr_cascade_platfrom/
├── configs/
│   ├── base/                # Default configuration for every pipeline stage
│   └── environments/        # mac.yaml, hpc.yaml overrides
├── data/
│   ├── raw/                 # Site CSVs (input)
│   ├── bronze/              # Ingested parquet
│   ├── silver/              # Cleaned parquet
│   ├── harmonized/          # Cross-site aligned parquet
│   ├── interim/             # Per-site comorbidity matrices
│   ├── gold/                # Analytical gold layer (per organism)
│   ├── features/            # Model-ready feature matrices
│   ├── artifacts/           # Cascade, prevalence, modeling outputs
│   ├── reference/           # Bundled reference tables (intrinsic resistance)
│   └── antibiotic_classification_complete.csv  # AWaRe lookup
├── outputs/
│   ├── tables/              # Manuscript tables (CSV)
│   ├── figures/             # Manuscript figures (PNG, PDF, SVG)
│   └── reports/             # Pipeline manifests
├── papers/                  # LaTeX manuscript and supplementary
├── scripts/                 # Pipeline runners and DAG submitters
├── src/amr_cascade_platform/
│   ├── cli/                 # CLI entry points
│   ├── core/                # Config, paths, statistics, logging
│   ├── data/                # Ingestion, preprocessing, harmonization, transformations
│   ├── domain/              # Domain services (eligibility, episode, AWaRe)
│   ├── cascade/             # Cascade detection, statistics, validation, outputs
│   ├── surveillance/        # Prevalence-shift analyser
│   ├── features/            # Feature engineering
│   ├── modeling/            # Estimators (LR, RF, XGBoost), workflows
│   ├── reporting/           # Table builders and reporting workflows
│   └── visualization/       # Plotting and report visuals
├── tests/                   # Unit tests (85 passing)
├── mini_command.md          # Pipeline command reference
└── README.md
```

---

## Citation

If you use this platform in your research, please cite:

> Awotoro E, et al. *Selective AST Testing, Diagnostic Cascade Structure, and Bias-Aware Resistance Surveillance.* [Journal], [Year].

---

## License

License terms will be added prior to public release. The repository currently has no formal licence; reuse, redistribution, and modification are not yet permitted without explicit permission.

Recommended target: MIT or Apache-2.0 for code; CC-BY-4.0 for documentation and figures.
