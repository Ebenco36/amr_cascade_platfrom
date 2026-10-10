# Both manuscripts: scoped HPC run

`submit_everything_hpc.sh` now defaults to `--publication manuscripts`. This
means the union of `paper1_detection_validation` and
`paper2_prevalence_surveillance`, not every analysis the platform can perform.
Use `--publication paper1` for Paper 1 only; `--publication all` explicitly
restores the broad platform wave. The lower-level DAG retains its `all` default.

## Coverage contract

| Manuscript support | Producer retained in the wave |
| --- | --- |
| Cohort, specimen/context, exclusions, eligibility, site/era coverage, panel breadth, patient multiplicity | Shared gold layers and report tables |
| Primary ER, permutation/FDR, bootstrap, site/temporal replication and labels | Combined cascade validation shards and merge |
| Independent site analyses, cross-site comparisons and Stanford prediction features | Three site cascades, before feature building and reporting |
| Adjusted associations and confounding-strength summaries | Combined adjusted regression |
| Availability support >=5, annual eras, joint variant | Three separate gold/validation chains, after primary harmonization; no variant adjusted fits or reports |
| Site-by-era permutation, patient-cluster bootstrap, ridge 1/10/100 | Supplementary sensitivity jobs |
| Support/PBI/continuity/heterogeneity sensitivity summaries | Saved-result supplementary table builders; no new primary validation runs |
| LR primary model, RF and XGBoost benchmarks | One full-feature site-split training job |
| Holdout versus patient-grouped OOF threshold comparison | Prespecified clinical-no-cascade LR comparison; not a test-set model-selection contest |
| ROC, PR, calibration, cross-split ROC, threshold and LR coefficient figures | Saved-model supplementary prediction exporter; no reporting-time retraining |
| Paper 2 triggered/non-triggered summaries, kappa, rho, standardisation and SMDs | Shared validated escalation edges and prevalence workflow |
| Model-free bounds and model-based MNAR odds-tilt curves | Prevalence workflow, including explicit non-estimable states |
| Tested-only antibiogram and prevalence/coverage figures | Shared report step |
| All cited CSVs, numerical macros and table bodies | Supplementary builder, then named-asset audit |

The exact figure contract is 17 Paper 1 report figures, 6 prediction figures,
and 10 Paper 2 figures. All have PDF/PNG/SVG; the three Matplotlib MNAR tier
figures have no HTML exporter. The other 30 have HTML. An empty tier has an
explicit empty-state panel, not an invented result. `figure_pipeline_overview.pdf`
is a manual asset checked by the local manuscript packager.

Omitted: ESKAPE expansion, SHAP, broad dataset-characterization/summary jobs,
duplicate robust-only versions of the validated figure suites, separate generic
prediction figures, interactive tipping-point explorer, unreferenced forest and
network variants, and the full-platform readiness audit that expects those
optional assets. This does **not** suppress exceptions or relax scientific gates.
Both-paper asset auditing replaces the broad asset check, not scientific review.

## Before submission

Keep the checkout and `.venv` fixed while jobs are queued/running. Do not delete
outputs or recreate the environment underneath a running wave. The archived HPC
logs include unavailable Python executables, training/threshold OOM kills and
timeouts; limiting the scope does not prove these infrastructure problems solved.
Check both partitions using the runtime commands in
[PAPER1_HPC_RECOVERY.md](PAPER1_HPC_RECOVERY.md). Keep the existing production
memory allocations and review `sacct` after the run; the local replica is not a
full-scale memory benchmark.

The wrapper checks saved validation provenance before submitting. If that check
rejects an earlier checkout, stop and decide on a documented rebuild. Do not
edit sidecars to force reuse. `--fresh` deliberately moves old gold/cascade
outputs aside; it is not the default. Follow-up sensitivity results now require
matching code, configuration, inputs, parameters and an intact output sidecar.
Old unrecorded follow-ups are recomputed once. Matching completed follow-ups are
not queued again when upstream validation is unchanged.

## Review the submission plan

From the HPC project root, with completed ingestion, harmonization and gold:

```bash
CASCADE_PARTITION=long bash scripts/submit_everything_hpc.sh \
  --publication manuscripts --partition main --cascade-partition long \
  --sites armd,armd_ecuh,armd_utsw --organisms "ESCHERICHIA COLI" \
  --skip-ingestion --skip-preprocessing --skip-harmonization \
  --skip-comorbidity --skip-gold --hide-figure-titles --skip-install --dry-run
```

Remove `--dry-run` after runtime/provenance checks pass. Omit the input-layer
skip flags for a genuinely fresh cohort. Omit `--skip-install` when dependencies
need installation. Do not add `--force-rerun-existing` just to refresh figures.
The wrapper still submits required supplementary exporters; existing matching
analyses are reused. Manual `--skip-*` options are recovery controls: skipped
requirements must already exist, or the final build/audit fails.

## Figures and tables only, from completed compatible results

Run these in a reporting compute allocation, in this order (reporting archives
old top-level figures, so prediction export must follow it):

```bash
.venv/bin/python scripts/check_publication_renderer.py
.venv/bin/python scripts/run_reporting.py --env hpc --scope combined \
  --organism "ESCHERICHIA COLI" --publication manuscripts --hide-figure-titles \
  --figure-format html --figure-format png --figure-format svg --figure-format pdf \
  --fail-on-missing-figures
.venv/bin/python scripts/run_supplementary_prediction.py --env hpc \
  --organism "ESCHERICHIA COLI" --hide-figure-titles
.venv/bin/python scripts/build_supplementary_tables.py --env hpc \
  --organism "ESCHERICHIA COLI" --publication manuscripts
.venv/bin/python scripts/audit_paper1_outputs.py --env hpc \
  --organism "ESCHERICHIA COLI" --publication manuscripts
```

The audit filename remains backward-compatible; with this profile it writes
`manuscripts_artifact_audit.json`. Missing data, figures, CSVs or supplementary
inputs fail rather than silently producing a partial publication package.

After transfer, `build_manuscript_package.py --publication manuscripts` scans
both paper folders and checks/copies the actually cited assets. Inspect its
`--dry-run` first. Inspect the production PDFs and resolve each manuscript's
result-dependent `\recheck{...}` statements. A passed asset audit establishes
completeness, not causal validity, clinical relevance or publication acceptance.
