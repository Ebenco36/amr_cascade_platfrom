# Blood-culture analysis (paper 1, Supplementary Note "Blood-culture analysis")

- The analysis script now lives in the code base: `scripts/run_blood_culture_validation.py` (24 Nov 2026). It reproduces this
  folder's run exactly (checked on 4 pairs: identical ER, permutation p, bootstrap stability, replication; identical co-testing
  table; check output in manuscript_edit_log/2026-11-24/blood_script_check/).
- `blood_run/`: the full run of 20 Nov (632 candidates, 188 validated). `blood_numbers.py` builds the paper's numbers and text from it
  and is used by hand_numbers.py. `apply_blood.py`: the edit that wrote the text in. Earlier copies of the script: `retired/`.
- First, descriptive look (before the full run): blood_check.py, analyse.py.

## Adjusted models for the blood-validated pairs (prepared 24 Nov 2026, to run on the HPC)
- `scripts/run_blood_culture_validation.py --adjust --validation-dir <blood_run>` fits the primary analysis's pair-specific
  adjusted models (DownstreamTestingRegression, explain off) to the 188 blood-validated pairs and summarises direction agreement
  with the same function as the primary 344/380 (adjusted_direction_summary). It recomputes screening and refuses to reuse a
  validation whose retained pairs differ, and stops if any site's harmonised covariate tables are missing.
- `scripts/submit_blood_adjustment_hpc.sh`: checks the cluster has the final run's data (gold pair file 115,967,231 bytes /
  111,247,965 rows; harmonised comorbidity rows 118,901,224 / 13,566,752 / 13,794,973 as in harmonization_run.json) and the
  copied blood run, then submits one job (8 CPUs, 128G, 4 h). Output: blood_adjusted/.
- Local tests (24 Nov): on the Mac the script stops at the missing Stanford tables (correct); a code-path test restricted to
  ECU Health and UT Southwestern with the OLD local tables ran in 42 s (results discarded, not used anywhere); the submit
  script's checks refuse the local data (correct).
