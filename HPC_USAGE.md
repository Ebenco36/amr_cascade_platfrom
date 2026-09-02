# Running the AMR Cascade Platform on HPC

A quick guide for anyone who needs to run this pipeline on a SLURM-based HPC.
This document is deliberately short. Follow it top to bottom.

For the full command reference and edge-case recipes, see `mini_command.md`.

---

## 1. Before you start

You need:

1. SSH access to an HPC cluster with **SLURM** as the workload manager
2. A user account on that cluster with the ability to submit jobs to a partition that allows multi-day jobs (**at least 7 days**) and high memory (~512 GB)
3. A working **Python 3.12** environment

Run these four checks on the HPC. Each one should succeed:

```bash
which sbatch         # SLURM is installed
which squeue         # ditto
which sacct          # SLURM accounting database is accessible
python3.12 --version # Python 3.12 is available
```

---

## 2. Set up the project (one-time)

```bash
# Clone the repository
git clone https://github.com/Ebenco36/amr_cascade_platfrom.git
cd amr_cascade_platfrom

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the platform and its dependencies
pip install -r requirements-hpc.txt
pip install -e .

# Quick sanity check
python -c "import amr_cascade_platform" && echo "Platform installed OK"
```

If the last line prints `Platform installed OK`, the environment is ready.

---

## 3. Place the input data

The pipeline needs three sites' worth of public ARMD-family clinical microbiology
data. There are two ways to get it.

**Fastest path — bundled zip (all three sites, pre-organised):**

Download from:

> <https://drive.google.com/file/d/1WExT5dRYE4F-OV3SwxuXMrsOeAVz9UNd/view?usp=sharing>

Unzip into the project so that the layout matches what the pipeline expects:

```bash
# From the project root, with the downloaded archive in the same directory
unzip <downloaded-archive>.zip -d data/raw/
```

**Alternative — individual Dryad downloads:**

If you prefer to fetch the canonical source files directly, download each
site's archive from the Dryad DOIs listed in the manuscript's Data Availability
section.

Either way, after extraction the layout must be:

```
data/raw/armd/microbiology_cultures_*.csv
data/raw/armd_ecuh/microbiology_cultures_*.csv
data/raw/armd_utsw/microbiology_cultures_*.csv
```

Verify:

```bash
ls data/raw/armd/microbiology_cultures_cohort.csv \
   data/raw/armd_ecuh/microbiology_cultures_cohort.csv \
   data/raw/armd_utsw/microbiology_cultures_cohort.csv
```

All three should print without errors.

---

## 4. Run the pipeline

**Step 1 — start a persistent terminal session so the submission script survives disconnection:**

```bash
tmux new -s amr_pipeline
```

> If `tmux` is not available, use `screen -S amr_pipeline` instead.
> **Do NOT run the submission script via `sbatch`** — the script itself is
> fast (seconds), but if you sbatch it the HPC will kill it after its partition
> wall-clock limit before all jobs are submitted.

**Step 2 — ensure the code is up to date:**

```bash
git pull
```

**Step 3 — submit:**

```bash
bash scripts/submit_pipeline_dag_hpc.sh \
  --organisms "ESCHERICHIA COLI" \
  --run-features --run-training \
  --force-rerun-existing
```

The script bootstraps the virtualenv automatically, submits every stage as its
own SLURM job with stage-appropriate memory and wall-clock limits, and exits.
Your jobs are now running in SLURM — you can safely close the terminal.

**Expected wall-clock:** up to 7 days, with the cascade stage as the long pole
(B=1000 permutations + B=1000 bootstraps over the full E. coli dataset).

**Detach from tmux:** `Ctrl-B D`

**Re-attach later:** `tmux attach -t amr_pipeline`

**Monitor from anywhere:**

```bash
bash scripts/submit_pipeline_dag_hpc.sh --status logs/pipeline_<timestamp>.jobs
```

The job-ID filename is printed by the submitter at the very start; note it down.

---

## 5. What to verify in the first 30 minutes

Three quick checks confirm the pipeline is healthy. If all three pass,
you can walk away.

| When | Command | Expected |
|---|---|---|
| 30 seconds after submit | `squeue -u $USER` | Lists your submitted jobs (PENDING or RUNNING) |
| 10 minutes after submit | `squeue -u $USER -t RUNNING` | At least one job is RUNNING |
| 30 minutes after submit | `tail -50 logs/amr_armd_ingest_<jobid>.out` | Normal Python log lines, not a traceback |

If any gate fails, see Section 8 below.

---

## 6. How to follow live progress

The `--wait` watcher prints one line per state transition, with timestamps:

```
[2026-05-30 09:15:31] STARTED   amr_armd_ingest           (jobid=12345)
[2026-05-30 11:47:08] FINISHED  amr_armd_ingest           (jobid=12345, elapsed 02:32:45)
[2026-05-30 11:48:11] STARTED   amr_armd_preprocess       (jobid=12348)
...
```

Every event is also written to `logs/pipeline_<timestamp>.events.log` so you
can `tail -f` it from a second SSH window or grep it after the fact:

```bash
# In a second window
tail -f logs/pipeline_*.events.log
```

```bash
# After the run
grep STARTED   logs/pipeline_*.events.log   # stage start times
grep FINISHED  logs/pipeline_*.events.log   # stage completion times
grep FAILED    logs/pipeline_*.events.log   # anything that broke
```

---

## 7. Where the outputs land

When the pipeline finishes successfully, the manuscript artefacts live here:

```
outputs/tables/combined/organisms/escherichia_coli/      # CSV tables
outputs/figures/combined/organisms/escherichia_coli/     # PDF + SVG + PNG + HTML figures
outputs/reports/combined/organisms/escherichia_coli/report_manifest.json
```

Vector figures (PDF, SVG) are publication-ready. PNG is high-resolution raster
for slides. HTML is interactive for review.

Intermediate analysis artefacts (parquet) live under:

```
data/gold/combined/organisms/escherichia_coli/
data/artifacts/cascade/combined/organisms/escherichia_coli/
data/artifacts/prevalence_shift/combined/organisms/escherichia_coli/
data/artifacts/modeling/downstream_testing/combined/organisms/escherichia_coli/
```

---

## 8. If something fails

The DAG submitter has `--kill-on-invalid-dep=yes` on every job, which means
**if one job fails, every downstream job is automatically cancelled**. You will
not waste days on jobs that depend on a broken input.

| Symptom | Likely cause | What to do |
|---|---|---|
| All jobs sit in PENDING for over 10 minutes | Partition is queue-saturated | `squeue -u $USER`; consider passing `--partition=<other>` |
| A specific job fails immediately | Missing data or Python import error | `cat logs/<jobname>_<jobid>.err` to see the traceback |
| Cascade job hits TIMEOUT | 7-day default not enough for your data | Resubmit with `CASCADE_TIME=10-00:00:00 bash scripts/...` |
| All downstream jobs CANCELLED | Upstream failed; this is expected | Fix the upstream root cause, then resubmit without `--force-rerun-existing` to resume from where it died |
| Pipeline finished but figures look wrong | Stale reference data or bad input | Inspect `logs/amr_escherichia_coli_report_<jobid>.out` for warnings |

After fixing the root cause, **drop `--force-rerun-existing` and resubmit**.
The script will skip stages whose outputs already exist and only re-run what
actually needs to be rebuilt:

```bash
bash scripts/submit_pipeline_dag_hpc.sh \
  --organisms "ESCHERICHIA COLI" \
  --run-features --run-training \
  --wait
```

---

## 9. Useful incantations

```bash
# See what would be submitted, without actually submitting (dry run)
bash scripts/submit_pipeline_dag_hpc.sh --organisms "ESCHERICHIA COLI" --dry-run

# Cancel every job from a specific submission
JOBS=$(awk '/^[0-9]+/ {print $1}' logs/pipeline_<TS>.jobs | paste -sd, -)
scancel "${JOBS}"

# After the run, see the full per-stage timing table
JOBS=$(awk '/^[0-9]+/ {print $1}' logs/pipeline_<TS>.jobs | paste -sd, -)
sacct -j "${JOBS}" --format=JobID,JobName%30,Submit,Start,End,Elapsed,State

# Run the full help text
bash scripts/submit_pipeline_dag_hpc.sh --help
```

---

## 10. Why this is safe

The DAG submitter replaced a previous monolithic design where every stage ran
inside a single 1500 GB / 5-day SLURM job. When one stage hung, the whole job
sat idle until the wall-clock limit expired. **The DAG submitter eliminates
that failure mode.**

Concretely, with the DAG submitter:

- Each stage has its own memory and time allocation
- A hung stage is killed by SLURM at its own time limit, not the global one
- A failed stage auto-cancels all downstream jobs via `--kill-on-invalid-dep`
- Parallel stages run concurrently across sites and organisms
- You always have visibility through `squeue`, `sacct`, the events log, and the timeline log

No silent hangs. No five-day waste runs. Everything visible.

---

## Quick reference card

```bash
# Submit (run inside tmux/screen — NOT via sbatch)
tmux new -s amr_pipeline
git pull
bash scripts/submit_pipeline_dag_hpc.sh --organisms "ESCHERICHIA COLI" \
     --run-features --run-training --force-rerun-existing
# Ctrl-B D to detach

# Monitor live
tail -f logs/pipeline_*.events.log

# Check status after dropping the terminal
bash scripts/submit_pipeline_dag_hpc.sh --status logs/pipeline_<TS>.jobs

# Cancel everything
scancel $(awk '/^[0-9]+/{print $1}' logs/pipeline_<TS>.jobs | paste -sd, -)

# Resume after fixing a failure (skips completed stages automatically)
bash scripts/submit_pipeline_dag_hpc.sh --organisms "ESCHERICHIA COLI" \
     --run-features --run-training
```

For all 11 recipes (cold start, partial re-run, ESKAPE generalisation,
per-organism, custom memory / time, re-attach, etc.) see `mini_command.md`.
