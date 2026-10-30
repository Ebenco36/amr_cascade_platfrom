"""Exercise the resume decision graph without Slurm or production data."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/submit_covariate_fix_hpc.sh"
ORIGINAL = "9077e93d9964253eac5fd13d39fb1dc6f86631c3"
REVIEWED = "d10dba23baa146709445cf66244c6f731c45fc75"


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    script = tmp_path / "scripts/submit_covariate_fix_hpc.sh"
    script.parent.mkdir()
    script.write_text(SCRIPT.read_text())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "covariate_fix_submitted.txt").write_text(f"2026-10-28 16:34:06\n{ORIGINAL}\n")
    executable(tmp_path / ".venv/bin/python", 'exit "${RUNTIME_FAILURE:-0}"\n')
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "git", f'''
case "$1" in
  status) [[ "${{DIRTY:-0}}" == 0 ]] || echo ' M src/example.py'; exit 0 ;;
  rev-parse) echo "${{CURRENT_COMMIT:-{REVIEWED}}}" ;;
  merge-base) exit 0 ;;
  diff)
    [[ "$3" == "{REVIEWED}" && "${{UNREVIEWED_CHANGE:-0}}" == 0 ]] ;;
  *) exit 90 ;;
esac
''')
    executable(bin_dir / "squeue", '[[ "${ACTIVE:-0}" == 0 ]] || echo fix_training; exit 0\n')
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
p = pathlib.Path(os.environ["SUBMISSION_LOG"])
with p.open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
print(9000 + len(p.read_text().splitlines()))
''')
    sbatch.chmod(0o755)
    executable(bin_dir / "sacct", '''
[[ "${ACCOUNTING_FAILURE:-0}" == 0 ]] || exit 1
[[ "${EMPTY_ACCOUNTING:-0}" == 0 ]] || exit 0
for arg in "$@"; do
  case "$arg" in --name=*) names=${arg#--name=} ;; esac
done
for name in ${names//,/ }; do
    [[ "$name" != "${MISSING_STAGE:-}" ]] || continue
    if [[ "${MISSING_DESCENDANTS:-0}" == 1 ]]; then
        case "$name" in fix_pred_figures|fix_temporal|fix_tables|fix_audit) continue ;; esac
    fi
    case "$name" in
      fix_report) value=${REPORT_STATE:-FAILED} ;;
      fix_pred_figures|fix_temporal|fix_tables|fix_audit) value=CANCELLED ;;
      *) value=COMPLETED ;;
    esac
    start=2026-10-29T09:00:00
    if [[ "$name" == fix_training && "${STALE_TRAINING:-0}" == 1 ]]; then
        start=2026-10-28T17:00:00
    fi
    echo "123|$name|$value|$start|2026-10-28T18:00:00|"
done
''')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", USER="test",
               SUBMISSION_LOG=str(tmp_path / "submissions.jsonl"))
    return tmp_path, script, env


def test_reviewed_resume_keeps_completed_stages(sandbox):
    root, script, env = sandbox
    marker = (root / "logs/covariate_fix_submitted.txt").read_bytes()
    result = subprocess.run(
        ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update", "--dry-run"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    submitted = [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]
    assert len(submitted) == 5
    for name in ("fix_report", "fix_pred_figures", "fix_temporal", "fix_tables", "fix_audit"):
        assert any(f"--job-name={name} " in line for line in submitted)
    for name in ("fix_training", "fix_threshold", "fix_cascade", "fix_features", "fix_prev"):
        assert f"kept   {name} (completed)" in result.stderr
    assert (root / "logs/covariate_fix_submitted.txt").read_bytes() == marker
    assert not (root / "logs/covariate_fix_resume_transitions.tsv").exists()
    assert not (root / "submissions.jsonl").exists()


@pytest.mark.parametrize("reason", ["no_opt_in", "new_code", "wrong_original"])
def test_unapproved_transitions_fail_closed(sandbox, reason):
    root, script, env = sandbox
    args = ["bash", str(script), "--resume", "--dry-run"]
    if reason != "no_opt_in":
        args.append("--accept-reviewed-eligibility-update")
    if reason == "new_code":
        env["UNREVIEWED_CHANGE"] = "1"
    if reason == "wrong_original":
        (root / "logs/covariate_fix_submitted.txt").write_text("2026-10-28 16:34:06\nother\n")
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Nothing submitted" in result.stderr
    assert not any(line.startswith("sbatch ") for line in result.stderr.splitlines())


def test_exception_requires_resume(sandbox):
    _, script, env = sandbox
    result = subprocess.run(
        ["bash", str(script), "--accept-reviewed-eligibility-update", "--dry-run"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires --resume" in result.stderr


@pytest.mark.parametrize("condition", ["ACCOUNTING_FAILURE", "EMPTY_ACCOUNTING", "ACTIVE", "RUNTIME_FAILURE", "DIRTY", "MISSING_STAGE"])
def test_preflight_failure_never_submits(sandbox, condition):
    root, script, env = sandbox
    env[condition] = "fix_features" if condition == "MISSING_STAGE" else "1"
    result = subprocess.run(
        ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (root / "submissions.jsonl").exists()


def test_submission_dependencies_and_guard_errors(sandbox):
    root, script, env = sandbox
    marker = (root / "logs/covariate_fix_submitted.txt").read_bytes()
    result = subprocess.run(
        ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    submissions = [json.loads(line) for line in (root / "submissions.jsonl").read_text().splitlines()]
    assert len(submissions) == 5
    assert not any(arg.startswith("--dependency=") for arg in submissions[0])
    for args, dependency in zip(submissions[1:], ["9001", "9001", "9001:9002:9003", "9004"]):
        assert f"--dependency=afterok:{dependency}" in args
    assert (root / "logs/covariate_fix_submitted.txt").read_bytes() == marker
    transition = (root / "logs/covariate_fix_resume_transitions.tsv").read_text()
    assert ORIGINAL in transition and REVIEWED in transition
    wrap = submissions[0][-1]
    subprocess.run(["sh", "-n", "-c", wrap], check=True)
    changed_env = dict(env, CURRENT_COMMIT="changed")
    guard = subprocess.run(["sh", "-c", wrap], env=changed_env, capture_output=True, text=True)
    assert guard.returncode == 1
    assert "checkout commit changed" in guard.stderr
    guard = subprocess.run(["sh", "-c", wrap], env=dict(env, DIRTY="1"), capture_output=True, text=True)
    assert guard.returncode == 1
    assert "uncommitted changes" in guard.stderr


def test_stale_training_reruns_and_propagates(sandbox):
    _, script, env = sandbox
    env["STALE_TRAINING"] = "1"
    result = subprocess.run(
        ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update", "--dry-run"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "re-run fix_training (completed, but started before fix_features had finished)" in result.stderr
    assert "kept   fix_threshold (completed)" in result.stderr
    assert len([line for line in result.stderr.splitlines() if line.startswith("sbatch ")]) == 6


@pytest.mark.parametrize("dry_run", [True, False])
def test_missing_blocked_descendants_after_failed_report(sandbox, dry_run):
    root, script, env = sandbox
    env["MISSING_DESCENDANTS"] = "1"
    args = ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update"]
    if dry_run:
        args.append("--dry-run")
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "kept   fix_training (completed)" in result.stderr
    if dry_run:
        assert len([line for line in result.stderr.splitlines() if line.startswith("sbatch ")]) == 5
        assert not (root / "submissions.jsonl").exists()
    else:
        submissions = [json.loads(line) for line in (root / "submissions.jsonl").read_text().splitlines()]
        assert len(submissions) == 5
        for args, dep in zip(submissions[1:], ["9001", "9001", "9001:9002:9003", "9004"]):
            assert f"--dependency=afterok:{dep}" in args


def test_missing_descendants_without_failed_report_still_stops(sandbox):
    root, script, env = sandbox
    env.update(MISSING_DESCENDANTS="1", REPORT_STATE="COMPLETED")
    result = subprocess.run(
        ["bash", str(script), "--resume", "--accept-reviewed-eligibility-update"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (root / "submissions.jsonl").exists()
