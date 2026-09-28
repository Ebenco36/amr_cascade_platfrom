import os
import subprocess

import pandas as pd
import pytest

from amr_cascade_platform.core.utils import provenance


def test_validation_fingerprint_is_stable_and_tracks_inputs_and_configuration(tmp_path):
    pairs = tmp_path / "drug_pair_episodes.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(pairs)
    first, components = provenance.validation_fingerprint(tmp_path, {"iterations": 1000, "seed": 7}, pairs)
    assert provenance.validation_fingerprint(tmp_path, {"iterations": 1000, "seed": 7}, pairs)[0] == first
    assert components["inputs"]["drug_pair_episodes"]["rows"] == 2
    assert provenance.validation_fingerprint(tmp_path, {"iterations": 50, "seed": 7}, pairs)[0] != first
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pairs)  # a rebuilt gold table
    assert provenance.validation_fingerprint(tmp_path, {"iterations": 1000, "seed": 7}, pairs)[0] != first


def test_atomic_parquet_write_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "result.parquet"

    def broken(self, *_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    with pytest.raises(OSError):
        provenance.atomic_write_parquet(pd.DataFrame({"a": [1]}), target)
    assert os.listdir(tmp_path) == []


def test_success_marker_lists_the_files_and_refuses_empty_ones(tmp_path):
    good = tmp_path / "edge_report.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(good)
    marker = provenance.write_success_marker(tmp_path, {"edge_report": good}, {"git": {"commit": "x"}})
    payload = provenance.read_json(marker)
    assert payload["files"]["edge_report"]["rows"] == 1
    (tmp_path / "empty.json").write_text("")
    with pytest.raises(RuntimeError):
        provenance.write_success_marker(tmp_path, {"empty": tmp_path / "empty.json"}, {})


def test_clearing_a_marker_removes_it_and_tolerates_its_absence(tmp_path):
    marker = tmp_path / provenance.SUCCESS_MARKER
    marker.write_text("{}")
    provenance.clear_success_marker(tmp_path)
    assert not marker.exists()
    provenance.clear_success_marker(tmp_path)  # nothing left to remove: no error


def test_success_marker_refuses_a_missing_listed_output(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        provenance.write_success_marker(tmp_path, {"edge_report": tmp_path / "absent.parquet"}, {})
    assert not (tmp_path / provenance.SUCCESS_MARKER).exists()


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_git_state_tells_different_uncommitted_versions_apart(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    code = tmp_path / "src" / "module.py"
    code.write_text("x = 1\n")
    _git(tmp_path, "add", "src/module.py")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "initial")

    clean = provenance.git_state(tmp_path)
    assert clean["code_modified"] is False and clean["code_diff_sha256"] is None
    assert provenance.code_identity(tmp_path) == clean["commit"]

    code.write_text("x = 2\n")
    first = provenance.git_state(tmp_path)
    code.write_text("x = 3\n")
    second = provenance.git_state(tmp_path)
    # A bare "modified" flag made these two versions look identical.
    assert first["code_modified"] and second["code_modified"]
    assert first["code_diff_sha256"] != second["code_diff_sha256"]
    assert provenance.code_identity(tmp_path).startswith(clean["commit"] + "+dirty.")

    code.write_text("x = 1\n")
    (tmp_path / "src" / "new_module.py").write_text("y = 1\n")
    # New, untracked code counts as a change too (the old check ignored it).
    assert provenance.git_state(tmp_path)["code_modified"] is True
