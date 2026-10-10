import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def reuse_check(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "check_validation_reuse_diagnostics", root / "scripts/check_validation_reuse.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = SimpleNamespace(cascade=SimpleNamespace(outputs=SimpleNamespace(result_dir="cascade")))
    paths = SimpleNamespace(paths=SimpleNamespace(artifacts=tmp_path / "artifacts", gold=tmp_path / "gold"))
    directory = paths.paths.artifacts / "cascade/combined/organisms/escherichia_coli"
    directory.mkdir(parents=True)
    components = {
        "git": {"commit": "current", "code_modified": False, "code_diff_sha256": None},
        "config_sha256": "config",
        "inputs": {"drug_pair_episodes": {"path": "pairs.parquet", "size": 42, "mtime_ns": 1, "rows": 3}},
        "parameters": {"stage": "cascade_validation"},
    }
    monkeypatch.setattr(module, "validation_fingerprint", lambda *args: ("expected", components))

    def write(name, record):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # The check reads only sidecars and current input metadata, not result rows.
        path.write_bytes(b"saved-result-not-loaded")
        module.sidecar_path(path).write_text(json.dumps(record))
        return path

    def run(explain=True):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
        errors = module.check(tmp_path, settings, paths, [], ["ESCHERICHIA COLI"], include_sites=False, explain=explain)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
        assert after == before
        return errors

    return module, components, write, run


def test_explain_code_only_mismatch_still_refuses_reuse(reuse_check):
    _, components, write, run = reuse_check
    record = copy.deepcopy(components)
    record.update(fingerprint="old")
    record["git"]["commit"] = "old-commit"
    write("validation_results.parquet", record)
    errors = run()
    assert len(errors) == 1
    assert 'git.commit: recorded="old-commit"; current="current"' in errors[0]
    assert "config_sha256:" not in errors[0]
    assert "inputs.drug_pair_episodes" not in errors[0]
    assert len(run(explain=False)) == 1


def test_explain_does_not_hide_different_shard_mismatches(reuse_check):
    _, components, write, run = reuse_check
    first = copy.deepcopy(components)
    first.update(fingerprint="old")
    first["git"]["commit"] = "old-commit"
    write("validation_shards/shard_0000_of_0002.parquet", first)
    second = copy.deepcopy(components)
    second.update(fingerprint="different")
    second["config_sha256"] = "old-config"
    second["inputs"]["drug_pair_episodes"]["mtime_ns"] = 2
    write("validation_shards/shard_0001_of_0002.parquet", second)
    errors = run()
    assert "2/2" in errors[0]
    assert "git.commit:" in errors[0]
    assert "config_sha256:" in errors[0]
    assert "inputs.drug_pair_episodes.mtime_ns: recorded=2; current=1" in errors[0]
    assert errors[0].count("1 artifact(s), example=") == 2


@pytest.mark.parametrize("payload", [None, "invalid json", "[]"])
def test_explain_missing_or_unreadable_sidecar(reuse_check, payload):
    module, _, write, run = reuse_check
    path = write("validation_results.parquet", {})
    sidecar = module.sidecar_path(path)
    if payload is None:
        sidecar.unlink()
    else:
        sidecar.write_text(payload)
    errors = run()
    assert "recorded commit=unrecorded" in errors[0]
    assert "<not recorded>" in errors[0]


def test_explain_identical_components_wrong_hash_is_not_accepted(reuse_check):
    _, components, write, run = reuse_check
    write("validation_results.parquet", {**components, "fingerprint": "wrong"})
    errors = run()
    assert "Fingerprint differs although the recorded components match" in errors[0]


def test_explain_matching_record_passes(reuse_check):
    _, components, write, run = reuse_check
    write("validation_results.parquet", {**components, "fingerprint": "expected", "rows": 3})
    assert run() == []
