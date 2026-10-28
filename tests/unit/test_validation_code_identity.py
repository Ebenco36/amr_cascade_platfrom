"""A validation result's code identity follows the validation code, not the whole checkout."""
import subprocess
from pathlib import Path

import pytest

from amr_cascade_platform.core.utils import provenance


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A small checkout with one validation module, one figure module and an anchor commit."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.org")
    _git(tmp_path, "config", "user.name", "Test")
    analyzer = tmp_path / "src/amr_cascade_platform/cascade/analyzers/validation.py"
    figure = tmp_path / "src/amr_cascade_platform/visualization/figure.py"
    own = tmp_path / "src/amr_cascade_platform/core/utils/provenance.py"
    for path, text in ((analyzer, "ITERATIONS = 1000\n"), (figure, "TITLE = 'a'\n"), (own, "x = 1\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/base.yaml").write_text("seed: 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "anchor")
    anchor = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(provenance, "_VALIDATION_CODE_ANCHORS", (anchor,))
    return tmp_path, anchor, analyzer, figure, own


def _anchor_state(anchor: str) -> dict:
    return {"commit": anchor, "code_modified": False, "code_diff_sha256": None}


def test_anchor_commit_itself_keeps_its_identity(repo):
    root, anchor, *_ = repo
    assert provenance.validation_code_state(root) == _anchor_state(anchor) == provenance.git_state(root)


def test_a_later_commit_outside_the_validation_code_keeps_the_anchor_identity(repo):
    root, anchor, _, figure, own = repo
    figure.write_text("TITLE = ''\n")
    own.write_text("x = 2\n")
    _git(root, "commit", "-q", "-am", "figures only")
    assert provenance.git_state(root)["commit"] != anchor
    assert provenance.validation_code_state(root) == _anchor_state(anchor)
    figure.write_text("TITLE = 'uncommitted'\n")
    assert provenance.validation_code_state(root) == _anchor_state(anchor)


@pytest.mark.parametrize("commit", [True, False])
def test_any_change_to_validation_code_or_configuration_ends_the_match(repo, commit):
    root, anchor, analyzer, *_ = repo
    analyzer.write_text("ITERATIONS = 10\n")
    if commit:
        _git(root, "commit", "-q", "-am", "validation changed")
    assert provenance.validation_code_state(root) == provenance.git_state(root) != _anchor_state(anchor)


def test_configuration_change_and_new_untracked_validation_file_end_the_match(repo):
    root, anchor, analyzer, *_ = repo
    (root / "configs/base.yaml").write_text("seed: 2\n")
    assert provenance.validation_code_state(root) != _anchor_state(anchor)
    _git(root, "checkout", "-q", "--", "configs/base.yaml")
    assert provenance.validation_code_state(root) == _anchor_state(anchor)
    (analyzer.parent / "new_rule.py").write_text("x = 1\n")
    assert provenance.validation_code_state(root) != _anchor_state(anchor)


def test_unknown_anchor_commit_falls_back_to_the_checkout_identity(repo, monkeypatch):
    root, *_ = repo
    monkeypatch.setattr(provenance, "_VALIDATION_CODE_ANCHORS", ("0" * 40,))
    assert provenance.validation_code_state(root) == provenance.git_state(root)


def test_fingerprint_equals_the_one_the_anchor_commit_recorded(repo):
    """Same four components as before the change, so old shard records still match."""
    root, anchor, _, figure, _ = repo
    pairs = root / "pairs.parquet"
    pairs.write_bytes(b"not parquet, identified by size and time")
    settings = {"seed": 1}
    recorded = provenance.fingerprint({
        "git": _anchor_state(anchor), "config_sha256": provenance.stable_hash(settings),
        "inputs": {"drug_pair_episodes": provenance.file_fingerprint(pairs)},
        "parameters": {"stage": "cascade_validation"},
    })
    figure.write_text("TITLE = ''\n")
    _git(root, "commit", "-q", "-am", "figures only")
    current, components = provenance.validation_fingerprint(root, settings, pairs)
    assert current == recorded
    assert components["executed_under"]["commit"] != anchor
    assert provenance.fingerprint({k: v for k, v in components.items() if k != "executed_under"}) == recorded


def test_gold_eligibility_builder_change_keeps_validation_code_identity(repo):
    root, anchor, *_ = repo
    eligibility = root / "src/amr_cascade_platform/domain/services/eligibility_service.py"
    eligibility.parent.mkdir(parents=True, exist_ok=True)
    eligibility.write_text("BUILDER_VERSION = 1\n")
    _git(root, "add", str(eligibility.relative_to(root)))
    _git(root, "commit", "-q", "-m", "upstream eligibility builder only")

    assert provenance.git_state(root)["commit"] != anchor
    assert provenance.validation_code_state(root) == _anchor_state(anchor)

    eligibility.write_text("BUILDER_VERSION = 2\n")
    assert provenance.validation_code_state(root) == _anchor_state(anchor)
