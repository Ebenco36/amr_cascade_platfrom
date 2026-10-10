"""Run provenance: which code, configuration and inputs produced an output.

Every stage that writes a result records this next to it, so a stale output
can be recognised instead of being reused silently: the validation shards
compare a fingerprint before skipping work, and each finished cascade
directory carries a ``_SUCCESS.json`` written only after all of its files are
in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

SUCCESS_MARKER = "_SUCCESS.json"
_PACKAGES = (
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "xgboost",
    "dask", "plotly", "kaleido", "matplotlib", "networkx", "PyYAML",
)
_CODE_PATHS = ("src", "scripts", "configs")
# What a cascade validation result depends on: the scripts and modules the shard and merge
# jobs execute, and the configuration. This file is left out (it names results, it does not
# compute them); reporting, figures, modelling and the submit scripts are outside it.
_VALIDATION_CODE_PATHS = (
    "configs",
    "scripts/run_cascade_validation_shard.py",
    "scripts/merge_cascade_validation_shards.py",
    "src/amr_cascade_platform/cascade/analyzers",
    "src/amr_cascade_platform/cascade/statistics",
    "src/amr_cascade_platform/cli/main.py",
    "src/amr_cascade_platform/core",
    "src/amr_cascade_platform/data",
    "src/amr_cascade_platform/domain",
    "src/amr_cascade_platform/features",
    "src/amr_cascade_platform/infrastructure",
    ":(exclude)src/amr_cascade_platform/core/utils/provenance.py",
)
# Commits whose validation results stay valid for as long as every path above is unchanged
# since them. Checked with git at run time, never assumed: one differing byte and the
# current checkout's own identity is used instead, so the results are recomputed.
_VALIDATION_CODE_ANCHORS = ("4e9a056ecc671ba12f01a5c12f16ad07d6e37082",)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_hash(value: object) -> str:
    """SHA-256 of a JSON rendering with sorted keys (dataclasses and paths included)."""
    text = json.dumps(_jsonable(value), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state(project_root: Path) -> dict[str, object]:
    """The checked-out commit and, when code or configuration differs from it, a hash of the difference.

    A bare "modified" flag made two different uncommitted versions of the code
    look identical, so a stage could reuse a result that other code produced;
    ``code_diff_sha256`` covers the uncommitted changes to tracked files and the
    contents of new, untracked files under src/, scripts/ and configs/.
    """

    def _git(*args: str) -> subprocess.CompletedProcess:
        # Bytes, not text: a diff can hold content that is not valid in the locale's encoding.
        return subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, timeout=60)

    unavailable = {"commit": "unavailable", "code_modified": None, "code_diff_sha256": None}
    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return unavailable
        diff = _git("diff", "HEAD", "--binary", "--", *_CODE_PATHS)
        untracked = _git("ls-files", "--others", "--exclude-standard", "-z", "--", *_CODE_PATHS)
        if diff.returncode != 0 or untracked.returncode != 0:
            return unavailable
        new_files = sorted(name for name in untracked.stdout.decode("utf-8", "surrogateescape").split("\0") if name)
        digest = hashlib.sha256(diff.stdout)
        for name in new_files:
            digest.update(name.encode("utf-8", "surrogateescape") + b"\0")
            try:
                digest.update((Path(project_root) / name).read_bytes())
            except OSError:
                digest.update(b"<unreadable>")
        modified = bool(diff.stdout.strip() or new_files)
        return {
            "commit": head.stdout.decode().strip(),
            "code_modified": modified,
            "code_diff_sha256": digest.hexdigest() if modified else None,
        }
    except (OSError, subprocess.SubprocessError):
        return unavailable


def validation_code_state(project_root: Path) -> dict[str, object]:
    """The code identity of a cascade validation result.

    The whole-checkout identity (git_state) changes with every commit, also one that only
    touches a figure, which made finished validation unusable. A validation result depends
    only on _VALIDATION_CODE_PATHS: when git shows those identical to an anchor commit --
    no difference in tracked files, no new untracked file -- the anchor's identity is
    returned, so results recorded under it keep matching. Otherwise, or when git or the
    anchor commit is unavailable, the current checkout's identity is returned unchanged.
    """
    current = git_state(project_root)

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, timeout=60)

    for anchor in _VALIDATION_CODE_ANCHORS:
        try:
            if _git("cat-file", "-e", f"{anchor}^{{commit}}").returncode != 0:
                continue
            same = _git("diff", "--quiet", anchor, "--", *_VALIDATION_CODE_PATHS)
            untracked = _git("ls-files", "--others", "--exclude-standard", "--", *_VALIDATION_CODE_PATHS)
            if same.returncode == 0 and untracked.returncode == 0 and not untracked.stdout.strip():
                return {"commit": anchor, "code_modified": False, "code_diff_sha256": None}
        except (OSError, subprocess.SubprocessError):
            continue
    return current


def code_identity(project_root: Path) -> str:
    """One string naming the code a run uses: the commit, plus a short hash of any uncommitted change.

    HPC submission records it for every job, and each job compares its own
    checkout against it at start-up (scripts/check_python_runtime.py).
    """
    state = git_state(project_root)
    identity = str(state["commit"])
    if state.get("code_diff_sha256"):
        identity += f"+dirty.{str(state['code_diff_sha256'])[:16]}"
    return identity


def file_fingerprint(path: Path) -> dict[str, object]:
    """Identity of an input file: size, modification time and, for parquet, row count."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    fingerprint: dict[str, object] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            fingerprint["rows"] = pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001 -- a directory dataset or unreadable footer: size and mtime still identify it
            pass
    return fingerprint


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_provenance(project_root: Path, settings: object, inputs: dict[str, Path] | None = None) -> dict[str, object]:
    """Everything needed to tell whether two outputs came from the same code, configuration and data."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": git_state(project_root),
        "config_sha256": stable_hash(settings),
        "inputs": {name: file_fingerprint(path) for name, path in (inputs or {}).items()},
        "packages": package_versions(),
        "platform": platform.platform(),
        "host": platform.node(),
    }


def stage_fingerprint(project_root: Path, settings: object, inputs: dict[str, Path], **parameters: object) -> tuple[str, dict]:
    """The fingerprint of a stage's result and its components: code, configuration, inputs, parameters."""
    components = {
        "git": git_state(project_root),
        "config_sha256": stable_hash(settings),
        "inputs": {name: file_fingerprint(path) for name, path in inputs.items()},
        "parameters": _jsonable(parameters),
    }
    return fingerprint(components), components


def validation_fingerprint(project_root: Path, settings: object, pair_path: Path) -> tuple[str, dict]:
    """Identity of a cascade validation result: shards, their merge and the cascade job all compare it.

    Its code component is validation_code_state, not the whole checkout; ``executed_under``
    records the checkout that actually ran and is not part of the fingerprint.
    """
    components = {
        "git": validation_code_state(project_root),
        "config_sha256": stable_hash(settings),
        "inputs": {"drug_pair_episodes": file_fingerprint(pair_path)},
        "parameters": _jsonable({"stage": "cascade_validation"}),
    }
    return fingerprint(components), {**components, "executed_under": git_state(project_root)}


def sidecar_path(path: Path) -> Path:
    """Where a result's provenance record lives: ``<name>.provenance.json`` beside it."""
    path = Path(path)
    return path.with_name(f"{path.stem}.provenance.json")


def fingerprint(components: dict[str, object]) -> str:
    """A short, stable identity for a set of provenance components (timestamps excluded)."""
    return stable_hash({key: value for key, value in components.items() if key not in ("generated_at_utc", "executed_under")})[:32]


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet file under a temporary name in the same directory, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(payload: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, default=str)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_success_marker(output_dir: Path) -> None:
    """Withdraw a stage directory's completion marker before the stage rewrites it.

    Called first thing by a stage that writes into an existing directory: if the
    rerun dies partway, no marker from the earlier run is left beside a mix of
    rewritten and stale files.
    """
    (Path(output_dir) / SUCCESS_MARKER).unlink(missing_ok=True)


def write_success_marker(output_dir: Path, outputs: dict[str, Path], provenance: dict[str, object]) -> Path:
    """Declare a stage's output directory complete: every listed file exists and is non-empty.

    Written last, so a directory without it -- a killed job, a partial write --
    is never mistaken for a finished one. Pass only the outputs the stage must
    produce: a listed file that is missing or empty is an error, not skipped.
    """
    missing = sorted(name for name, path in outputs.items() if not Path(path).is_file())
    if missing:
        raise RuntimeError(f"Refusing to mark {output_dir} complete: missing output files {missing}")
    files = {}
    for name, path in outputs.items():
        path = Path(path)
        entry: dict[str, object] = {"path": str(path), "size": path.stat().st_size}
        if path.suffix == ".parquet":
            entry.update({key: value for key, value in file_fingerprint(path).items() if key == "rows"})
        files[name] = entry
    empty = [name for name, entry in files.items() if not entry["size"]]
    if empty:
        raise RuntimeError(f"Refusing to mark {output_dir} complete: empty output files {empty}")
    marker = Path(output_dir) / SUCCESS_MARKER
    atomic_write_json({"completed_at_utc": datetime.now(UTC).isoformat(), "files": files, "provenance": provenance}, marker)
    return marker
