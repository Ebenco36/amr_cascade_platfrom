"""Provenance-checked reuse of expensive supplementary sensitivity results."""

from pathlib import Path

import pandas as pd

from amr_cascade_platform.core.utils.provenance import (
    atomic_write_json, atomic_write_parquet, file_fingerprint, read_json,
    sidecar_path, stage_fingerprint,
)


class SensitivityCache:
    def __init__(self, root: Path, settings, inputs: dict[str, Path], **parameters):
        self.fingerprint, self.components = stage_fingerprint(root, settings, inputs, **parameters)

    def ready(self, path: Path, *, shard: tuple[int, int] | None = None) -> bool:
        record = read_json(sidecar_path(path)) or {}
        return (
            path.is_file()
            and record.get("fingerprint") == self.fingerprint
            and record.get("shard") == (list(shard) if shard is not None else None)
            and record.get("output") == file_fingerprint(path)
            and "rows" in record.get("output", {})
        )

    def save(self, frame: pd.DataFrame, path: Path, *, shard: tuple[int, int] | None = None):
        atomic_write_parquet(frame, path)
        atomic_write_json({
            **self.components, "fingerprint": self.fingerprint,
            "shard": shard, "output": file_fingerprint(path),
        }, sidecar_path(path))
