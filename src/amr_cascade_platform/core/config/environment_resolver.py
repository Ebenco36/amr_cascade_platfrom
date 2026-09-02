"""Environment resolution helpers."""

from __future__ import annotations

import os


class EnvironmentResolver:
    """Resolve the active runtime environment."""

    def __init__(self, env_var_name: str = "AMR_CASCADE_ENV") -> None:
        self._env_var_name = env_var_name

    def resolve(self, default_environment: str) -> str:
        return os.getenv(self._env_var_name, default_environment)
