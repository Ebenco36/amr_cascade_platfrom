"""Gold build pipeline."""

from __future__ import annotations

from amr_cascade_platform.data.gold.gold_build_manager import GoldBuildManager, GoldBuildRequest


class GoldPipeline:
    """Pipeline wrapper for gold dataset construction."""

    def __init__(self, gold_build_manager: GoldBuildManager) -> None:
        self._gold_build_manager = gold_build_manager

    def run(self, request: GoldBuildRequest):
        return self._gold_build_manager.build(request)
