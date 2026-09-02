"""Cross-site harmonization pipeline."""

from __future__ import annotations

from amr_cascade_platform.data.harmonization.harmonization_manager import (
    HarmonizationManager,
    HarmonizationRequest,
)


class HarmonizedPipeline:
    """Pipeline wrapper for harmonization."""

    def __init__(self, harmonization_manager: HarmonizationManager) -> None:
        self._harmonization_manager = harmonization_manager

    def run(self, request: HarmonizationRequest):
        return self._harmonization_manager.run(request)
