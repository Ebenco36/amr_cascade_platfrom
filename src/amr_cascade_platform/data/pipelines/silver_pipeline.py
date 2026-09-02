"""Silver cleaning pipeline."""

from __future__ import annotations

from amr_cascade_platform.data.cleaning.site_cleaning_manager import (
    SilverCleaningRequest,
    SiteCleaningManager,
)


class SilverPipeline:
    """Pipeline wrapper for bronze-to-silver cleaning."""

    def __init__(self, cleaning_manager: SiteCleaningManager) -> None:
        self._cleaning_manager = cleaning_manager

    def run(self, request: SilverCleaningRequest):
        return self._cleaning_manager.clean_site(request)
