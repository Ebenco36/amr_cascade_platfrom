"""Bronze ingestion pipeline."""

from __future__ import annotations

from amr_cascade_platform.data.ingestion.raw_ingestion_manager import (
    RawIngestionManager,
    RawIngestionRequest,
)


class BronzePipeline:
    """Thin pipeline wrapper around raw ingestion."""

    def __init__(self, ingestion_manager: RawIngestionManager) -> None:
        self._ingestion_manager = ingestion_manager

    def run(self, request: RawIngestionRequest):
        return self._ingestion_manager.ingest_site(request)
