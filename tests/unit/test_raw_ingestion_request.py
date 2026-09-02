from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.ingestion.raw_ingestion_manager import RawIngestionRequest
from pathlib import Path


def test_mac_environment_defaults_to_sample_layer() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    request = RawIngestionRequest(site="armd", source_layer=settings.environment.default_data_layer)
    assert request.source_layer == "sample"
