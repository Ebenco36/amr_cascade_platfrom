from pathlib import Path

from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.data.schemas.processed_schema_registry import ProcessedSchemaRegistry


def test_required_columns_include_default_and_specific_fields() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    registry = ProcessedSchemaRegistry(settings)
    columns = registry.required_columns_for("cohort")
    assert "anon_id" in columns
    assert "source_site" in columns
    assert "organism" in columns
