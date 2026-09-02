from pathlib import Path

from amr_cascade_platform.core.config.config_loader import ConfigLoader


def test_config_loader_reads_mac_environment() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    assert settings.app.name == "amr_cascade_platform"
    assert "armd" in settings.platform.sites
    assert settings.environment.name == "mac"
    assert "ESCHERICHIA COLI" in settings.organisms.priority_organisms
