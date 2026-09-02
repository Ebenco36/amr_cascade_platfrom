from pathlib import Path

from amr_cascade_platform.core.utils.scopes import organism_slug, scoped_output_dir


def test_scoped_output_dir_adds_organism_subdirectory() -> None:
    root = Path("/tmp/example")
    path = scoped_output_dir(root=root, scope="combined", organism="ESCHERICHIA COLI")
    assert path == root / "combined" / "organisms" / "escherichia_coli"


def test_organism_slug_normalizes_labels() -> None:
    assert organism_slug(" Staph aureus {MRSA} ") == "staph_aureus_mrsa"
