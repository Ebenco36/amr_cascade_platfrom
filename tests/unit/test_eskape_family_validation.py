from pathlib import Path

import pandas as pd

from amr_cascade_platform.cascade.workflows.eskape_family_validation import ArmdEskapeCascadeRunner, ArmdEskapeRequest
from amr_cascade_platform.core.config.config_loader import ConfigLoader
from amr_cascade_platform.core.paths.path_manager import PathManager
from amr_cascade_platform.core.utils.eskape import get_eskape_target
from amr_cascade_platform.core.utils.organism_names import normalize_organism_label


def test_a_family_target_is_analysed_one_canonical_organism_at_a_time(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    settings = ConfigLoader(project_root).load("mac")
    runner = ArmdEskapeCascadeRunner(settings, PathManager(project_root, settings))
    labels = ["ENTEROBACTER CLOACAE", "ENTEROBACTER CLOACAE COMPLEX", "ENTEROBACTER AEROGENES", "ENTEROBACTER SPECIES", "ESCHERICHIA COLI"]
    cohort = pd.DataFrame({"organism": labels, "antibiotic": "CEFEPIME", "susceptibility": "SUSCEPTIBLE"})
    monkeypatch.setattr(runner, "_load_cohort", lambda request: cohort)
    calls = []

    def fake_run(request, subset, organism, organism_dir):
        calls.append((organism, set(subset["organism"])))
        return {}, pd.DataFrame([{"eskape_target": request.target.display_name, "organism": organism}])

    monkeypatch.setattr(runner, "_run_organism", fake_run)
    outputs = runner.run(ArmdEskapeRequest(target=get_eskape_target("enterobacter"), source_scope="combined", output_root=tmp_path))

    expected = sorted({normalize_organism_label(label) for label in labels if label.startswith("ENTEROBACTER")})
    assert [organism for organism, _ in calls] == expected
    assert all(values == {organism} for organism, values in calls)  # one organism label per analysis, never the family name
    summary = pd.read_csv(outputs["existence_summary_csv"])
    assert sorted(summary["organism"]) == expected
