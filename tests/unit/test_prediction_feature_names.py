"""Antibiotic one-hot features are labelled with display names, not truncated source labels."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("supp_prediction_names", ROOT / "scripts" / "run_supplementary_prediction.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_truncated_source_labels_get_display_names():
    assert module._clean_feature_name("downstream_antibiotic_MINOCYCLIN") == "Downstream Antibiotic Minocycline"
    assert module._clean_feature_name("upstream_antibiotic_CEFEPIM") == "Upstream Antibiotic Cefepime"
    assert module._clean_feature_name("downstream_antibiotic_CO-TRIMOXAZOL") == "Downstream Antibiotic Sulfamethoxazole/Trimethoprim"


def test_other_features_are_unchanged():
    assert module._clean_feature_name("train_downstream_out_degree") == "Train Downstream Out Degree"
    assert module._clean_feature_name("comorbidity_count") == "Total comorbidity count"
