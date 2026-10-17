"""ROC/PR curves are thinned for drawing only: ends kept, shape kept, AUC untouched."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("supp_prediction_thin", ROOT / "scripts" / "run_supplementary_prediction.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _curve(n=400_000, seed=3):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    score = rng.normal(target * 0.8, 1.0)
    return roc_curve(target, score)[:2]


def test_short_curves_are_left_alone():
    x, y = np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.7, 1.0])
    tx, ty = module._thin_curve(x, y)
    np.testing.assert_array_equal(tx, x)
    np.testing.assert_array_equal(ty, y)


def test_long_curve_keeps_ends_and_existing_vertices_only():
    fpr, tpr = _curve()
    assert len(fpr) > 10 * module._CURVE_DISPLAY_POINTS
    tx, ty = module._thin_curve(fpr, tpr)
    assert len(tx) <= module._CURVE_DISPLAY_POINTS + 2
    assert (tx[0], ty[0], tx[-1], ty[-1]) == (fpr[0], tpr[0], fpr[-1], tpr[-1])
    vertices = set(zip(fpr.tolist(), tpr.tolist()))
    assert all(point in vertices for point in zip(tx.tolist(), ty.tolist()))
    assert np.all(np.diff(tx) >= 0)


def test_drawn_curve_stays_within_a_fraction_of_a_percent_of_the_full_one():
    fpr, tpr = _curve()
    tx, ty = module._thin_curve(fpr, tpr)
    assert abs(np.interp(fpr, tx, ty) - tpr).max() < 0.002
    assert abs(auc(tx, ty) - auc(fpr, tpr)) < 1e-4
