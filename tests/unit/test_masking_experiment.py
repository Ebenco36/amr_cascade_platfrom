"""Helpers of scripts/run_masking_experiment.py: synthetic keep probabilities and curve reading."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("masking_experiment", ROOT / "scripts" / "run_masking_experiment.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.mark.parametrize("lam", [-1.0, 0.0, 1.0, 2.5])
def test_keep_probabilities_give_the_requested_odds_tilt(lam):
    q_r, q_s = module.keep_probabilities(lam)
    # Observed-versus-unobserved odds ratio of resistance implied by keeping R with q_r and S with q_s.
    assert math.log(q_r * (1 - q_s) / (q_s * (1 - q_r))) == pytest.approx(lam)


def test_keep_probabilities_recover_the_tilt_by_simulation():
    rng = np.random.default_rng(1)
    y = rng.random(400_000) < 0.2
    q_r, q_s = module.keep_probabilities(1.0)
    kept = rng.random(y.size) < np.where(y, q_r, q_s)
    odds = lambda p: p / (1 - p)
    assert math.log(odds(y[kept].mean()) / odds(y[~kept].mean())) == pytest.approx(1.0, abs=0.03)


def test_curve_reading_and_implied_lambda():
    curve = pd.DataFrame({"mnar_lambda": [-2.0, 0.0, 2.0], "mnar_prevalence": [0.30, 0.20, 0.10]})
    assert module.curve_value(curve, 1.0) == pytest.approx(0.15)
    assert module.implied_lambda(curve, 0.25) == pytest.approx(-1.0)
    assert math.isnan(module.implied_lambda(curve, 0.05))
