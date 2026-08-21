from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pypanelsim as pps

pytest.importorskip("pandas")
pytest.importorskip("pyfixest")

sys.path.insert(0, str(Path(__file__).parents[1]))
from examples.baker_twfe import fit_baker_event_studies


def _post_rmse(results) -> float:
    post = results.loc[results["event_time"] >= 0.0]
    return float(np.sqrt(np.mean((post["estimate"] - post["truth"]) ** 2)))


def test_saturated_event_study_recovers_noiseless_baker_truth() -> None:
    config = pps.BakerPanelConfig(n_units=100, noise_scale=0.0)
    panel = pps.baker(config=config, seed=9)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid value encountered in sqrt",
            category=RuntimeWarning,
        )
        estimates = fit_baker_event_studies(panel, config)

    assert _post_rmse(estimates["twfe"]) > 0.2
    assert _post_rmse(estimates["saturated"]) < 1e-10
