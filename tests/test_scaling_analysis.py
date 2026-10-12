from __future__ import annotations

import math

import pytest

from aegis_qec.scaling import fit_surface_code_scaling


def _synthetic_campaign():
    critical_probability = 0.01
    nu = 1.2
    beta_0 = -2.0
    beta_1 = 180.0
    shots = 50000
    rows = []

    for distance in [3, 5, 7]:
        for p in [0.005, 0.008, 0.010, 0.012, 0.015]:
            x = (p - critical_probability) * distance ** (1.0 / nu)
            probability = 1.0 / (1.0 + math.exp(-(beta_0 + beta_1 * x)))
            failures = int(round(shots * probability))
            rows.append(
                {
                    "decoder": "test-decoder",
                    "metadata": {
                        "distance": distance,
                        "physical_error_rate": p,
                    },
                    "shots": shots,
                    "accepted_shots": shots,
                    "errors": failures,
                }
            )
    return {"rows": rows}


def test_finite_size_scaling_recovers_known_crossing():
    result = fit_surface_code_scaling(
        _synthetic_campaign(),
        decoder="test-decoder",
    )

    assert result["analysis_type"] == "first_order_finite_size_scaling"
    assert result["points_used"] == 15
    assert abs(result["critical_probability"] - 0.01) < 0.001
    assert result["critical_probability_ci95_profile"][0] <= 0.01
    assert result["critical_probability_ci95_profile"][1] >= 0.01
    assert result["delta_aic_null_minus_scaling"] > 0
    assert result["beta_1"] > 0


def test_scaling_rejects_too_few_distances():
    campaign = _synthetic_campaign()
    campaign["rows"] = [
        row
        for row in campaign["rows"]
        if row["metadata"]["distance"] in {3, 5}
    ]

    with pytest.raises(ValueError, match="at least three code distances"):
        fit_surface_code_scaling(campaign, decoder="test-decoder")


def test_scaling_rejects_unknown_decoder():
    with pytest.raises(ValueError):
        fit_surface_code_scaling(
            _synthetic_campaign(),
            decoder="missing-decoder",
        )



def test_profile_interval_contains_refined_maximum_likelihood_fit():
    result = fit_surface_code_scaling(
        _synthetic_campaign(),
        decoder="test-decoder",
        pc_grid_points=21,
        nu_grid_points=11,
    )
    lo, hi = result["critical_probability_ci95_profile"]
    assert lo <= result["critical_probability"] <= hi
    assert lo <= 0.01 <= hi
    assert result["nu"] > 0


def test_scaling_refuses_all_successes_without_inventing_threshold():
    campaign = _synthetic_campaign()
    for row in campaign["rows"]:
        row["errors"] = 0
    with pytest.raises(ValueError, match="both logical successes and failures"):
        fit_surface_code_scaling(campaign, decoder="test-decoder")
