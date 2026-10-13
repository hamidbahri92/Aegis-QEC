"""Scientific rate controls for the synthetic correlated noise model."""
from __future__ import annotations

import numpy as np
import pytest

from a3d.graph import RotatedSurfaceLayout
from a3d.noise_physical import (
    generate_correlated_pauli_errors,
    generate_pauli_errors,
    syndromes_from_pauli_errors,
)


@pytest.mark.parametrize(("distance", "rounds"), [(3, 120), (17, 6)])
def test_correlated_error_probability_respects_requested_rate(distance, rounds):
    layout = RotatedSurfaceLayout(distance)
    total_trials = distance * distance * rounds

    def sample(rate, seed=20261013):
        return generate_correlated_pauli_errors(
            layout,
            rounds,
            {"base_error_rate": rate, "correlation_length": 2.0},
            np.random.default_rng(seed),
        )

    assert sample(0.0) == {}
    low = sample(0.005)
    high = sample(0.18)
    assert sample(0.18) == high
    assert len(high) > max(len(low) * 3, int(total_trials * 0.05))
    assert len(high) < int(total_trials * 0.4)
    assert all(value in {"X", "Y", "Z"} for value in high.values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_error_rate", -0.1),
        ("base_error_rate", 1.5),
        ("base_error_rate", float("nan")),
        ("correlation_length", 0.0),
        ("correlation_length", float("inf")),
    ],
)
def test_correlated_model_rejects_invalid_scientific_parameters(field, value):
    with pytest.raises(ValueError):
        generate_correlated_pauli_errors(
            RotatedSurfaceLayout(3), 3,
            {field: value}, np.random.default_rng(2),
        )


def test_iid_pauli_and_measurement_noise_probabilities_are_validated():
    layout = RotatedSurfaceLayout(3)
    with pytest.raises(ValueError, match="p_phys"):
        generate_pauli_errors(
            layout, 3, 1.1, np.random.default_rng(1)
        )
    with pytest.raises(ValueError, match="p_meas"):
        syndromes_from_pauli_errors(layout, 3, {}, p_meas=float("nan"))
    assert generate_pauli_errors(
        layout, 3, 0.0, np.random.default_rng(1)
    ) == {}
    assert len(generate_pauli_errors(
        layout, 3, 1.0, np.random.default_rng(1)
    )) == 27
