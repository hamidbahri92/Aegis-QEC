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



def test_correlated_covariance_rejects_ambiguous_hardware_qubit_mapping():
    layout = RotatedSurfaceLayout(3)
    with pytest.raises(ValueError, match="must match"):
        generate_correlated_pauli_errors(
            layout, 1,
            {"base_error_rate": 0.2, "crosstalk_matrix": np.eye(64)},
            np.random.default_rng(11),
        )


@pytest.mark.parametrize("mode", ("asymmetric", "nan", "not_psd"))
def test_correlated_covariance_refuses_unphysical_matrices(mode):
    layout = RotatedSurfaceLayout(3)
    matrix = np.zeros((9, 9), dtype=float)
    if mode == "asymmetric":
        matrix[0, 1] = 0.3
    elif mode == "nan":
        matrix[0, 0] = float("nan")
    else:
        matrix = -100 * np.eye(9)
    with pytest.raises(ValueError, match="crosstalk_matrix|covariance"):
        generate_correlated_pauli_errors(
            layout, 1,
            {"base_error_rate": 0.2, "crosstalk_matrix": matrix},
            np.random.default_rng(11),
        )


def test_valid_explicit_correlated_covariance_is_seed_reproducible():
    layout = RotatedSurfaceLayout(3)
    params = {
        "base_error_rate": 0.2,
        "crosstalk_matrix": np.zeros((9, 9)),
    }
    a = generate_correlated_pauli_errors(
        layout, 3, params, np.random.default_rng(11)
    )
    b = generate_correlated_pauli_errors(
        layout, 3, params, np.random.default_rng(11)
    )
    assert a == b
