# FILE: a3d/noise_physical.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .graph import RotatedSurfaceLayout


def generate_pauli_errors(
    layout: RotatedSurfaceLayout, rounds: int, p_phys: float, rng: np.random.Generator
) -> Dict[Tuple[Tuple[int, int], int], str]:
    """IID per-qubit-per-round Pauli errors with bias Z>X>Y."""
    if not np.isfinite(p_phys) or not 0.0 <= p_phys <= 1.0:
        raise ValueError("p_phys must be a finite probability in [0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    errors: Dict[Tuple[Tuple[int, int], int], str] = {}
    data_qubits = layout.data_qubits()
    for t in range(rounds):
        for dq in data_qubits:
            if rng.random() < p_phys:
                r = rng.random()
                pauli = "Z" if r < 0.6 else ("X" if r < 0.9 else "Y")
                errors[(dq, t)] = pauli
    return errors


def syndromes_from_pauli_errors(
    layout: RotatedSurfaceLayout,
    rounds: int,
    pauli_errors: Dict[Tuple[Tuple[int, int], int], str],
    p_meas: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[int]]:
    """
    Compute stabilizer syndrome time-series from physical errors with optional measurement noise:
      raw_t(stab) = parity(stab on error@t) XOR mflip_t
      s_t = raw_t XOR raw_{t-1}
    """
    if rng is None:
        rng = np.random.default_rng(0)

    stabs = layout.stabilizer_coords()
    syndromes_X: List[int] = []
    syndromes_Z: List[int] = []

    # For simplicity, a stabilizer at (i,j) is flipped by:
    #  - X-stab: Z or Y error on its four data positions (i,j),(i-1,j),(i,j-1),(i-1,j-1)
    #  - Z-stab: X or Y error on same
    def parity_at(pauli: str, coord: Tuple[int, int], t: int) -> int:
        i, j = coord
        total = 0
        for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            dq = (i + di, j + dj)
            err = pauli_errors.get((dq, t), None)
            if err is None:
                continue
            if pauli == "X":
                total += 1 if err in ("Z", "Y") else 0
            else:
                total += 1 if err in ("X", "Y") else 0
        return total % 2

    # produce raw measurement outcomes with optional flips
    rawX: List[int] = []
    rawZ: List[int] = []
    for t in range(rounds):
        for coord in stabs["X"]:
            pt = parity_at("X", coord, t)
            if p_meas > 0.0 and rng.random() < p_meas:
                pt ^= 1
            rawX.append(pt)
        for coord in stabs["Z"]:
            pt = parity_at("Z", coord, t)
            if p_meas > 0.0 and rng.random() < p_meas:
                pt ^= 1
            rawZ.append(pt)

    # differencing across time
    nX_per_t = len(stabs["X"])
    nZ_per_t = len(stabs["Z"])
    for t in range(rounds):
        for k in range(nX_per_t):
            idx = t * nX_per_t + k
            prev = rawX[idx - nX_per_t] if t > 0 else 0
            syndromes_X.append(rawX[idx] ^ prev)
        for k in range(nZ_per_t):
            idx = t * nZ_per_t + k
            prev = rawZ[idx - nZ_per_t] if t > 0 else 0
            syndromes_Z.append(rawZ[idx] ^ prev)

    return syndromes_X, syndromes_Z


def generate_correlated_pauli_errors(
    layout: RotatedSurfaceLayout,
    rounds: int,
    noise_params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[Tuple[Tuple[int, int], int], str]:
    """
    Correlated model:
      - small grids (≤256 qubits): dense covariance multivariate normal
      - large grids: local Gaussian smoothing (O(n)) at each round
    """
    errors: Dict[Tuple[Tuple[int, int], int], str] = {}
    data_qubits = layout.data_qubits()
    n_qubits = len(data_qubits)
    base_rate = float(noise_params.get("base_error_rate", 1e-3))
    corr_len = float(noise_params.get("correlation_length", 2.0))
    if not np.isfinite(base_rate) or not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_error_rate must be a finite probability")
    if not np.isfinite(corr_len) or corr_len <= 0:
        raise ValueError("correlation_length must be positive and finite")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if n_qubits == 0 or base_rate == 0:
        return errors
    use_dense = n_qubits <= 256

    if use_dense:
        kernel = np.zeros((n_qubits, n_qubits), dtype=np.float64)
        for i, q1 in enumerate(data_qubits):
            for j, q2 in enumerate(data_qubits):
                dist = abs(q1[0] - q2[0]) + abs(q1[1] - q2[1])
                kernel[i, j] = np.exp(-dist / max(corr_len, 0.1))
        C = noise_params.get("crosstalk_matrix", None)
        if C is not None:
            C = np.asarray(C, dtype=np.float64)
            m = min(C.shape[0], n_qubits)
            kernel[:m, :m] += C[:m, :m]
        kernel += np.eye(n_qubits) * 1e-6
        prev = np.zeros(n_qubits, dtype=np.float64)
        for t in range(rounds):
            vec = rng.multivariate_normal(
                mean=np.zeros(n_qubits), cov=kernel
            )
            if t > 0:
                vec += 0.25 * prev * rng.normal(0, 0.1, size=n_qubits)
            prev = vec
            intensity = np.abs(vec)
            mean_intensity = max(float(np.mean(intensity)), 1e-12)
            event_probability = np.clip(
                base_rate * intensity / mean_intensity, 0.0, 1.0
            )
            for i, dq in enumerate(data_qubits):
                if rng.random() < event_probability[i]:
                    r = rng.random()
                    pauli = "Z" if r < 0.6 else ("X" if r < 0.9 else "Y")
                    errors[(dq, t)] = pauli
        return errors

    # Large systems: local smoothing via tiny 2D convolution
    d = layout.d

    def gauss_kernel(sigma=1.0, size=5):
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        k /= k.sum()
        return k

    K = gauss_kernel(sigma=max(0.5, corr_len / 3.0), size=5)

    for t in range(rounds):
        white = rng.normal(0.0, 1.0, size=(d, d))
        pad = 2
        padded = np.pad(white, ((pad, pad), (pad, pad)), mode="reflect")
        smooth = np.zeros_like(white)
        for i in range(d):
            for j in range(d):
                block = padded[i : i + 5, j : j + 5]
                smooth[i, j] = float((block * K).sum())

        # Normalize the spatially correlated intensity before applying the
        # requested base rate. Previously amplitude and variance cancelled,
        # making large-grid event probabilities independent of base_error_rate.
        intensity = np.abs(smooth)
        mean_intensity = max(float(np.mean(intensity)), 1e-12)
        p_local = np.clip(
            base_rate * intensity / mean_intensity, 0.0, 1.0
        )

        for dq in data_qubits:
            i, j = dq
            if rng.random() < p_local[i, j]:
                r = rng.random()
                pauli = "Z" if r < 0.6 else ("X" if r < 0.9 else "Y")
                errors[(dq, t)] = pauli
    return errors
