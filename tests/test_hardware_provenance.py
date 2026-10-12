from __future__ import annotations

import pytest

from a3d.hardware_interface import IBMQuantumInterface, QuantumHardwareInterface


def test_explicit_simulator_is_labeled_synthetic():
    result = QuantumHardwareInterface().get_real_noise_parameters()
    assert result["source"] == "synthetic-simulator"
    assert result["synthetic"] is True
    assert result["backend_name"] == "simulator"


def test_ibm_interface_fails_closed_when_backend_is_unavailable(monkeypatch):
    interface = IBMQuantumInterface("missing-backend")
    monkeypatch.setattr(interface, "_init_ibm_connection", lambda: None)

    with pytest.raises(RuntimeError, match="Synthetic data was not substituted"):
        interface.get_real_noise_parameters()


def test_ibm_synthetic_fallback_requires_explicit_opt_in(monkeypatch):
    interface = IBMQuantumInterface(
        "missing-backend",
        allow_synthetic_fallback=True,
    )
    monkeypatch.setattr(interface, "_init_ibm_connection", lambda: None)

    result = interface.get_real_noise_parameters()
    assert result["synthetic"] is True
    assert result["source"] == "synthetic-simulator"
    assert "explicit IBM fallback" in result["provenance_reason"]



def test_ibm_partial_backend_record_exposes_every_unmeasured_field():
    from types import SimpleNamespace

    class Props:
        gates = [
            SimpleNamespace(
                gate="cx",
                parameters=[SimpleNamespace(value=0.008)],
            )
        ]

        def t1(self, qubit):
            return 45e-6 + qubit * 1e-6

        def t2(self, qubit):
            return 28e-6 + qubit * 1e-6

    config = SimpleNamespace(n_qubits=2, coupling_map=[(0, 1)])
    result = IBMQuantumInterface("test-backend")._extract_calibration(
        Props(), config
    )
    assert result["source"] == "ibm-backend"
    assert result["hardware_measurements_present"] is True
    assert result["calibration_quality"] == "mixed_observed_and_modeled"
    assert result["synthetic"] is True
    assert result["field_provenance"]["T1_times"] == "backend_measured"
    assert result["field_provenance"]["crosstalk_matrix"] == "model_estimate_not_measured"
    assert result["field_provenance"]["leakage_rates"] == "unmeasured_placeholder"
    assert result["base_error_rate"] == pytest.approx(0.008)


def test_ibm_refuses_to_fabricate_absent_gate_calibration():
    from types import SimpleNamespace

    class Props:
        gates = []

        def t1(self, index):
            return 50e-6

        def t2(self, index):
            return 25e-6

    interface = IBMQuantumInterface("empty-gates")
    with pytest.raises(RuntimeError, match="no measured two-qubit gate"):
        interface._extract_calibration(
            Props(), SimpleNamespace(n_qubits=1, coupling_map=[])
        )


def test_cached_calibration_reports_staleness():
    interface = IBMQuantumInterface("dummy", cache_duration=10)
    interface._last_fetch_time = 100.0
    interface._last_cal = {"synthetic": False, "source": "ibm-backend"}
    fresh = interface._cached(105.0)
    stale = interface._cached(120.0)
    assert fresh["stale"] is False
    assert stale["stale"] is True
    assert stale["cache_age_seconds"] == 20.0
