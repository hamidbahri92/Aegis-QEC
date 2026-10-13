# FILE: a3d/hardware_interface.py
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


class QuantumHardwareInterface:
    def __init__(self, backend_name: str = "simulator"):
        self.backend_name = backend_name

    def _simulate_realistic_noise(
        self,
        *,
        reason: str = "explicit simulator",
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(123)
        return {
            "T1_times": rng.exponential(50e-6, size=64),
            "T2_times": rng.exponential(25e-6, size=64),
            "gate_errors": rng.beta(2, 1000, size=64) * 0.01,
            "crosstalk_matrix": rng.normal(0, 0.001, (64, 64)),
            "leakage_rates": rng.beta(1, 1000, size=64) * 0.001,
            "base_error_rate": 1e-3,
            "timestamp": time.time(),
            "source": "synthetic-simulator",
            "synthetic": True,
            "backend_name": self.backend_name,
            "provenance_reason": reason,
        }

    def get_real_noise_parameters(self) -> Dict[str, Any]:
        return self._simulate_realistic_noise()


class IBMQuantumInterface(QuantumHardwareInterface):
    """Fetch IBM calibration data without silently inventing hardware evidence.

    Real-backend mode fails closed by default. Synthetic fallback is available
    only when explicitly enabled by the caller.
    """

    def __init__(
        self,
        backend_name: str,
        token: str | None = None,
        cache_duration: float = 300.0,
        timeout_s: float = 30.0,
        allow_synthetic_fallback: bool = False,
    ):
        super().__init__(backend_name)
        self.token = token
        self.cache_duration = float(cache_duration)
        self.timeout_s = float(timeout_s)
        self.allow_synthetic_fallback = bool(allow_synthetic_fallback)
        self._last_cal: Dict[str, Any] = {}
        self._last_fetch_time = 0.0
        self._api_call_count = 0
        self._window_start_time = time.time()
        self._max_api_calls_per_hour = 100
        self._lock = threading.Lock()
        self.backend = None

    def _fallback_or_raise(self, reason: str) -> Dict[str, Any]:
        if self.allow_synthetic_fallback:
            return self._simulate_realistic_noise(
                reason=f"explicit IBM fallback: {reason}"
            )
        raise RuntimeError(
            f"IBM calibration unavailable for {self.backend_name!r}: {reason}. "
            "Synthetic data was not substituted. Set "
            "allow_synthetic_fallback=True only when a simulated fallback is "
            "explicitly acceptable."
        )

    def _cached(self, now: float) -> Dict[str, Any]:
        result = dict(self._last_cal)
        result["cached"] = True
        result["cache_age_seconds"] = float(now - self._last_fetch_time)
        result["stale"] = bool(
            now - self._last_fetch_time >= self.cache_duration
        )
        return result

    def _init_ibm_connection(self) -> None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.token,
            )
            self.backend = service.backend(self.backend_name)
        except Exception:
            self.backend = None

    def _extract_calibration(self, props, config) -> Dict[str, Any]:
        try:
            n = int(config.n_qubits)
            t1 = np.array([props.t1(q) for q in range(n)], dtype=np.float64)
            t2 = np.array([props.t2(q) for q in range(n)], dtype=np.float64)
            gate_errors = []
            for gate in props.gates:
                if gate.gate not in ("cx", "cz"):
                    continue
                try:
                    # IBM's documented accessor is order-independent. The
                    # first gate parameter might instead be gate_length.
                    accessor = getattr(props, "gate_error", None)
                    if callable(accessor) and hasattr(gate, "qubits"):
                        value = float(accessor(gate.gate, gate.qubits))
                    else:
                        error_property = next(
                            item for item in gate.parameters
                            if getattr(item, "name", None) == "gate_error"
                        )
                        value = float(error_property.value)
                    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError("invalid backend gate error probability")
                    gate_errors.append(value)
                except (AttributeError, StopIteration, TypeError, ValueError):
                    continue
            if not gate_errors:
                raise ValueError(
                    "backend supplied no measured two-qubit gate error parameters"
                )
            gate_errors_array = np.array(gate_errors, dtype=np.float64)
            crosstalk = np.zeros((n, n), dtype=np.float64)
            if hasattr(config, "coupling_map") and config.coupling_map:
                estimate = float(np.mean(gate_errors_array) * 0.01)
                for left, right in config.coupling_map:
                    crosstalk[left, right] = estimate
                    crosstalk[right, left] = estimate
            return {
                "T1_times": t1,
                "T2_times": t2,
                "gate_errors": gate_errors_array,
                "crosstalk_matrix": crosstalk,
                "leakage_rates": np.full(n, 1e-3),
                "base_error_rate": float(np.mean(gate_errors_array)),
                "timestamp": time.time(),
                "source": "ibm-backend",
                # This is a *mixed* record. T1/T2 and gate errors are measured;
                # crosstalk and leakage fields are heuristics/placeholders.
                "synthetic": True,
                "hardware_measurements_present": True,
                "calibration_quality": "mixed_observed_and_modeled",
                "field_provenance": {
                    "T1_times": "backend_measured",
                    "T2_times": "backend_measured",
                    "gate_errors": "backend_measured",
                    "base_error_rate": "derived_from_backend_gate_errors",
                    "crosstalk_matrix": "model_estimate_not_measured",
                    "leakage_rates": "unmeasured_placeholder",
                },
                "backend_name": self.backend_name,
                "cached": False,
            }
        except Exception as exc:
            raise RuntimeError(
                f"could not parse IBM backend calibration: {type(exc).__name__}: {exc}"
            ) from exc

    def _fetch_props_and_config(self) -> Optional[Tuple[Any, Any]]:
        try:
            props = self.backend.properties()
            config = self.backend.configuration()
            return props, config
        except Exception:
            return None

    def get_real_noise_parameters(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()

            if (now - self._window_start_time) >= 3600.0:
                self._window_start_time = now
                self._api_call_count = 0

            if self._last_cal and (
                now - self._last_fetch_time
            ) < self.cache_duration:
                return self._cached(now)

            if self._api_call_count >= self._max_api_calls_per_hour:
                if self._last_cal:
                    return self._cached(now)
                return self._fallback_or_raise("API rate limit reached")

            if self.backend is None:
                self._init_ibm_connection()
            if self.backend is None:
                return self._fallback_or_raise("backend connection unavailable")

            result_box: Dict[str, Optional[Tuple[Any, Any]]] = {"val": None}
            done = threading.Event()

            def worker():
                result_box["val"] = self._fetch_props_and_config()
                done.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            done.wait(self.timeout_s)

            if not done.is_set():
                if self._last_cal:
                    return self._cached(now)
                return self._fallback_or_raise("calibration request timed out")

            if result_box["val"] is None:
                if self._last_cal:
                    return self._cached(now)
                return self._fallback_or_raise("backend returned no calibration")

            props, config = result_box["val"]
            self._api_call_count += 1
            try:
                calibration = self._extract_calibration(props, config)
            except RuntimeError as exc:
                if self._last_cal:
                    return self._cached(now)
                return self._fallback_or_raise(str(exc))

            self._last_cal = calibration
            self._last_fetch_time = now
            return calibration
