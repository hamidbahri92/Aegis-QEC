"""Validated analog syndrome information for research decoder inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SoftSyndrome:
    llr_X: List[float]
    llr_Z: List[float]
    p_erase_time: Optional[List[float]] = None
    leakage_flags: Optional[List[int]] = None


def validate_soft(s: SoftSyndrome) -> None:
    """Reject malformed or nonfinite confidence inputs in all Python modes.

    Assertions are unsuitable here: Python optimization removes them entirely.
    """
    if not isinstance(s.llr_X, list) or not isinstance(s.llr_Z, list):
        raise TypeError("LLR arrays must be lists")
    for axis, arr in (("X", s.llr_X), ("Z", s.llr_Z)):
        for index, value in enumerate(arr):
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {axis} LLR at index {index}") from exc
            if not math.isfinite(converted):
                raise ValueError(f"nonfinite {axis} LLR at index {index}")

    if s.p_erase_time is not None:
        for index, value in enumerate(s.p_erase_time):
            try:
                probability = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid erasure probability at {index}") from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"erasure probability must be finite and in [0, 1] at {index}"
                )

    if s.leakage_flags is not None:
        for index, value in enumerate(s.leakage_flags):
            if value not in (0, 1):
                raise ValueError(f"leakage flag must be binary at {index}")
