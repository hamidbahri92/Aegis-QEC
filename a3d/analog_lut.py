"""A deliberately synthetic GKP lookup-table example, not a physical decoder."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class GKPLUTBuilder:
    """Build synthetic teaching-only curves on a finite, increasing grid.

    This placeholder is not a calibrated GKP likelihood model and must never
    be mistaken for a physical device decoder or a research-quality threshold.
    """

    def __init__(
        self, x_min: float = -3.5, x_max: float = 3.5, n: int = 1024
    ):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.n = int(n)
        if not math.isfinite(self.x_min) or not math.isfinite(self.x_max):
            raise ValueError("lookup-table bounds must be finite")
        if self.x_min >= self.x_max:
            raise ValueError("lookup-table minimum must be below maximum")
        if self.n < 2:
            raise ValueError("lookup-table must contain at least two samples")

    def build_lut(self) -> Dict[str, Any]:
        x = np.linspace(self.x_min, self.x_max, self.n, dtype=np.float64)
        Lq = -np.abs(x)
        Lp = -(x**2) * 0.1
        return {
            "x": x,
            "Lq": Lq,
            "Lp": Lp,
            "model": "synthetic_not_calibrated",
            "scientifically_validated": False,
        }
