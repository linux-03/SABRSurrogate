"""Combined per-surface arbitrage summary.

Composes the butterfly (:mod:`sabrfem.metrics.butterfly`), calendar
(:mod:`sabrfem.metrics.calendar`) and price-bound (:mod:`sabrfem.metrics.bounds`)
checks into the single dict the thesis reports per (parameter, surface): the
violation rates, the displaced negative mass and the worst density value.
"""

from __future__ import annotations

import numpy as np

from .bounds import bound_violations
from .calendar import calendar_derivative
from .density import breeden_litzenberger_density


def violation_summary(C: np.ndarray, K: np.ndarray, T: np.ndarray,
                      F: float = 1.0) -> dict:
    """All arbitrage metrics for one (n_K, n_T) call-price surface.

    Returns rates in [0, 1] for each violation type, the displaced negative
    mass, the minimum density and whether any violation is present.
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    p = breeden_litzenberger_density(C, K)                 # (n_K-2, n_T)
    dCdT = calendar_derivative(C, T)                       # (n_K, n_T-1)
    below, above = bound_violations(C, K, F)

    butterfly = p < 0.0
    calendar = dCdT < 0.0
    h = K[1] - K[0]
    displaced_mass = float(np.sum(np.maximum(-p, 0.0)) * h)

    return {
        "butterfly_rate": float(butterfly.mean()),
        "calendar_rate": float(calendar.mean()),
        "below_intrinsic_rate": float(below.mean()),
        "above_forward_rate": float(above.mean()),
        "displaced_mass": displaced_mass,
        "min_density": float(p.min()),
        "any_violation": bool(butterfly.any() or calendar.any()
                              or below.any() or above.any()),
    }
