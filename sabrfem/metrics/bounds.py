"""No-arbitrage price-bound metrics.

A call price must lie in the model-free band  max(F - K, 0) <= C <= F. Prices
below intrinsic or above the forward are hard static-arbitrage violations.
"""

from __future__ import annotations

import numpy as np


def bound_violations(C: np.ndarray, K: np.ndarray, F: float = 1.0,
                     tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for lower- and upper-bound violations.

    Returns ``(below_intrinsic, above_forward)``, each the shape of ``C``:
    ``below_intrinsic`` where C < (F - K)+ and ``above_forward`` where C > F.
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    intrinsic = np.maximum(F - K, 0.0)
    if C.ndim == 2:
        intrinsic = intrinsic[:, None]
    below = C < intrinsic - tol
    above = C > float(F) + tol
    return below, above


def bound_violation_rate(C: np.ndarray, K: np.ndarray, F: float = 1.0,
                         tol: float = 1e-12) -> dict:
    """Fractions of cells violating the lower and upper no-arbitrage bounds."""
    below, above = bound_violations(C, K, F, tol)
    return {
        "below_intrinsic_rate": float(below.mean()),
        "above_forward_rate": float(above.mean()),
    }
