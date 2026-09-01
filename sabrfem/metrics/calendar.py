"""Calendar-spread arbitrage metrics.

An undiscounted call is non-decreasing in maturity; a slice that falls as T
grows is a calendar-spread arbitrage. The metric is the first difference
dC/dT on the maturity grid and the fraction of cells where it is negative.
"""

from __future__ import annotations

import numpy as np


def calendar_derivative(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """First-difference dC/dT on the maturity grid.

    Parameters
    ----------
    C : (n_K, n_T) call prices.
    T : (n_T,) maturities, increasing.

    Returns
    -------
    dCdT : (n_K, n_T - 1) at midpoint maturities (T[i] + T[i+1]) / 2.
    """
    C = np.asarray(C, dtype=float)
    T = np.asarray(T, dtype=float)
    dT = np.diff(T)
    return (C[:, 1:] - C[:, :-1]) / dT[None, :]


def calendar_violation_rate(C: np.ndarray, T: np.ndarray, tol: float = 1e-12) -> float:
    """Fraction of (strike, maturity) cells where dC/dT < -tol."""
    return float((calendar_derivative(C, T) < -tol).mean())
