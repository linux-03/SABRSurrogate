"""Arbitrage metrics on a call-price surface.

Extracted verbatim from the thesis code (hagan_arbitrage.py, sanos_repair.py); the metric function bodies
are unchanged.
"""
from __future__ import annotations

import numpy as np


def butterfly_density(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Second-difference Breeden-Litzenberger density along the strike axis.

    Returns p of shape (len(K)-2, len(T)) on the interior strike nodes.
    Standard three-point central second difference; the strike grid is
    uniform, so the formula reduces to (C[i+1] - 2 C[i] + C[i-1]) / h^2.
    """
    h = K[1] - K[0]
    return (C[2:, :] - 2.0 * C[1:-1, :] + C[:-2, :]) / (h * h)


def calendar_derivative(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """First-difference dC/dT on the (interior) maturity grid.

    Returns dCdT of shape (len(K), len(T)-1) at midpoint times
    T_mid = (T[i] + T[i+1])/2.
    """
    dT = np.diff(T)
    return (C[:, 1:] - C[:, :-1]) / dT[None, :]


def bound_violations(C: np.ndarray, K: np.ndarray,
                     F: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Indicator masks for intrinsic and upper bound violations.

    Returns (V_lo, V_hi) of shape (len(K), len(T)).
    """
    intrinsic = np.maximum(F - K, 0.0)[:, None]
    upper = float(F)
    V_lo = C < intrinsic - 1e-12
    V_hi = C > upper + 1e-12
    return V_lo, V_hi


def violation_summary_per_tuple(
    C: np.ndarray, K: np.ndarray, T: np.ndarray, F: float = 1.0
) -> dict:
    """Compute all violation metrics for a single (K, T) surface."""
    p = butterfly_density(C, K)
    dCdT = calendar_derivative(C, T)
    V_lo, V_hi = bound_violations(C, K, F)

    B = p < 0
    D = dCdT < 0

    # Negative mass: integral of max(-p, 0) on (interior K, full T)
    # Use simple Riemann sum on the strike axis
    h = K[1] - K[0]
    M_neg = float(np.sum(np.maximum(-p, 0.0)) * h)

    n_cells_B = int(B.size)
    n_cells_D = int(D.size)
    n_cells_V = int(V_lo.size)

    any_violation = bool(
        B.any() or D.any() or V_lo.any() or V_hi.any()
    )

    return {
        "butterfly_rate": float(B.sum()) / max(n_cells_B, 1),
        "calendar_rate": float(D.sum()) / max(n_cells_D, 1),
        "intrinsic_rate": float(V_lo.sum()) / max(n_cells_V, 1),
        "upper_rate": float(V_hi.sum()) / max(n_cells_V, 1),
        "negative_mass": M_neg,
        "any_violation": any_violation,
        "butterfly_mask": B,        # for the per-(K,T) heatmap
        "calendar_mask": D,
        "intrinsic_mask": V_lo,
        "upper_mask": V_hi,
        "density_min": float(p.min()),
        "density_argmin": tuple(np.unravel_index(int(np.argmin(p)), p.shape)),
    }


def butterfly_stats(C_col: np.ndarray, K: np.ndarray, tol: float = 0.0) -> dict:
    """Butterfly diagnostics for one maturity column of call prices.

    Parameters
    ----------
    tol : float
        Density tolerance: a cell is flagged only if p < -tol.  Use tol=0
        (default) for raw surrogate / Hagan output.  Use tol=1e-10 for
        repaired output to avoid flagging machine-precision artefacts from
        the QP solver (SLSQP satisfies constraints to ~1e-12 in objective
        but constraints themselves can sit at ~1e-14, which is not real
        arbitrage).
    """
    p = butterfly_density(C_col[:, None], K)[:, 0]
    neg = p < -tol
    dK = K[1] - K[0]
    return {
        "any": bool(neg.any()),
        "cells": int(neg.sum()),
        "min_density": float(p.min()),
        "neg_mass": float(np.sum(np.maximum(-p - tol, 0.0)) * dK),
    }
