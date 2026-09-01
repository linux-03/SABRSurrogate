"""Butterfly (static) arbitrage metrics.

A call slice is butterfly-arbitrage-free iff its Breeden-Litzenberger density
(:mod:`sabrfem.metrics.density`) is non-negative everywhere. These are the
headline arbitrage numbers of the thesis:

* ``min_density``    -- the most negative density value (0 if clean).
* ``displaced_mass`` -- integral of the negative part of the density,
  int max(-p, 0) dK; the amount of probability mass sitting below zero, the
  scalar that lets FE (which converges to 0) be compared against the
  asymptotics (which have an irreducible floor).
* ``cells``          -- number of strike nodes with a negative density.

A small ``tol`` avoids flagging machine-precision artefacts from a QP repair
(constraints can sit at ~1e-14, which is not real arbitrage); use ``tol=0`` for
raw surrogate / Hagan output.
"""

from __future__ import annotations

import numpy as np

from .density import breeden_litzenberger_density


def butterfly_stats(C_col: np.ndarray, K: np.ndarray, tol: float = 0.0) -> dict:
    """Butterfly diagnostics for one maturity column of call prices.

    Returns a dict with ``any`` (bool), ``cells`` (int), ``min_density`` and
    ``displaced_mass`` (floats).
    """
    p = breeden_litzenberger_density(np.asarray(C_col, dtype=float), K)
    neg = p < -tol
    dK = K[1] - K[0]
    return {
        "any": bool(neg.any()),
        "cells": int(neg.sum()),
        "min_density": float(p.min()),
        "displaced_mass": float(np.sum(np.maximum(-p - tol, 0.0)) * dK),
    }


def butterfly_surface_stats(C: np.ndarray, K: np.ndarray, tol: float = 0.0) -> dict:
    """Aggregate butterfly diagnostics over a whole (n_K, n_T) price surface.

    ``min_density`` is the worst over all maturities, ``displaced_mass`` the max
    over maturities (the reported per-surface number), and ``violation_rate``
    the fraction of interior (strike, maturity) cells with a negative density.
    """
    C = np.asarray(C, dtype=float)
    p = breeden_litzenberger_density(C, K)
    neg = p < -tol
    dK = K[1] - K[0]
    mass_per_T = np.sum(np.maximum(-p - tol, 0.0), axis=0) * dK
    return {
        "any": bool(neg.any()),
        "violation_rate": float(neg.mean()),
        "min_density": float(p.min()),
        "displaced_mass": float(mass_per_T.max()),
    }


def is_butterfly_free(C: np.ndarray, K: np.ndarray, tol: float = 0.0) -> bool:
    """True iff the density is non-negative everywhere (within ``tol``)."""
    return bool((breeden_litzenberger_density(np.asarray(C, float), K) >= -tol).all())
