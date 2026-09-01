"""Finite-element convergence metric.

Empirical order of accuracy: the slope of log(error) against log(mesh size)
over a refinement sequence. A slope of ~p means error ~ O(h^p), so it
quantifies how fast the FE solution converges to the reference as the mesh is
refined.
"""

from __future__ import annotations

import numpy as np


def estimate_order(maxh_values, error_values) -> float | None:
    """Power-law convergence order from a log-log least-squares fit.

    Parameters
    ----------
    maxh_values  : mesh sizes h (any order).
    error_values : the corresponding errors (same length).

    Returns
    -------
    The fitted slope p in  error ~ h^p, or None if fewer than two positive
    (h, error) pairs are available.
    """
    xs = np.asarray(maxh_values, dtype=float)
    ys = np.asarray(error_values, dtype=float)
    mask = (xs > 0.0) & (ys > 0.0)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 2:
        return None
    slope, _ = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(slope)
