"""Finite-element convergence metric.

Extracted verbatim from the thesis code (convergence_study.py); the metric function bodies
are unchanged.
"""
from __future__ import annotations

import numpy as np


def estimate_rate(maxh_values, error_values):
    """Estimate a power-law rate from log-log data."""
    xs = np.asarray(maxh_values, dtype=float)
    ys = np.asarray(error_values, dtype=float)
    mask = (xs > 0.0) & (ys > 0.0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return None
    slope, _ = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(slope)
