"""Surrogate / method accuracy metrics.

Error of a predicted surface against a reference (finite-element) surface, in
price or in implied-vol space. Percentile absolute errors are reported
alongside the means because the surrogate error distribution is heavy-tailed,
so p95 / p99 are more informative than the mean. NaN cells (un-invertible
deep-wing IVs) are dropped from every reduction.
"""

from __future__ import annotations

import numpy as np


def _finite(err: np.ndarray) -> np.ndarray:
    err = np.asarray(err, dtype=float).ravel()
    return err[np.isfinite(err)]


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Root-mean-square error over finite cells."""
    e = _finite(np.asarray(pred, float) - np.asarray(ref, float))
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute error over finite cells."""
    e = _finite(np.asarray(pred, float) - np.asarray(ref, float))
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def abs_percentiles(pred: np.ndarray, ref: np.ndarray,
                    q=(50, 95, 99)) -> dict:
    """Percentiles of the absolute error (default p50 / p95 / p99)."""
    e = np.abs(_finite(np.asarray(pred, float) - np.asarray(ref, float)))
    return {f"p{int(qi)}": float(np.percentile(e, qi)) for qi in q}


def error_summary(pred: np.ndarray, ref: np.ndarray,
                  q=(50, 95, 99)) -> dict:
    """MAE, RMSE, absolute percentiles and the finite-cell rate in one dict."""
    diff = np.asarray(pred, float) - np.asarray(ref, float)
    finite = np.isfinite(diff)
    out = {
        "mae": mae(pred, ref),
        "rmse": rmse(pred, ref),
        "finite_rate": float(finite.mean()),
    }
    out.update({f"abs_{k}": v for k, v in abs_percentiles(pred, ref, q).items()})
    return out
