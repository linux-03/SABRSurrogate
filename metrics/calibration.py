"""Calibration-fit metrics.

Extracted verbatim from the thesis code (calibrate_market_v2.py); the metric function bodies
are unchanged.
"""
from __future__ import annotations

import numpy as np


def weighted_rmse(model_iv, market_iv, weight, mask):
    r = (model_iv - market_iv)[mask]
    w = weight[mask]
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def rmse_in_spreads(model_iv, market_iv, weight, mask):
    """RMSE measured in units of the bid-ask half-spread (weight = 1/hs^2)."""
    r = (model_iv - market_iv)[mask]
    hs = 1.0 / np.sqrt(weight[mask])
    return float(np.sqrt(np.mean((r / hs) ** 2)))
