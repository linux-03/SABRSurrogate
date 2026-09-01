"""Calibration-fit metrics.

Quality of a fitted implied-vol surface against market quotes. Two numbers:

* ``weighted_rmse``   -- RMSE weighted by liquidity/vega (weight = 1 / hs^2 with
  hs the bid-ask half-spread), so tight ATM quotes count more than wide wings.
* ``rmse_in_spreads`` -- the same residuals expressed in bid-ask half-spread
  units; ~1 means the fit typically lands on the edge of the spread, < 1 means
  inside it. This is the metric that actually says whether a quote is usable.
"""

from __future__ import annotations

import numpy as np


def weighted_rmse(model_iv, market_iv, weight, mask=None) -> float:
    """Liquidity-weighted RMSE of model vs market IV over the masked cells."""
    model_iv = np.asarray(model_iv, float)
    market_iv = np.asarray(market_iv, float)
    weight = np.asarray(weight, float)
    if mask is None:
        mask = np.ones(model_iv.shape, dtype=bool)
    r = (model_iv - market_iv)[mask]
    w = weight[mask]
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def rmse_in_spreads(model_iv, market_iv, weight, mask=None) -> float:
    """RMSE measured in units of the bid-ask half-spread (weight = 1 / hs^2)."""
    model_iv = np.asarray(model_iv, float)
    market_iv = np.asarray(market_iv, float)
    weight = np.asarray(weight, float)
    if mask is None:
        mask = np.ones(model_iv.shape, dtype=bool)
    r = (model_iv - market_iv)[mask]
    hs = 1.0 / np.sqrt(weight[mask])
    return float(np.sqrt(np.mean((r / hs) ** 2)))
