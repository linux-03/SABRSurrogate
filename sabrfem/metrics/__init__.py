"""Metrics used in the thesis, one file per metric.

Arbitrage (on a call-price surface)
    density       Breeden-Litzenberger risk-neutral density  d^2C/dK^2
    butterfly     butterfly / static arbitrage: min density, displaced mass
    calendar      calendar-spread monotonicity  dC/dT
    bounds        no-arbitrage price bounds  max(F-K,0) <= C <= F
    arbitrage     combined per-surface violation summary

Accuracy / fit
    accuracy      RMSE, MAE, absolute-error percentiles (price or IV space)
    calibration   liquidity-weighted RMSE and RMSE in bid-ask spread units

Solver
    convergence   empirical finite-element order of accuracy

All metrics are pure NumPy.
"""

from .accuracy import abs_percentiles, error_summary, mae, rmse
from .arbitrage import violation_summary
from .bounds import bound_violation_rate, bound_violations
from .butterfly import butterfly_stats, butterfly_surface_stats, is_butterfly_free
from .calendar import calendar_derivative, calendar_violation_rate
from .calibration import rmse_in_spreads, weighted_rmse
from .convergence import estimate_order
from .density import breeden_litzenberger_density

__all__ = [
    "breeden_litzenberger_density",
    "butterfly_stats", "butterfly_surface_stats", "is_butterfly_free",
    "calendar_derivative", "calendar_violation_rate",
    "bound_violations", "bound_violation_rate",
    "violation_summary",
    "rmse", "mae", "abs_percentiles", "error_summary",
    "weighted_rmse", "rmse_in_spreads",
    "estimate_order",
]
