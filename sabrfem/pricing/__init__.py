"""SABR pricing methods, one module per method.

    hagan        Hagan (2002) + Oblój implied-vol asymptotics
    hagan2014    arbitrage-free SABR density PDE (Hagan et al. 2014)
    montecarlo   Monte Carlo + Sobol-QMC
    finite_diff  ADI / Craig-Sneyd finite differences
    surrogate    the trained neural surrogate (fast pricer)

Every method prices on the same (strikes x maturities) grid under the shared
forward-measure, zero-rate convention. ``SABRParams`` (in ``params``) is the
common parameter object.
"""

from . import hagan2014, surrogate
from .finite_diff import FDConfig, FDReport, fd_call_surface
from .hagan import (
    hagan_call_surface,
    hagan_implied_vol,
    obloj_call_surface,
    obloj_implied_vol,
)
from .montecarlo import (
    MCConfig,
    MCReport,
    QMCConfig,
    mc_call_surface,
    mc_call_surface_qmc,
)
from .params import SABRParams
from .surrogate import load_surrogate, predict

__all__ = [
    "SABRParams",
    "hagan_call_surface", "obloj_call_surface",
    "hagan_implied_vol", "obloj_implied_vol",
    "mc_call_surface", "mc_call_surface_qmc", "MCConfig", "QMCConfig", "MCReport",
    "fd_call_surface", "FDConfig", "FDReport",
    "hagan2014",
    "surrogate", "load_surrogate", "predict",
]
