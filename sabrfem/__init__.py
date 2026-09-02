"""sabrfem — SABR pricing methods and a trained neural surrogate.

Classical asymptotic and grid-based SABR pricers plus a neural surrogate that
prices a whole implied-vol surface at calibration speed. The surrogate ships
pre-trained, so the package is light: no finite-element / NGSolve dependency.

Layout
------
    sabrfem.black                  Black-Scholes price/vega + implied-vol inversion
    sabrfem.pricing.params         the shared SABRParams object
    sabrfem.pricing.hagan          Hagan (2002) + Oblój asymptotics
    sabrfem.pricing.hagan2014      arbitrage-free SABR density PDE
    sabrfem.pricing.montecarlo     Monte Carlo + Sobol-QMC
    sabrfem.pricing.finite_diff    ADI / Craig-Sneyd finite differences
    sabrfem.pricing.surrogate      the trained neural surrogate (fast pricer)
    sabrfem.calibration            inverse-problem calibration (surrogate speedup)
    sabrfem.arbitrage              butterfly-arbitrage scan + SANOS / QP repair
    sabrfem.metrics                the thesis metrics, one file per metric

The light core (black, pricing, metrics) is imported eagerly; the analysis
subpackages are imported explicitly when needed. PyTorch is required only to
run the surrogate and is imported lazily.
"""

from . import black, metrics, pricing
from .pricing import SABRParams

__all__ = ["black", "pricing", "metrics", "SABRParams"]
__version__ = "0.2.0"
