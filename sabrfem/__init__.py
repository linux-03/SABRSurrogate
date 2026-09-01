"""sabrfem — finite-element SABR pricing and a neural surrogate.

A weighted-Sobolev finite-element solver for the degenerate SABR pricing PDE,
the classical asymptotic and grid-based competitors it is benchmarked against,
and a neural surrogate trained on FE prices for calibration-speed pricing that
keeps FE accuracy.

Layout
------
    sabrfem.black                  Black-Scholes price/vega + implied-vol inversion
    sabrfem.pricing.fem            weighted-Sobolev finite-element solver
    sabrfem.pricing.hagan          Hagan (2002) + Oblój asymptotics
    sabrfem.pricing.hagan2014      arbitrage-free SABR density PDE
    sabrfem.pricing.montecarlo     Monte Carlo + Sobol-QMC
    sabrfem.pricing.finite_diff    ADI / Craig-Sneyd finite differences
    sabrfem.surrogate              the MLP surrogate: model / train / predict
    sabrfem.calibration            inverse-problem calibration (surrogate speedup)
    sabrfem.arbitrage              butterfly-arbitrage scan + SANOS / QP repair
    sabrfem.eval                   convergence studies + method benchmarks
    sabrfem.metrics                the thesis metrics, one file per metric

Only the light core (black, pricing, surrogate, metrics) is imported eagerly;
the analysis subpackages are imported explicitly when needed, and heavy
dependencies (NGSolve, PyTorch) are imported lazily inside them.
"""

from . import black, metrics, pricing, surrogate
from .pricing import SABRParams

__all__ = ["black", "pricing", "surrogate", "metrics", "SABRParams"]
__version__ = "0.1.0"
