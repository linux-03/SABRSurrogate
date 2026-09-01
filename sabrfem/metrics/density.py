"""Risk-neutral density by Breeden-Litzenberger.

The state-price density implied by a call-price slice is its second derivative
in strike, p(K) = d^2 C / dK^2. On the uniform strike grid used throughout the
thesis this is the three-point central second difference. A density that dips
negative is exactly a butterfly-arbitrage violation, so this quantity is the
building block for the arbitrage metrics in :mod:`sabrfem.metrics.butterfly`.
"""

from __future__ import annotations

import numpy as np


def breeden_litzenberger_density(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Second-difference Breeden-Litzenberger density along the strike axis.

    Parameters
    ----------
    C : (n_K, n_T) or (n_K,) call prices on an equally spaced strike grid.
    K : (n_K,) strikes, uniform spacing.

    Returns
    -------
    p : density on the interior strike nodes, shape (n_K - 2, n_T) (or
        (n_K - 2,) for a single slice). p = (C[i+1] - 2 C[i] + C[i-1]) / h^2.
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    squeeze = C.ndim == 1
    if squeeze:
        C = C[:, None]
    h = K[1] - K[0]
    p = (C[2:, :] - 2.0 * C[1:-1, :] + C[:-2, :]) / (h * h)
    return p[:, 0] if squeeze else p
