"""Hagan (2002) and Oblój implied-volatility asymptotics for SABR.

These are the closed-form Black implied-vol expansions the thesis benchmarks the
finite-element solver against. The Oblój (2008) variant replaces Hagan's
log-moneyness with the scaled CEV distance, curing part of the wing behaviour.
Both return call prices via the Black formula (:mod:`sabr_lib.black`).

Convention: forward measure,
zero rate, undiscounted prices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..black import bs_call_price
from .params import SABRParams


def _obloj_log_moneyness(F: np.ndarray, K: np.ndarray, beta: float) -> np.ndarray:
    """Dimensionless moneyness used in the Oblój correction.

    This replaces log(F/K) with the scaled CEV distance, which collapses to
    log(F/K) as beta -> 1.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)

    if abs(1.0 - beta) < 1e-10:
        return np.log(F / K)

    one_beta = 1.0 - beta
    fk = np.power(F * K, 0.5 * one_beta)
    cev = (np.power(F, one_beta) - np.power(K, one_beta)) / one_beta
    out = cev / np.maximum(fk, 1e-16)
    return out


def _hagan_vol(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    obloj: bool,
) -> np.ndarray:
    """Vectorized Hagan implied volatility (Black) with optional Oblój correction."""
    F, K, T = np.broadcast_arrays(F, K, T)
    F = F.astype(float)
    K = K.astype(float)
    T = T.astype(float)

    one_beta = 1.0 - beta
    logm = np.where(
        obloj, _obloj_log_moneyness(F, K, beta), np.log(F / K)
    )

    fk = np.power(F * K, 0.5 * one_beta)
    fk = np.maximum(fk, 1e-16)

    logm2 = logm * logm
    logm4 = logm2 * logm2
    denom = fk * (
        1.0
        + (one_beta * one_beta / 24.0) * logm2
        + (one_beta**4 / 1920.0) * logm4
    )

    z = (nu / alpha) * fk * logm
    sqrt_term = np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z * z, 1e-16))
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))

    # Both z and x_z have the same sign (both flip for K > F), so z/x_z > 0.
    # Guard against x_z ≈ 0 (ATM limit) but preserve the sign.
    safe_x_z = np.where(np.abs(x_z) < 1e-16, np.copysign(1e-16, z + 1e-30), x_z)
    z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / safe_x_z)

    term1 = (one_beta * one_beta / 24.0) * (alpha * alpha) / (fk * fk)
    term2 = 0.25 * rho * beta * nu * alpha / fk
    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    corr = 1.0 + (term1 + term2 + term3) * T

    sigma = (alpha / denom) * z_over_x * corr
    sigma = np.where(T <= 0.0, 0.0, sigma)
    return np.maximum(sigma, 0.0)


def hagan_implied_vol(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    forward: float | None = None,
) -> np.ndarray:
    """Return Hagan (2002) implied vols on a strike x maturity grid."""
    params.validate()
    F = float(params.x0 if forward is None else forward)
    K = np.asarray(strikes, dtype=float)[:, None]
    T = np.asarray(maturities, dtype=float)[None, :]
    Fg = np.full_like(K, F) * np.ones_like(T)
    return _hagan_vol(
        Fg,
        K * np.ones_like(T),
        T,
        alpha=params.y0,
        beta=params.beta,
        rho=params.rho,
        nu=params.nu,
        obloj=False,
    )


def obloj_implied_vol(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    forward: float | None = None,
) -> np.ndarray:
    """Return Oblój-corrected Hagan implied vols on a strike x maturity grid."""
    params.validate()
    F = float(params.x0 if forward is None else forward)
    K = np.asarray(strikes, dtype=float)[:, None]
    T = np.asarray(maturities, dtype=float)[None, :]
    Fg = np.full_like(K, F) * np.ones_like(T)
    return _hagan_vol(
        Fg,
        K * np.ones_like(T),
        T,
        alpha=params.y0,
        beta=params.beta,
        rho=params.rho,
        nu=params.nu,
        obloj=True,
    )


def hagan_call_surface(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    forward: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return call prices and implied vols from the Hagan formula."""
    iv = hagan_implied_vol(params, strikes, maturities, forward=forward)
    F = float(params.x0 if forward is None else forward)
    K = np.asarray(strikes, dtype=float)[:, None]
    T = np.asarray(maturities, dtype=float)[None, :]
    prices = bs_call_price(F, K, T, iv)
    return prices, iv


def obloj_call_surface(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    forward: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return call prices and implied vols from the Oblój-corrected formula."""
    iv = obloj_implied_vol(params, strikes, maturities, forward=forward)
    F = float(params.x0 if forward is None else forward)
    K = np.asarray(strikes, dtype=float)[:, None]
    T = np.asarray(maturities, dtype=float)[None, :]
    prices = bs_call_price(F, K, T, iv)
    return prices, iv
