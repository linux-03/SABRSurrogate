"""
Black-Scholes implied volatility inversion.

Given a (non-discounted) European call price C, a forward F, strike K and
maturity T, solve for the log-normal volatility sigma such that

    C = F * N(d_plus) - K * N(d_minus),
    d_pm = (log(F/K) +/- 0.5 * sigma^2 * T) / (sigma * sqrt(T)).

We assume zero interest rate throughout (matching Horvath-Reichmann's
convention); a non-zero rate can be absorbed into the forward F = S0 * exp(r*T).

Algorithm: rational initial guess followed by Newton iteration on the vega.
Robust to deep out-of-the-money and short-maturity pricing arising from the
SABR finite-element output.

Vectorised over arrays of (F, K, T, C) of identical broadcastable shape.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf, ndtr
from scipy.stats import norm as _scipy_norm

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / SQRT2PI


def _Phi(x):
    """Standard normal CDF (vectorised)."""
    return 0.5 * (1.0 + erf(x / SQRT2))


def bs_call_price(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Black-Scholes call price on a forward, zero rate.

    All inputs broadcastable.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        v = sigma * np.sqrt(T)  # total volatility
        d1 = np.where(v > 0.0, (np.log(F / K) + 0.5 * v * v) / v, np.inf)
        d2 = d1 - v
        price = F * _Phi(d1) - K * _Phi(d2)

    # Handle degenerate limits cleanly.
    intrinsic = np.maximum(F - K, 0.0)
    price = np.where(v <= 0.0, intrinsic, price)
    return price


def bs_vega(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Black-Scholes vega on a forward, zero rate: dC/dsigma.

    Vega is the same for calls and puts, so this function doubles as the
    put-price derivative w.r.t. sigma.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        v = sigma * np.sqrt(T)
        d1 = np.where(v > 0.0, (np.log(F / K) + 0.5 * v * v) / v, 0.0)
        out = F * _phi(d1) * np.sqrt(T)
    return np.where(v > 0.0, out, 0.0)


def bs_put_price(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Black-Scholes put price on a forward, zero rate.

    P = K * Phi(-d2) - F * Phi(-d1)
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        v = sigma * np.sqrt(T)
        d1 = np.where(v > 0.0, (np.log(F / K) + 0.5 * v * v) / v, np.inf)
        d2 = d1 - v
        price = K * _Phi(-d2) - F * _Phi(-d1)

    intrinsic = np.maximum(K - F, 0.0)
    price = np.where(v <= 0.0, intrinsic, price)
    return price


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------


def _rational_guess(F, K, T, C):
    """Coarse rational initial guess for sigma.

    Uses the Brenner-Subrahmanyam approximation near ATM and blends with a
    log-moneyness-based estimate for deep wings. Good enough to give Newton
    a clean start point for all reasonable SABR outputs.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)

    # Brenner-Subrahmanyam: sigma ~ sqrt(2*pi/T) * C / F   for ATM
    atm = np.sqrt(2.0 * math.pi / np.maximum(T, 1e-12)) * C / np.maximum(F, 1e-12)

    # Log-moneyness guess (Corrado-Miller-style):
    # sigma_hat^2 * T ~ 2 * |log(F/K)|  when we're very far in/out of the money.
    logm = np.log(np.maximum(F, 1e-12) / np.maximum(K, 1e-12))
    far = np.sqrt(2.0 * np.abs(logm) / np.maximum(T, 1e-12))

    # Blend: ATM guess dominates when |logm| is small.
    weight_atm = np.exp(-5.0 * logm * logm)
    guess = weight_atm * atm + (1.0 - weight_atm) * np.maximum(far, atm)
    return np.clip(guess, 1e-4, 5.0)


def implied_vol(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    C: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 60,
    sigma_floor: float = 1e-5,
    sigma_ceiling: float = 5.0,
    parity_for_itm: bool = True,
) -> np.ndarray:
    """Invert Black-Scholes for implied volatility.

    Parameters
    ----------
    F, K, T, C : array_like
        Forward, strike, maturity, call price. Broadcastable.
    tol : float
        Absolute price tolerance for convergence.
    max_iter : int
        Maximum Newton iterations.
    sigma_floor, sigma_ceiling : float
        Clipping bounds for the iterate.
    parity_for_itm : bool, optional
        For ITM calls (K < F) convert to puts via put-call parity
        (P = C + K - F) and invert the put instead. Puts on ITM-call
        strikes are OTM and dominated by *time value*, so a small absolute
        pricing error from the FE solver does not push the price outside
        the put arbitrage band [(K - F)+, K]. This greatly reduces the
        number of spurious NaNs from FE truncation error. Default: True.

    Returns
    -------
    sigma : np.ndarray
        Implied volatility, NaN where the price is outside the arbitrage
        bounds [(F - K)+, F] (or the analogous put bounds) or where Newton
        failed to converge.
    """
    F, K, T, C = np.broadcast_arrays(
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(C, dtype=float),
    )
    shape = C.shape
    F = F.ravel().copy()
    K = K.ravel().copy()
    T = T.ravel().copy()
    C = C.ravel().copy()

    # Decide which instrument to invert per grid point: use the put when the
    # option is ITM as a call (K < F), else keep the call.
    use_put = parity_for_itm & (K < F)
    # Put price via parity: P = C + K - F (note: zero rate)
    P = C + K - F

    # Per-cell working price and arbitrage bounds:
    #   calls: lower = (F - K)+, upper = F
    #   puts : lower = (K - F)+, upper = K
    price_work = np.where(use_put, P, C)
    intr_call = np.maximum(F - K, 0.0)
    intr_put = np.maximum(K - F, 0.0)
    lower = np.where(use_put, intr_put, intr_call)
    upper = np.where(use_put, K, F)

    arb_lower = price_work >= lower - 1e-10
    arb_upper = price_work <= upper + 1e-10
    time_ok = T > 1e-10
    feasible = arb_lower & arb_upper & time_ok

    sigma = np.full_like(C, np.nan)
    work = feasible.copy()
    # Initial guess: use the call's rational guess on a "virtual call" with
    # effective (F, K) chosen so that the instrument is OTM. For put-inversion
    # cells we swap F and K in the guess — this gives a symmetric log-moneyness
    # which is a fine initial iterate (Brenner-Subrahmanyam is a function of
    # |log F/K| only).
    F_guess = np.where(use_put, K, F)
    K_guess = np.where(use_put, F, K)
    # Note: the initial guess only reads from (F_guess, K_guess, T, price_work)
    # to produce a sigma; Newton then corrects using the correct pricing fn.
    sigma[work] = _rational_guess(
        F_guess[work], K_guess[work], T[work], price_work[work]
    )

    def _price_fn(F_, K_, T_, s_, use_put_):
        """Call or put price elementwise according to use_put_."""
        call = bs_call_price(F_, K_, T_, s_)
        put = bs_put_price(F_, K_, T_, s_)
        return np.where(use_put_, put, call)

    for _ in range(max_iter):
        if not np.any(work):
            break
        s = sigma[work]
        price = _price_fn(F[work], K[work], T[work], s, use_put[work])
        diff = price - price_work[work]
        # Convergence check
        done = np.abs(diff) <= tol
        if np.all(done):
            sigma[work] = s
            break
        vega = bs_vega(F[work], K[work], T[work], s)
        # Protect against zero vega (vanishing in deep wings).
        vega = np.where(vega < 1e-14, 1e-14, vega)
        step = diff / vega
        # Damp the step to prevent overshoot.
        step = np.clip(step, -0.25, 0.25)
        s = s - step
        s = np.clip(s, sigma_floor, sigma_ceiling)
        sigma[work] = s

    # Final sanity check: if we ended up at the boundary with a residual, mark NaN.
    final_diff = np.abs(
        _price_fn(F, K, T, np.nan_to_num(sigma, nan=0.5), use_put) - price_work
    )
    failed = feasible & (final_diff > 1e-4)
    sigma[failed] = np.nan

    return sigma.reshape(shape)
