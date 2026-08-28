"""Monte Carlo pricing of the SABR call surface.

Two estimators, both log-Euler with an absorbing boundary at the forward's zero
and an optional lognormal control variate driven by the same Brownian path:

* :func:`mc_call_surface`      -- pseudo-random with antithetic variates.
* :func:`mc_call_surface_qmc`  -- Sobol quasi-Monte-Carlo.

Conventions match the FE solver (:mod:`sabr_lib.pricing.fem`): forward measure,
zero rate, undiscounted prices.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..black import bs_call_price
from .fem import SABRParams


@dataclass
class MCConfig:
    n_paths: int = 200_000
    n_steps_per_year: int = 200
    seed: int = 1
    antithetic: bool = True
    control_variate: bool = True


@dataclass
class QMCConfig:
    n_paths: int = 65_536
    n_steps_per_year: int = 200
    seed: int = 1
    scramble: bool = True
    control_variate: bool = True


@dataclass
class MCReport:
    n_paths: int
    n_steps: int
    runtime_seconds: float
    stderr: np.ndarray


def mc_call_surface(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cfg: MCConfig | None = None,
) -> Tuple[np.ndarray, MCReport]:
    """Monte Carlo call prices with antithetic and optional control variate.

    The control variate uses a lognormal proxy driven by the same Brownian
    path, with sigma = y0 and known Black price.
    """
    params.validate()
    cfg = cfg or MCConfig()

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    T_max = float(np.max(maturities))
    n_steps = max(1, int(round(cfg.n_steps_per_year * T_max)))
    dt = T_max / n_steps
    sqrt_dt = math.sqrt(dt)

    n_paths = int(cfg.n_paths)
    if cfg.antithetic:
        n_half = (n_paths + 1) // 2
        n_paths = 2 * n_half
    else:
        n_half = n_paths

    rng = np.random.default_rng(cfg.seed)

    X = np.full(n_paths, params.x0, dtype=float)
    Y = np.full(n_paths, params.y0, dtype=float)

    if cfg.control_variate:
        sigma_cv = params.y0
        X_cv = np.full(n_paths, params.x0, dtype=float)
    else:
        sigma_cv = 0.0
        X_cv = None

    snap_steps: Dict[int, int] = {
        int(round(T / dt)): j for j, T in enumerate(maturities)
    }
    snapshots = {}
    snapshots_cv = {}

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_half)
        z2 = rng.standard_normal(n_half)
        if cfg.antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])

        dW = z1
        dZ = params.rho * z1 + math.sqrt(1.0 - params.rho**2) * z2

        X_new = X + Y * np.power(np.maximum(X, 0.0), params.beta) * sqrt_dt * dW
        Y_new = Y * np.exp(-0.5 * params.nu**2 * dt + params.nu * sqrt_dt * dZ)
        X = np.where(X_new <= 0.0, 0.0, X_new)
        Y = Y_new

        if cfg.control_variate:
            X_cv = X_cv * np.exp(-0.5 * sigma_cv**2 * dt + sigma_cv * sqrt_dt * dW)

        if step in snap_steps:
            idx = snap_steps[step]
            snapshots[idx] = X.copy()
            if cfg.control_variate:
                snapshots_cv[idx] = X_cv.copy()

    prices = np.zeros((strikes.size, maturities.size), dtype=float)
    stderr = np.zeros_like(prices)

    for j, T in enumerate(maturities):
        XT = snapshots[j]
        if cfg.control_variate:
            XT_cv = snapshots_cv[j]
            bs = bs_call_price(params.x0, strikes[:, None], T, params.y0).ravel()
        else:
            XT_cv = None
            bs = None

        for i, K in enumerate(strikes):
            payoff = np.maximum(XT - K, 0.0)
            if cfg.control_variate:
                payoff_cv = np.maximum(XT_cv - K, 0.0)
                cov = np.cov(payoff, payoff_cv, ddof=1)
                b = cov[0, 1] / max(cov[1, 1], 1e-16)
                adj = payoff - b * (payoff_cv - bs[i])
                prices[i, j] = np.mean(adj)
                stderr[i, j] = np.std(adj, ddof=1) / math.sqrt(n_paths)
            else:
                prices[i, j] = np.mean(payoff)
                stderr[i, j] = np.std(payoff, ddof=1) / math.sqrt(n_paths)

    report = MCReport(
        n_paths=n_paths,
        n_steps=n_steps,
        runtime_seconds=time.perf_counter() - t0,
        stderr=stderr,
    )
    return prices, report


def mc_call_surface_qmc(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cfg: QMCConfig | None = None,
) -> Tuple[np.ndarray, MCReport]:
    """Sobol-QMC Monte Carlo call prices (log-Euler, absorbing boundary)."""
    from scipy.stats import qmc as _qmc
    from scipy.stats import norm as _norm

    params.validate()
    cfg = cfg or QMCConfig()

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    T_max = float(np.max(maturities))
    n_steps = max(1, int(round(cfg.n_steps_per_year * T_max)))
    dt = T_max / n_steps
    sqrt_dt = math.sqrt(dt)

    n_paths = int(cfg.n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    # Sobol prefers power-of-two draws; we generate the next power-of-two and slice.
    m = int(math.ceil(math.log2(n_paths))) if n_paths > 1 else 0
    n_draw = 1 << m

    sobol = _qmc.Sobol(d=2 * n_steps, scramble=cfg.scramble, seed=cfg.seed)
    u = sobol.random_base2(m=m) if n_draw > 1 else sobol.random(n_draw)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = _norm.ppf(u)
    z = z[:n_paths]

    z1_all = z[:, :n_steps]
    z2_all = z[:, n_steps:]

    X = np.full(n_paths, params.x0, dtype=float)
    Y = np.full(n_paths, params.y0, dtype=float)

    if cfg.control_variate:
        sigma_cv = params.y0
        X_cv = np.full(n_paths, params.x0, dtype=float)
    else:
        sigma_cv = 0.0
        X_cv = None

    snap_steps: Dict[int, int] = {
        int(round(T / dt)): j for j, T in enumerate(maturities)
    }
    snapshots = {}
    snapshots_cv = {}

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        z1 = z1_all[:, step - 1]
        z2 = z2_all[:, step - 1]
        dW = z1
        dZ = params.rho * z1 + math.sqrt(1.0 - params.rho**2) * z2

        X_new = X + Y * np.power(np.maximum(X, 0.0), params.beta) * sqrt_dt * dW
        Y_new = Y * np.exp(-0.5 * params.nu**2 * dt + params.nu * sqrt_dt * dZ)
        X = np.where(X_new <= 0.0, 0.0, X_new)
        Y = Y_new

        if cfg.control_variate:
            X_cv = X_cv * np.exp(-0.5 * sigma_cv**2 * dt + sigma_cv * sqrt_dt * dW)

        if step in snap_steps:
            idx = snap_steps[step]
            snapshots[idx] = X.copy()
            if cfg.control_variate:
                snapshots_cv[idx] = X_cv.copy()

    prices = np.zeros((strikes.size, maturities.size), dtype=float)
    stderr = np.zeros_like(prices)

    for j, T in enumerate(maturities):
        XT = snapshots[j]
        if cfg.control_variate:
            XT_cv = snapshots_cv[j]
            bs = bs_call_price(params.x0, strikes[:, None], T, params.y0).ravel()
        else:
            XT_cv = None
            bs = None

        for i, K in enumerate(strikes):
            payoff = np.maximum(XT - K, 0.0)
            if cfg.control_variate:
                payoff_cv = np.maximum(XT_cv - K, 0.0)
                cov = np.cov(payoff, payoff_cv, ddof=1)
                b = cov[0, 1] / max(cov[1, 1], 1e-16)
                adj = payoff - b * (payoff_cv - bs[i])
                prices[i, j] = np.mean(adj)
                stderr[i, j] = np.std(adj, ddof=1) / math.sqrt(n_paths)
            else:
                prices[i, j] = np.mean(payoff)
                stderr[i, j] = np.std(payoff, ddof=1) / math.sqrt(n_paths)

    report = MCReport(
        n_paths=n_paths,
        n_steps=n_steps,
        runtime_seconds=time.perf_counter() - t0,
        stderr=stderr,
    )
    return prices, report
