"""
Benchmark SABR pricing methods: Monte Carlo, finite differences, and
Hagan-style implied-volatility formulas.

The PDE and SDE conventions match sabr_fem.py:
- Forward process X has absorbing boundary at 0.
- Volatility process Y uses a log-Euler update.
- Prices are undiscounted (zero rate, forward measure).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from implied_vol import bs_call_price
from sabr_fem import SABRParams


# ---------------------------------------------------------------------------
# Hagan and Obloj implied-volatility formulas
# ---------------------------------------------------------------------------


def _obloj_log_moneyness(F: np.ndarray, K: np.ndarray, beta: float) -> np.ndarray:
    """Dimensionless moneyness used in the Obloj correction.

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
    """Vectorized Hagan implied volatility (Black) with optional Obloj correction."""
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
    """Return Obloj-corrected Hagan implied vols on a strike x maturity grid."""
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
    """Return call prices and implied vols from the Obloj-corrected formula."""
    iv = obloj_implied_vol(params, strikes, maturities, forward=forward)
    F = float(params.x0 if forward is None else forward)
    K = np.asarray(strikes, dtype=float)[:, None]
    T = np.asarray(maturities, dtype=float)[None, :]
    prices = bs_call_price(F, K, T, iv)
    return prices, iv


# ---------------------------------------------------------------------------
# Monte Carlo benchmark
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Finite-difference benchmark (ADI + Crank-Nicolson, Craig-Sneyd)
# ---------------------------------------------------------------------------


@dataclass
class FDConfig:
    x_min: float = 1e-3
    x_max: float = 12.0
    y_min: float = -4.0
    y_max: float = 4.0
    nx: int = 161
    ny: int = 121
    n_steps_per_year: int = 200
    theta: float = 0.5
    right_bc: str = "payoff"  # "payoff" or "zero"
    y_bc: str = "neumann"  # "zero" (Dirichlet) or "neumann" (zero-gradient)
    clip_prices: bool = True


@dataclass
class FDReport:
    nx: int
    ny: int
    n_steps: int
    runtime_seconds: float


def _apply_boundary(u: np.ndarray, x_grid: np.ndarray, strike: float, cfg: FDConfig) -> None:
    u[0, :] = 0.0
    if cfg.right_bc == "payoff":
        u[-1, :] = max(x_grid[-1] - strike, 0.0)
    else:
        u[-1, :] = 0.0
    if cfg.y_bc == "zero":
        u[:, 0] = 0.0
        u[:, -1] = 0.0
    elif cfg.y_bc == "neumann":
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
    else:
        raise ValueError("y_bc must be 'zero' or 'neumann'")


def _clip_prices(u: np.ndarray, x_grid: np.ndarray, cfg: FDConfig) -> None:
    if cfg.clip_prices:
        np.clip(u, 0.0, x_grid[-1], out=u)


def _tridiag_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using scipy's banded solver (LAPACK dgbsv)."""
    from scipy.linalg import solve_banded

    n = int(diag.size)
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([rhs[0] / diag[0]], dtype=float)

    # Allow lower/upper of length n or n-1
    if lower.size == n:
        lower = lower[:-1]
    if upper.size == n:
        upper = upper[:-1]

    # Pack into banded format: ab[0] = upper (shifted), ab[1] = diag, ab[2] = lower (shifted)
    ab = np.zeros((3, n), dtype=float)
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return solve_banded((1, 1), ab, rhs)


def _apply_ax(u: np.ndarray, a: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        a[1:-1, 1:-1]
        * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        / (dx * dx)
    )
    return out


def _apply_ay(u: np.ndarray, c: float, d: float, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        c * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
        + d * (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
    )
    return out


def _apply_axy(u: np.ndarray, b: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        b[1:-1, 1:-1]
        * (
            u[2:, 2:]
            - u[2:, :-2]
            - u[:-2, 2:]
            + u[:-2, :-2]
        )
        / (4.0 * dx * dy)
    )
    return out


def _bilinear(x: np.ndarray, y: np.ndarray, u: np.ndarray, x0: float, y0: float) -> float:
    i = int(np.searchsorted(x, x0) - 1)
    j = int(np.searchsorted(y, y0) - 1)
    i = min(max(i, 0), x.size - 2)
    j = min(max(j, 0), y.size - 2)

    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[j], y[j + 1]
    wx = (x0 - x1) / max(x2 - x1, 1e-16)
    wy = (y0 - y1) / max(y2 - y1, 1e-16)

    u11 = u[i, j]
    u21 = u[i + 1, j]
    u12 = u[i, j + 1]
    u22 = u[i + 1, j + 1]

    return (
        (1.0 - wx) * (1.0 - wy) * u11
        + wx * (1.0 - wy) * u21
        + (1.0 - wx) * wy * u12
        + wx * wy * u22
    )


def _batch_tridiag_solve_y(lower_s, diag_s, upper_s, rhs_batch):
    """Solve n_batch independent tridiagonal systems of the same structure.

    lower_s, diag_s, upper_s are 1-D arrays of length n (the y-system size).
    rhs_batch is (n_batch, n).  Returns (n_batch, n).
    """
    from scipy.linalg import solve_banded
    n = diag_s.size
    ab = np.zeros((3, n), dtype=float)
    ab[0, 1:] = upper_s[:n - 1] if upper_s.size >= n else upper_s
    ab[1, :] = diag_s
    ab[2, :-1] = lower_s[:n - 1] if lower_s.size >= n else lower_s
    # solve_banded with multiple RHS columns
    return solve_banded((1, 1), ab, rhs_batch.T).T


def fd_call_surface(
    params: SABRParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cfg: FDConfig | None = None,
) -> Tuple[np.ndarray, FDReport]:
    """ADI / Crank-Nicolson benchmark for SABR call prices.

    Uses a Craig-Sneyd ADI scheme with the mixed derivative treated explicitly.
    Vectorised: all strikes are evolved simultaneously as a batch.
    """
    params.validate()
    cfg = cfg or FDConfig()

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    order = np.argsort(maturities)
    maturities_sorted = maturities[order]

    T_max = float(maturities_sorted[-1])
    n_steps = max(1, int(round(cfg.n_steps_per_year * T_max)))
    dt = T_max / n_steps

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    xx, yy = np.meshgrid(x, y, indexing="ij")
    a = 0.5 * np.power(xx, 2.0 * params.beta) * np.exp(2.0 * yy)
    b = params.rho * params.nu * np.power(xx, params.beta) * np.exp(yy)
    c_coeff = 0.5 * params.nu * params.nu
    d_coeff = -0.5 * params.nu * params.nu

    y0 = math.log(params.y0)
    if not (cfg.y_min <= y0 <= cfg.y_max):
        raise ValueError("y0 outside FD log-vol grid; widen y-range")
    if not (cfg.x_min <= params.x0 <= cfg.x_max):
        raise ValueError("x0 outside FD grid; widen x-range")

    snap_steps = {int(round(T / dt)): j for j, T in enumerate(maturities_sorted)}

    n_K = strikes.size
    prices = np.zeros((n_K, maturities.size), dtype=float)

    # Batch: u has shape (n_K, nx, ny) — all strikes at once
    u = np.maximum(x[None, :, None] - strikes[:, None, None], 0.0)
    u = np.broadcast_to(u, (n_K, cfg.nx, cfg.ny)).copy()
    for ik in range(n_K):
        _apply_boundary(u[ik], x, strikes[ik], cfg)

    # Precompute x-sweep coefficients per y-column
    # alpha[j] shape (nx-2,) for interior x nodes
    alpha_all = cfg.theta * dt * a[1:-1, :] / (dx * dx)  # (nx-2, ny)

    # y-sweep constants
    cy = c_coeff / (dy * dy)
    dy1 = d_coeff / (2.0 * dy)
    lower_y_s = -cfg.theta * dt * (cy - dy1)
    diag_y_s = 1.0 + cfg.theta * dt * 2.0 * cy
    upper_y_s = -cfg.theta * dt * (cy + dy1)
    n_yi = cfg.ny - 2
    y_lower = np.full(n_yi, lower_y_s)
    y_diag = np.full(n_yi, diag_y_s)
    y_upper = np.full(n_yi, upper_y_s)

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Apply operators (vectorised over n_K batch)
        ax_u = np.zeros_like(u)
        ax_u[:, 1:-1, 1:-1] = (
            a[None, 1:-1, 1:-1]
            * (u[:, 2:, 1:-1] - 2.0 * u[:, 1:-1, 1:-1] + u[:, :-2, 1:-1])
            / (dx * dx)
        )
        ay_u = np.zeros_like(u)
        ay_u[:, 1:-1, 1:-1] = (
            c_coeff * (u[:, 1:-1, 2:] - 2.0 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dy * dy)
            + d_coeff * (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dy)
        )
        axy_u = np.zeros_like(u)
        axy_u[:, 1:-1, 1:-1] = (
            b[None, 1:-1, 1:-1]
            * (u[:, 2:, 2:] - u[:, 2:, :-2] - u[:, :-2, 2:] + u[:, :-2, :-2])
            / (4.0 * dx * dy)
        )

        y0_arr = u + dt * (ax_u + ay_u + axy_u)

        def _x_sweep(rhs_full):
            """ADI x-sweep: solve tridiag per y-column, batched over strikes."""
            out = u.copy()
            for j in range(1, cfg.ny - 1):
                al = alpha_all[:, j]  # (nx-2,)
                lo = -al[1:]
                di = 1.0 + 2.0 * al
                up = -al[:-1]
                rr = rhs_full[:, 1:-1, j] - cfg.theta * dt * ax_u[:, 1:-1, j]
                rr[:, 0] += al[0] * u[:, 0, j]
                rr[:, -1] += al[-1] * u[:, -1, j]
                # Batch solve: each strike is an independent RHS
                from scipy.linalg import solve_banded
                n_x = di.size
                ab = np.zeros((3, n_x), dtype=float)
                ab[0, 1:] = up
                ab[1, :] = di
                ab[2, :-1] = lo
                out[:, 1:-1, j] = solve_banded((1, 1), ab, rr.T).T
            for ik in range(n_K):
                _apply_boundary(out[ik], x, strikes[ik], cfg)
                _clip_prices(out[ik], x, cfg)
            return out

        def _y_sweep(u_in):
            """ADI y-sweep: solve tridiag per x-row, batched over strikes."""
            out = u_in.copy()
            # Build batch RHS: (n_K * (nx-2), ny-2)
            rr = u_in[:, 1:-1, 1:-1] - cfg.theta * dt * ay_u[:, 1:-1, 1:-1]
            rr[:, :, 0] += lower_y_s * u_in[:, 1:-1, 0]
            rr[:, :, -1] += upper_y_s * u_in[:, 1:-1, -1]
            # Reshape to (n_K*(nx-2), ny-2) for batch solve
            n_rows = n_K * (cfg.nx - 2)
            rr_flat = rr.reshape(n_rows, n_yi)
            sol = _batch_tridiag_solve_y(y_lower, y_diag, y_upper, rr_flat)
            out[:, 1:-1, 1:-1] = sol.reshape(n_K, cfg.nx - 2, n_yi)
            for ik in range(n_K):
                _apply_boundary(out[ik], x, strikes[ik], cfg)
                _clip_prices(out[ik], x, cfg)
            return out

        # First half of Craig-Sneyd
        u1 = _x_sweep(y0_arr)
        u2 = _y_sweep(u1)

        # Craig-Sneyd correction for mixed derivative
        diff = u2 - u
        axy_diff = np.zeros_like(u)
        axy_diff[:, 1:-1, 1:-1] = (
            b[None, 1:-1, 1:-1]
            * (diff[:, 2:, 2:] - diff[:, 2:, :-2] - diff[:, :-2, 2:] + diff[:, :-2, :-2])
            / (4.0 * dx * dy)
        )
        y0_tilde = y0_arr + 0.5 * dt * axy_diff

        # Second half of Craig-Sneyd
        u1 = _x_sweep(y0_tilde)
        u2 = _y_sweep(u1)

        u = u2

        if step in snap_steps:
            idx = snap_steps[step]
            for ik in range(n_K):
                prices[ik, order[idx]] = _bilinear(x, y, u[ik], params.x0, y0)

    report = FDReport(
        nx=cfg.nx,
        ny=cfg.ny,
        n_steps=n_steps,
        runtime_seconds=time.perf_counter() - t0,
    )
    return prices, report
