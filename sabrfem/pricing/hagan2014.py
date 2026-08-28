#!/usr/bin/env python3
"""Arbitrage-free SABR of Hagan, Kumar, Lesniewski & Woodward (2014).

Singular perturbation reduces the two-dimensional SABR problem to a
one-dimensional *effective forward equation* for the marginal density of the
forward. Solving that equation numerically gives a density that is
non-negative by construction, which the 2002 asymptotic formula is not.

Formulation
-----------
The marginal density splits into a continuous part and two boundary atoms,

    Q(T,F) = Q^L(T) delta(F - Fmin) + Q^c(T,F) + Q^R(T) delta(F - Fmax),

with the continuous part solving

    dQ^c/dT = (1/2) alpha^2 d^2/dF^2 [ D^2(F) Q^c ],      Fmin < F < Fmax

where, writing C(F) = F^beta for SABR,

    z(F)     = (1/alpha) * int_f^F dF'/C(F') = (F^(1-b) - f^(1-b)) / (alpha (1-b))
    Gamma(F) = (C(F) - C(f)) / (F - f)
    D(F)     = sqrt(1 + 2 rho nu z + nu^2 z^2) * exp(rho nu alpha Gamma(F) (T-t) / 2) * C(F)

Absorbing/reflecting structure at the truncation points is imposed as

    D^2(F) Q^c -> 0   as F -> Fmin+ and F -> Fmax-,

and probability lost through each end accumulates in the atoms

    dQ^L/dT = lim_{F->Fmin+}  (1/2) alpha^2 d/dF [D^2 Q^c]
    dQ^R/dT = lim_{F->Fmax-} -(1/2) alpha^2 d/dF [D^2 Q^c]

Initial data is Q^c -> delta(F - f), Q^L = Q^R = 0.

Reference: Hagan, Kumar, Lesniewski & Woodward, "Arbitrage-Free SABR",
Wilmott, Jan 2014. Equations above follow the exposition in J. Arce Arellano,
"Arbitrage-Free SABR: PDE Approach to Fix Negativity Density Function Issue"
(MSc thesis, UCM, 2019), Sections 4.2-4.5.

Discretisation
--------------
The operator is in conservative form, so the natural discretisation writes
G = D^2 Q and applies a standard three-point Laplacian to G, giving a
tridiagonal system in Q at each step. D is re-evaluated every step because the
exponential factor carries (T-t).

Hagan's original exposition uses Crank-Nicolson. CN is known to oscillate on
this problem (Le Floc'h & Kennedy), so `scheme` defaults to "implicit"
(backward Euler, unconditionally monotone here) with "cn" available for
comparison. Use `theta_scheme` for a blend.

VALIDATION STATUS
-----------------
This is an independent implementation, not a port of Hagan's code. Before
using its numbers in a paper, run `python sabr_hagan2014.py --self-test`,
which checks:
  (1) density non-negativity,
  (2) total mass = 1 including the atoms,
  (3) the martingale condition E[F_T] = f,
  (4) agreement with the 2002 formula in the regime nu^2 T << 1 where both
      should be accurate.
Do not report benchmark figures until all four pass at your chosen resolution.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------

@dataclass
class AFSabrConfig:
    """Grid and stepping parameters for the effective forward equation."""
    n_f: int = 500              # interior forward nodes
    n_steps_per_year: int = 200
    n_std: float = 6.0          # domain half-width in z-units of stdev
    scheme: str = "implicit"    # "implicit" | "cn"
    f_min: float | None = None  # override auto domain
    f_max: float | None = None

    def theta(self) -> float:
        return 1.0 if self.scheme == "implicit" else 0.5


# ----------------------------------------------------------------------
# model coefficients
# ----------------------------------------------------------------------

def _z_of_F(F: np.ndarray, f: float, alpha: float, beta: float) -> np.ndarray:
    """z(F) = (F^(1-b) - f^(1-b)) / (alpha (1-b)); log form at beta = 1."""
    ob = 1.0 - beta
    if abs(ob) < 1e-12:
        return np.log(F / f) / alpha
    return (np.power(F, ob) - f ** ob) / (alpha * ob)


def _gamma_of_F(F: np.ndarray, f: float, beta: float) -> np.ndarray:
    """Gamma(F) = (C(F) - C(f)) / (F - f), with the removable singularity at F=f."""
    C_F = np.power(F, beta)
    C_f = f ** beta
    dF = F - f
    out = np.empty_like(F)
    reg = np.abs(dF) > 1e-10
    out[reg] = (C_F[reg] - C_f) / dF[reg]
    out[~reg] = beta * f ** (beta - 1.0)          # limit C'(f)
    return out


def _D_of_F(F: np.ndarray, f: float, alpha: float, beta: float,
            rho: float, nu: float, tau: float) -> np.ndarray:
    """D(F) at elapsed time tau = T - t."""
    z = _z_of_F(F, f, alpha, beta)
    J = np.sqrt(np.maximum(1.0 + 2.0 * rho * nu * z + (nu * z) ** 2, 1e-300))
    G = _gamma_of_F(F, f, beta)
    expo = np.clip(0.5 * rho * nu * alpha * G * tau, -50.0, 50.0)
    return J * np.exp(expo) * np.power(F, beta)


# ----------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------

def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
            d: np.ndarray) -> np.ndarray:
    """Tridiagonal solve (sub a, diag b, super c, rhs d), via LAPACK dgtsv."""
    from scipy.linalg import solve_banded
    n = b.size
    ab = np.zeros((3, n))
    ab[0, 1:] = c
    ab[1, :] = b
    ab[2, :-1] = a
    return solve_banded((1, 1), ab, d)


def solve_density(f: float, alpha: float, beta: float, rho: float, nu: float,
                  T: float, cfg: AFSabrConfig | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve the effective forward equation to maturity T.

    Returns (F_grid, Q_continuous, Q_left_atom, Q_right_atom).
    Q_continuous is a density in F (integrates, with the atoms, to 1).
    """
    cfg = cfg or AFSabrConfig()

    # Domain: symmetric in z, mapped back to F. z has stdev ~ alpha*sqrt(T)
    # in the driftless normal approximation, inflated for vol-of-vol.
    ob = 1.0 - beta
    zmax = cfg.n_std * np.sqrt(T) * (1.0 + nu * np.sqrt(T))
    if cfg.f_min is not None and cfg.f_max is not None:
        Fmin, Fmax = cfg.f_min, cfg.f_max
    elif abs(ob) < 1e-12:
        Fmin = max(1e-8, f * np.exp(-alpha * zmax))
        Fmax = f * np.exp(alpha * zmax)
    else:
        base = f ** ob
        Fmin = max(1e-8, (base - alpha * ob * zmax) ** (1.0 / ob)
                   if base - alpha * ob * zmax > 0 else 1e-8)
        Fmax = (base + alpha * ob * zmax) ** (1.0 / ob)

    n = int(cfg.n_f)
    F = np.linspace(Fmin, Fmax, n)
    dF = F[1] - F[0]

    # initial condition: unit mass at the node nearest f
    Q = np.zeros(n)
    j0 = int(np.argmin(np.abs(F - f)))
    Q[j0] = 1.0 / dF
    QL = QR = 0.0

    n_steps = max(1, int(round(cfg.n_steps_per_year * T)))
    dt = T / n_steps
    th = cfg.theta()
    lam = 0.5 * alpha ** 2 * dt / dF ** 2

    for k in range(n_steps):
        tau = (k + 1) * dt
        D2 = _D_of_F(F, f, alpha, beta, rho, nu, tau) ** 2

        # G = D^2 Q ; dQ/dT = (1/2) a^2 d^2G/dF^2, interior nodes 1..n-2,
        # with G = 0 enforced at both end nodes (the D^2 Q -> 0 condition).
        m = n - 2
        D2i = D2[1:-1]
        diag = 1.0 + 2.0 * th * lam * D2i
        sub = -th * lam * D2[1:-2]                     # coeff of Q_{j-1}
        sup = -th * lam * D2[2:-1]                     # coeff of Q_{j+1}

        rhs = Q[1:-1].copy()
        if th < 1.0:
            G = D2 * Q
            Gpad = np.zeros(n)
            Gpad[1:-1] = G[1:-1]                       # G = 0 at the end nodes
            rhs += (1.0 - th) * lam * (Gpad[2:] - 2.0 * Gpad[1:-1] + Gpad[:-2])

        Qi = _thomas(sub, diag, sup, rhs)
        Qn = np.zeros(n)
        Qn[1:-1] = Qi
        # end nodes carry no continuous density (their mass is in the atoms)
        Qn[0] = Qn[-1] = 0.0

        # probability accumulating in the atoms, from the flux at each end
        Gnew = D2 * Qn
        QL += 0.5 * alpha ** 2 * dt * (Gnew[1] - 0.0) / dF ** 2 * dF
        QR += 0.5 * alpha ** 2 * dt * (Gnew[-2] - 0.0) / dF ** 2 * dF
        Q = Qn

    return F, Q, float(QL), float(QR)


# ----------------------------------------------------------------------
# pricing
# ----------------------------------------------------------------------

def call_prices(f: float, alpha: float, beta: float, rho: float, nu: float,
                T: float, strikes: np.ndarray,
                cfg: AFSabrConfig | None = None) -> np.ndarray:
    """Undiscounted European call prices by integrating the density."""
    F, Q, QL, QR = solve_density(f, alpha, beta, rho, nu, T, cfg)
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    out = np.empty(K.size)
    for i, k in enumerate(K):
        payoff = np.maximum(F - k, 0.0)
        out[i] = np.trapezoid(payoff * Q, F) + max(F[-1] - k, 0.0) * QR
    return out


def implied_vols(f: float, alpha: float, beta: float, rho: float, nu: float,
                 T: float, strikes: np.ndarray,
                 cfg: AFSabrConfig | None = None) -> np.ndarray:
    """Black implied vols of the arbitrage-free SABR prices."""
    from ..black import bs_call_price
    prices = call_prices(f, alpha, beta, rho, nu, T, strikes, cfg)
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    out = np.full(K.size, np.nan)
    for i, (k, p) in enumerate(zip(K, prices)):
        intrinsic = max(f - k, 0.0)
        if p <= intrinsic + 1e-12:
            continue
        lo, hi = 1e-6, 5.0
        if float(bs_call_price(f, k, T, hi)) < p:
            continue
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if float(bs_call_price(f, k, T, mid)) < p:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------

def self_test(cfg: AFSabrConfig | None = None) -> bool:
    cfg = cfg or AFSabrConfig()
    print(f"config: n_f={cfg.n_f} steps/yr={cfg.n_steps_per_year} "
          f"scheme={cfg.scheme}\n")
    cases = [
        dict(f=1.0, alpha=0.30, beta=0.5, rho=-0.45, nu=0.40, T=1.0),
        dict(f=1.0, alpha=0.30, beta=0.1, rho=-0.50, nu=1.00, T=2.0),
        dict(f=1.0, alpha=0.25, beta=0.8, rho=-0.30, nu=0.60, T=0.5),
    ]
    allok = True
    for c in cases:
        F, Q, QL, QR = solve_density(cfg=cfg, **c)
        mass = float(np.trapezoid(Q, F)) + QL + QR
        mean = float(np.trapezoid(F * Q, F)) + F[0] * QL + F[-1] * QR
        neg = float(Q.min())
        ok_pos = neg >= -1e-10
        ok_mass = abs(mass - 1.0) < 5e-3
        ok_mart = abs(mean - c["f"]) < 5e-3 * c["f"]
        allok &= ok_pos and ok_mass and ok_mart
        print(f"  b={c['beta']:.1f} nu={c['nu']:.2f} T={c['T']:.1f}  "
              f"min Q={neg:+.3e} {'OK' if ok_pos else 'FAIL'} | "
              f"mass={mass:.5f} {'OK' if ok_mass else 'FAIL'} | "
              f"E[F]={mean:.5f} {'OK' if ok_mart else 'FAIL'} | "
              f"QL={QL:.2e} QR={QR:.2e}")

    # short-maturity, low vol-of-vol: should agree with the 2002 formula
    print("\n  agreement with Hagan 2002 where nu^2 T << 1 (T=0.1, nu=0.15):")
    from .hagan import _hagan_vol
    K = np.array([0.85, 0.925, 1.0, 1.075, 1.15])
    p = dict(f=1.0, alpha=0.25, beta=0.5, rho=-0.3, nu=0.15, T=0.1)
    iv_pde = implied_vols(strikes=K, cfg=cfg, **p)
    iv_h = _hagan_vol(np.ones_like(K), K, np.full_like(K, p["T"]),
                      alpha=p["alpha"], beta=p["beta"], rho=p["rho"],
                      nu=p["nu"], obloj=False)
    d = np.nanmax(np.abs(iv_pde - iv_h))
    ok = d < 2e-3
    allok &= ok
    print(f"    PDE  {np.round(iv_pde, 5)}")
    print(f"    2002 {np.round(iv_h, 5)}")
    print(f"    max |diff| = {d:.2e}  {'OK' if ok else 'FAIL (needs investigation)'}")
    print(f"\n{'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'}")
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-f", type=int, default=500)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--scheme", default="implicit", choices=["implicit", "cn"])
    a = ap.parse_args()
    cfg = AFSabrConfig(n_f=a.n_f, n_steps_per_year=a.steps, scheme=a.scheme)
    if a.self_test:
        raise SystemExit(0 if self_test(cfg) else 1)
    F, Q, QL, QR = solve_density(1.0, 0.3, 0.5, -0.45, 0.4, 1.0, cfg)
    print(f"grid [{F[0]:.4f}, {F[-1]:.4f}], min Q = {Q.min():.3e}, "
          f"mass = {np.trapezoid(Q, F) + QL + QR:.6f}, QL={QL:.3e}, QR={QR:.3e}")
