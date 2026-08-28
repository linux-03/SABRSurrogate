#!/usr/bin/env python3
"""Arbitrage repair layer for the DLV surrogate output.

Two repair methods are implemented:

  QP (default) — minimal-perturbation projection
  ------------------------------------------------
  Per maturity slice, solve the small quadratic program

      min  (1/2) ||c - c_surrogate||²_w
      s.t. c[i-1] - 2c[i] + c[i+1] >= 0   for i=1..n-2  (convexity / butterfly-free)
           c[i] - c[i+1]             >= 0   for i=0..n-2  (monotone decreasing)
           c[i] >= (F - K[i])^+              for all i     (intrinsic lower bound)
           c[i] <= F                          for all i     (upper bound)

  with w = 1/max(ν_i, ε)² (inverse-vega-squared) so the objective is
  approximately Σ(ΔIV_i)² — the natural IV-space loss.  This finds the
  closest arbitrage-free price vector to the surrogate output, so
  perturbation is exactly zero for already-clean slices and minimal for
  violating ones.  Typical IV perturbation: median ~1e-4, orders of
  magnitude below the SANOS basis approach.

  SANOS (--method sanos) — basis projection
  ------------------------------------------
  Fits a non-negative combination of Black-Scholes call payoffs (SANOS
  basis, Buehler 2026) to the quoted prices via NNLS.  Provides a globally
  convex surface between and beyond quoted strikes, at the cost of a larger
  IV perturbation (~3e-2 median) because the SANOS manifold is a restricted
  function class.

Usage:
    python sanos_repair.py                        # 512 tuples, QP default
    python sanos_repair.py --method sanos         # SANOS basis projection
    python sanos_repair.py --n 2048
    python sanos_repair.py --sweep-anchors        # anchor-count study (SANOS only)
    python sanos_repair.py --ch4-tuples           # same tuples as Ch.4 table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent

from ..calibration.calibrate import load_nn, theta_to_x, PRIOR, PARAM_NAMES  # noqa: E402
from ..black import bs_call_price, implied_vol, bs_vega  # noqa: E402
from .hagan_scan import call_surface_from_iv, butterfly_density  # noqa: E402

STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
F = 1.0

# Anchor range: anchors act as "forwards" in BS calls, so an anchor below the
# minimum quoted strike K_min produces a deep-OTM call (value ≈ 0) at every
# quoted strike — a zero column that makes AᵀA singular in NNLS.  Start just
# above K_min = 0.50.  Extend well past K_max = 1.50 so that high-strike
# basis calls have appreciable value across the full quoted grid.
ANCHOR_LO = 0.55
ANCHOR_HI = 2.20


def make_anchors(n: int) -> np.ndarray:
    """Uniformly spaced anchor strikes covering [ANCHOR_LO, ANCHOR_HI]."""
    return np.linspace(ANCHOR_LO, ANCHOR_HI, n)


def sanos_basis(anchors: np.ndarray, K_eval: np.ndarray, V: float) -> np.ndarray:
    """Build the SANOS design matrix B of shape (len(K_eval), len(anchors)).

    B[m, i] = BS-call(forward=anchor_i, strike=K_eval_m, total_var=V).
    Each column is a convex decreasing function of K_eval, so any non-negative
    linear combination is also convex and decreasing — butterfly-free by
    construction.

    The backbone variance V is set per-slice as V = atm_iv^2 * T (ATM total
    variance). This anchors the basis scale to the local smile level so that
    only a small number of weights are appreciably non-zero.
    """
    a = anchors[None, :]           # (1, N_anchors)
    K = np.asarray(K_eval)[:, None]  # (n_K, 1)
    # bs_call_price(F, K, T, sigma): use T=1, sigma=sqrt(V) -> total var V
    return bs_call_price(a, K, 1.0, np.sqrt(V))   # (n_K, N_anchors)


def repair_slice(
    C_quoted: np.ndarray,
    K_quoted: np.ndarray,
    V: float,
    anchors: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative LSQ fit of the SANOS basis to surrogate quoted prices.

    Parameters
    ----------
    C_quoted : (n_K,) call prices from the surrogate
    K_quoted : (n_K,) strike grid
    V        : backbone total variance (atm_iv^2 * T)
    anchors  : (N_anchors,) anchor strikes
    w        : (n_K,) optional non-negative weights (e.g. inverse bid-ask)

    Returns
    -------
    q          : (N_anchors,) NNLS weights, q >= 0
    C_repaired : (n_K,) repaired call prices, butterfly-free by construction
    residual   : scalar NNLS residual norm ||B q - C_quoted|| (or weighted)
    """
    B = sanos_basis(anchors, K_quoted, V)
    # Drop columns whose max value is negligible — these are anchors so far
    # below the quoted strikes that the basis call is effectively zero, which
    # makes AᵀA singular.  We reconstruct q on the full anchor set afterwards.
    col_scale = B.max(axis=0)
    active = col_scale > 1e-7
    if not active.any():
        # Degenerate: return the unrepaired prices with zero residual warning
        return np.zeros(len(anchors)), C_quoted.copy(), float(np.linalg.norm(C_quoted))
    B_act = B[:, active]
    # Final guard: replace any residual NaN/Inf in the basis (should not occur
    # after the iv sanitisation above, but protects against edge cases in BS).
    B_act = np.nan_to_num(B_act, nan=0.0, posinf=0.0, neginf=0.0)
    # lsq_linear with bounds=(0, inf) solves the same non-negative least-squares
    # problem as nnls but uses a more robust bounded-variable active-set method
    # (BVLS) that avoids the AᵀA singularity issues in newer scipy's nnls.
    if w is not None:
        sw = np.sqrt(w)
        result = lsq_linear(B_act * sw[:, None], C_quoted * sw,
                             bounds=(0, np.inf), method='bvls')
    else:
        result = lsq_linear(B_act, C_quoted, bounds=(0, np.inf), method='bvls')
    q_act = result.x
    residual = float(np.linalg.norm(B_act @ q_act - C_quoted))
    q = np.zeros(len(anchors))
    q[active] = q_act
    return q, B @ q, float(residual)


def butterfly_stats(C_col: np.ndarray, K: np.ndarray, tol: float = 0.0) -> dict:
    """Butterfly diagnostics for one maturity column of call prices.

    Parameters
    ----------
    tol : float
        Density tolerance: a cell is flagged only if p < -tol.  Use tol=0
        (default) for raw surrogate / Hagan output.  Use tol=1e-10 for
        repaired output to avoid flagging machine-precision artefacts from
        the QP solver (SLSQP satisfies constraints to ~1e-12 in objective
        but constraints themselves can sit at ~1e-14, which is not real
        arbitrage).
    """
    p = butterfly_density(C_col[:, None], K)[:, 0]
    neg = p < -tol
    dK = K[1] - K[0]
    return {
        "any": bool(neg.any()),
        "cells": int(neg.sum()),
        "min_density": float(p.min()),
        "neg_mass": float(np.sum(np.maximum(-p - tol, 0.0)) * dK),
    }


def min_perturb_repair(
    C_quoted: np.ndarray,
    K: np.ndarray,
    F: float = 1.0,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Minimal-perturbation projection onto the arbitrage-free price manifold.

    Solves the QP
        min  (1/2) ||c - C_quoted||²_w
        s.t. c[i-1] - 2c[i] + c[i+1] >= 0    (convexity / butterfly-free)
             c[i] - c[i+1]             >= 0    (monotone decreasing in K)
             (F - K[i])^+  <=  c[i]  <=  F    (no-arbitrage bounds)

    With w = 1/max(ν_i, ε)², the objective equals approximately Σ(ΔIV_i)²
    (since ΔC ≈ ν·ΔIV), so the solver minimises total IV perturbation.

    Unlike the SANOS basis projection this finds the *nearest* arbitrage-free
    point to the surrogate output — perturbation is zero for already-clean
    slices and minimal (typically ~1e-4 in IV) for violating ones.

    Parameters
    ----------
    C_quoted : (n,) call prices from the surrogate
    K        : (n,) strike grid (must be equally spaced and increasing)
    F        : forward (default 1.0)
    w        : (n,) non-negative weights; None = uniform

    Returns
    -------
    c_repaired : (n,) call prices satisfying all arbitrage constraints
    residual   : ||c_repaired - C_quoted||_w  (weighted L2 norm)
    """
    from scipy.optimize import minimize

    n = len(C_quoted)
    if w is None:
        w = np.ones(n)

    def obj(c: np.ndarray) -> float:
        r = c - C_quoted
        return 0.5 * float(np.dot(w * r, r))

    def jac(c: np.ndarray) -> np.ndarray:
        return w * (c - C_quoted)

    # Convexity: c[i-1] - 2c[i] + c[i+1] >= 0  for i = 1..n-2
    A_cvx = np.zeros((n - 2, n))
    for i in range(1, n - 1):
        A_cvx[i - 1, i - 1] =  1.0
        A_cvx[i - 1, i    ] = -2.0
        A_cvx[i - 1, i + 1] =  1.0

    # Monotone decreasing: c[i] - c[i+1] >= 0  for i = 0..n-2
    A_mon = np.zeros((n - 1, n))
    for i in range(n - 1):
        A_mon[i, i    ] =  1.0
        A_mon[i, i + 1] = -1.0

    constraints = [
        {'type': 'ineq', 'fun': lambda c, A=A_cvx: A @ c,
                         'jac': lambda c, A=A_cvx: A},
        {'type': 'ineq', 'fun': lambda c, A=A_mon: A @ c,
                         'jac': lambda c, A=A_mon: A},
    ]

    intrinsic = np.maximum(F - K, 0.0)
    bounds = list(zip(intrinsic.tolist(), np.full(n, F).tolist()))

    result = minimize(
        obj, C_quoted.copy(), jac=jac,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )

    c_rep = np.clip(result.x, intrinsic, F)   # safety clip against numerics
    residual = float(np.sqrt(max(2.0 * result.fun, 0.0)))
    return c_rep, residual


def best_V(C_quoted: np.ndarray, K_quoted: np.ndarray,
           anchors: np.ndarray, V_init: float,
           w: np.ndarray | None = None) -> float:
    """Golden-section search for the backbone variance that minimises the
    NNLS residual.  Searches over [V_init/16, V_init*16] in log-space."""
    from scipy.optimize import minimize_scalar

    def obj(logV):
        V = float(np.exp(logV))
        B = sanos_basis(anchors, K_quoted, V)
        col_scale = B.max(axis=0)
        active = col_scale > 1e-7
        if not active.any():
            return float(np.linalg.norm(C_quoted))
        B_act = np.nan_to_num(B[:, active], nan=0.0, posinf=0.0, neginf=0.0)
        if w is not None:
            sw = np.sqrt(w)
            r = lsq_linear(B_act * sw[:, None], C_quoted * sw,
                           bounds=(0, np.inf), method='bvls')
        else:
            r = lsq_linear(B_act, C_quoted, bounds=(0, np.inf), method='bvls')
        return float(np.linalg.norm(B_act @ r.x - C_quoted))

    lo, hi = np.log(V_init / 16), np.log(V_init * 16)
    result = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                             options={'xatol': 1e-3})
    return float(np.exp(result.x))


def run_scan(
    nn,
    theta: np.ndarray,
    anchors: np.ndarray,
    torch,
    with_hagan: bool = True,
    opt_v: bool = False,
    method: str = 'qp',
) -> dict:
    """Scan theta tuples through the surrogate and repair layer.

    Parameters
    ----------
    method : 'qp' (default) or 'sanos'
        'qp'    — minimal-perturbation QP projection (recommended)
        'sanos' — SANOS basis NNLS projection

    Optionally computes Hagan on the same grid as a paired baseline so that
    all three columns (Hagan / NN / NN+repair) use identical tuples, identical
    strikes, and identical ΔK — the only apples-to-apples comparison possible
    on the HMT grid.

    Returns a dict of aggregate statistics suitable for JSON serialisation.
    """
    from .hagan_scan import hagan_iv_surface

    n_slice = 0
    raw_viol = rep_viol = hag_viol = 0
    raw_minp, rep_minp, hag_minp = [], [], []
    raw_negm, rep_negm, hag_negm = [], [], []
    iv_dev = []           # IV RMSE introduced by the repair
    repair_residuals = [] # repair residual per slice (QP or NNLS)

    for th in theta:
        x = torch.tensor(theta_to_x(th), dtype=torch.float32)
        iv = nn.forward_iv(x).detach().numpy()          # (11, 8)
        # np.clip propagates NaN silently; replace non-finite values with a
        # reasonable fallback before clipping so downstream V is always finite.
        iv = np.where(np.isfinite(iv), iv, 0.3)
        iv = np.clip(iv, 1e-4, 5.0)
        C = call_surface_from_iv(iv, STRIKES, MATURITIES, F)  # (11, 8)

        # Hagan on the same grid — same tuples, same strikes, same ΔK
        if with_hagan:
            try:
                iv_hag = np.clip(
                    hagan_iv_surface(th, STRIKES, MATURITIES, formula="hagan"),
                    1e-4, 5.0,
                )
                C_hag = call_surface_from_iv(iv_hag, STRIKES, MATURITIES, F)
            except Exception:
                C_hag = None
        else:
            C_hag = None

        for j, T in enumerate(MATURITIES):
            atm_iv = float(iv[5, j])                    # K=1.0 is index 5
            V = max(atm_iv ** 2 * T, 1e-6)             # ATM total variance; guard zero

            # Inverse-vega-squared weights: minimises Σ(ΔIV_i)² directly.
            # Derivation: ΔC_i ≈ ν_i · ΔIV_i, so Σ w_i (ΔC_i)² = Σ(ΔIV_i)²
            # requires w_i = 1/ν_i². Cap at 1/ε² to avoid numerics in deep
            # OTM where ν → 0.  Equivalent to fitting in IV space.
            vega = bs_vega(
                np.full_like(STRIKES, F), STRIKES,
                np.full_like(STRIKES, T),
                np.full_like(STRIKES, atm_iv),
            )
            eps = float(vega[5]) * 0.05      # 5% of ATM vega as floor
            inv_vega2_w = 1.0 / np.maximum(vega, eps) ** 2

            raw = butterfly_stats(C[:, j], STRIKES)
            if raw["any"]:
                # Only project violating slices — preserves the surrogate's
                # accuracy on the 96% of already-clean slices.
                if method == 'qp':
                    Crep, res_norm = min_perturb_repair(
                        C[:, j], STRIKES, F=F, w=inv_vega2_w)
                else:  # 'sanos'
                    V_use = best_V(C[:, j], STRIKES, anchors, V,
                                   w=inv_vega2_w) if opt_v else V
                    _, Crep, res_norm = repair_slice(C[:, j], STRIKES, V_use,
                                                     anchors, w=inv_vega2_w)
            else:
                Crep, res_norm = C[:, j].copy(), 0.0
            # tol=1e-10: SLSQP satisfies constraints to ~1e-12 in objective
            # value; the density itself can sit at ~1e-14, which is machine
            # epsilon and not real arbitrage.  Raw / Hagan checks use tol=0.
            rep = butterfly_stats(Crep, STRIKES, tol=1e-10)

            if C_hag is not None:
                hag = butterfly_stats(C_hag[:, j], STRIKES)
                hag_viol += hag["any"]
                hag_minp.append(hag["min_density"])
                hag_negm.append(hag["neg_mass"])

            # IV deviation introduced by repair
            iv_rep = implied_vol(
                np.full_like(STRIKES, F),
                STRIKES,
                np.full_like(STRIKES, T),
                Crep,
            )
            ok = np.isfinite(iv_rep) & np.isfinite(iv[:, j])
            iv_dev.append(float(np.sqrt(np.mean((iv_rep[ok] - iv[ok, j]) ** 2)))
                          if ok.any() else np.nan)
            repair_residuals.append(res_norm)

            n_slice += 1
            raw_viol += raw["any"]
            rep_viol += rep["any"]
            raw_minp.append(raw["min_density"])
            rep_minp.append(rep["min_density"])
            raw_negm.append(raw["neg_mass"])
            rep_negm.append(rep["neg_mass"])

    pct = lambda x: 100.0 * x / max(n_slice, 1)  # noqa: E731
    iv_dev_finite = [v for v in iv_dev if np.isfinite(v)]
    has_hagan = len(hag_minp) > 0

    out = {
        "method": method,
        "n_anchors": int(len(anchors)),
        "n_tuples": int(len(theta)),
        "n_slices": n_slice,
        "butterfly_slice_pct": {
            "hagan": pct(hag_viol) if has_hagan else None,
            "raw": pct(raw_viol),
            "repaired": pct(rep_viol),
        },
        "min_density": {
            "hagan_min": float(np.min(hag_minp)) if has_hagan else None,
            "raw_min": float(np.min(raw_minp)),
            "repaired_min": float(np.min(rep_minp)),
            "hagan_p1": float(np.percentile(hag_minp, 1)) if has_hagan else None,
            "raw_p1": float(np.percentile(raw_minp, 1)),
            "repaired_p1": float(np.percentile(rep_minp, 1)),
        },
        "neg_mass": {
            "hagan_max": float(np.max(hag_negm)) if has_hagan else None,
            "raw_max": float(np.max(raw_negm)),
            "repaired_max": float(np.max(rep_negm)),
            "hagan_p90": float(np.percentile(hag_negm, 90)) if has_hagan else None,
            "raw_p90": float(np.percentile(raw_negm, 90)),
            "repaired_p90": float(np.percentile(rep_negm, 90)),
        },
        "repair_iv_rmse": {
            "median": float(np.median(iv_dev_finite)) if iv_dev_finite else np.nan,
            "p90": float(np.percentile(iv_dev_finite, 90)) if iv_dev_finite else np.nan,
        },
        "repair_residual": {
            "median": float(np.median(repair_residuals)),
            "p90": float(np.percentile(repair_residuals, 90)),
            "max": float(np.max(repair_residuals)),
        },
    }
    return out


def print_summary(s: dict) -> None:
    method = s.get("method", "qp")
    rep_label = "QP-repaired" if method == 'qp' else "SANOS-repaired"
    has_hagan = s["butterfly_slice_pct"]["hagan"] is not None
    if has_hagan:
        print(f"\n{'':22}{'Hagan':>12}{'DLV surrogate':>15}{rep_label:>17}")
        print("-" * 66)
        print(f"{'butterfly slice %':22}"
              f"{s['butterfly_slice_pct']['hagan']:>12.1f}"
              f"{s['butterfly_slice_pct']['raw']:>15.1f}"
              f"{s['butterfly_slice_pct']['repaired']:>17.1f}")
        print(f"{'min density (worst)':22}"
              f"{s['min_density']['hagan_min']:>12.3f}"
              f"{s['min_density']['raw_min']:>15.3f}"
              f"{s['min_density']['repaired_min']:>17.3f}")
        print(f"{'neg mass (max)':22}"
              f"{s['neg_mass']['hagan_max']:>12.2e}"
              f"{s['neg_mass']['raw_max']:>15.2e}"
              f"{s['neg_mass']['repaired_max']:>17.2e}")
    else:
        print(f"\n{'':22}{'DLV surrogate':>15}{rep_label:>17}")
        print("-" * 54)
        print(f"{'butterfly slice %':22}"
              f"{s['butterfly_slice_pct']['raw']:>15.1f}"
              f"{s['butterfly_slice_pct']['repaired']:>17.1f}")
        print(f"{'min density (worst)':22}"
              f"{s['min_density']['raw_min']:>15.3f}"
              f"{s['min_density']['repaired_min']:>17.3f}")
        print(f"{'neg mass (max)':22}"
              f"{s['neg_mass']['raw_max']:>15.2e}"
              f"{s['neg_mass']['repaired_max']:>17.2e}")
    if method == 'sanos':
        print(f"\nN_anchors = {s['n_anchors']}")
    print(f"IV perturbation  : median {s['repair_iv_rmse']['median']:.2e}"
          f"  p90 {s['repair_iv_rmse']['p90']:.2e}")
    print(f"repair residual  : median {s['repair_residual']['median']:.2e}"
          f"  p90 {s['repair_residual']['p90']:.2e}"
          f"  max {s['repair_residual']['max']:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="Sobol' tuples")
    ap.add_argument("--model", default=str(HERE / "data" / "nn_iv" / "nn_model.pt"))
    ap.add_argument("--config", default=None,
                    help="Path to nn_config.json (auto-detected from model dir).")
    ap.add_argument("--method", choices=["qp", "sanos"], default="qp",
                    help="Repair method: 'qp' (default) = minimal-perturbation "
                         "QP projection; 'sanos' = SANOS basis NNLS projection.")
    ap.add_argument("--n-anchors", type=int, default=40,
                    help="Number of SANOS anchor strikes (default 40, sanos method only).")
    ap.add_argument("--opt-v", action="store_true",
                    help="(SANOS method only) Optimise backbone variance V per slice "
                         "via golden-section search. Improves fit at ~20x cost per slice.")
    ap.add_argument("--ch4-tuples", action="store_true",
                    help="Use the identical Sobol tuples as Ch.4 (seed=0, "
                         "|rho|nu²<1.9 filter), so rates are directly "
                         "comparable with tab:hagan-violations.")
    ap.add_argument("--sweep-anchors", action="store_true",
                    help="Sweep N_anchors in {10,15,20,30,40,60} and report "
                         "NNLS residual + IV RMSE to help choose N.")
    ap.add_argument("--out", default=str(HERE / "data" / "arbitrage" /
                                         "sanos_repair_summary.json"))
    args = ap.parse_args()

    # Auto-detect model config
    model_dir = Path(args.model).parent
    config_path = Path(args.config) if args.config else model_dir / "nn_config.json"
    hidden, depth, target = 30, 4, "iv"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        hidden = cfg.get("hidden", hidden)
        depth  = cfg.get("depth", depth)
        target = cfg.get("target", target)

    import torch
    nn = load_nn(torch, Path(args.model), hidden, depth,
                 STRIKES.size, MATURITIES.size, target,
                 STRIKES, MATURITIES, torch.device("cpu"))

    # Sobol' prior sample — optionally use the *identical* tuples as Ch. 4
    if args.ch4_tuples:
        from .hagan_scan import sobol_params
        theta = sobol_params(args.n, seed=0)
        print(f"Using Ch.4 Sobol tuples (seed=0, |rho|nu²<1.9 filter): "
              f"{len(theta)} tuples retained")
    else:
        lb = np.array([PRIOR[n][0] for n in PARAM_NAMES])
        ub = np.array([PRIOR[n][1] for n in PARAM_NAMES])
        n2 = int(2 ** np.ceil(np.log2(args.n)))
        theta = lb + (ub - lb) * qmc.Sobol(d=4, scramble=True, seed=0).random(n2)[:args.n]

    print(f"Scanning {args.n} Sobol' tuples × {MATURITIES.size} maturities "
          f"= {args.n * MATURITIES.size} slices")

    if args.sweep_anchors:
        # Empirical study: how does repair quality depend on N_anchors?
        # Use a smaller n for speed.
        n_sweep = min(args.n, 128)
        theta_s = theta[:n_sweep]
        sweep_counts = [10, 15, 20, 30, 40, 60]
        sweep_results = []
        print(f"\nAnchor sweep ({n_sweep} tuples):")
        print(f"{'N_anchors':>10}  {'IV RMSE (med)':>14}  {'IV RMSE (p90)':>14}  "
              f"{'NNLS res (med)':>15}  {'NNLS res (max)':>15}")
        for N in sweep_counts:
            anch = make_anchors(N)
            s = run_scan(nn, theta_s, anch, torch, method='sanos')
            sweep_results.append(s)
            print(f"{N:>10}  {s['repair_iv_rmse']['median']:>14.2e}  "
                  f"{s['repair_iv_rmse']['p90']:>14.2e}  "
                  f"{s['repair_residual']['median']:>15.2e}  "
                  f"{s['repair_residual']['max']:>15.2e}")

        out_sweep = Path(args.out).parent / "sanos_anchor_sweep.json"
        out_sweep.parent.mkdir(parents=True, exist_ok=True)
        json.dump(sweep_results, open(out_sweep, "w"), indent=2)
        print(f"\nwrote {out_sweep}")
        return 0

    # Main scan — always include Hagan for the three-way comparison
    anchors = make_anchors(args.n_anchors)
    summary = run_scan(nn, theta, anchors, torch,
                       with_hagan=True, opt_v=args.opt_v, method=args.method)
    print_summary(summary)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
