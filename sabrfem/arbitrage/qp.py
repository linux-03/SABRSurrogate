#!/usr/bin/env python3
"""Joint-surface arbitrage repair for the DLV surrogate (sec. 4.4).

Minimal-perturbation projection of a full SABR call-price surface onto the
*static-arbitrage-free* set: the closest surface (inverse-vega-weighted L2)
to the surrogate output that satisfies, jointly over all n_K x n_T cells,

    convexity in strike      C[i-1,j] - 2 C[i,j] + C[i+1,j] >= 0   (butterfly)
    monotone decreasing in K  C[i,j]   - C[i+1,j]           >= 0
    calendar monotonicity     C[i,j+1] - C[i,j]             >= 0   (across T)
    static bounds             (F-K)^+ <= C[i,j] <= F .

All constraints are linear, so this is a convex QP (identity Hessian) solved
by SLSQP. If the surrogate surface is already arbitrage-free the projection
returns it unchanged (zero perturbation): the repair acts only where a
violation exists. Unlike the per-slice SANOS repair, this enforces the
calendar constraints across maturities, so the repaired surface is free of
butterfly AND calendar AND bound arbitrage by construction.

Usage:
    python arbfree_surface_qp.py --selftest        # synthetic check, no torch
    python arbfree_surface_qp.py --ch4-tuples --n 512   # scan the surrogate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
F = 1.0
TOL = 1e-12


# ----------------------------------------------------------------------------
# core projection
# ----------------------------------------------------------------------------
def _bs_vega(iv, K, T, F=1.0):
    iv = np.maximum(np.asarray(iv, float), 1e-8)
    T = np.asarray(T, float)
    sw = iv * np.sqrt(np.maximum(T, 1e-12))                      # (nK, nT)
    d1 = (np.log(F / np.asarray(K, float)[:, None]) + 0.5 * sw ** 2) / np.where(sw > 0, sw, 1.0)
    return F * norm.pdf(d1) * np.sqrt(np.maximum(T, 1e-12))


def vega_weights(iv, K, T, F=1.0, eps=1e-3):
    """Inverse-vega-squared cell weights so the price objective approximates
    sum (delta_sigma)^2 (matches the per-slice SANOS repair's weighting)."""
    with np.errstate(all="ignore"):
        v = _bs_vega(iv, K, T, F)
    v = np.nan_to_num(v, nan=eps, posinf=1e12, neginf=eps)
    return np.clip(1.0 / np.maximum(v, eps) ** 2, 0.0, 1e8)


def repair_surface_qp(C_surr, K, T, weights=None, F=1.0, maxiter=500):
    """Project a (n_K, n_T) call surface onto the static-arbitrage-free set.

    Returns (C_repaired, perturbation_L2, success).
    """
    C0 = np.nan_to_num(np.asarray(C_surr, float), nan=0.0, posinf=F, neginf=0.0)
    nK, nT = C0.shape
    K = np.asarray(K, float)
    intr = np.maximum(F - K, 0.0)
    C0 = np.clip(C0, intr[:, None], F)
    n = nK * nT
    x0 = C0.flatten(order="F")                                  # x[j*nK + i] = C[i, j]
    W = np.ones((nK, nT)) if weights is None else np.asarray(weights, float)
    wflat = np.clip(np.nan_to_num(W.flatten(order="F"), nan=1.0, posinf=1e8, neginf=1.0), 0.0, 1e8)
    idx = lambda i, j: j * nK + i                               # noqa: E731

    fun = lambda x: 0.5 * np.sum(wflat * (x - x0) ** 2)         # noqa: E731
    jac = lambda x: wflat * (x - x0)                            # noqa: E731

    rows = []
    for j in range(nT):                                         # butterfly: convex in K
        for i in range(1, nK - 1):
            a = np.zeros(n); a[idx(i - 1, j)] = 1; a[idx(i, j)] = -2; a[idx(i + 1, j)] = 1
            rows.append(a)
    for j in range(nT):                                         # monotone decreasing in K
        for i in range(nK - 1):
            a = np.zeros(n); a[idx(i, j)] = 1; a[idx(i + 1, j)] = -1
            rows.append(a)
    for j in range(nT - 1):                                     # calendar: C[i,j+1] >= C[i,j]
        for i in range(nK):
            a = np.zeros(n); a[idx(i, j + 1)] = 1; a[idx(i, j)] = -1
            rows.append(a)
    A = np.asarray(rows)                                        # (m, n): A x >= 0
    cons = [{"type": "ineq", "fun": (lambda x: A @ x), "jac": (lambda x: A)}]
    bounds = [(intr[i], F) for j in range(nT) for i in range(nK)]

    x0c = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    with np.errstate(all="ignore"):
        res = minimize(fun, x0c, jac=jac, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
    xr = res.x if res.success else x0c
    return xr.reshape((nK, nT), order="F"), float(np.linalg.norm(xr - x0)), bool(res.success)


# ----------------------------------------------------------------------------
# diagnostics (butterfly + calendar + bounds), per surface
# ----------------------------------------------------------------------------
def surface_stats(C, K):
    C = np.asarray(C, float)
    K = np.asarray(K, float)
    dK = K[1] - K[0]
    d2 = C[:-2, :] - 2 * C[1:-1, :] + C[2:, :]                  # (nK-2, nT)
    dens = d2 / dK ** 2
    cal = C[:, 1:] - C[:, :-1]                                  # (nK, nT-1)
    bf_slices = (d2 < -TOL).any(axis=0)                         # per maturity
    return {
        "bf_slice_flags": bf_slices,                           # bool (nT,)
        "calendar_any": bool((cal < -TOL).any()),
        "calendar_cells": int((cal < -TOL).sum()),
        "min_density": float(dens.min()),
        "neg_mass_slice_max": float(np.max(np.sum(np.maximum(-dens, 0.0), axis=0) * dK)),
    }


# ----------------------------------------------------------------------------
# self-test (no torch required)
# ----------------------------------------------------------------------------
def _bs_call(K, var, F=1.0):
    K = np.asarray(K, float); sw = np.sqrt(np.maximum(var, 1e-300))
    d1 = (np.log(F / K) + 0.5 * var) / sw; d2 = d1 - sw
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def selftest():
    K, T = STRIKES, MATURITIES
    C = np.column_stack([_bs_call(K, 0.2 ** 2 * t) for t in T])     # clean, arb-free
    _, p, ok = repair_surface_qp(C, K, T)
    print(f"[clean]    projection perturbation = {p:.2e}  success={ok}  (expect 0)")
    Cv = C.copy()
    Cv[2, 1] += 0.012; Cv[3, 1] += 0.004                            # butterfly dent
    Cv[7, 4] = Cv[7, 3] - 0.01                                      # calendar inversion
    s0 = surface_stats(Cv, K)
    Cr, p, ok = repair_surface_qp(Cv, K, T)
    s1 = surface_stats(Cr, K)
    print(f"[violated] min_density={s0['min_density']:.3e}  calendar_cells={s0['calendar_cells']}")
    print(f"[repaired] min_density={s1['min_density']:.2e}  calendar_cells={s1['calendar_cells']}  "
          f"pert={p:.2e}  cells_moved={int((np.abs(Cr - Cv) > 1e-5).sum())}/{Cv.size}")
    assert s1["min_density"] > -1e-9 and s1["calendar_cells"] == 0, "repair failed"
    print("OK: butterfly and calendar both repaired to machine precision.")


# ----------------------------------------------------------------------------
# surrogate scan (needs torch + project modules)
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--model", default=str(HERE / "data" / "nn_iv" / "nn_model.pt"))
    ap.add_argument("--ch4-tuples", action="store_true",
                    help="identical Sobol tuples as Ch.4 (seed=0, |rho|nu^2<1.9).")
    ap.add_argument("--out", default=str(HERE / "data" / "arbitrage" / "qp_surface_repair_summary.json"))
    args = ap.parse_args()

    if args.selftest:
        selftest(); return 0

    # lazy imports so --selftest needs no torch / project deps
    import torch
    from ..calibration.calibrate import load_nn, theta_to_x, PRIOR, PARAM_NAMES
    from ..black import implied_vol
    from .hagan_scan import call_surface_from_iv
    from scipy.stats import qmc

    cfg_path = Path(args.model).parent / "nn_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    nn = load_nn(torch, Path(args.model), cfg.get("hidden", 30), cfg.get("depth", 4),
                 STRIKES.size, MATURITIES.size, cfg.get("target", "iv"),
                 STRIKES, MATURITIES, torch.device("cpu"))

    if args.ch4_tuples:
        from .hagan_scan import sobol_params
        theta = sobol_params(args.n, seed=0)
    else:
        lb = np.array([PRIOR[n][0] for n in PARAM_NAMES])
        ub = np.array([PRIOR[n][1] for n in PARAM_NAMES])
        n2 = int(2 ** np.ceil(np.log2(args.n)))
        theta = lb + (ub - lb) * qmc.Sobol(d=4, scramble=True, seed=0).random(n2)[:args.n]

    nT = MATURITIES.size
    n_slice = 0
    raw_bf = rep_bf = 0
    raw_cal = rep_cal = 0                      # tuples with ANY calendar violation
    raw_minp, rep_minp, raw_negm, rep_negm, iv_dev = [], [], [], [], []
    n_ok = 0

    for th in theta:
        x = torch.tensor(theta_to_x(th), dtype=torch.float32)
        iv = nn.forward_iv(x).detach().numpy()
        iv = np.clip(np.where(np.isfinite(iv), iv, 0.3), 1e-4, 5.0)
        C = call_surface_from_iv(iv, STRIKES, MATURITIES, F)
        W = vega_weights(iv, STRIKES, MATURITIES, F)
        Crep, _, ok = repair_surface_qp(C, STRIKES, MATURITIES, weights=W, F=F)
        n_ok += ok

        s0, s1 = surface_stats(C, STRIKES), surface_stats(Crep, STRIKES)
        n_slice += nT
        raw_bf += int(s0["bf_slice_flags"].sum()); rep_bf += int(s1["bf_slice_flags"].sum())
        raw_cal += int(s0["calendar_any"]);        rep_cal += int(s1["calendar_any"])
        raw_minp.append(s0["min_density"]); rep_minp.append(s1["min_density"])
        raw_negm.append(s0["neg_mass_slice_max"]); rep_negm.append(s1["neg_mass_slice_max"])
        for j, T in enumerate(MATURITIES):
            iv_rep = implied_vol(np.full_like(STRIKES, F), STRIKES,
                                 np.full_like(STRIKES, T), Crep[:, j])
            m = np.isfinite(iv_rep) & np.isfinite(iv[:, j])
            if m.any():
                iv_dev.append(float(np.sqrt(np.mean((iv_rep[m] - iv[m, j]) ** 2))))

    nt = len(theta)
    summary = {
        "n_tuples": nt, "n_slices": n_slice, "repair_success_rate": n_ok / nt,
        "butterfly_slice_pct": {"raw": 100 * raw_bf / n_slice, "repaired": 100 * rep_bf / n_slice},
        "calendar_tuple_pct":  {"raw": 100 * raw_cal / nt,     "repaired": 100 * rep_cal / nt},
        "min_density": {"raw_min": float(np.min(raw_minp)), "repaired_min": float(np.min(rep_minp))},
        "neg_mass":    {"raw_max": float(np.max(raw_negm)), "repaired_max": float(np.max(rep_negm))},
        "repair_iv_rmse": {"median": float(np.median(iv_dev)),
                           "p90": float(np.percentile(iv_dev, 90)),
                           "max": float(np.max(iv_dev))},
    }
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
