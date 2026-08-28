"""
Comprehensive benchmark of all SABR pricing methods for Chapter 4.

Runs FEM, FD-ADI, MC (crude), MC (QMC), Hagan, Oblój, and DLV surrogate
on a Sobol parameter grid.  Reports accuracy (vs. tight FEM reference)
and timing for each method.

Usage:
    # Full run (needs NGSolve for FEM reference — run on your workstation)
    python run_benchmarks.py

    # Quick test with fewer parameter tuples
    python run_benchmarks.py --n-params 5

    # Skip FEM reference recompute if cached
    python run_benchmarks.py --cache-ref

Outputs:
    data/benchmarks/benchmark_results.json    Per-case results
    data/benchmarks/benchmark_summary.json    Aggregate tables for ch04
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

log = logging.getLogger("run_benchmarks")

PRIOR = {
    "beta": (0.1, 0.9),
    "rho": (-0.9, 0.1),
    "nu": (0.1, 1.0),
    "y0": (0.1, 0.5),
}
PARAM_NAMES = ["beta", "rho", "nu", "y0"]

# HMT grid
STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])


def sobol_params(n: int, seed: int = 0) -> np.ndarray:
    """Generate n parameter tuples via Sobol sequence on the prior box."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    m = max(1, int(np.ceil(np.log2(n))))
    u = sampler.random_base2(m)[:n]
    params = np.empty_like(u)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        params[:, j] = lo + u[:, j] * (hi - lo)
    return params


def make_sabr_params(row):
    from ..pricing.fem import SABRParams
    return SABRParams(
        beta=float(row[0]), rho=float(row[1]),
        nu=float(row[2]), y0=float(row[3]), x0=1.0,
    )


def compute_reference_fem(params_grid, strikes, maturities, cache_path=None):
    """Compute tight FEM reference prices for all parameter tuples."""
    from ..pricing.fem import FEConfig, SABRSolver

    if cache_path and Path(cache_path).exists():
        log.info("Loading cached reference from %s", cache_path)
        arc = np.load(cache_path)
        return arc["ref_prices"], arc["ref_iv"]

    ref_cfg = FEConfig(
        Rx=8.0, Ry=4.0, x_min=0.0, maxh=0.075,
        order=2, n_time=400, right_bc="payoff",
    )
    n = len(params_grid)
    ref_prices = np.full((n, len(strikes), len(maturities)), np.nan)
    ref_iv = np.full_like(ref_prices, np.nan)

    from ..black import implied_vol
    F = 1.0
    K_grid = np.broadcast_to(strikes[:, None], (len(strikes), len(maturities)))
    T_grid = np.broadcast_to(maturities[None, :], (len(strikes), len(maturities)))

    for i in range(n):
        t0 = time.perf_counter()
        sp = make_sabr_params(params_grid[i])
        try:
            solver = SABRSolver(sp, ref_cfg)
            prices, _ = solver.price_call_surface(strikes, maturities)
            ref_prices[i] = prices
            intr = np.maximum(F - K_grid, 0.0)
            prices_clip = np.clip(prices, intr, F)
            ref_iv[i] = implied_vol(
                np.full_like(prices, F), K_grid, T_grid, prices_clip
            )
        except Exception as e:
            log.warning("Reference FEM failed for param %d: %s", i, e)
        elapsed = time.perf_counter() - t0
        log.info("  Reference %d/%d done in %.1fs", i + 1, n, elapsed)

    if cache_path:
        np.savez_compressed(
            cache_path, ref_prices=ref_prices, ref_iv=ref_iv,
            params=params_grid, strikes=strikes, maturities=maturities,
        )
        log.info("Cached reference to %s", cache_path)

    return ref_prices, ref_iv


def run_method(name, func, params_grid, strikes, maturities, n_timing_runs=1):
    """Run a pricing method on all param tuples, return prices, ivs, timings.

    If the method returns (prices, extra), prices are stored and ivs are None.
    If the method returns (None, ivs), ivs are stored directly (IV-target NN).
    """
    n = len(params_grid)
    all_prices = np.full((n, len(strikes), len(maturities)), np.nan)
    all_ivs = np.full((n, len(strikes), len(maturities)), np.nan)
    timings = np.full(n, np.nan)
    has_direct_iv = False

    for i in range(n):
        sp = make_sabr_params(params_grid[i])
        best_time = float("inf")
        result = None
        for r in range(n_timing_runs):
            t0 = time.perf_counter()
            try:
                result = func(sp, strikes, maturities)
            except Exception as e:
                log.warning("  %s failed for param %d: %s", name, i, e)
                break
            elapsed = time.perf_counter() - t0
            best_time = min(best_time, elapsed)

        if result is not None:
            if isinstance(result, tuple) and result[0] is None:
                # IV-target model: (None, ivs)
                all_ivs[i] = result[1]
                has_direct_iv = True
            elif isinstance(result, tuple):
                all_prices[i] = result[0]
            else:
                all_prices[i] = result
        timings[i] = best_time if best_time < float("inf") else np.nan
        if (i + 1) % 10 == 0 or i == 0:
            log.info("  %s: %d/%d done (last %.2fs)", name, i + 1, n, timings[i])

    return all_prices, all_ivs if has_direct_iv else None, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-params", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-ref", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Method names to skip, e.g. --skip FEM 'MC (crude)'")
    ap.add_argument("--out-dir", default=str(HERE / "data" / "benchmarks"))
    ap.add_argument("--n-timing-runs", type=int, default=3)
    ap.add_argument("--nn-model", default=str(HERE / "data" / "nn_iv" / "nn_model.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strikes = STRIKES
    maturities = MATURITIES
    n_K, n_T = len(strikes), len(maturities)

    # Generate parameter grid
    params_grid = sobol_params(args.n_params, seed=args.seed)
    # Filter coercivity: |rho| * nu^2 < 1.5
    coercivity = np.abs(params_grid[:, 1]) * params_grid[:, 2] ** 2
    keep = coercivity < 1.5
    params_grid = params_grid[keep]
    log.info("Parameter grid: %d tuples (after coercivity filter)", len(params_grid))

    # Compute reference
    cache_path = out_dir / "ref_cache.npz" if args.cache_ref else None
    log.info("Computing tight FEM reference...")
    ref_prices, ref_iv = compute_reference_fem(
        params_grid, strikes, maturities, cache_path=cache_path,
    )

    from ..black import implied_vol
    F = 1.0
    K_grid = np.broadcast_to(strikes[:, None], (n_K, n_T))
    T_grid = np.broadcast_to(maturities[None, :], (n_K, n_T))

    # Define methods
    from ..pricing.fem import FEConfig, SABRSolver
    from ..pricing.finite_diff import fd_call_surface, FDConfig
    from ..pricing.montecarlo import (
        mc_call_surface, MCConfig, mc_call_surface_qmc, QMCConfig,
    )
    from ..pricing.hagan import hagan_call_surface, obloj_call_surface

    fe_cfg = FEConfig(
        Rx=8.0, Ry=4.0, x_min=0.0, maxh=0.15,
        order=2, n_time=200, right_bc="payoff",
    )
    fd_cfg = FDConfig(nx=161, ny=121, n_steps_per_year=200)
    mc_cfg = MCConfig(n_paths=200_000, n_steps_per_year=200)
    qmc_cfg = QMCConfig(n_paths=65_536, n_steps_per_year=200)

    # Timing runs: fast methods get multiple, slow methods get 1
    timing_runs = {
        "FEM": 1,
        "FD-ADI": 1,
        "MC (crude)": 1,
        "MC (QMC)": 1,
        "Hagan": args.n_timing_runs,
        "Obloj": args.n_timing_runs,
        "DLV surrogate": args.n_timing_runs,
    }

    def fem_func(sp, K, T):
        solver = SABRSolver(sp, fe_cfg)
        return solver.price_call_surface(K, T)

    def fd_func(sp, K, T):
        return fd_call_surface(sp, K, T, fd_cfg)

    def mc_func(sp, K, T):
        prices, report = mc_call_surface(sp, K, T, mc_cfg)
        return prices, report

    def qmc_func(sp, K, T):
        prices, report = mc_call_surface_qmc(sp, K, T, qmc_cfg)
        return prices, report

    def hagan_func(sp, K, T):
        return hagan_call_surface(sp, K, T)

    def obloj_func(sp, K, T):
        return obloj_call_surface(sp, K, T)

    # NN surrogate
    nn_func = None
    model_path = Path(args.nn_model)
    config_path = model_path.parent / "nn_config.json"
    if model_path.exists() and config_path.exists():
        import torch
        nn_cfg = json.loads(config_path.read_text())
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        from ..calibration.calibrate import build_mlp, theta_to_x
        model = build_mlp(
            torch, 4, nn_cfg["n_strikes"] * nn_cfg["n_maturities"],
            nn_cfg["hidden"], nn_cfg["depth"],
        ).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        nn_target = nn_cfg.get("target", "price")

        def nn_func(sp, K, T):
            theta = np.array([sp.beta, sp.rho, sp.nu, sp.y0])
            x = theta_to_x(theta.reshape(1, -1)).astype(np.float32)
            with torch.no_grad():
                x_t = torch.from_numpy(x).to(device)
                out = model(x_t).cpu().numpy().reshape(n_K, n_T)
            if nn_target == "iv":
                # Return (None, ivs) to signal that output is IV, not prices
                return None, out
            return out, None

        log.info("NN surrogate loaded from %s (device=%s, target=%s)",
                 model_path, device, nn_target)
    else:
        log.warning("NN model not found at %s; skipping DLV surrogate", model_path)

    methods = {
        "FEM": fem_func,
        "FD-ADI": fd_func,
        "MC (crude)": mc_func,
        "MC (QMC)": qmc_func,
        "Hagan": hagan_func,
        "Obloj": obloj_func,
    }
    if nn_func is not None:
        methods["DLV surrogate"] = nn_func

    # Filter out skipped methods
    for s in args.skip:
        if s in methods:
            del methods[s]
            log.info("Skipping %s", s)

    # Load previous results if they exist (so skipped methods keep their data)
    summary_path = out_dir / "benchmark_summary.json"
    if summary_path.exists():
        results = json.loads(summary_path.read_text())
    else:
        results = {}

    for name, func in methods.items():
        n_runs = timing_runs.get(name, 1)
        log.info("Running %s (%d timing run(s) per param)...", name, n_runs)
        prices, direct_ivs, timings = run_method(
            name, func, params_grid, strikes, maturities,
            n_timing_runs=n_runs,
        )

        if direct_ivs is not None:
            # IV-target model: IVs are already computed, derive prices via BS
            ivs = direct_ivs
            from ..black import bs_call_price
            prices = np.full_like(ivs, np.nan)
            for i in range(len(params_grid)):
                if np.all(np.isfinite(ivs[i])):
                    prices[i] = bs_call_price(F, K_grid, T_grid, np.maximum(ivs[i], 1e-8))
        else:
            # Price-target: invert prices to IVs
            ivs = np.full_like(prices, np.nan)
            for i in range(len(params_grid)):
                if np.all(np.isfinite(prices[i])):
                    intr = np.maximum(F - K_grid, 0.0)
                    p_clip = np.clip(prices[i], intr, F)
                    ivs[i] = implied_vol(
                        np.full_like(p_clip, F), K_grid, T_grid, p_clip,
                    )

        # Compute errors vs reference
        price_err = prices - ref_prices
        iv_err = ivs - ref_iv
        finite_price = np.isfinite(price_err)
        finite_iv = np.isfinite(iv_err)

        price_rmse_per_sample = np.sqrt(
            np.nanmean(price_err ** 2, axis=(1, 2))
        )
        iv_rmse_per_sample = np.sqrt(
            np.nanmean(iv_err ** 2, axis=(1, 2))
        )

        summary = {
            "name": name,
            "n_valid": int(np.sum(np.all(np.isfinite(prices.reshape(len(params_grid), -1)), axis=1))),
            "timing_median_s": float(np.nanmedian(timings)),
            "timing_mean_s": float(np.nanmean(timings)),
            "timing_p95_s": float(np.nanpercentile(timings, 95)),
            "price_rmse_median": float(np.nanmedian(price_rmse_per_sample)),
            "price_rmse_p90": float(np.nanpercentile(price_rmse_per_sample, 90)),
            "price_rmse_max": float(np.nanmax(price_rmse_per_sample)),
            "iv_rmse_median": float(np.nanmedian(iv_rmse_per_sample)),
            "iv_rmse_p90": float(np.nanpercentile(iv_rmse_per_sample, 90)),
            "iv_rmse_max": float(np.nanmax(iv_rmse_per_sample)),
        }

        results[name] = summary

        # Save per-sample data for plotting
        np.savez_compressed(
            out_dir / f"bench_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz",
            prices=prices.astype(np.float32),
            ivs=ivs.astype(np.float32),
            iv_rmse_per_sample=iv_rmse_per_sample.astype(np.float32),
            price_rmse_per_sample=price_rmse_per_sample.astype(np.float32),
            timings=timings.astype(np.float32),
        )

        log.info("  %s: price RMSE median=%.2e, IV RMSE median=%.2e, time=%.4fs",
                 name, summary["price_rmse_median"], summary["iv_rmse_median"],
                 summary["timing_median_s"])

    # Save
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    (out_dir / "benchmark_params.json").write_text(
        json.dumps({
            "n_params": len(params_grid),
            "params": params_grid.tolist(),
            "strikes": strikes.tolist(),
            "maturities": maturities.tolist(),
            "fe_config": {"maxh": fe_cfg.maxh, "n_time": fe_cfg.n_time, "order": fe_cfg.order},
            "fd_config": {"nx": fd_cfg.nx, "ny": fd_cfg.ny, "n_steps_per_year": fd_cfg.n_steps_per_year},
            "mc_config": {"n_paths": mc_cfg.n_paths},
            "qmc_config": {"n_paths": qmc_cfg.n_paths},
        }, indent=2)
    )
    log.info("Wrote results to %s", out_dir)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<16} {'Price RMSE':>12} {'IV RMSE':>12} "
          f"{'Price p90':>12} {'IV p90':>12} {'Time (s)':>10}")
    print("-" * 90)
    for name, s in results.items():
        print(f"{name:<16} {s['price_rmse_median']:>12.2e} {s['iv_rmse_median']:>12.2e} "
              f"{s['price_rmse_p90']:>12.2e} {s['iv_rmse_p90']:>12.2e} "
              f"{s['timing_median_s']:>10.4f}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    sys.exit(main())
