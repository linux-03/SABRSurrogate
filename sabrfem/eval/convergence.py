"""
Empirical convergence and timing study for the FE-SABR solver.

Produces the evidence needed to defend the substitution of nodal P2 elements
for HR (2018)'s biorthogonal wavelet basis:

  (1) Convergence rate on a fixed parameter tuple. We solve at decreasing
      maxh values and measure the L2 price-error against the finest grid as
      reference. For the weighted SABR operator the rate is expected to be
      singularity-limited, roughly O(h^(2(1-beta))) in L2 rather than the
      smooth P2 rate O(h^3); if we see that rate, the post-lift discrete
      problem is resolving the degeneracy correctly.

  (2) Per-solve wall time at each mesh resolution. Demonstrates that the
      P2-on-graded-mesh implementation runs at comparable cost to HR's
      reported wavelet timings (see HR 2018, Table 6.1).

  (3) Convergence in mesh-grading exponent. We hold maxh fixed and vary
      the grading exponent gamma to confirm that the choice gamma = 0.4 is
      well-inside the regime where the boundary singularity at F = 0 is
      properly resolved.

All three results are produced for the same fixed reference parameter tuple
so the analysis is self-consistent.

Usage:
    python convergence_study.py
    python convergence_study.py --params beta=0.5,rho=-0.3,nu=0.6,y0=0.3
    python convergence_study.py --skip-grading      # just (1) + (2)

Outputs:
    data/convergence/h_study.json          mesh-refinement convergence
    data/convergence/grading_study.json    grading-exponent sensitivity
    data/convergence/summary.txt           ASCII tables for the dissertation
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

from ..pricing.fem import FEConfig, SABRParams, SABRSolver  # noqa: E402

log = logging.getLogger("convergence_study")

# Reference parameter tuple — chosen mid-prior, well-conditioned, no extreme nu.
DEFAULT_PARAMS = {
    "beta": 0.5,
    "rho": -0.3,
    "nu": 0.6,
    "y0": 0.3,
    "x0": 1.0,
}

# A single representative (K, T) tuple for the headline number, plus a small
# diagnostic grid for the surface error.
HEADLINE_K = 1.0
HEADLINE_T = 1.0
DIAG_STRIKES = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
DIAG_MATURITIES = np.array([0.1, 0.6, 1.0, 2.0])


def parse_params(s: str) -> dict:
    out = dict(DEFAULT_PARAMS)
    for tok in s.split(","):
        if "=" not in tok:
            continue
        k, v = tok.split("=")
        out[k.strip()] = float(v)
    return out


def run_one(params_dict, fe_kwargs):
    """Run a single FE solve and return (price_grid, wall_seconds)."""
    params = SABRParams(**params_dict)
    cfg = FEConfig(**fe_kwargs)
    t0 = time.perf_counter()
    solver = SABRSolver(params, cfg)
    prices, _ = solver.price_call_surface(DIAG_STRIKES, DIAG_MATURITIES)
    wall = time.perf_counter() - t0
    return prices, wall


def solve_grid(params_dict, fe_kwargs, strikes, maturities):
    """Run one FE solve on an arbitrary strike/maturity grid."""
    params = SABRParams(**params_dict)
    cfg = FEConfig(**fe_kwargs)
    solver = SABRSolver(params, cfg)
    t0 = time.perf_counter()
    prices, _ = solver.price_call_surface(strikes, maturities)
    wall = time.perf_counter() - t0
    return prices, wall


def estimate_rate(maxh_values, error_values):
    """Estimate a power-law rate from log-log data."""
    xs = np.asarray(maxh_values, dtype=float)
    ys = np.asarray(error_values, dtype=float)
    mask = (xs > 0.0) & (ys > 0.0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return None
    slope, _ = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(slope)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="", help="comma-separated param overrides")
    ap.add_argument("--out-dir", default=str(HERE / "data" / "convergence"))
    ap.add_argument("--skip-grading", action="store_true")
    ap.add_argument(
        "--maxh-values", default="0.30,0.22,0.15,0.10,0.075,0.055",
        help="comma-separated maxh values, finest last (used as reference)",
    )
    ap.add_argument(
        "--grading-values", default="0.20,0.30,0.40,0.50,0.70,1.00",
        help="comma-separated grading exponents (1.0 = uniform)",
    )
    ap.add_argument(
        "--reference-maxh", type=float, default=0.05,
        help="mesh width for the dedicated reference solve",
    )
    ap.add_argument(
        "--reference-n-time", type=int, default=800,
        help="time steps per unit maturity for the dedicated reference solve",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    params_dict = parse_params(args.params) if args.params else dict(DEFAULT_PARAMS)
    log.info(
        "Reference params: beta=%.3f rho=%+.3f nu=%.3f y0=%.3f x0=%.2f",
        params_dict["beta"], params_dict["rho"], params_dict["nu"],
        params_dict["y0"], params_dict["x0"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_cfg = {
        "Rx": 8.0,
        "Ry": 4.0,
        "x_min": 1e-3,
        "maxh": float(args.reference_maxh),
        "order": 2,
        "n_time": int(args.reference_n_time),
        "theta": 0.5,
        "grade": 0.4,
        "right_bc": "payoff",
    }

    # -----------------------------------------------------------------------
    # Study 1: mesh refinement convergence (h-study)
    # -----------------------------------------------------------------------
    maxh_values = sorted(
        [float(v) for v in args.maxh_values.split(",")], reverse=True
    )
    log.info("=" * 60)
    log.info("Study 1: mesh refinement convergence")
    log.info("=" * 60)

    log.info(
        "Reference solve: maxh=%.3f, n_time=%d",
        reference_cfg["maxh"], reference_cfg["n_time"],
    )
    ref_prices, ref_wall = solve_grid(
        params_dict, reference_cfg, DIAG_STRIKES, DIAG_MATURITIES
    )
    log.info("Reference solve done in %.2f s", ref_wall)

    h_results = []
    for maxh in maxh_values:
        cfg_kwargs = {
            "Rx": 8.0, "Ry": 4.0, "x_min": 1e-3,
            "maxh": maxh, "order": 2, "n_time": 200,
            "theta": 0.5, "grade": 0.4,
            "right_bc": "payoff",
        }
        log.info("Solving at maxh=%.3f ...", maxh)
        try:
            prices, wall = run_one(params_dict, cfg_kwargs)
            h_results.append({
                "maxh": maxh,
                "wall_seconds": wall,
                "prices": prices.tolist(),
                "ok": True,
            })
            log.info("  done in %.2f s, headline price (K=1, T=1) = %.6f",
                     wall, prices[2, 2])
        except Exception as exc:  # noqa: BLE001
            log.error("  FAILED: %s", exc)
            h_results.append({
                "maxh": maxh, "ok": False, "error": str(exc),
            })

    # Use finest successful solve as reference; compute L2 grid error vs it.
    successful = [r for r in h_results if r["ok"]]
    for r in h_results:
        if not r["ok"]:
            r["L2_err_vs_reference"] = None
            r["Linf_err_vs_reference"] = None
            continue
        this = np.asarray(r["prices"])
        r["L2_err_vs_reference"] = float(np.sqrt(np.mean((this - ref_prices) ** 2)))
        r["Linf_err_vs_reference"] = float(np.max(np.abs(this - ref_prices)))

    # Empirical convergence rate from successive coarser meshes
    rates = []
    for a, b in zip(successful[:-1], successful[1:]):
        h_ratio = a["maxh"] / b["maxh"]
        if a.get("L2_err_vs_reference") and b.get("L2_err_vs_reference"):
            err_ratio = a["L2_err_vs_reference"] / max(b["L2_err_vs_reference"], 1e-16)
            rate = float(np.log(err_ratio) / np.log(h_ratio))
            rates.append({"from_h": a["maxh"], "to_h": b["maxh"], "rate": rate})

    expected_rate_L2 = 2.0 * (1.0 - params_dict["beta"])
    usable_h = [r["maxh"] for r in successful if r.get("L2_err_vs_finest") not in (None, 0.0)]
    usable_err = [r["L2_err_vs_finest"] for r in successful if r.get("L2_err_vs_finest") not in (None, 0.0)]
    estimated_rate_L2 = estimate_rate(usable_h, usable_err)
    rate_margin = 0.25
    rate_pass = (
        estimated_rate_L2 is not None
        and estimated_rate_L2 >= expected_rate_L2 - rate_margin
    )

    h_study = {
        "params": params_dict,
        "reference_config": reference_cfg,
        "config_template": {
            "Rx": 8.0, "Ry": 4.0, "order": 2, "n_time": 200,
            "grade": 0.4, "right_bc": "payoff",
        },
        "diag_strikes": DIAG_STRIKES.tolist(),
        "diag_maturities": DIAG_MATURITIES.tolist(),
        "reference_prices": ref_prices.tolist(),
        "results": h_results,
        "rates_L2_vs_reference": rates,
        "expected_rate_L2": expected_rate_L2,
        "estimated_rate_L2_loglog": estimated_rate_L2,
        "rate_margin": rate_margin,
        "rate_pass": rate_pass,
    }
    (out_dir / "h_study.json").write_text(json.dumps(h_study, indent=2))
    log.info("Wrote %s", out_dir / "h_study.json")

    # -----------------------------------------------------------------------
    # Study 2: grading exponent sensitivity
    # -----------------------------------------------------------------------
    grading_results = []
    if not args.skip_grading:
        grading_values = [float(v) for v in args.grading_values.split(",")]
        log.info("=" * 60)
        log.info("Study 2: grading exponent sensitivity (maxh fixed at 0.15)")
        log.info("=" * 60)

        for grade in grading_values:
            cfg_kwargs = {
                "Rx": 8.0, "Ry": 4.0, "x_min": 1e-3,
                "maxh": 0.15, "order": 2, "n_time": 200,
                "theta": 0.5, "grade": grade,
                "right_bc": "payoff",
            }
            log.info("Solving at grade=%.2f ...", grade)
            try:
                prices, wall = run_one(params_dict, cfg_kwargs)
                grading_results.append({
                    "grade": grade,
                    "wall_seconds": wall,
                    "prices": prices.tolist(),
                    "ok": True,
                })
                log.info("  done in %.2f s, headline price = %.6f",
                         wall, prices[2, 2])
            except Exception as exc:  # noqa: BLE001
                log.error("  FAILED: %s", exc)
                grading_results.append({
                    "grade": grade, "ok": False, "error": str(exc),
                })

        # Reference = the finest mesh from study 1, same params, evaluated above.
        if successful:
            ref = np.asarray(successful[-1]["prices"])
            for r in grading_results:
                if not r["ok"]:
                    continue
                this = np.asarray(r["prices"])
                r["L2_err_vs_finest_h"] = float(
                    np.sqrt(np.mean((this - ref) ** 2))
                )

    grading_study = {
        "params": params_dict,
        "config_template": {
            "Rx": 8.0, "Ry": 4.0, "order": 2, "n_time": 200,
            "maxh": 0.15, "right_bc": "payoff",
        },
        "results": grading_results,
        "interpretation": (
            "L2_err_vs_finest_h compares each grading choice against the "
            "finest h-study reference. A flat-or-decreasing curve as grade "
            "decreases from 1.0 (uniform) toward 0.4 confirms the boundary "
            "singularity at F=0 is being resolved correctly. A re-increase "
            "for very small grade (<0.2) would indicate over-grading wastes "
            "DOFs."
        ),
    }
    (out_dir / "grading_study.json").write_text(json.dumps(grading_study, indent=2))
    log.info("Wrote %s", out_dir / "grading_study.json")

    # -----------------------------------------------------------------------
    # Pretty-printed summary for the dissertation
    # -----------------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("CONVERGENCE STUDY SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Reference params: {params_dict}")
    lines.append("")
    lines.append("Study 1: mesh refinement (h-study)")
    lines.append("-" * 70)
    lines.append(
        f"  Reference solve: maxh={reference_cfg['maxh']:.3f}, n_time={reference_cfg['n_time']}"
    )
    lines.append(
        f"  {'maxh':>6}  {'wall (s)':>10}  {'L2 err':>12}  "
        f"{'Linf err':>12}  {'price K=1 T=1':>14}"
    )
    for r in h_results:
        if r["ok"]:
            l2 = r.get("L2_err_vs_reference")
            li = r.get("Linf_err_vs_reference")
            ph = r["prices"][2][2]  # headline price
            l2s = f"{l2:.3e}" if l2 is not None else "n/a"
            lis = f"{li:.3e}" if li is not None else "n/a"
            lines.append(
                f"  {r['maxh']:>6.3f}  {r['wall_seconds']:>10.2f}  "
                f"{l2s:>12}  {lis:>12}  {ph:>14.6f}"
            )
        else:
            lines.append(f"  {r['maxh']:>6.3f}  FAILED ({r.get('error','?')})")
    lines.append("")
    lines.append("  Empirical convergence rates (L2 vs reference):")
    for rt in rates:
        lines.append(
            f"    h={rt['from_h']:.3f} -> h={rt['to_h']:.3f}: rate = {rt['rate']:.2f}"
        )
    lines.append(
        f"  Log-log fit rate: {estimated_rate_L2:.2f}" if estimated_rate_L2 is not None else "  Log-log fit rate: n/a"
    )
    lines.append(
        f"  Expected singularity-limited L2 rate: 2*(1-beta) = {expected_rate_L2:.2f}"
    )
    lines.append(
        f"  Rate check: {'PASS' if rate_pass else 'FAIL'} (margin {rate_margin:.2f})"
    )
    lines.append("")
    if grading_results:
        lines.append("Study 2: grading exponent sensitivity (maxh = 0.15)")
        lines.append("-" * 70)
        lines.append(
            f"  {'grade':>6}  {'wall (s)':>10}  {'L2 err vs finest h':>20}  "
            f"{'price K=1 T=1':>14}"
        )
        for r in grading_results:
            if r["ok"]:
                err = r.get("L2_err_vs_finest_h")
                errs = f"{err:.3e}" if err is not None else "n/a"
                lines.append(
                    f"  {r['grade']:>6.2f}  {r['wall_seconds']:>10.2f}  "
                    f"{errs:>20}  {r['prices'][2][2]:>14.6f}"
                )
            else:
                lines.append(f"  {r['grade']:>6.2f}  FAILED")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary)
    log.info("Wrote %s", out_dir / "summary.txt")
    log.info("\n%s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
