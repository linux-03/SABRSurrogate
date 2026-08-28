"""
Arbitrage-violation analysis of the Hagan and Obloj implied-volatility
formulas across the parameter prior of the Chapter 4 benchmark.

For each parameter tuple drawn from the Sobol prior used in
run_benchmarks.py, evaluate three classes of no-arbitrage indicators on a
fine (K, T) analysis grid:

  1. Butterfly:  d^2 C / dK^2 < 0  (negative Breeden-Litzenberger density)
  2. Calendar:   dC/dT < 0          (zero-rate calendar arbitrage)
  3. Bounds:     C < (F - K)+ or C > F  (model-free arbitrage bounds)

Severity:
  - per-surface violation rate  phi  = #{violating cells} / #cells
  - per-surface negative mass    M_neg = int max(-p, 0) dK dT
  - per-tuple any-violation flag A

Aggregated across the prior, we report the violation rates and severity
distributions in JSON form for ingestion into Chapter 5, plus two figures:

  figures/hagan_density_examples.pdf  -- representative density slices
  figures/hagan_violation_heatmap.pdf -- per-(K,T) violation frequency

Usage:
    python hagan_arbitrage.py
    python hagan_arbitrage.py --n-params 200 --formula obloj
    python hagan_arbitrage.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

log = logging.getLogger("hagan_arbitrage")

# Same prior as run_benchmarks.py and generate_dataset_parallel.py
PRIOR = {
    "beta": (0.1, 0.9),
    "rho": (-0.9, 0.1),
    "nu": (0.1, 1.45),     # matches the augmented training prior
    "y0": (0.1, 0.5),
}
PARAM_NAMES = ["beta", "rho", "nu", "y0"]


def sobol_params(n: int, seed: int = 0) -> np.ndarray:
    """Sobol prior over the parameter box, with |rho|*nu^2 < 1.9 filter.

    The 1.9 cap matches the training-prior coercivity guard used to generate
    the FEM dataset (Horvath-Reichmann bound is |rho|*nu^2 < 2; 1.9 keeps a
    margin), so the Hagan/FEM scans sample the exact region the surrogate is
    trained on.
    """
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    m = max(1, int(np.ceil(np.log2(n))))
    u = sampler.random_base2(m)[:n]
    params = np.empty_like(u)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        params[:, j] = lo + u[:, j] * (hi - lo)
    coercivity = np.abs(params[:, 1]) * params[:, 2] ** 2
    return params[coercivity < 1.9]


def hagan_iv_surface(
    params_row: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    formula: str = "hagan",
) -> np.ndarray:
    """Evaluate Hagan or Obloj IV on an outer-product (K, T) grid.

    Returns sigma shape (len(K), len(T)).
    """
    from ..pricing.fem import SABRParams
    from ..pricing.hagan import hagan_implied_vol, obloj_implied_vol

    sp = SABRParams(
        beta=float(params_row[0]),
        rho=float(params_row[1]),
        nu=float(params_row[2]),
        y0=float(params_row[3]),
        x0=1.0,
    )
    fn = obloj_implied_vol if formula == "obloj" else hagan_implied_vol
    return fn(sp, K, T)


def call_surface_from_iv(iv: np.ndarray, K: np.ndarray, T: np.ndarray,
                         F: float = 1.0) -> np.ndarray:
    """Black-Scholes call prices on the (K, T) outer product."""
    from ..black import bs_call_price

    K_grid = K[:, None] * np.ones_like(T)[None, :]
    T_grid = np.ones_like(K)[:, None] * T[None, :]
    F_arr = np.full_like(K_grid, F)
    return bs_call_price(F_arr, K_grid, T_grid, iv)


def butterfly_density(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Second-difference Breeden-Litzenberger density along the strike axis.

    Returns p of shape (len(K)-2, len(T)) on the interior strike nodes.
    Standard three-point central second difference; the strike grid is
    uniform, so the formula reduces to (C[i+1] - 2 C[i] + C[i-1]) / h^2.
    """
    h = K[1] - K[0]
    return (C[2:, :] - 2.0 * C[1:-1, :] + C[:-2, :]) / (h * h)


def calendar_derivative(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """First-difference dC/dT on the (interior) maturity grid.

    Returns dCdT of shape (len(K), len(T)-1) at midpoint times
    T_mid = (T[i] + T[i+1])/2.
    """
    dT = np.diff(T)
    return (C[:, 1:] - C[:, :-1]) / dT[None, :]


def bound_violations(C: np.ndarray, K: np.ndarray,
                     F: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Indicator masks for intrinsic and upper bound violations.

    Returns (V_lo, V_hi) of shape (len(K), len(T)).
    """
    intrinsic = np.maximum(F - K, 0.0)[:, None]
    upper = float(F)
    V_lo = C < intrinsic - 1e-12
    V_hi = C > upper + 1e-12
    return V_lo, V_hi


def violation_summary_per_tuple(
    C: np.ndarray, K: np.ndarray, T: np.ndarray, F: float = 1.0
) -> dict:
    """Compute all violation metrics for a single (K, T) surface."""
    p = butterfly_density(C, K)
    dCdT = calendar_derivative(C, T)
    V_lo, V_hi = bound_violations(C, K, F)

    B = p < 0
    D = dCdT < 0

    # Negative mass: integral of max(-p, 0) on (interior K, full T)
    # Use simple Riemann sum on the strike axis
    h = K[1] - K[0]
    M_neg = float(np.sum(np.maximum(-p, 0.0)) * h)

    n_cells_B = int(B.size)
    n_cells_D = int(D.size)
    n_cells_V = int(V_lo.size)

    any_violation = bool(
        B.any() or D.any() or V_lo.any() or V_hi.any()
    )

    return {
        "butterfly_rate": float(B.sum()) / max(n_cells_B, 1),
        "calendar_rate": float(D.sum()) / max(n_cells_D, 1),
        "intrinsic_rate": float(V_lo.sum()) / max(n_cells_V, 1),
        "upper_rate": float(V_hi.sum()) / max(n_cells_V, 1),
        "negative_mass": M_neg,
        "any_violation": any_violation,
        "butterfly_mask": B,        # for the per-(K,T) heatmap
        "calendar_mask": D,
        "intrinsic_mask": V_lo,
        "upper_mask": V_hi,
        "density_min": float(p.min()),
        "density_argmin": tuple(np.unravel_index(int(np.argmin(p)), p.shape)),
    }


def scan_prior(params_grid: np.ndarray, K_fine: np.ndarray,
               T_fine: np.ndarray, formula: str) -> dict:
    """Evaluate all violation metrics across the parameter prior."""
    n = len(params_grid)
    butterfly = np.zeros(n)
    calendar = np.zeros(n)
    intrinsic = np.zeros(n)
    upper = np.zeros(n)
    M_neg = np.zeros(n)
    any_viol = np.zeros(n, dtype=bool)
    density_min = np.zeros(n)

    n_K = len(K_fine)
    n_T = len(T_fine)
    butterfly_mask_sum = np.zeros((n_K - 2, n_T), dtype=np.int32)
    calendar_mask_sum = np.zeros((n_K, n_T - 1), dtype=np.int32)
    bound_mask_sum = np.zeros((n_K, n_T), dtype=np.int32)

    for i in range(n):
        try:
            iv = hagan_iv_surface(params_grid[i], K_fine, T_fine, formula)
            C = call_surface_from_iv(iv, K_fine, T_fine)
            r = violation_summary_per_tuple(C, K_fine, T_fine)
        except Exception as exc:
            log.warning("formula=%s tuple %d failed: %s", formula, i, exc)
            butterfly[i] = np.nan
            continue

        butterfly[i] = r["butterfly_rate"]
        calendar[i] = r["calendar_rate"]
        intrinsic[i] = r["intrinsic_rate"]
        upper[i] = r["upper_rate"]
        M_neg[i] = r["negative_mass"]
        any_viol[i] = r["any_violation"]
        density_min[i] = r["density_min"]
        butterfly_mask_sum += r["butterfly_mask"].astype(np.int32)
        calendar_mask_sum += r["calendar_mask"].astype(np.int32)
        bound_mask_sum += (
            r["intrinsic_mask"].astype(np.int32)
            + r["upper_mask"].astype(np.int32)
        )

        if (i + 1) % 25 == 0:
            log.info("[%s] %d/%d scanned", formula, i + 1, n)

    return {
        "formula": formula,
        "n_tuples": n,
        "K_fine": K_fine,
        "T_fine": T_fine,
        "butterfly_rate": butterfly,
        "calendar_rate": calendar,
        "intrinsic_rate": intrinsic,
        "upper_rate": upper,
        "negative_mass": M_neg,
        "any_violation": any_viol,
        "density_min": density_min,
        "butterfly_mask_sum": butterfly_mask_sum,
        "calendar_mask_sum": calendar_mask_sum,
        "bound_mask_sum": bound_mask_sum,
        "params": params_grid,
    }


def aggregate(scan: dict) -> dict:
    """Reduce a scan to a small JSON-friendly summary."""
    finite = np.isfinite(scan["butterfly_rate"])
    n = int(finite.sum())
    return {
        "formula": scan["formula"],
        "n_tuples_evaluated": n,
        "any_violation_fraction": float(scan["any_violation"][finite].mean()),
        "butterfly": {
            "fraction_with_any": float((scan["butterfly_rate"][finite] > 0).mean()),
            "median_cell_rate": float(np.median(scan["butterfly_rate"][finite])),
            "p90_cell_rate": float(np.percentile(scan["butterfly_rate"][finite], 90)),
            "max_cell_rate": float(scan["butterfly_rate"][finite].max()),
        },
        "calendar": {
            "fraction_with_any": float((scan["calendar_rate"][finite] > 0).mean()),
            "median_cell_rate": float(np.median(scan["calendar_rate"][finite])),
            "p90_cell_rate": float(np.percentile(scan["calendar_rate"][finite], 90)),
            "max_cell_rate": float(scan["calendar_rate"][finite].max()),
        },
        "intrinsic": {
            "fraction_with_any": float((scan["intrinsic_rate"][finite] > 0).mean()),
            "median_cell_rate": float(np.median(scan["intrinsic_rate"][finite])),
            "max_cell_rate": float(scan["intrinsic_rate"][finite].max()),
        },
        "upper": {
            "fraction_with_any": float((scan["upper_rate"][finite] > 0).mean()),
            "median_cell_rate": float(np.median(scan["upper_rate"][finite])),
            "max_cell_rate": float(scan["upper_rate"][finite].max()),
        },
        "negative_mass": {
            "median": float(np.median(scan["negative_mass"][finite])),
            "p90": float(np.percentile(scan["negative_mass"][finite], 90)),
            "max": float(scan["negative_mass"][finite].max()),
        },
        "density_min": {
            "median": float(np.median(scan["density_min"][finite])),
            "p1": float(np.percentile(scan["density_min"][finite], 1)),
            "min": float(scan["density_min"][finite].min()),
        },
    }


def make_figures(scans: list, fig_dir: Path) -> None:
    """Two figures: density examples and violation-rate heatmaps."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- Figure 1: representative density slices -----
    # Use Hagan formula on three representative parameter tuples chosen to
    # span benign / moderate / severe regimes (low/mid/high vol-of-vol).
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    examples = [
        ("Benign", np.array([0.5, -0.3, 0.30, 0.25])),
        ("Moderate", np.array([0.3, -0.5, 0.70, 0.30])),
        ("Severe", np.array([0.2, -0.85, 1.00, 0.20])),
    ]
    K_fine = np.linspace(0.30, 1.80, 151)
    T_show = np.array([0.5, 1.0, 2.0])
    for ax, (label, p) in zip(axes, examples):
        iv = hagan_iv_surface(p, K_fine, T_show, formula="hagan")
        C = call_surface_from_iv(iv, K_fine, T_show)
        p_dens = butterfly_density(C, K_fine)
        K_int = K_fine[1:-1]
        for j, T_j in enumerate(T_show):
            ax.plot(K_int, p_dens[:, j], label=f"T = {T_j}")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel("Strike $K$")
        ax.set_ylabel(r"Implied density $\partial_K^2 C$")
        coercivity = abs(p[1]) * p[2] ** 2
        ax.set_title(
            f"{label}: $\\beta={p[0]:.1f}$, $\\rho={p[1]:+.2f}$, "
            f"$\\nu={p[2]:.2f}$, $|\\rho|\\nu^2={coercivity:.2f}$"
        )
        ax.legend(loc="best", fontsize=8)
    fig.savefig(fig_dir / "hagan_density_examples.pdf")
    plt.close(fig)
    log.info("Wrote %s", fig_dir / "hagan_density_examples.pdf")

    # ----- Figure 2: violation-rate heatmaps (Hagan vs Obloj) -----
    fig, axes = plt.subplots(1, len(scans), figsize=(5.5 * len(scans), 4.0),
                              constrained_layout=True)
    if len(scans) == 1:
        axes = [axes]
    for ax, scan in zip(axes, scans):
        K_int = scan["K_fine"][1:-1]
        T_fine = scan["T_fine"]
        rate = scan["butterfly_mask_sum"] / max(scan["n_tuples"], 1)
        im = ax.imshow(
            rate.T, origin="lower", aspect="auto",
            extent=[K_int.min(), K_int.max(), T_fine.min(), T_fine.max()],
            cmap="viridis", vmin=0.0, vmax=1.0,
        )
        ax.set_xlabel("Strike $K$")
        ax.set_ylabel("Maturity $T$ (years)")
        ax.set_title(f"{scan['formula'].capitalize()}: butterfly violation rate")
        plt.colorbar(im, ax=ax)
    fig.savefig(fig_dir / "hagan_violation_heatmap.pdf")
    plt.close(fig)
    log.info("Wrote %s", fig_dir / "hagan_violation_heatmap.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-params", type=int, default=512,
                    help="Sobol draws over the prior (default 512).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-K", type=int, default=151,
                    help="Strike-axis resolution for the analysis grid.")
    ap.add_argument("--K-min", type=float, default=0.30)
    ap.add_argument("--K-max", type=float, default=1.80)
    ap.add_argument("--n-T", type=int, default=21,
                    help="Maturity-axis resolution.")
    ap.add_argument("--T-min", type=float, default=0.10)
    ap.add_argument("--T-max", type=float, default=2.00)
    ap.add_argument("--formula", choices=["hagan", "obloj", "both"],
                    default="both")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out-dir", type=str,
        default=str(HERE / "data" / "arbitrage"),
    )
    ap.add_argument(
        "--fig-dir", type=str,
        default=str(HERE.parent / "thesis" / "figures"),
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_grid = sobol_params(args.n_params, args.seed)
    log.info("Sobol prior: %d tuples (post-coercivity filter)", len(params_grid))

    K_fine = np.linspace(args.K_min, args.K_max, args.n_K)
    T_fine = np.linspace(args.T_min, args.T_max, args.n_T)
    log.info(
        "Analysis grid: K %d in [%.2f, %.2f], T %d in [%.2f, %.2f]",
        args.n_K, args.K_min, args.K_max,
        args.n_T, args.T_min, args.T_max,
    )

    formulas = (["hagan", "obloj"] if args.formula == "both"
                else [args.formula])
    scans = []
    summary = {}
    for formula in formulas:
        log.info("Scanning %s ...", formula)
        scan = scan_prior(params_grid, K_fine, T_fine, formula)
        scans.append(scan)
        summary[formula] = aggregate(scan)

    summary_path = out_dir / "hagan_arbitrage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)

    # Persist per-tuple arrays for downstream analysis
    for scan in scans:
        np.savez_compressed(
            out_dir / f"hagan_arbitrage_per_tuple_{scan['formula']}.npz",
            **{
                k: v for k, v in scan.items()
                if isinstance(v, np.ndarray) or isinstance(v, (int, float))
            },
        )

    if not args.no_figures:
        make_figures(scans, Path(args.fig_dir))

    # Pretty-print a quick summary table to stdout
    print()
    print("=" * 78)
    print(f"{'Formula':<10s} {'AnyViol%':>10s} {'Butterfly%':>12s} "
          f"{'Calendar%':>12s} {'NegMass p90':>14s} {'min(p)':>10s}")
    print("-" * 78)
    for formula, s in summary.items():
        print(
            f"{formula:<10s} "
            f"{100*s['any_violation_fraction']:>10.1f} "
            f"{100*s['butterfly']['fraction_with_any']:>12.1f} "
            f"{100*s['calendar']['fraction_with_any']:>12.1f} "
            f"{s['negative_mass']['p90']:>14.3e} "
            f"{s['density_min']['min']:>10.3e}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
