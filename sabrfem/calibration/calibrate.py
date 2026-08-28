"""
Calibrate the SABR model to implied-volatility surfaces using either the
trained NN surrogate or the FE solver as the forward map.

This is the headline experiment of the dissertation: the NN is orders of
magnitude faster than the FE solver, so calibration via the NN is essentially
free, while FE-in-the-loop calibration is infeasible in real-time settings
(cf. Horvath, Muguruza & Tomas 2019 §4.2).

Protocol
--------
For each of K test-set parameter tuples theta* = (beta, rho, nu, y0):

  1. The "market" IV surface is the FE-solver output on the HMT grid (already
     computed and stored in the test split).
  2. Two calibrators attempt to recover theta* from the surface:
       - NN calibrator : minimises ||NN(theta) - market_iv||^2  via L-BFGS,
                         using PyTorch autograd for the Jacobian.
       - FE calibrator : minimises ||FE(theta) - market_iv||^2  via
                         scipy.optimize.least_squares (Levenberg-Marquardt
                         with finite differences), calling the FE solver
                         on every evaluation.
  3. We report: parameter-recovery error, final surface residual, and
     wall-clock per calibration. Speedup = FE wall-clock / NN wall-clock.

The NN calibrator works in normalised input space x in [0, 1]^4 (same space
the NN was trained on); theta is recovered from x via the PRIOR. The FE
calibrator works directly in theta space, with box bounds = PRIOR.

Both calibrators use the same loss function (uniform MSE on IV surface) and
the same random initialisation (drawn from the prior, seeded for
reproducibility). We run both with an early-termination tolerance on the
residual norm so the timings are comparable.

Usage
-----

    # Default: calibrate with the IV-target surrogate against 20 test cases,
    # no FE baseline (too slow for interactive use).
    python calibrate.py

    # Include the FE baseline. Each FE calibration takes ~2-10 min, so use
    # a small K here.
    python calibrate.py --with-fe --n-cases 3

    # Different surrogate:
    python calibrate.py --model data/nn/nn_model.pt --target price

Outputs
-------
    data/calibration/results.json          Per-case calibration summary
    data/calibration/summary.json          Aggregate metrics + speedup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent

log = logging.getLogger("calibrate")

# Must stay in sync with prepare_training_data.py
PRIOR = {
    "beta": (0.1, 0.9),
    "rho": (-0.9, 0.1),
    "nu": (0.1, 1.45),   # extended to the coercivity limit; was (0.1, 1.0)
    "y0": (0.1, 0.5),
}
PARAM_NAMES = ["beta", "rho", "nu", "y0"]


# ---------------------------------------------------------------------------
# Parameter-space conversions
# ---------------------------------------------------------------------------


def theta_to_x(theta: np.ndarray) -> np.ndarray:
    """Map raw params to normalised [0, 1]^4 coords."""
    out = np.empty_like(theta)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        out[..., j] = (theta[..., j] - lo) / (hi - lo)
    return out


def x_to_theta(x: np.ndarray) -> np.ndarray:
    """Inverse of theta_to_x."""
    out = np.empty_like(x)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        out[..., j] = lo + x[..., j] * (hi - lo)
    return out


# ---------------------------------------------------------------------------
# NN wrapper
# ---------------------------------------------------------------------------


def build_mlp(torch, in_dim: int, out_dim: int, hidden: int, depth: int):
    """Matches train_nn.build_mlp — kept identical so state_dicts load cleanly."""
    layers = []
    prev = in_dim
    for _ in range(depth):
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ELU())
        prev = hidden
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


@dataclass
class NNForward:
    """Differentiable forward map theta -> surface, backed by the trained NN.

    The net predicts whatever it was trained on (price or IV). For calibration
    we want an IV surface regardless; if the net is price-target, we convert
    via differentiable BS inversion using a fixed-point Newton unroll.

    For simplicity and faithfulness to HMT, we recommend using an IV-target
    surrogate and setting `target='iv'` so no conversion is needed.
    """

    torch: object
    model: object
    device: object
    n_K: int
    n_T: int
    target: str  # "iv" or "price"
    strikes: np.ndarray
    maturities: np.ndarray

    def forward_raw(self, x):
        """x : (..., 4) torch tensor on self.device, returns (..., n_K, n_T) raw output."""
        return self.model(x).reshape(*x.shape[:-1], self.n_K, self.n_T)

    def forward_iv(self, x):
        """x : (..., 4) torch tensor on self.device, returns (..., n_K, n_T) IVs."""
        out = self.forward_raw(x)
        if self.target == "iv":
            return out
        # Price-target: convert to IV via differentiable Newton on BS
        raise NotImplementedError(
            "Differentiable price->IV inversion not implemented. "
            "Please use an IV-target surrogate for calibration, "
            "or calibrate in price space via calibrate_nn_price()."
        )


def load_nn(
    torch, model_path: Path, hidden: int, depth: int,
    n_K: int, n_T: int, target: str, strikes, maturities, device,
):
    model = build_mlp(torch, 4, n_K * n_T, hidden, depth).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return NNForward(
        torch=torch, model=model, device=device,
        n_K=n_K, n_T=n_T, target=target,
        strikes=strikes, maturities=maturities,
    )


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    theta_true: list
    theta_hat: list
    x_true: list
    x_hat: list
    success: bool
    n_iter: int
    n_func_eval: int
    surface_rmse_initial: float
    surface_rmse_final: float
    param_abs_err: list
    param_rel_err: list
    wall_seconds: float
    method: str
    x0: list


def calibrate_nn(
    nn: NNForward,
    market_iv: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> CalibrationResult:
    """L-BFGS calibration in normalised input space x in [0, 1]^4.

    Uses PyTorch's torch.optim.LBFGS which exposes the classic quasi-Newton
    loop with a strong-Wolfe line search. This is the standard choice for
    inverse problems with smooth differentiable forward maps.
    """
    torch = nn.torch
    device = nn.device

    y_target = torch.from_numpy(market_iv.astype(np.float32)).to(device)
    x = torch.from_numpy(x0.astype(np.float32)).to(device).clone()
    x.requires_grad_(True)

    optim = torch.optim.LBFGS(
        [x],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    # Initial residual for the report
    with torch.no_grad():
        init_pred = nn.forward_iv(x)
        init_rmse = float(
            torch.sqrt(((init_pred - y_target) ** 2).mean()).cpu()
        )

    n_func_eval = {"n": 0}

    def closure():
        optim.zero_grad()
        n_func_eval["n"] += 1
        # Clamp x into the feasible box [0, 1] in-place between iterations.
        # Using no-grad so the clamp is part of the state, not the graph.
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
        pred = nn.forward_iv(x)
        loss = ((pred - y_target) ** 2).mean()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    optim.step(closure)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        x.clamp_(0.0, 1.0)
        final_pred = nn.forward_iv(x)
        final_rmse = float(
            torch.sqrt(((final_pred - y_target) ** 2).mean()).cpu()
        )
        x_hat = x.detach().cpu().numpy().astype(np.float64)

    theta_hat = x_to_theta(x_hat)
    return CalibrationResult(
        theta_true=[],
        theta_hat=theta_hat.tolist(),
        x_true=[],
        x_hat=x_hat.tolist(),
        success=True,
        n_iter=-1,  # LBFGS does not expose a clean iter count across step()
        n_func_eval=int(n_func_eval["n"]),
        surface_rmse_initial=init_rmse,
        surface_rmse_final=final_rmse,
        param_abs_err=[],
        param_rel_err=[],
        wall_seconds=elapsed,
        method="nn-lbfgs",
        x0=x0.tolist(),
    )


def calibrate_nn_price(
    nn: NNForward,
    market_price: np.ndarray,
    vega_weight: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> CalibrationResult:
    """L-BFGS calibration in price space with vega weighting.

    Minimises  sum_ij  w_ij * (NN(x)_ij - C^mkt_ij)^2
    where w_ij = 1 / (vega_ij + eps), normalised to mean 1.
    This makes the price-space loss approximately equivalent to IV-MSE.
    """
    torch = nn.torch
    device = nn.device

    y_target = torch.from_numpy(market_price.astype(np.float32)).to(device)
    w = torch.from_numpy(vega_weight.astype(np.float32)).to(device)
    x = torch.from_numpy(x0.astype(np.float32)).to(device).clone()
    x.requires_grad_(True)

    optim = torch.optim.LBFGS(
        [x],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    with torch.no_grad():
        init_pred = nn.forward_raw(x)
        init_rmse = float(
            torch.sqrt(((init_pred - y_target) ** 2).mean()).cpu()
        )

    n_func_eval = {"n": 0}

    def closure():
        optim.zero_grad()
        n_func_eval["n"] += 1
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
        pred = nn.forward_raw(x)
        loss = (w * (pred - y_target) ** 2).mean()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    optim.step(closure)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        x.clamp_(0.0, 1.0)
        final_pred = nn.forward_raw(x)
        final_rmse = float(
            torch.sqrt(((final_pred - y_target) ** 2).mean()).cpu()
        )
        x_hat = x.detach().cpu().numpy().astype(np.float64)

    theta_hat = x_to_theta(x_hat)
    return CalibrationResult(
        theta_true=[],
        theta_hat=theta_hat.tolist(),
        x_true=[],
        x_hat=x_hat.tolist(),
        success=True,
        n_iter=-1,
        n_func_eval=int(n_func_eval["n"]),
        surface_rmse_initial=init_rmse,
        surface_rmse_final=final_rmse,
        param_abs_err=[],
        param_rel_err=[],
        wall_seconds=elapsed,
        method="nn-lbfgs-price",
        x0=x0.tolist(),
    )


def calibrate_fe(
    fe_cfg,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_iv: np.ndarray,
    theta0: np.ndarray,
    max_nfev: int = 60,
) -> CalibrationResult:
    """Levenberg-Marquardt calibration calling the FE solver on every eval.

    We minimise residuals in IV space (not price space) to match the NN
    objective. Each residual evaluation: one FE call-price surface, then
    Newton-inversion to get IVs. scipy's least_squares with method='trf'
    gives box constraints at minimal cost.
    """
    from scipy.optimize import least_squares

    from ..black import implied_vol
    from ..pricing.fem import FEConfig, SABRParams, SABRSolver

    n_K, n_T = len(strikes), len(maturities)
    y_target = market_iv.ravel()
    F_arr = np.ones((n_K, n_T))
    K_grid = np.broadcast_to(strikes[:, None], (n_K, n_T))
    T_grid = np.broadcast_to(maturities[None, :], (n_K, n_T))
    finite_mask = np.isfinite(y_target)

    lb = np.array([PRIOR[n][0] for n in PARAM_NAMES])
    ub = np.array([PRIOR[n][1] for n in PARAM_NAMES])

    n_func_eval = {"n": 0}

    def residuals(theta):
        n_func_eval["n"] += 1
        params = SABRParams(
            beta=float(theta[0]), rho=float(theta[1]),
            nu=float(theta[2]), y0=float(theta[3]),
            x0=1.0,
        )
        try:
            solver = SABRSolver(params, fe_cfg)
            prices, _ = solver.price_call_surface(strikes, maturities)
        except Exception as exc:  # noqa: BLE001
            log.warning("FE solver failed at theta=%s: %s", theta.tolist(), exc)
            # Large residual drives LM away from this region
            return np.full_like(y_target, 1.0)

        # Clip sub-intrinsic prices (same rule as prepare_training_data)
        intr = np.maximum(F_arr - K_grid, 0.0)
        prices_clipped = np.clip(prices, intr, 1.0)
        iv = implied_vol(F_arr, K_grid, T_grid, prices_clipped)
        iv_flat = iv.ravel()
        # Replace NaN with target (zero residual) — equivalent to masking them
        # out, but scipy least_squares needs a fixed residual length.
        iv_flat = np.where(np.isfinite(iv_flat), iv_flat, y_target)
        res = iv_flat - y_target
        # Zero-out cells where market was NaN (shouldn't happen post-prep)
        res = np.where(finite_mask, res, 0.0)
        return res

    # Initial surface RMSE
    r0 = residuals(theta0)
    init_rmse = float(np.sqrt(np.mean(r0[finite_mask] ** 2)))
    # Reset counter so LM call sees its own budget
    n_func_eval["n"] = 0

    t0 = time.perf_counter()
    res = least_squares(
        residuals,
        x0=theta0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    theta_hat = res.x
    final_rmse = float(
        np.sqrt(np.mean(res.fun[finite_mask] ** 2))
    )
    x_hat = theta_to_x(theta_hat)
    return CalibrationResult(
        theta_true=[],
        theta_hat=theta_hat.tolist(),
        x_true=[],
        x_hat=x_hat.tolist(),
        success=bool(res.success),
        n_iter=int(res.nfev),
        n_func_eval=int(n_func_eval["n"]),
        surface_rmse_initial=init_rmse,
        surface_rmse_final=final_rmse,
        param_abs_err=[],
        param_rel_err=[],
        wall_seconds=elapsed,
        method="fe-lm",
        x0=theta_to_x(theta0).tolist(),
    )


def calibrate_hagan(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_iv: np.ndarray,
    theta0: np.ndarray,
    max_nfev: int = 200,
) -> CalibrationResult:
    """Hagan-formula calibration via scipy least_squares (fast baseline)."""
    from scipy.optimize import least_squares
    from ..pricing.fem import SABRParams
    from ..pricing.hagan import hagan_implied_vol

    y_target = market_iv.ravel()
    finite_mask = np.isfinite(y_target)
    lb = np.array([PRIOR[n][0] for n in PARAM_NAMES])
    ub = np.array([PRIOR[n][1] for n in PARAM_NAMES])
    n_func_eval = {"n": 0}

    def residuals(theta):
        n_func_eval["n"] += 1
        params = SABRParams(
            beta=float(theta[0]), rho=float(theta[1]),
            nu=float(theta[2]), y0=float(theta[3]), x0=1.0,
        )
        try:
            iv = hagan_implied_vol(params, strikes, maturities)
        except Exception:
            return np.full_like(y_target, 1.0)
        iv_flat = iv.ravel()
        iv_flat = np.where(np.isfinite(iv_flat), iv_flat, y_target)
        res = iv_flat - y_target
        return np.where(finite_mask, res, 0.0)

    r0 = residuals(theta0)
    init_rmse = float(np.sqrt(np.mean(r0[finite_mask] ** 2)))
    n_func_eval["n"] = 0

    t0 = time.perf_counter()
    res = least_squares(
        residuals, x0=theta0, bounds=(lb, ub),
        method="trf", max_nfev=max_nfev,
        xtol=1e-8, ftol=1e-8, gtol=1e-8, verbose=0,
    )
    elapsed = time.perf_counter() - t0

    theta_hat = res.x
    final_rmse = float(np.sqrt(np.mean(res.fun[finite_mask] ** 2)))
    return CalibrationResult(
        theta_true=[], theta_hat=theta_hat.tolist(),
        x_true=[], x_hat=theta_to_x(theta_hat).tolist(),
        success=bool(res.success),
        n_iter=int(res.nfev), n_func_eval=int(n_func_eval["n"]),
        surface_rmse_initial=init_rmse, surface_rmse_final=final_rmse,
        param_abs_err=[], param_rel_err=[],
        wall_seconds=elapsed, method="hagan-lm",
        x0=theta_to_x(theta0).tolist(),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "training_data.npz"))
    ap.add_argument("--model", default=str(HERE / "data" / "nn" / "nn_model.pt"))
    ap.add_argument("--config", default=None,
                    help="Path to nn_config.json (auto-detected from model dir if omitted).")
    ap.add_argument("--hidden", type=int, default=30)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument(
        "--target", default="price", choices=["iv", "price"],
        help="Must match the training target of the loaded model.",
    )
    ap.add_argument("--n-cases", type=int, default=20,
                    help="Number of test-set cases to calibrate.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for random initialisations (shared by NN and FE).")
    ap.add_argument("--with-fe", action="store_true",
                    help="Also run the FE baseline (slow: ~2-10 min per case).")
    ap.add_argument("--with-hagan", action="store_true",
                    help="Also run Hagan-formula calibration baseline.")
    ap.add_argument("--fe-max-nfev", type=int, default=60,
                    help="Max FE evaluations per calibration.")
    ap.add_argument("--n-restarts", type=int, default=1,
                    help="Random restarts per case (NN only; FE uses 1).")
    ap.add_argument("--out-dir", default=str(HERE / "data" / "calibration"))
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    import torch  # noqa: E402

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # Auto-detect config from model directory
    model_dir = Path(args.model).parent
    config_path = Path(args.config) if args.config else model_dir / "nn_config.json"
    if config_path.exists():
        nn_cfg = json.loads(config_path.read_text())
        args.hidden = nn_cfg.get("hidden", args.hidden)
        args.depth = nn_cfg.get("depth", args.depth)
        args.target = nn_cfg.get("target", args.target)
        log.info("Loaded NN config from %s: hidden=%d depth=%d target=%s",
                 config_path, args.hidden, args.depth, args.target)

    # Load data + NN
    arc = np.load(args.data, allow_pickle=True)
    X = arc["X"].astype(np.float32)
    Y_price = arc["Y_price"].astype(np.float32)
    Y_iv = arc["Y_iv"].astype(np.float32)
    W_vega = arc["W_vega"].astype(np.float32)
    test_idx = arc["test_idx"]
    strikes = arc["strikes"].astype(np.float64)
    maturities = arc["maturities"].astype(np.float64)
    n_K, n_T = Y_iv.shape[1], Y_iv.shape[2]
    params_raw = arc["params_raw"].astype(np.float64)  # (N, 4) in raw units

    nn = load_nn(
        torch, Path(args.model), args.hidden, args.depth,
        n_K, n_T, args.target, strikes, maturities, device,
    )

    # Optional FE setup
    fe_cfg = None
    if args.with_fe:
        # Same config the dataset was generated with (read from sidecar JSON).
        for candidate in ["sabr_fe_40k.json", "sabr_fe_pilot.json"]:
            pilot_meta_path = HERE / "data" / candidate
            if pilot_meta_path.exists():
                break
        if pilot_meta_path.exists():
            meta = json.loads(pilot_meta_path.read_text())
            from ..pricing.fem import FEConfig
            fe_cfg = FEConfig(**meta["fe_config"])
            log.info("FE config (from pilot): %s", fe_cfg)
        else:
            from ..pricing.fem import FEConfig
            fe_cfg = FEConfig()
            log.warning(
                "No sabr_fe_pilot.json found; using default FEConfig: %s", fe_cfg
            )

    # Select n_cases test surfaces
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(test_idx, size=min(args.n_cases, test_idx.size), replace=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    nn_wall_times = []
    fe_wall_times = []
    hagan_wall_times = []
    nn_param_errs = []
    fe_param_errs = []
    hagan_param_errs = []

    for case_i, idx in enumerate(chosen):
        theta_true = params_raw[idx]
        market_iv = Y_iv[idx]
        finite_mask = np.isfinite(market_iv)
        log.info(
            "Case %d/%d  idx=%d  theta*=(beta=%.3f rho=%+.3f nu=%.3f y0=%.3f)",
            case_i + 1, len(chosen), int(idx),
            theta_true[0], theta_true[1], theta_true[2], theta_true[3],
        )

        # Random initial guesses in [0.1, 0.9] of [0, 1] box (avoid prior edges)
        x0s = rng.uniform(0.1, 0.9, size=(args.n_restarts, 4))

        # --- NN calibration with restarts ---
        best_nn = None
        nn_total_wall = 0.0
        for r in range(args.n_restarts):
            if args.target == "price":
                market_price = Y_price[idx]
                vega_w = W_vega[idx]
                res = calibrate_nn_price(nn, market_price, vega_w, x0s[r])
            else:
                res = calibrate_nn(nn, market_iv, x0s[r])
            nn_total_wall += res.wall_seconds
            if best_nn is None or res.surface_rmse_final < best_nn.surface_rmse_final:
                best_nn = res
        nn_wall_times.append(nn_total_wall)
        # Fill in the case-specific fields
        best_nn.theta_true = theta_true.tolist()
        best_nn.x_true = theta_to_x(theta_true).tolist()
        theta_hat_nn = np.asarray(best_nn.theta_hat)
        abs_err_nn = np.abs(theta_hat_nn - theta_true)
        rel_err_nn = abs_err_nn / np.maximum(np.abs(theta_true), 1e-12)
        best_nn.param_abs_err = abs_err_nn.tolist()
        best_nn.param_rel_err = rel_err_nn.tolist()
        nn_param_errs.append(abs_err_nn)

        log.info(
            "  NN   : RMSE %.2e -> %.2e  nfev=%d  wall=%.2fs  "
            "abs_err=(b=%.3f r=%.3f n=%.3f y=%.3f)",
            best_nn.surface_rmse_initial, best_nn.surface_rmse_final,
            best_nn.n_func_eval, nn_total_wall,
            abs_err_nn[0], abs_err_nn[1], abs_err_nn[2], abs_err_nn[3],
        )

        case_record = {"nn": asdict(best_nn)}

        # --- FE calibration (1 restart for time reasons) ---
        if args.with_fe:
            theta0 = x_to_theta(x0s[0])
            res_fe = calibrate_fe(
                fe_cfg, strikes, maturities, market_iv, theta0,
                max_nfev=args.fe_max_nfev,
            )
            fe_wall_times.append(res_fe.wall_seconds)
            res_fe.theta_true = theta_true.tolist()
            res_fe.x_true = theta_to_x(theta_true).tolist()
            theta_hat_fe = np.asarray(res_fe.theta_hat)
            abs_err_fe = np.abs(theta_hat_fe - theta_true)
            rel_err_fe = abs_err_fe / np.maximum(np.abs(theta_true), 1e-12)
            res_fe.param_abs_err = abs_err_fe.tolist()
            res_fe.param_rel_err = rel_err_fe.tolist()
            fe_param_errs.append(abs_err_fe)

            log.info(
                "  FE   : RMSE %.2e -> %.2e  nfev=%d  wall=%.1fs  "
                "abs_err=(b=%.3f r=%.3f n=%.3f y=%.3f)",
                res_fe.surface_rmse_initial, res_fe.surface_rmse_final,
                res_fe.n_func_eval, res_fe.wall_seconds,
                abs_err_fe[0], abs_err_fe[1], abs_err_fe[2], abs_err_fe[3],
            )
            case_record["fe"] = asdict(res_fe)

        # --- Hagan formula calibration ---
        if args.with_hagan:
            theta0 = x_to_theta(x0s[0])
            res_hagan = calibrate_hagan(
                strikes, maturities, market_iv, theta0,
            )
            hagan_wall_times.append(res_hagan.wall_seconds)
            res_hagan.theta_true = theta_true.tolist()
            res_hagan.x_true = theta_to_x(theta_true).tolist()
            theta_hat_h = np.asarray(res_hagan.theta_hat)
            abs_err_h = np.abs(theta_hat_h - theta_true)
            rel_err_h = abs_err_h / np.maximum(np.abs(theta_true), 1e-12)
            res_hagan.param_abs_err = abs_err_h.tolist()
            res_hagan.param_rel_err = rel_err_h.tolist()
            hagan_param_errs.append(abs_err_h)

            log.info(
                "  Hagan: RMSE %.2e -> %.2e  nfev=%d  wall=%.4fs  "
                "abs_err=(b=%.3f r=%.3f n=%.3f y=%.3f)",
                res_hagan.surface_rmse_initial, res_hagan.surface_rmse_final,
                res_hagan.n_func_eval, res_hagan.wall_seconds,
                abs_err_h[0], abs_err_h[1], abs_err_h[2], abs_err_h[3],
            )
            case_record["hagan"] = asdict(res_hagan)

        all_results.append(case_record)

    # Aggregate summary
    summary = {
        "n_cases": len(chosen),
        "n_restarts_nn": args.n_restarts,
        "target": args.target,
        "model": str(args.model),
        "device": str(device),
        "nn": {
            "wall_mean_s": float(np.mean(nn_wall_times)),
            "wall_median_s": float(np.median(nn_wall_times)),
            "wall_p95_s": float(np.percentile(nn_wall_times, 95)),
            "wall_total_s": float(np.sum(nn_wall_times)),
            "param_mae": np.mean(nn_param_errs, axis=0).tolist(),
            "surface_rmse_final_mean": float(np.mean(
                [c["nn"]["surface_rmse_final"] for c in all_results]
            )),
        },
    }
    if args.with_fe:
        summary["fe"] = {
            "wall_mean_s": float(np.mean(fe_wall_times)),
            "wall_median_s": float(np.median(fe_wall_times)),
            "wall_p95_s": float(np.percentile(fe_wall_times, 95)),
            "wall_total_s": float(np.sum(fe_wall_times)),
            "param_mae": np.mean(fe_param_errs, axis=0).tolist(),
            "surface_rmse_final_mean": float(np.mean(
                [c["fe"]["surface_rmse_final"] for c in all_results]
            )),
        }
        summary["speedup_vs_fe_mean"] = (
            summary["fe"]["wall_mean_s"] / summary["nn"]["wall_mean_s"]
        )
        summary["speedup_vs_fe_median"] = (
            summary["fe"]["wall_median_s"] / summary["nn"]["wall_median_s"]
        )
    if args.with_hagan and hagan_wall_times:
        summary["hagan"] = {
            "wall_mean_s": float(np.mean(hagan_wall_times)),
            "wall_median_s": float(np.median(hagan_wall_times)),
            "wall_total_s": float(np.sum(hagan_wall_times)),
            "param_mae": np.mean(hagan_param_errs, axis=0).tolist(),
            "surface_rmse_final_mean": float(np.mean(
                [c["hagan"]["surface_rmse_final"] for c in all_results if "hagan" in c]
            )),
        }

    log.info("Aggregate summary:\n%s", json.dumps(summary, indent=2))

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Wrote per-case results to %s and summary to %s",
        out_dir / "results.json", out_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
