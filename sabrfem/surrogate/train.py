"""Train the HMT-style neural surrogate on an FE-SABR call-price grid.

Loss is a vega-weighted MSE in price space (equivalent to IV-space MSE while
training on the finite, well-defined price), Adam with ReduceLROnPlateau, and
early stopping on validation loss. See :mod:`sabr_lib.surrogate.model` for the
architecture and the input-normalisation contract.

Expects a training ``.npz`` with X, Y_price, Y_iv, W_vega, strikes, maturities,
and train/val/test index arrays (as produced by the project's data-prep step).

Usage:
    python -m sabr_lib.surrogate.train --data data/training_data.npz
    python -m sabr_lib.surrogate.train --epochs 500 --lr 5e-4 --hidden 30 --depth 4

Outputs (in --out-dir): nn_model.pt, nn_config.json, nn_history.npz,
nn_eval_test.json, nn_test_predictions.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ..black import bs_call_price, implied_vol
from .model import build_mlp, _import_torch

log = logging.getLogger("sabr_lib.surrogate.train")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the HMT-style NN surrogate on FE-SABR data."
    )
    parser.add_argument("--data", default="data/training_data.npz")
    parser.add_argument("--out-dir", default="data/nn")
    parser.add_argument("--hidden", type=int, default=30)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early-stop after N epochs without val improvement.",
    )
    parser.add_argument(
        "--target", default="price", choices=["price", "iv"],
        help="Train on call prices (default; recommended) or IVs.",
    )
    parser.add_argument(
        "--use-vega-weight", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1/vega weighting for the price-MSE loss (no-op if --target iv).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to nn_model.pt to resume training from.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    torch = _import_torch()

    # Device selection. MPS (Apple GPU) is supported on macOS 12+ with
    # PyTorch 1.12+; CUDA is preferred where available.
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    arc = np.load(args.data, allow_pickle=True)
    X = arc["X"].astype(np.float32)                       # (N, 4)
    Y_price = arc["Y_price"].astype(np.float32)           # (N, n_K, n_T)
    Y_iv = arc["Y_iv"].astype(np.float32)                 # (N, n_K, n_T)
    W_vega = arc["W_vega"].astype(np.float32)             # (N, n_K, n_T)
    train_idx = arc["train_idx"]
    val_idx = arc["val_idx"]
    test_idx = arc["test_idx"]
    n_K, n_T = Y_price.shape[1], Y_price.shape[2]
    out_dim = n_K * n_T

    if args.target == "price":
        Y = Y_price
        W = W_vega if args.use_vega_weight else np.ones_like(W_vega)
    else:  # IV target
        # Black-Scholes inversion fails for a small fraction of deep-OTM cells
        # where the FE call price sits outside the inversion's validity range.
        # Use the precomputed mask_iv as a binary weight: NaN cells -> weight 0
        # (dropped from the loss); finite cells -> weight 1 (renormalised so the
        # mean weight matches the magnitude of the price-target loss).
        if "mask_iv" not in arc:
            raise RuntimeError(
                "--target iv requires 'mask_iv' in the dataset."
            )
        mask_iv = arc["mask_iv"]
        n_finite = int(mask_iv.sum())
        n_total = int(mask_iv.size)
        log.info(
            "IV target: %d/%d cells (%.2f%%) finite; %d masked.",
            n_finite, n_total, 100.0 * n_finite / max(n_total, 1),
            n_total - n_finite,
        )
        Y = np.where(mask_iv, Y_iv, 0.0).astype(np.float32)  # safe NaN sub
        W = mask_iv.astype(np.float32)
        W *= W.size / max(W.sum(), 1.0)
    Y = Y.reshape(-1, out_dim)
    W = W.reshape(-1, out_dim)

    log.info("Loaded data: X %s, Y %s (target=%s)", X.shape, Y.shape, args.target)
    log.info("Splits: train=%d val=%d test=%d",
             train_idx.size, val_idx.size, test_idx.size)

    def to_device(*arrays):
        return tuple(torch.from_numpy(a).to(device) for a in arrays)

    X_train, Y_train, W_train = to_device(X[train_idx], Y[train_idx], W[train_idx])
    X_val, Y_val, W_val = to_device(X[val_idx], Y[val_idx], W[val_idx])
    X_test = to_device(X[test_idx])[0]

    model = build_mlp(torch, in_dim=4, out_dim=out_dim,
                      hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: depth=%d, hidden=%d, total params=%d",
             args.depth, args.hidden, n_params)

    if args.resume:
        log.info("Resuming from %s", args.resume)
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=10)

    def weighted_mse(y_pred, y_true, w):
        return ((y_pred - y_true) ** 2 * w).mean()

    n_train = X_train.shape[0]
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i: i + args.batch_size]
            yb_pred = model(X_train[idx])
            loss = weighted_mse(yb_pred, Y_train[idx], W_train[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.detach())
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(weighted_mse(model(X_val), Y_val, W_val))

        sched.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optim.param_groups[0]["lr"])

        improved = val_loss < best_val * (1 - 1e-6)
        if improved:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            log.info(
                "epoch %4d  train=%.3e  val=%.3e  lr=%.1e  patience=%d/%d%s",
                epoch, train_loss, val_loss,
                optim.param_groups[0]["lr"], patience_counter, args.patience,
                "  *best*" if improved else "",
            )

        if patience_counter >= args.patience:
            log.info("Early stop: no val improvement for %d epochs "
                     "(best %.3e at epoch %d)", args.patience, best_val, best_epoch)
            break

    elapsed = time.perf_counter() - t0
    log.info("Training finished in %.1fs (%d epochs)",
             elapsed, len(history["train_loss"]))

    model.load_state_dict(best_state)
    model.eval()

    # Test-set evaluation: MSE in price AND IV space plus percentile errors.
    with torch.no_grad():
        Y_test_pred = model(X_test).cpu().numpy().reshape(-1, n_K, n_T)
    Y_test_true_price = Y_price[test_idx]
    Y_test_true_iv = Y_iv[test_idx]

    K_grid = np.broadcast_to(arc["strikes"][None, :, None].astype(np.float32),
                             Y_test_pred.shape)
    T_grid = np.broadcast_to(arc["maturities"][None, None, :].astype(np.float32),
                             Y_test_pred.shape)
    F_arr = np.full_like(Y_test_pred, 1.0)

    if args.target == "price":
        pred_price = Y_test_pred
        intr = np.maximum(F_arr - K_grid, 0.0)
        pred_price_clipped = np.clip(pred_price, intr, np.ones_like(pred_price))
        pred_iv = implied_vol(F_arr, K_grid, T_grid, pred_price_clipped)
    else:
        pred_iv = Y_test_pred
        pred_price = bs_call_price(F_arr, K_grid, T_grid, pred_iv)

    err_price = pred_price - Y_test_true_price
    err_iv = pred_iv - Y_test_true_iv
    finite_iv = np.isfinite(err_iv)

    eval_summary = {
        "target": args.target,
        "n_test_samples": int(test_idx.size),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "n_epochs": int(len(history["train_loss"])),
        "train_seconds": float(elapsed),
        "device": str(device),
        "price_mae": float(np.mean(np.abs(err_price))),
        "price_rmse": float(np.sqrt(np.mean(err_price ** 2))),
        "price_p50_abs": float(np.percentile(np.abs(err_price), 50)),
        "price_p95_abs": float(np.percentile(np.abs(err_price), 95)),
        "price_p99_abs": float(np.percentile(np.abs(err_price), 99)),
        "iv_mae": float(np.mean(np.abs(err_iv[finite_iv]))),
        "iv_rmse": float(np.sqrt(np.mean(err_iv[finite_iv] ** 2))),
        "iv_p50_abs": float(np.percentile(np.abs(err_iv[finite_iv]), 50)),
        "iv_p95_abs": float(np.percentile(np.abs(err_iv[finite_iv]), 95)),
        "iv_p99_abs": float(np.percentile(np.abs(err_iv[finite_iv]), 99)),
        "iv_finite_rate": float(finite_iv.mean()),
    }
    log.info("Test-set summary:\n%s", json.dumps(eval_summary, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "nn_model.pt")
    np.savez(
        out_dir / "nn_history.npz",
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
        lr=np.array(history["lr"]),
    )
    (out_dir / "nn_eval_test.json").write_text(json.dumps(eval_summary, indent=2))
    (out_dir / "nn_config.json").write_text(json.dumps({
        "hidden": args.hidden,
        "depth": args.depth,
        "in_dim": 4,
        "out_dim": out_dim,
        "n_strikes": n_K,
        "n_maturities": n_T,
        "target": args.target,
        "use_vega_weight": bool(args.use_vega_weight),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }, indent=2))
    np.savez_compressed(
        out_dir / "nn_test_predictions.npz",
        test_idx=test_idx,
        params_raw=arc["params_raw"][test_idx] if "params_raw" in arc else X[test_idx],
        pred_price=pred_price.astype(np.float32),
        true_price=Y_test_true_price.astype(np.float32),
        pred_iv=pred_iv.astype(np.float32),
        true_iv=Y_test_true_iv.astype(np.float32),
        strikes=arc["strikes"],
        maturities=arc["maturities"],
    )
    log.info("Wrote model to %s", out_dir / "nn_model.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
