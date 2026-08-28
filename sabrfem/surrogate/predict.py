"""Load a trained surrogate and evaluate it on SABR parameters.

    from sabr_lib.surrogate import load_surrogate, predict_ivs
    model, cfg, torch = load_surrogate("data/nn/nn_model.pt")
    iv = predict_ivs(model, cfg, [[0.5, -0.3, 1.0, 0.3]], strikes, maturities)

The surrogate's native target (call price or implied vol) is recorded in its
``nn_config.json``; :func:`predict_prices` / :func:`predict_ivs` convert to the
requested quantity via the Black formula (:mod:`sabr_lib.black`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..black import bs_call_price, implied_vol
from .model import _import_torch, build_mlp, normalise_params


def load_surrogate(model_path, config_path=None, device: str = "cpu"):
    """Build the network from its config and load trained weights.

    ``config_path`` defaults to ``nn_config.json`` beside ``model_path``.
    Returns ``(model, config, torch)``.
    """
    torch = _import_torch()
    model_path = Path(model_path)
    if config_path is None:
        config_path = model_path.with_name("nn_config.json")
    config = json.loads(Path(config_path).read_text())

    model = build_mlp(
        torch, config["in_dim"], config["out_dim"],
        config["hidden"], config["depth"],
    )
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, config, torch


def predict_grid(model, config, params_raw, *, prior=None,
                 device: str = "cpu", torch=None) -> np.ndarray:
    """Predict the surrogate's NATIVE grid (price or IV per ``config['target']``).

    ``params_raw`` is (N, 4) raw SABR params [beta, rho, nu, y0]; the result is
    (N, n_strikes, n_maturities).
    """
    torch = torch or _import_torch()
    X = normalise_params(params_raw, prior).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return out.reshape(-1, int(config["n_strikes"]), int(config["n_maturities"]))


def predict_prices(model, config, params_raw, strikes, maturities, *,
                   forward: float = 1.0, prior=None, device: str = "cpu",
                   torch=None) -> np.ndarray:
    """Predicted call prices (N, n_K, n_T), converting from IV if needed."""
    grid = predict_grid(model, config, params_raw, prior=prior,
                        device=device, torch=torch)
    if config["target"] == "price":
        return grid
    K = np.asarray(strikes, float)[None, :, None]
    T = np.asarray(maturities, float)[None, None, :]
    return bs_call_price(forward, K, T, grid)


def predict_ivs(model, config, params_raw, strikes, maturities, *,
                forward: float = 1.0, prior=None, device: str = "cpu",
                torch=None) -> np.ndarray:
    """Predicted implied vols (N, n_K, n_T), converting from price if needed.

    Price-target surrogates are clipped to the no-arbitrage band before
    inversion (the network can drift a hair below intrinsic), so deep-wing
    cells outside the band come back as NaN.
    """
    grid = predict_grid(model, config, params_raw, prior=prior,
                        device=device, torch=torch)
    if config["target"] == "iv":
        return grid
    K = np.asarray(strikes, float)[None, :, None]
    T = np.asarray(maturities, float)[None, None, :]
    intrinsic = np.maximum(forward - K, 0.0)
    grid = np.clip(grid, intrinsic, forward)
    return implied_vol(np.full_like(grid, forward), np.broadcast_to(K, grid.shape),
                       np.broadcast_to(T, grid.shape), grid)
