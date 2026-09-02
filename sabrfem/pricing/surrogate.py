"""Neural surrogate — the fast pricer, shipped pre-trained.

A small MLP trained on finite-element prices that maps the four SABR parameters
to the implied-vol grid in microseconds. The weights (``nn_model.pt``) and
architecture (``nn_config.json``) sit beside this module; ``load_surrogate()``
rebuilds the network and loads them, ``predict()`` prices raw SABR parameters.

    from sabrfem.pricing import load_surrogate, predict
    model, cfg = load_surrogate()
    iv = predict(model, cfg, [[0.5, -0.3, 1.0, 0.3]])   # (1, 11, 8) implied-vol grid

The FE training loop is not part of the package — the model is shipped already
trained (target = implied vol, 4x30 ELU MLP, IV RMSE ~2.7e-3; see
``nn_eval_test.json``). PyTorch is required only to load/run it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Prior box used to normalise the four SABR inputs to [0, 1]. Must match the box
# the surrogate was trained on.
PRIOR = {
    "beta": (0.1, 0.9),
    "rho": (-0.9, 0.1),
    "nu": (0.1, 1.45),
    "y0": (0.1, 0.5),
}
PARAM_NAMES = ["beta", "rho", "nu", "y0"]


def normalise_params(params: np.ndarray) -> np.ndarray:
    """Map each (beta, rho, nu, y0) column to [0, 1] using the prior's lo/hi."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    out = np.empty_like(params)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        out[:, j] = (params[:, j] - lo) / (hi - lo)
    return out


def build_mlp(torch, in_dim: int, out_dim: int, hidden: int, depth: int):
    """HMT-style MLP: `depth` hidden layers of `hidden` ELU units."""
    layers = []
    prev = in_dim
    for _ in range(depth):
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ELU())
        prev = hidden
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


def load_surrogate(model_path=None, config_path=None):
    """Rebuild the network from ``nn_config.json`` and load ``nn_model.pt``.

    Both default to the files bundled next to this module. Returns
    ``(model, config)`` with the model in eval mode.
    """
    import torch

    model_path = Path(model_path) if model_path else HERE / "nn_model.pt"
    config_path = Path(config_path) if config_path else HERE / "nn_config.json"
    cfg = json.loads(Path(config_path).read_text())

    model = build_mlp(torch, cfg["in_dim"], cfg["out_dim"],
                      cfg["hidden"], cfg["depth"])
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model, cfg


def predict(model, cfg, params: np.ndarray) -> np.ndarray:
    """Predict the (N, n_strikes, n_maturities) grid for raw SABR parameters.

    ``params`` is (N, 4) in [beta, rho, nu, y0]. The output is the surrogate's
    native target (implied vol for the bundled model).
    """
    import torch

    X = normalise_params(params).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(X)).numpy()
    return out.reshape(-1, int(cfg["n_strikes"]), int(cfg["n_maturities"]))
