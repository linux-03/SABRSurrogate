"""Neural surrogate architecture and input normalisation.

The surrogate follows Horvath, Muguruza & Tomas (2019): a small fully-connected
network mapping the four SABR parameters to the flattened call-price (or IV)
grid.

    input   (4,)      normalised SABR params (beta, rho, nu, y0) in [0, 1]
    hidden  depth x   `hidden` ELU units  (default 4 x 30)
    output  (n_K*n_T) the flattened strike x maturity grid

The network consumes *normalised* inputs, so the prior used for normalisation is
part of the model's contract and lives here alongside the architecture.
"""

from __future__ import annotations

import sys

import numpy as np

# Prior box used to map raw SABR parameters to the unit cube. Must match the box
# the surrogate was trained on (see the training dataset's metadata).
DEFAULT_PRIOR = {
    "beta": (0.1, 0.9),
    "rho": (-0.9, 0.1),
    "nu": (0.1, 1.45),
    "y0": (0.1, 0.5),
}
PARAM_NAMES = ["beta", "rho", "nu", "y0"]


def _import_torch():
    """Import torch lazily with a clear message if it is missing."""
    try:
        import torch
        return torch
    except ImportError as exc:  # noqa: BLE001
        sys.stderr.write(
            "PyTorch is required for the neural surrogate. Install with:\n"
            "    pip install torch\n"
            f"Original error: {exc}\n"
        )
        raise


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


def normalise_params(params: np.ndarray, prior: dict | None = None) -> np.ndarray:
    """Map raw (beta, rho, nu, y0) columns to [0, 1] using the prior's lo/hi."""
    prior = prior or DEFAULT_PRIOR
    params = np.atleast_2d(np.asarray(params, dtype=float))
    out = np.empty_like(params)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = prior[name]
        out[:, j] = (params[:, j] - lo) / (hi - lo)
    return out


def denormalise_params(unit: np.ndarray, prior: dict | None = None) -> np.ndarray:
    """Inverse of :func:`normalise_params`: [0, 1] columns back to raw params."""
    prior = prior or DEFAULT_PRIOR
    unit = np.atleast_2d(np.asarray(unit, dtype=float))
    out = np.empty_like(unit)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = prior[name]
        out[:, j] = lo + unit[:, j] * (hi - lo)
    return out
