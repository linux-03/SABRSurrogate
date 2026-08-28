"""Neural surrogate for the FE-SABR grid: architecture, training, inference.

    model    the HMT-style MLP and the input-normalisation contract
    train    training loop / CLI  (python -m sabr_lib.surrogate.train)
    predict  load a trained model and price parameters

``train`` is intentionally not imported here so that ``import sabr_lib`` does not
require PyTorch; import it explicitly (or run it as a module) when training.
"""

from .model import (
    DEFAULT_PRIOR,
    PARAM_NAMES,
    build_mlp,
    denormalise_params,
    normalise_params,
)
from .predict import (
    load_surrogate,
    predict_grid,
    predict_ivs,
    predict_prices,
)

__all__ = [
    "build_mlp", "normalise_params", "denormalise_params",
    "DEFAULT_PRIOR", "PARAM_NAMES",
    "load_surrogate", "predict_grid", "predict_prices", "predict_ivs",
]
