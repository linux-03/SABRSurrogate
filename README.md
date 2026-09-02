# sabrfem

SABR pricing methods and a **trained neural surrogate** that prices a full
implied-vol surface at calibration speed. The package is light — the surrogate
ships pre-trained, so there is no finite-element / NGSolve dependency; only
NumPy and SciPy are needed for the classical methods, and PyTorch to run the
surrogate.

This is the code behind my MSc dissertation. Every pricing method exposes the
same `(params, strikes, maturities) -> surface` interface, so the asymptotic
formulas, Monte Carlo, finite differences and the neural surrogate are directly
comparable on the same grid.

## The problem

The Hagan (2002) asymptotic formula is the market standard for SABR: a closed
form, so it is instant. But it is only an expansion — in the wings and at short
maturities the implied density it induces goes negative, i.e. it admits
butterfly arbitrage, so a surface quoted straight from it is not tradeable.

The accurate alternative is to solve the pricing PDE with finite elements, which
(unlike the asymptotics) converges to an arbitrage-free surface — but an FE solve
per parameter set is far too slow to sit inside a calibration loop. So the thesis
trains a small neural network on finite-element prices: once trained it prices a
whole surface in microseconds, fast enough to calibrate, while inheriting the FE
solver's accuracy.

That trained network is what ships here (`pricing.surrogate`, with its weights).
The FE solver that generated its training labels is a separate, heavier component
and is **not** included in this package.

## What is in here

Every method prices on the same `strikes × maturities` grid under one convention
(forward measure, zero rate, undiscounted prices).

| Family        | Module                    | What it is                                             |
| ------------- | ------------------------- | ------------------------------------------------------ |
| Surrogate     | `pricing.surrogate`       | trained MLP, prices a surface in microseconds (bundled weights) |
| Asymptotic    | `pricing.hagan`           | Hagan (2002) and Oblój implied-vol expansions          |
| Arbitrage-free| `pricing.hagan2014`       | Hagan–Kumar–Lesniewski–Woodward (2014) density PDE     |
| Monte Carlo   | `pricing.montecarlo`      | log-Euler MC with antithetic / control-variate + Sobol-QMC |
| Finite diff.  | `pricing.finite_diff`     | ADI / Craig–Sneyd two-dimensional scheme               |

Around them:

- **`calibration`** — the inverse problem: fit SABR parameters to a quoted
  surface, using the surrogate as the forward map for the speedup.
- **`arbitrage`** — a butterfly-density scan (`hagan_scan`) and two arbitrage-free
  *repair* layers for a raw surrogate surface: a SANOS-style least-squares
  projection (`sanos`) and a QP projection (`qp`).
- **`metrics`** — the thesis metrics, one file per metric (Breeden-Litzenberger
  density, butterfly / calendar / bound arbitrage, calibration fit, convergence).

## Layout

```
sabrfem/
  black.py            Black-Scholes price / vega / put + implied-vol inversion
  pricing/
    params.py         the shared SABRParams object
    hagan.py          Hagan (2002) + Oblój
    hagan2014.py      arbitrage-free SABR density PDE
    montecarlo.py     Monte Carlo + Sobol-QMC
    finite_diff.py    ADI / Craig–Sneyd
    surrogate.py      the trained neural surrogate + nn_model.pt / nn_config.json
  calibration/        calibrate  (surrogate-accelerated inverse problem)
  arbitrage/          hagan_scan, sanos, qp
  metrics/            density, butterfly, calendar, bounds, arbitrage, accuracy,
                      calibration, convergence
figures/              headline plots
README.md   requirements.txt   .gitignore
```

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Price with the classical methods:

```python
import numpy as np
from sabrfem.pricing import SABRParams, hagan_call_surface, mc_call_surface

p = SABRParams(beta=0.5, rho=-0.3, nu=1.0, y0=0.3, x0=1.0)
K, T = np.array([0.8, 0.9, 1.0, 1.1, 1.2]), np.array([0.5, 1.0])
hagan_px, hagan_iv = hagan_call_surface(p, K, T)
mc_px, mc_report   = mc_call_surface(p, K, T)
```

Price with the trained surrogate (microseconds per surface):

```python
from sabrfem.pricing import load_surrogate, predict
model, cfg = load_surrogate()                       # loads the bundled weights
iv = predict(model, cfg, [[0.5, -0.3, 1.0, 0.3]])   # (1, 11, 8) implied-vol grid
```

Calibrate SABR parameters to a surface with the surrogate as the forward map:

```bash
python -m sabrfem.calibration.calibrate --model sabrfem/pricing/nn_model.pt
```

## Figures

Surrogate accuracy across the parameter prior (RMSE in vol points), and the
implied-vol surface from each method side by side:

![rmse vs params](figures/rmse_vs_params.png)
![iv surface comparison](figures/iv_surface_comparison.png)

Why the asymptotics are not enough — a Hagan (2002) surface with a negative
implied density (butterfly arbitrage), and a calibration to a market smile:

![severe density violation](figures/severe_density_violation.png)
![market calibration](figures/market_calibration.png)

## The bundled model

`pricing/nn_model.pt` (+ `nn_config.json`, `nn_eval_test.json`) is the trained
surrogate: a 4×30 ELU MLP, target = implied vol, IV RMSE ≈ 2.7e-3 on a 6000-case
held-out test set. It consumes SABR parameters normalised to the prior box in
`pricing/surrogate.py` and outputs the 11×8 implied-vol grid.
