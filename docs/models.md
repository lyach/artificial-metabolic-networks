# Models — AMN Reservoir and Ensemble

The model code lives in `src/Build_Model.py`, adapted from [Faure et al. (2023)](https://www.nature.com/articles/s41467-023-40380-0). `src/Build_Model_Wrappers.py` adds per-fold logging. This page explains what the scripts build. It does not document every option of the library.

## The problem

Flux Balance Analysis (FBA) needs **uptake flux bounds** (`Vin`) for each medium compound. Experiments report **concentrations** (`C_med`), and the mapping from concentrations to bounds is unknown and non-linear. The AMN-Reservoir approach learns that mapping from measured growth rates.

```
               trainable                          frozen
  C_med ─► [ Media Network ] ─► Vin ─► [ AMN Reservoir ] ─► V (all fluxes) ─► growth
                   ▲                                                            │
                   └──────────────── loss vs. measured growth ◄─────────────────┘
```

## 1. AMN Reservoir (step 4)

The reservoir is trained on the pFBA-simulated dataset to behave like a differentiable FBA solver. It has two parts:

1. **Neural layer:** dense layers (`n_hidden` × `hidden_dim`, `dropout`) that map the scaled medium bounds to an initial flux vector `V0`.
2. **Mechanistic layer:** `timestep` iterations that move `V` toward a solution satisfying the GEM constraints. Each iteration reduces these losses:

| Loss | Constraint | Formula |
|---|---|---|
| `Loss_SV` | Steady state | ‖S·V‖ |
| `Loss_Vin` | Respect uptake bounds | ‖ReLU(Pin·V − Vin)‖ for `UB` (‖Pin·V − Vin‖ for `EB`) |
| `Loss_Vpos` | Non-negative fluxes | ‖ReLU(−V)‖ |
| objective | Match target fluxes | ‖Pout·V − Vout‖ |

`model_type` selects the mechanistic layer:

| `model_type` | Mechanistic layer |
|---|---|
| `AMN_QP` | Gradient descent on the quadratic constraint losses (default) |
| `AMN_LP` | LP-based iterative solver |
| `AMN_Wt` | RNN-style layer with trainable weights |
| `ANN_Dense` | No mechanistic layer: plain dense network, used as a baseline |

Inputs are scaled by their maximum (`scaler: true`), or column by column with `per_feature_norm: true`. Zero-biomass samples are removed before training because they distort this scaling.

## 2. Ensemble: Media Network + frozen Reservoir (step 5)

`RC_Model` loads the trained reservoir and freezes it (its `model_type` becomes `RC`). It then puts a trainable **Media Network** in front: dense layers (`n_hidden_prior` × `hidden_dim_prior`, `activation_prior`) that map `log10(1 + C_med)` to `Vin`. Only the Media Network's weights are updated, using the error between predicted and measured growth.

The prediction array `pred` returned for each seed is laid out as:

```
[ objective | Loss_SV | Loss_Vin | Loss_Vpos | V (n_reactions) | Vin (scaled) ]
```

`scripts/train_eval_ensemble.py` takes the `Vin` block, undoes the reservoir's input scaling, and writes it with `GR_AVG` to `result_medium/pred_medium_seed<k>.csv`.

## 3. Metrics

| Name | Where | Meaning |
|---|---|---|
| R² | training folds, test set, step 6 | `sklearn.metrics.r2_score` |
| Q² | CV | R² on out-of-fold predictions |
| Collated Q² | step 5, per seed | R² of all out-of-fold growth predictions against measured growth, `multioutput='variance_weighted'` |

With multiple targets, `variance_weighted` gives more weight to targets with larger variance. Keep this in mind when comparing scores computed with a different `multioutput` setting.

## 4. Reading results

```python
from src.plots import plot_learning_curve, plot_r2

plot_learning_curve('models/aidaying_iml1515_UB_AMN_QP/result')            # reservoir CV
plot_learning_curve('models/aidaying_iml1515_UB_AMN_QP/result_ensemble')   # ensemble, seed 0
```

The tutorial notebooks show these plots for past runs.
