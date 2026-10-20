# Pipeline — From Raw Data to Evaluated Media Predictions

## Overview

The pipeline has six steps. Each one is a command-line script under `scripts/`, driven by one YAML file under `configs/`. Steps 1–2 prepare the inputs, and are also walked through interactively in `notebooks/Tutorial_Aida_LC.ipynb`.

```
 data/raw/iML1515.xml ──► [1] duplicate GEM ──► data/interim/iML1515_duplicated.xml
                                              └► data/interim/iML1515_medium_corrected.csv
                                                              │
 data/external/aidaying_*.csv ──► [2] preprocess ─────────────┤
                                   ├► data/interim/aidaying_media_growth_amns_padded.csv  (experimental X, Y)
                                   └► data/processed/aidaying_parameters.csv              (sampling ranges)
                                                              │
                         [3] create_reservoir_training_dataset.py   (pFBA simulations, CPU)
                                                              ▼
                              data/processed/<run_id>/<prefix>_<UB|EB>.npz
                                                              │
                         [4] train_eval_reservoir.py                 (AMN Reservoir, GPU)
                                                              ▼
                              models/<name>_<model_type>/reservoir/
                                                              │
                         [5] train_eval_ensemble.py                  (Media Network + frozen Reservoir, GPU)
                                                              ▼
                              models/<name>_<model_type>/result_medium/pred_medium_seed<k>.csv
                                                              │
                         [6] evaluate_medium_pred.py                 (COBRApy check, CPU)
                                                              ▼
                              models/<name>_<model_type>/result_evaluation/
```

All scripts resolve paths relative to the repository root, so they can be launched from any directory. Every script accepts `--help`.

---

## Step 1 — Make the GEM AMN-compatible

```bash
python scripts/prepare_gem.py configs/gem_preparation.yaml
```

Uses `src/Duplicate_Model.py`. Same as *Step 1* of `notebooks/Tutorial_Aida_LC.ipynb`.

AMNs require every flux to be non-negative. Reversible reactions in iML1515 are therefore split into a forward and a backward reaction, and exchange reactions are split into uptake (`_i`) and secretion (`_o`) reactions.

| Input | Output |
|---|---|
| `data/raw/iML1515.xml` | `data/interim/iML1515_duplicated.xml`: the irreversible model used for simulation |
| | `data/interim/iML1515_medium_corrected.csv`: default medium of the duplicated model (`rxn`, `value`) |

The config also sets which compartment transitions count as uptake or secretion (`io_dict`) and which small molecules to ignore when deciding a reaction's direction (`unsignificant_mols`).

---

## Step 2 — Pre-process the Aida & Ying dataset

```bash
python scripts/prepare_experimental_dataset.py configs/experimental_dataset.yaml
```

Runs the three sub-steps below in order. Same as *Step 2* of `notebooks/Tutorial_Aida_LC.ipynb`. Needs the corrected medium from step 1.

### 2.1 Map compounds to GEM exchange reactions

`src/data_preprocess.py` maps each experimental compound (e.g. `Glucose (mM)`) to one or more iML1515 exchange reactions, with stoichiometry for salts. It then averages replicate growth rates (keeping `r_info == 0`), removes duplicate media, and renames columns to the `_i` uptake reactions of the duplicated model. The mapping rules and these file names are set in `src/data_preprocess.py`, not in the config.

Outputs, in `data/interim/`:

| File | Content |
|---|---|
| `aidaying_compound_mapping.csv` | `AIDA_compound, gem_rxn, stoichiometry, match_type` |
| `aidaying_media_growth.csv` | Media (as `_i` reactions) + `avg_growth`, with `Condition ID` |
| `aidaying_media_growth_amns.csv` | Same data without IDs: compound columns, then `GR_AVG` |

### 2.2 Build the medium parameter file

Step 3 uses this file to decide which compounds can vary, and over what range. It is built from the parameter file of Faure et al. (`data/external/faure_parameters.csv`):

1. Drop Faure's extra carbon sources (`faure_drop`) and the compounds it shares with Aida & Ying.
2. Keep the remaining environmental compounds (O₂, CO₂, H⁺, trace metals, …) with Faure's values.
3. Add every Aida & Ying compound, with `level` and `max_value` taken from its class in `compound_classes`:

| Class | `level` | `max_value` |
|---|---|---|
| carbon sources | 100 | 10 |
| vitamins | 100 | 0.0001 |
| major salts | 100 | 5.0 |
| minor ions | 100 | 0.0001 |
| essential (always on) | 1 | 2.2 |
| anything unlisted (amino acids) | 100 | 0.1 |

See [data.md](data.md#medium-parameter-file) for the file format.

Output: `data/processed/aidaying_parameters.csv`.

### 2.3 Pad the experimental dataset

The reservoir's input layer has one column per compound in the parameter file, so the experimental dataset must contain the same compounds. The environmental compounds that Aida & Ying did not measure are added as constant columns, before `GR_AVG`, using the concentrations under `padding` in the config. The first condition (zero growth) is then dropped (`drop_first_row`). At the end, the script prints a warning if the padded dataset and the parameter file contain different compounds.

Output: `data/interim/aidaying_media_growth_amns_padded.csv` (48 compounds + `GR_AVG`).

> `notebooks/Tutorial_Aida_YZ.ipynb` is a variant that keeps extra carbon sources. It produces `*_with_carbon.csv` files, which are used by `configs/reservoir_training_dataset_with_carbon.yaml`.

---

## Step 3 — Generate the pFBA-simulated training set

```bash
python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml
```

The reservoir has to learn the GEM's constraints, which takes far more examples than the ~900 experimental conditions. This step builds that dataset with pFBA:

1. Load the experimental media (`expname`) and the parameter file (`mediumfile`).
2. For each condition, list its **variable** compounds (`level > 1` in the parameter file and non-zero in the experiment). Duplicate combinations are removed, giving the `varmed` list.
3. For each entry of `varmed`, draw `size_i` random media in which only those compounds vary, then solve each with pFBA on the duplicated GEM. A draw whose objective is below `cobra_min_objective` is redrawn, up to 20 attempts. Conditions are spread over `num_workers` processes, each seeded with `seed + condition_index`.
4. Optionally drop samples with biomass below `biomass_threshold` (`filter_zero_biomass: true`).
5. Save `X` (uptake bounds) and `Y` (all fluxes), together with the model matrices (S, Pin, Pout, …).

| Input | Output |
|---|---|
| `data/interim/<cobraname>.xml` | `data/processed/<run_id>/<output_prefix>_<mediumbound>.npz` |
| `data/interim/<expname>.csv` | `data/processed/<run_id>/<output_prefix>_<mediumbound>.xml` |
| `data/processed/<mediumfile>.csv` | |

This step is CPU-bound and runs for hours at production sizes. Run it as a batch job ([hpc.md](hpc.md)), with `--num-workers` set to `--cpus-per-task`.

---

## Step 4 — Train the AMN Reservoir

```bash
python scripts/train_eval_reservoir.py configs/reservoir_params.yaml
python scripts/train_eval_reservoir.py --seed 1 configs/reservoir_params.yaml
```

1. Load the `.npz` from step 3 (`trainingfile`, relative to `data/processed/`).
2. Drop zero-biomass samples, then hold out `ratio_test` of the rest as an independent test set.
3. Run `xfold`-fold cross-validation on the remaining samples (stratified on binned growth if `stratify: true`).
4. Save the trained reservoir and evaluate it on the held-out test set.

Outputs, under `models/<reservoirname>_<model_type>/`:

| Path | Content |
|---|---|
| `reservoir/<name>_model.keras`, `reservoir/<name>_param.csv` | Trained reservoir (weights + hyperparameters, including the training-set path) |
| `result/cv_fold<k>_history.csv` | Per-epoch training/validation losses per fold |
| `result/cv_fold<k>_predictions.csv` | Out-of-fold predictions |
| `result/cv_fold_metrics.csv` | R² / Q² per fold |
| `result/test_set_predictions.csv` | Predictions on the independent test set |
| `result/console_log_<timestamp>.txt` | Full console output of the run |

See [models.md](models.md) for the architecture and the `model_type` options.

---

## Step 5 — Train the Media Network + frozen Reservoir ensemble

```bash
python scripts/train_eval_ensemble.py configs/ensemble_params.yaml
python scripts/train_eval_ensemble.py --seeds 3 configs/ensemble_params.yaml
```

1. Load the padded experimental dataset (`trainname`, relative to `data/interim/`) and reorder its columns to the reservoir's medium order.
2. Transform the media as `X ← log10(1 + X)`.
3. For each seed in `0 … number_of_seeds-1`, build an `RC_Model`: a trainable Media Network (`n_hidden_prior` × `hidden_dim_prior`) feeding the frozen reservoir. Train it with `xfold`-fold CV on measured growth.
4. Report Q² per seed and the mean ± std across seeds.
5. Rescale each seed's predicted uptake fluxes `Vin` back to flux units and save them with the measured growth.

Outputs, under `models/<reservoirname>/`:

| Path | Content |
|---|---|
| `result_ensemble/seed<k>_cv_*` | Per-seed fold histories, predictions and metrics |
| `result_ensemble/<reservoirname>_RC/Q2_across_seeds.csv` | One Q² per seed |
| `result_ensemble/<reservoirname>_RC/PRED_objective_across_seeds.csv` | Predicted growth, seeds × conditions |
| `result_medium/pred_medium_seed<k>.csv` | Predicted `Vin` + `GR_AVG`, with the experimental CSV header |
| `result_ensemble/console_log_<timestamp>.txt` | Console output |

> **Note:** `stratify` is passed to `RC_Model`, but `train_evaluate_with_logging` (which builds the folds) is not called with it. Ensemble CV therefore uses plain KFold even when `stratify: true`.

---

## Step 6 — Check the predicted media with COBRApy

```bash
python scripts/evaluate_medium_pred.py configs/evaluate_medium_pred.yaml
python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml
```

This step tests whether the Media Network's uptake fluxes, used as bounds in a standard GEM, give accurate growth predictions. For each value in `scaler_list`, every condition's fluxes are multiplied by the scaler (fluxes below 1e-4 get +1e-4), FBA/pFBA is run, and R² against measured growth is computed.

With `use_experimental: true`, the raw experimental concentrations (`expname`) are used instead. This gives the baseline without the Media Network.

Outputs, under `models/<reservoirname>/result_evaluation/`: `evaluation_results_<timestamp>.csv` (`scaler,r2`) and `evaluation_log_<timestamp>.txt`.

---

## How names connect between steps

Each config refers to the output of the previous step by name. The table shows how these names line up for the default configs:

| Config key | Value | Resolves to |
|---|---|---|
| `reservoir_training_dataset.yaml` → `run_id`, `output_prefix`, `mediumbound` | `aidaying_iml1515_UB`, `aidaying_iml1515`, `UB` | `data/processed/aidaying_iml1515_UB/aidaying_iml1515_UB.npz` |
| `reservoir_params.yaml` → `trainingfile` | `aidaying_iml1515_UB/aidaying_iml1515_UB` | the `.npz` above |
| `reservoir_params.yaml` → `reservoirname` + `model_type` | `aidaying_iml1515_UB` + `AMN_QP` | `models/aidaying_iml1515_UB_AMN_QP/` |
| `ensemble_params.yaml` → `reservoirname` | `aidaying_iml1515_UB_AMN_QP` | the folder above, *including* the `_AMN_QP` suffix |
| `evaluate_medium_pred.yaml` → `reservoirname`, `predfilename` | `aidaying_iml1515_UB_AMN_QP`, `pred_medium_seed1` | `models/aidaying_iml1515_UB_AMN_QP/result_medium/pred_medium_seed1.csv` |

> `predfilename: pred_medium_seed1` only exists when step 5 ran with `number_of_seeds ≥ 2`, because seeds are numbered from 0.

---

## Diagnostics

```bash
python scripts/investigate_datasets.py
```

Compares two step-3 training sets: the original Faure et al. `iML1515_UB` and the new `aidaying_iml1515_UB`. It prints X/Y statistics, zero-biomass fraction, matrix sparsity, and how much the first dense layer must amplify `X @ Pin` to reach the flux scale. To compare other datasets, edit `FILES` at the top of the script.
