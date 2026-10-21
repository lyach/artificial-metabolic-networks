# Configuration Reference

Each script under `scripts/` reads one YAML file:

- [Prepare GEM](#prepare_gempy)
- [Prepare experimental dataset](#prepare_experimental_datasetpy)
- [Create reservoir training dataset](#create_reservoir_training_datasetpy)
- [Train and evaluate reservoir](#train_eval_reservoirpy)
- [Train and evaluate ensemble](#train_eval_ensemblepy)
- [Evaluate medium prediction](#evaluate_medium_predpy)

### Adding a new experiment

Copy the closest config, change the names that identify the run (`run_id`, `reservoirname`, `trainname`, …), and keep the names consistent across steps ([pipeline.md](pipeline.md#how-names-connect-between-steps)). Step 4 does not version its outputs: re-running with the same `reservoirname` and `model_type` overwrites the saved reservoir and the CV files.

---

## `prepare_gem.py`

Config: `configs/gem_preparation.yaml`

| Key | Required | Default | Description |
|---|---|---|---|
| `cobraname` | ✓ | | Input GEM in `data/raw/` |
| `output_model` | ✓ | | Duplicated model written to `data/interim/<output_model>.xml` |
| `output_medium` | ✓ | | Corrected medium written to `data/interim/<output_medium>.csv`. Step 2 expects the name `iML1515_medium_corrected` |
| `io_dict` | ✓ | | `{_i: [[from, to], …], _o: […]}`: compartment transitions that define uptake and secretion. `null` = outside the system |
| `unsignificant_mols` | ✓ | | Metabolites ignored when deciding a reaction's direction |
| `data_dir` | | `<project>/data` | Root holding `raw/` and `interim/`. CLI: `--data-dir` |

---

## `prepare_experimental_dataset.py`

Config: `configs/experimental_dataset.yaml`

| Key | Required | Default | Description |
|---|---|---|---|
| `cobraname` | ✓ | | GEM in `data/raw/`, used to validate the compound mapping |
| `faure_parameters` | ✓ | | Faure et al. parameter file in `data/external/` |
| `output_parameters` | ✓ | | Parameter file written to `data/processed/` |
| `faure_drop` | ✓ | | Faure compounds removed from the parameter file |
| `compound_classes` | ✓ | | `{class: {level, max_value, compounds: […]}}` for the Aida & Ying compounds |
| `default_class` | ✓ | | `{level, max_value}` for Aida & Ying compounds not listed in any class |
| `output_padded` | ✓ | | Padded dataset written to `data/interim/` |
| `padding` | ✓ | | `{compound: value}` columns added to the dataset, in this order |
| `drop_first_row` | | `true` | Drop the first condition (zero growth) |
| `data_dir` | | `<project>/data` | Root holding `raw/`, `external/`, `interim/`, `processed/`. CLI: `--data-dir` |

The compound mapping rules and the step 2.1 file names are set in `src/data_preprocess.py`.

---

## `create_reservoir_training_dataset.py`

Configs: `configs/reservoir_training_dataset.yaml`, `configs/reservoir_training_dataset_with_carbon.yaml`

| Key | Required | Default | Description |
|---|---|---|---|
| `run_id` | | random 8-char hex | Output sub-folder under `data/processed/` |
| `seed` | ✓ | | Global seed; condition *i* uses `seed + i` |
| `cobraname` | ✓ | | GEM in `data/interim/` (e.g. `iML1515_duplicated`) |
| `mediumbound` | ✓ | | `UB` (medium values are upper bounds) or `EB` (exact bounds) |
| `expname` | ✓ | | Experimental CSV in `data/interim/` |
| `mediumsize` | ✓ | | Number of compound columns in `expname` (before `GR_AVG`) |
| `mediumfile` | ✓ | | Medium parameter file in `data/processed/` ([format](data.md#medium-parameter-file)) |
| `size_i` | ✓ | | pFBA samples per unique variable-compound combination |
| `output_prefix` | ✓ | | Output file name is `<output_prefix>_<mediumbound>.npz` |
| `num_workers` | | `1` | Worker processes. CLI: `--num-workers / -w` |
| `chunk_size` | | `10` | Samples per task. Smaller chunks balance the load better, so slow conditions don't stall the end of the run |
| `cobra_min_objective` | | `0.01` | Random media with a lower objective are redrawn (max 20 attempts) |
| `cap_varmed` | | `false` | Cap the per-compound scale multiplier |
| `max_varmed_scale` | | `5` | Cap value, used only if `cap_varmed: true` |
| `filter_zero_biomass` | | `false` | Drop samples with biomass below `biomass_threshold` after simulation |
| `biomass_threshold` | | `0.01` | Used only if `filter_zero_biomass: true` |
| `reduce` | | `false` | Reduce the model before saving |
| `verbose` | | `true` | Print progress |

`size` (number of experimental conditions) appears in the shipped configs for reference only. The script does not read it.

---

## `train_eval_reservoir.py`

Configs: `configs/reservoir_params.yaml`, `configs/reservoir_params_og.yaml` (original Faure et al. `iML1515_UB` dataset)

| Key | Required | Default | Description |
|---|---|---|---|
| `seed` | ✓ | | NumPy seed (test split, CV shuffling). CLI: `--seed / -s` |
| `ratio_test` | ✓ | | Fraction held out as independent test set |
| `reservoirname` | ✓ | | Output folder is `models/<reservoirname>_<model_type>/` |
| `trainingfile` | ✓ | | Step-3 `.npz`, relative to `data/processed/` |
| `objective` | ✓ | | List of objective reaction IDs (biomass) |
| `model_type` | ✓ | | `AMN_QP`, `AMN_LP`, `AMN_Wt` or `ANN_Dense` ([models.md](models.md)) |
| `timestep` | ✓ | | Iterations of the mechanistic layer |
| `learn_rate` | ✓ | | Step size of the mechanistic layer's gradient descent |
| `scaler` | ✓ | | Scale inputs by their max |
| `n_hidden`, `hidden_dim` | ✓ | | Neural layer depth and width |
| `train_rate` | ✓ | | Adam learning rate |
| `dropout` | ✓ | | Dropout rate |
| `epochs` | ✓ | | Maximum epochs |
| `xfold` | ✓ | | Number of CV folds |
| `batch_size` | ✓ | | Batch size |
| `es` | ✓ | | Early stopping |
| `y_scale` | | `false` | Scale Y |
| `per_feature_norm` | | `false` | Normalise each input column by its own max instead of the global max |
| `stratify` | | `false` | Stratify CV folds on binned growth |
| `stratify_bins` | | `10` | Quantile bins for stratification |
| `verbose` | | `true` | CLI: `--verbose / -v` |

---

## `train_eval_ensemble.py`

Config: `configs/ensemble_params.yaml`

| Key | Required | Default | Description |
|---|---|---|---|
| `number_of_seeds` | ✓ | | Repeat training with seeds `0 … n-1`. CLI: `--seeds / -n` |
| `trainname` | ✓ | | Padded experimental CSV in `data/interim/` |
| `reservoirname` | ✓ | | Full reservoir folder name in `models/`, e.g. `aidaying_iml1515_UB_AMN_QP` |
| `n_input` | ✓ | | Number of compound columns in `trainname` |
| `n_hidden_prior`, `hidden_dim_prior` | ✓ | | Media Network depth and width |
| `activation_prior` | ✓ | | Media Network activation (e.g. `relu`) |
| `epochs`, `train_rate`, `batch_size`, `xfold`, `es` | ✓ | | Same meaning as for the reservoir |
| `stratify`, `stratify_bins` | | `false`, `10` | Stored on `RC_Model`; currently **not** used for fold creation ([details](pipeline.md#step-5--train-the-media-network--frozen-reservoir-ensemble)) |
| `verbose` | | `true` | CLI: `--verbose / -v` |

---

## `evaluate_medium_pred.py`

Config: `configs/evaluate_medium_pred.yaml`

| Key | Required | Default | Description |
|---|---|---|---|
| `cobraname` | ✓ | | GEM in `data/raw/` (the original, non-duplicated model) |
| `reservoirname` | ✓ | | Reservoir folder in `models/` |
| `predfilename` | ✓ | | File in `models/<reservoirname>/result_medium/` |
| `mediumsize` | ✓ | | Number of compound columns |
| `seed` | | `10` | CLI: `--seed / -s` |
| `mediumbound` | | `UB` | Bound type |
| `method` | | `EXP` | How the CSV is read (`EXP` = columns are X then Y) |
| `method_simulation` | | `pFBA` | `FBA`, `pFBA` or `EXP`. CLI: `--method` |
| `scaler_list` | | `[1.0]` | Flux multipliers to try. CLI: `--scaler 0.5 1.0 …` |
| `use_experimental` | | `false` | Use `expname` instead of the predictions |
| `expname` | | `''` | Experimental CSV in `data/interim/` |
| `verbose` | | `true` | Prints every compound of every sample, which is very long. CLI: `--verbose / -v` |

---
