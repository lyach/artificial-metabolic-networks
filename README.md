# Artificial Metabolic Networks 

A re-implementation of the **AMN-Reservoir** architecture of [Faure et al. (2023)](https://www.nature.com/articles/s41467-023-40380-0), trained on a large-scale, experimental dataset of *Escherichia coli* growth [(Aida & Ying, 2025)](https://doi.org/10.1038/s41597-025-05356-3). The task is:

```text
(medium composition)  -->  uptake flux bounds  -->  growth rate
```

## Navigation

1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Framework components](#framework-components)
4. [Quick start](#quick-start)
5. [Project structure](#project-structure)
6. [Documentation](#documentation)

## Motivation

Genome-scale models (GEMs) aim to predict an organism's phenotype, such as growth, but require uptake flux bounds for each nutrient in its growth environment, like the culture medium. Experiments report medium in units of concentration, not uptake fluxes, and the mapping between the two is an open research question.

The AMN-Reservoir aims to learn the concentration-to-flux mapping from measured growth rates, in two stages:

1. **The frozen Reservoir.** An Artificial Metabolic Network (AMN) is pre-trained on a large set of parsimonious Flux Balance Analysis (pFBA) simulations, so that it learns the stoichiometric and bound constraints of the GEM. Its weights are then frozen. The result behaves like an FBA solver but lets gradients flow through it.
2. **The Media Network.** A trainable network is placed in front of the Reservoir. It takes the experimental medium composition ($C_{med}$) and predicts uptake flux bounds ($V_{in}$). The error between predicted and measured growth is backpropagated through the frozen Reservoir, so only the Media Network is updated. It learns how nutrient concentrations translate into uptake rates.

This project contributes with:
- **A data pipeline** that maps the Aida & Ying media (~900 conditions, 48 compounds) onto the exchange reactions of *E. coli* iML1515.
- **Scripts for configurable training** of the Reservoir and the Media Network ensemble.
- **Validation** of the predicted uptake fluxes, by feeeding them back into COBRApy to test whether they improve plain FBA growth predictions.

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/lyach/BactoCat-AMN.git
   cd BactoCat-AMN
   ```

2. Create the environment (Python 3.9).

   **Using `uv`**
   ```bash
   # Install uv (first time only)
   curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
   # or
   pip install uv                                    # Windows

   # Create .venv and install the dependencies from pyproject.toml
   uv venv .venv --python 3.9
   uv sync
   source .venv/bin/activate                         # Windows: .venv\Scripts\activate
   ```


## Framework components

| Component | Role | Code | Docs |
|---|---|---|---|
| **GEM preparation** | Split reversible and exchange reactions so that all fluxes are non-negative (AMN-compatible) | `scripts/prepare_gem.py`, `src/Duplicate_Model.py` | [pipeline.md-1](docs/pipeline.md#step-1--make-the-gem-amn-compatible) |
| **Experimental dataset** | Map Aida & Ying compounds to exchange reactions, average growth, pad to the reservoir's inputs | `scripts/prepare_experimental_dataset.py`, `src/data_preprocess.py` | [pipeline.md-2](docs/pipeline.md#step-2--pre-process-the-aida--ying-dataset), [data.md](docs/data.md) |
| **Reservoir training set** | Sample random media around each experimental condition and solve them with pFBA | `scripts/create_reservoir_training_dataset.py`, `src/Build_Dataset.py` | [pipeline.md-3](docs/pipeline.md#step-3--generate-the-pfba-simulated-training-set) |
| **AMN Reservoir** | Train the AMN to mimic FBA (k-fold CV + held-out test set) | `scripts/train_eval_reservoir.py`, `src/Build_Model.py` | [models.md](docs/models.md) |
| **Media Network ensemble** | Learn medium → uptake bounds through the frozen Reservoir | `scripts/train_eval_ensemble.py` | [models.md](docs/models.md#2-ensemble-media-network--frozen-reservoir-step-5) |
| **COBRApy evaluation** | Constrain the GEM with the predicted uptake bounds and compare FBA growth with measurements | `scripts/evaluate_medium_pred.py` | [pipeline.md-6](docs/pipeline.md#step-6--check-the-predicted-media-with-cobrapy) |



## Quick start

Every step is a script configured by a YAML file in `configs/`.

### 1. Prepare the inputs

```bash
# Make the GEM AMN-compatible
python scripts/prepare_gem.py configs/gem_preparation.yaml
# → data/interim/iML1515_duplicated.xml, data/interim/iML1515_medium_corrected.csv

# Map, pad and parametrise the Aida & Ying dataset
python scripts/prepare_experimental_dataset.py configs/experimental_dataset.yaml
# → data/interim/aidaying_media_growth_amns_padded.csv   (experimental media + growth)
# → data/processed/aidaying_parameters.csv               (sampling ranges for step 3)
```

### 2. Generate the Reservoir training set (CPU)

```bash
python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml
# → data/processed/aidaying_iml1515_UB/aidaying_iml1515_UB.npz
```

### 3. Train the Reservoir (GPU)

```bash
python scripts/train_eval_reservoir.py configs/reservoir_params.yaml
# → models/aidaying_iml1515_UB_AMN_QP/{reservoir,result}/
```

### 4. Train the Media Network ensemble (GPU)

```bash
python scripts/train_eval_ensemble.py --seeds 3 configs/ensemble_params.yaml
# → models/aidaying_iml1515_UB_AMN_QP/{result_ensemble,result_medium}/
```

### 5. Evaluate the predicted media with COBRApy

```bash
python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml
# → models/aidaying_iml1515_UB_AMN_QP/result_evaluation/
```


### Utils
```bash
python utils/investigate_datasets.py    # compare two Reservoir training sets (X/Y stats, sparsity, scaling)
```


## Project structure
```
configs/
  gem_preparation.yaml              ← Step 1: GEM duplication
  experimental_dataset.yaml         ← Step 2: compound classes, padding values
  reservoir_training_dataset*.yaml  ← Step 3: pFBA training-set generation
  reservoir_params*.yaml            ← Step 4: Reservoir hyperparameters 
  ensemble_params.yaml              ← Step 5: Media Network hyperparameters
  evaluate_medium_pred.yaml         ← Step 6: COBRApy evaluation

data/
  raw/                              ← iML1515.xml
  external/                         ← Aida & Ying media/growth, Faure et al. parameter file
  interim/                          ← Duplicated GEM, mapped and padded experimental CSVs
  processed/                        ← Medium parameter files and training sets (.npz)

models/
  <reservoir>_<model_type>/
    reservoir/                      ← Trained Reservoir
    result/                         ← Reservoir results
    result_ensemble/                ← Ensemble results
    result_medium/                  ← Predicted uptake fluxes
    result_evaluation/              ← COBRApy results

src/
  Build_Dataset.py                  ← TrainingSet, COBRApy sampling (pFBA)
  Build_Model.py                    ← AMN / RC models, mechanistic layers, training and CV
  Build_Model_Wrappers.py           ← Wrapper with per-fold logging
  Duplicate_Model.py                ← GEM duplication for non-negative fluxes
  data_preprocess.py                ← Aida & Ying - GEM reaction mapping
  plots.py, utils.py                ← Learning curves, plots, parameter-file helpers

scripts/
  _common.py                        ← Shared helpers for paths, configs, logging
  prepare_gem.py
  prepare_experimental_dataset.py
  create_reservoir_training_dataset.py
  train_eval_reservoir.py
  train_eval_ensemble.py
  evaluate_medium_pred.py
  investigate_datasets.py
docs/                               ← Documentation
```


## Documentation
- [Pipeline walkthrough](docs/pipeline.md)
- [Configuration reference](docs/configuration.md)
- [Data formats](docs/data.md)
- [Models](docs/models.md)


## References
- Faure, L., Mollet, B., Liebermeister, W., & Faulon, J. (2023). A neural-mechanistic hybrid approach improving the predictive power of genome-scale metabolic models. *Nature Communications*, *14*(1), 4669. https://doi.org/10.1038/s41467-023-40380-0
- Aida, H., & Ying, B. (2025). Population Dynamics of Escherichia coli Growing under Chemically Defined Media. *Scientific Data*, *12*(1), 984. https://doi.org/10.1038/s41597-025-05356-3 
