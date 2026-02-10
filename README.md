# Artificial Metabolic Networks for BactoCat

This repository provides a re-implementation of the AMN-Reservoir architecture described by [Faure et al. (2023)](https://www.nature.com/articles/s41467-023-40380-0), using a large-scale dataset of *Escherichia coli* growth measurements under well-defined laboratory conditions [(Aida & Ying, 2025)](https://doi.org/10.1038/s41597-025-05356-3).

This hybrid framework is designed to bridge the gap between experimental media compositions and metabolic flux predictions. This addresses the **unknown uptake flux** problem in Constraint-Based Modeling.

### The AMN-Reservoir

The AMN-Reservoir solves this by embedding a metabolic model within a machine learning architecture. It uses a two-step learning process to map realistic uptake fluxes to its experimental media.

There are 2 main components to this framework: 

1. **The Frozen Reservoir**:
First, an Artificial Metabolic Network (AMN) is pre-trained on a large dataset of FBA simulations. This model learns the stoichiometric constraints of the metabolic network. Once trained, its parameters are frozen. This frozen model acts as a differentiable "Reservoir" that mimics the behavior of an FBA solver but allows for gradient backpropagation.
2. **The Media Network**:
A trainable neural network is attached prior to the Reservoir. It takes experimental media composition ($C_{med}$) as input and predicts uptake flux bounds ($V_{in}$). The architecture is trained on measured growth rates. The error between the predicted and measured growth is backpropagated through the frozen Reservoir, but only updates the Media Network. This forces the neural layer to learn the complex, non-linear relationship between the presence of nutrients and their specific uptake rates.


## Environment set-up 

The following steps will set up a virtual environment using the `uv` Python package manager.

**1. Install uv**

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows (via `pip`):
```bash
pip install uv
```

**2. Navigate to the repository**

```bash
cd ~/BactoCat-AMN
```

**3. Create a virtual environment**

```bash
uv venv .venv --python 3.9
```

**4. Activate venv**

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```
uv sync
```
   - This command synchronizes and installs all dependencies listed in `pyproject.toml`