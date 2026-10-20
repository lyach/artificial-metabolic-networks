# Data — Sources, Formats and Layout

## Folder layout

```
data/
  raw/         ← Unmodified inputs that are versioned (iML1515.xml)
  external/    ← Third-party tables (Aida & Ying 2025, Faure et al. 2023)
  interim/     ← Outputs of steps 1–2: duplicated GEM, mapped/padded experimental CSVs   [git-ignored]
  processed/   ← Medium parameter files and step-3 training sets (<run_id>/*.npz)       [git-ignored]
models/        ← Outputs of steps 4–6, one folder per reservoir                         [git-ignored]
```

`data/interim/`, `data/processed/` and `models/` are not versioned. On a fresh clone, regenerate them with steps 1–3 ([pipeline.md](pipeline.md)).

---

## External sources

| File | Source | Content |
|---|---|---|
| `data/raw/iML1515.xml` | BiGG | *E. coli* K-12 MG1655 genome-scale model (SBML) |
| `data/external/aidaying_medium.csv` | [Aida & Ying, 2025](https://doi.org/10.1038/s41597-025-05356-3) | One row per assay: `Label, Assay ID, Condition ID` + compound concentrations in mM (e.g. `Glucose (mM)`) |
| `data/external/aidaying_growth.csv` | Aida & Ying, 2025 | One row per growth curve: `Label, Assay ID, Condition ID, K, r, K_info, r_info`. `r` is the growth rate, and `r_info == 0` marks a reliable fit |
| `data/external/faure_parameters.csv` | [Faure et al., 2023](https://www.nature.com/articles/s41467-023-40380-0) | Original medium parameter file for iML1515 |

---

## Experimental dataset (`data/interim/aidaying_media_growth_amns_padded.csv`)

The table the models train on. It has one row per unique condition, about 900 in total.

| Columns | Content |
|---|---|
| `EX_<met>_e_i` (×48) | Concentration (mM) mapped onto the uptake reaction of the duplicated GEM. Padding columns hold fixed default values or 0 |
| `GR_AVG` | Mean measured growth rate across reliable replicates |

Compound columns come first and growth comes last. This is the `EXP` layout expected by `TrainingSet(..., method='EXP', mediumsize=48)` and `read_XY(file, 48)`.

---

## Medium parameter file

`data/processed/aidaying_parameters.csv` tells step 3 how to draw random media. It is a transposed table: the first column holds row labels, and each further column is one uptake reaction.

```
name,          EX_co2_e_i, ..., EX_glc__D_e_i, EX_ala__L_e_i, ...
level,         1.0,        ..., 100.0,         100.0,         ...
max_value,     2.2,        ..., 10.0,          0.1,           ...
ratio_drawing, ...
comment,       ...
```

| Row | Meaning |
|---|---|
| `level` (`levmed`) | `1` → **fixed** compound, always present at `max_value`. `> 1` → **variable** compound, drawn uniformly from `k · max_value / (level − 1)` for `k = 1 … level − 2` |
| `max_value` (`valmed`) | Upper value of the uptake bound |
| `ratio_drawing` (`ratmed`) | Fraction of variable compounds switched on when no explicit list is given. Unused in this pipeline, because step 3 always passes the experiment's own list of variable compounds |
| `comment` | Free text |

When a condition has *n* variable compounds switched on, each value is also multiplied by *n + 1*. Setting `cap_varmed` limits this multiplier to `max_varmed_scale`.

---

## Reservoir training set (`data/processed/<run_id>/<prefix>_<UB|EB>.npz`)

Written by `TrainingSet.save()` in step 3, next to an `.xml` copy of the model. Main arrays:

| Key | Shape | Content |
|---|---|---|
| `X` | samples × medium | Uptake bounds used for each pFBA run |
| `Y` | samples × reactions | All pFBA fluxes (the reservoir later keeps only `objective` as its target) |
| `medium` | medium | Ordered uptake reaction IDs (the column order of `X`) |
| `levmed`, `valmed`, `ratmed` | | Copied from the parameter file |
| `S` | metabolites × reactions | Stoichiometric matrix |
| `Pin` | medium × reactions | Projects fluxes onto the medium uptake reactions |
| `Pout` | measure × reactions | Projects fluxes onto the measured reactions |
| `S_int, S_ext, Q, P, b_int, b_ext, Sb, c` | | Matrices for the LP/QP mechanistic layers |
| `cobraname`, `mediumname`, `mediumbound`, `objective`, `method`, `size`, `reduce` | | Metadata |

To load one:

```python
from src.Build_Dataset import TrainingSet
ts = TrainingSet()
ts.load('data/processed/aidaying_iml1515_UB/aidaying_iml1515_UB')   # no extension
```
