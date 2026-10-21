#!/usr/bin/env python3
"""
Diagnostic: compare two AMN Reservoir training sets (``.npz`` from step 3).

Loads each dataset with ``TrainingSet`` and prints statistics on X, Y, the
mechanistic matrices (S, Pin, Pout) and the scale of the V0 initialisation,
followed by a side-by-side summary table.

Usage:
    python scripts/investigate_datasets.py
"""

import numpy as np

from _common import DATA_PROCESSED_DIR

from src.Build_Dataset import TrainingSet, MaxScaler

# Datasets to compare: label -> path without .npz
FILES = {
    'OLD (iML1515_UB)': DATA_PROCESSED_DIR / 'iML1515' / 'iML1515_UB',
    'NEW (aidaying_UB)': DATA_PROCESSED_DIR / 'aidaying_iml1515_UB' / 'aidaying_iml1515_UB',
}
OBJECTIVE = ['BIOMASS_Ec_iML1515_core_75p37M']

# (results key, row label) for the summary table
SUMMARY_METRICS = [
    ('n_samples', 'Dataset size'),
    ('n_features', 'Input features (X cols)'),
    ('x_scaler', 'X scaler (max X)'),
    ('y_scaler', 'Y scaler (max biomass)'),
    ('Y_mean', 'Biomass mean'),
    ('Y_std', 'Biomass std'),
    ('Y_max', 'Biomass max'),
    ('Y_zero_frac', 'Fraction zero-biomass'),
    ('Yall_mean_abs', 'Mean |flux| (all reactions)'),
    ('V0_scl_mean_abs', 'Mean |V0| after X scaling'),
    ('amplification_needed', 'Dense amplification needed'),
]


# ── Printing helpers ─────────────────────────────────────────────────────────

def print_array_stats(arr, name=''):
    """Print shape, range, mean/std and non-zero fraction of ``arr``."""
    flat = np.asarray(arr, dtype=float).flatten()
    nonzero = flat[flat != 0]
    print(f"  {name}: shape={np.shape(arr)}  "
          f"min={flat.min():.4g}  max={flat.max():.4g}  "
          f"mean={flat.mean():.4g}  std={flat.std():.4g}  "
          f"nnz={len(nonzero)}  frac_nonzero={len(nonzero) / len(flat):.3f}")


def print_histogram(arr, n_bins=10, label=''):
    """Print a text histogram of ``arr``."""
    counts, edges = np.histogram(np.asarray(arr).flatten(), bins=n_bins)
    print(f"\n  Histogram of {label}:")
    for i in range(n_bins):
        bar = '█' * int(40 * counts[i] / max(counts, default=1))
        print(f"    [{edges[i]:8.3f}, {edges[i + 1]:8.3f})  {counts[i]:6d}  {bar}")


def print_sparsity(mat, name=''):
    """Print the number and fraction of non-zero entries of ``mat``."""
    mat = np.asarray(mat, dtype=float)
    nz = np.count_nonzero(mat)
    print(f"  {name}: shape={mat.shape}  nnz={nz}  sparsity={1 - nz / mat.size:.4f}  "
          f"frac_nonzero={nz / mat.size:.4f}")


# ── Analysis ─────────────────────────────────────────────────────────────────

def investigate(label, filepath):
    """
    Print diagnostics for one training set and return summary metrics.

    Args:
        label (str): Display name.
        filepath (Path): Dataset path without ``.npz``.

    Returns:
        dict: Metrics keyed as in ``SUMMARY_METRICS``.
    """
    print("\n" + "=" * 70)
    print(f"  DATASET: {label}")
    print(f"  File   : {filepath}.npz")
    print("=" * 70)

    ts = TrainingSet()
    ts.load(str(filepath))
    all_reactions = list(ts.measure)  # filter_measure overwrites ts.measure
    ts.filter_measure(measure=OBJECTIVE, verbose=True)

    print("\n-- Medium metadata --")
    print(f"  mediumbound : {ts.mediumbound}")
    print(f"  medium size : {len(ts.medium)}")
    print(f"  levmed      : min={min(ts.levmed):.1f}  max={max(ts.levmed):.1f}  "
          f"n_variable (>1)={sum(lev > 1 for lev in ts.levmed)}")
    print(f"  valmed      : min={min(ts.valmed):.4g}  max={max(ts.valmed):.4g}  "
          f"mean={np.mean(ts.valmed):.4g}")
    print(f"  ratmed      : {ts.ratmed}")

    print("\n-- X (medium inputs) --")
    print_array_stats(ts.X, 'X raw')
    X_scaled, x_scaler = MaxScaler(ts.X.copy())
    print(f"  X scaler (max)  : {x_scaler:.4g}")
    print_array_stats(X_scaled, 'X scaled')

    print("\n-- Y (biomass = filtered measure) --")
    print_array_stats(ts.Y, 'Y (biomass)')
    _, y_scaler = MaxScaler(ts.Y.copy())
    print(f"  Y scaler (max)  : {y_scaler:.4g}")
    zero_biomass = int(np.sum(ts.Y.flatten() == 0))
    n_samples = ts.Y.shape[0]
    print(f"  Zero-biomass samples: {zero_biomass} / {n_samples} "
          f"({100 * zero_biomass / n_samples:.1f}%)")
    print_histogram(ts.Y, n_bins=10, label='Y (biomass)')

    print("\n-- Yall (all fluxes) --")
    print_array_stats(ts.Yall, 'Yall')
    print("  Yall max per reaction (top 10):")
    rx_max = np.max(np.abs(ts.Yall), axis=0)
    for idx in np.argsort(rx_max)[-10:][::-1]:
        rxn = all_reactions[idx] if idx < len(all_reactions) else str(idx)
        print(f"    reaction={rxn}  max_abs={rx_max[idx]:.4g}")
    print_sparsity(ts.Yall, 'Yall')

    print("\n-- Mechanistic matrices --")
    print_sparsity(ts.S, 'S   (stoichiometric)')
    print_sparsity(ts.Pin, 'Pin (medium→V)')
    print_sparsity(ts.Pout, 'Pout (V→measure)')

    # V0 = X @ Pin, as in get_V0 (tf.linalg.matmul(inputs, Pin)).
    # Compare its scale with the target fluxes Yall.
    print("\n-- V0 initialization check --")
    print_array_stats(ts.X @ ts.Pin, 'V0_raw = X @ Pin')
    V0_scaled = X_scaled @ ts.Pin
    print_array_stats(V0_scaled, 'V0_scl = Xscaled @ Pin')
    yall_mean_abs = np.mean(np.abs(ts.Yall))
    v0_mean_abs = np.mean(np.abs(V0_scaled))
    ratio = yall_mean_abs / (v0_mean_abs + 1e-12)
    print(f"  Yall magnitude (mean abs): {yall_mean_abs:.4g}")
    print(f"  V0_scl magnitude (mean abs): {v0_mean_abs:.4g}")
    print(f"  --> Yall / V0_scl ratio: {ratio:.2f}  (Dense layer must amplify by this factor)")

    return {
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'Y_mean': float(np.mean(ts.Y)),
        'Y_std': float(np.std(ts.Y)),
        'Y_max': float(np.max(ts.Y)),
        'Y_zero_frac': zero_biomass / n_samples,
        'Yall_mean_abs': float(yall_mean_abs),
        'V0_scl_mean_abs': float(v0_mean_abs),
        'amplification_needed': ratio,
        'n_samples': ts.X.shape[0],
        'n_features': ts.X.shape[1],
    }


def print_comparison(results):
    """Print a side-by-side table of the first two datasets in ``results``."""
    print("\n\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    first, second = list(results)[:2]
    print(f"  {'Metric':<35} {first:<25} {second:<25}")
    print("  " + "-" * 85)
    for key, label in SUMMARY_METRICS:
        v0, v1 = results[first][key], results[second][key]
        ratio_str = f"  ratio={v1 / (v0 + 1e-12):.2f}x" if isinstance(v0, float) else ''
        print(f"  {label:<35} {v0:<25.4g} {v1:<25.4g}{ratio_str}")


def main():
    results = {label: investigate(label, path) for label, path in FILES.items()}
    print_comparison(results)


if __name__ == '__main__':
    main()
