#!/usr/bin/env python3
"""
Investigate and compare the two AMN training datasets (.npz files).

Usage:
    python scripts/investigate_datasets.py

Reads both .npz files using the TrainingSet class and prints a comprehensive
comparison of X, Y, matrices, and data distributions.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

from src.Build_Dataset import TrainingSet, MaxScaler

# ── File paths ──────────────────────────────────────────────────────────────
FILES = {
    'OLD (iML1515_UB)':     str(DATA_DIR / 'iML1515'            / 'iML1515_UB'),
    'NEW (aidaying_UB)':    str(DATA_DIR / 'aidaying_iml1515_UB' / 'aidaying_iml1515_UB'),
}
OBJECTIVE = ['BIOMASS_Ec_iML1515_core_75p37M']

# ── Helper ───────────────────────────────────────────────────────────────────
def stats(arr, name=''):
    arr = np.asarray(arr, dtype=float)
    flat = arr.flatten()
    nonzero = flat[flat != 0]
    print(f"  {name}: shape={arr.shape}  "
          f"min={flat.min():.4g}  max={flat.max():.4g}  "
          f"mean={flat.mean():.4g}  std={flat.std():.4g}  "
          f"nnz={len(nonzero)}  frac_nonzero={len(nonzero)/len(flat):.3f}")

def hist_bins(arr, n_bins=10, label=''):
    arr = np.asarray(arr).flatten()
    counts, edges = np.histogram(arr, bins=n_bins)
    print(f"\n  Histogram of {label}:")
    for i in range(n_bins):
        bar = '█' * int(40 * counts[i] / max(counts, default=1))
        print(f"    [{edges[i]:8.3f}, {edges[i+1]:8.3f})  {counts[i]:6d}  {bar}")

def sparsity(mat, name=''):
    mat = np.asarray(mat, dtype=float)
    nz = np.count_nonzero(mat)
    total = mat.size
    print(f"  {name}: shape={mat.shape}  nnz={nz}  sparsity={1-nz/total:.4f}  "
          f"frac_nonzero={nz/total:.4f}")

# ── Main ─────────────────────────────────────────────────────────────────────
results = {}

for label, filepath in FILES.items():
    print("\n" + "=" * 70)
    print(f"  DATASET: {label}")
    print(f"  File   : {filepath}.npz")
    print("=" * 70)

    ts = TrainingSet()
    ts.load(filepath)
    all_reactions = list(ts.measure)  # capture full reaction list before it is overwritten
    ts.filter_measure(measure=OBJECTIVE, verbose=True)

    print(f"\n-- Medium metadata --")
    print(f"  mediumbound : {ts.mediumbound}")
    print(f"  medium size : {len(ts.medium)}")
    print(f"  levmed      : min={min(ts.levmed):.1f}  max={max(ts.levmed):.1f}  "
          f"n_variable (>1)={sum(l>1 for l in ts.levmed)}")
    print(f"  valmed      : min={min(ts.valmed):.4g}  max={max(ts.valmed):.4g}  "
          f"mean={np.mean(ts.valmed):.4g}")
    print(f"  ratmed      : {ts.ratmed}")

    print(f"\n-- X (medium inputs) --")
    stats(ts.X, 'X raw')
    X_scaled, x_scaler = MaxScaler(ts.X.copy())
    print(f"  X scaler (max)  : {x_scaler:.4g}")
    stats(X_scaled, 'X scaled')

    print(f"\n-- Y (biomass = filtered measure) --")
    stats(ts.Y, 'Y (biomass)')
    Y_scaled, y_scaler = MaxScaler(ts.Y.copy())
    print(f"  Y scaler (max)  : {y_scaler:.4g}")
    # Fraction of zero biomass samples
    zero_biomass = np.sum(ts.Y.flatten() == 0)
    print(f"  Zero-biomass samples: {zero_biomass} / {ts.Y.shape[0]} "
          f"({100*zero_biomass/ts.Y.shape[0]:.1f}%)")
    hist_bins(ts.Y, n_bins=10, label='Y (biomass)')

    print(f"\n-- Yall (all fluxes) --")
    stats(ts.Yall, 'Yall')
    # Per-reaction statistics
    print(f"  Yall max per reaction (top 10):")
    rx_max = np.max(np.abs(ts.Yall), axis=0)
    top10_idx = np.argsort(rx_max)[-10:][::-1]
    for idx in top10_idx:
        rxn = all_reactions[idx] if idx < len(all_reactions) else str(idx)
        print(f"    reaction={rxn}  max_abs={rx_max[idx]:.4g}")
    # Fraction of Yall that is zero (overall sparsity)
    sparsity(ts.Yall, 'Yall')

    print(f"\n-- Mechanistic matrices --")
    sparsity(ts.S,   'S   (stoichiometric)')
    sparsity(ts.Pin, 'Pin (medium→V)')
    sparsity(ts.Pout,'Pout (V→measure)')

    print(f"\n-- V0 initialization check --")
    # V0 = X @ Pin  — matches get_V0: tf.linalg.matmul(inputs, Pin)
    # Pin shape: (n_medium, n_reactions)
    # Check the scale of this initialization vs. actual Yall
    V0_init = ts.X @ ts.Pin
    stats(V0_init, 'V0_raw = X @ Pin')
    V0_scaled = X_scaled @ ts.Pin
    stats(V0_scaled, 'V0_scl = Xscaled @ Pin')
    print(f"  Yall magnitude (mean abs): {np.mean(np.abs(ts.Yall)):.4g}")
    print(f"  V0_scl magnitude (mean abs): {np.mean(np.abs(V0_scaled)):.4g}")
    ratio = np.mean(np.abs(ts.Yall)) / (np.mean(np.abs(V0_scaled)) + 1e-12)
    print(f"  --> Yall / V0_scl ratio: {ratio:.2f}  "
          f"(Dense layer must amplify by this factor)")

    results[label] = {
        'x_scaler': x_scaler, 'y_scaler': y_scaler,
        'Y_mean': float(np.mean(ts.Y)), 'Y_std': float(np.std(ts.Y)),
        'Y_max': float(np.max(ts.Y)), 'Y_zero_frac': zero_biomass/ts.Y.shape[0],
        'Yall_mean_abs': float(np.mean(np.abs(ts.Yall))),
        'V0_scl_mean_abs': float(np.mean(np.abs(V0_scaled))),
        'amplification_needed': ratio,
        'n_samples': ts.X.shape[0], 'n_features': ts.X.shape[1],
    }

# ── Summary comparison ────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
keys = list(results.keys())
metrics = [
    ('n_samples',           'Dataset size'),
    ('n_features',          'Input features (X cols)'),
    ('x_scaler',            'X scaler (max X)'),
    ('y_scaler',            'Y scaler (max biomass)'),
    ('Y_mean',              'Biomass mean'),
    ('Y_std',               'Biomass std'),
    ('Y_max',               'Biomass max'),
    ('Y_zero_frac',         'Fraction zero-biomass'),
    ('Yall_mean_abs',       'Mean |flux| (all reactions)'),
    ('V0_scl_mean_abs',     'Mean |V0| after X scaling'),
    ('amplification_needed','Dense amplification needed'),
]
print(f"  {'Metric':<35} {keys[0]:<25} {keys[1]:<25}")
print("  " + "-" * 85)
for key, label in metrics:
    v0 = results[keys[0]][key]
    v1 = results[keys[1]][key]
    ratio_str = f"  ratio={v1/(v0+1e-12):.2f}x" if isinstance(v0, float) else ''
    print(f"  {label:<35} {v0:<25.4g} {v1:<25.4g}{ratio_str}")