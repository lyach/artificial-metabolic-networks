# Helper functions for clean logging and result storage
# for the original Build_Model.py file

import os
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.Build_Model import (
    model_input, train_model, evaluate_model, ReturnStats,
    get_stratified_kfold
)

def print_stats(name, stats, elapsed_time=None):
    """Print cross-validation statistics in a clean format."""
    header = f"--- Results: {name} "
    if elapsed_time is not None:
        header += f"(elapsed: {elapsed_time:.1f}s) "
    header += "-" * max(0, 72 - len(header))
    print(header)
    print(f"  R2 (train) = {stats.train_objective[0]:.4f} "
          f"(+/- {stats.train_objective[1]:.4f})  |  "
          f"Constraint = {stats.train_loss[0]:.6f} "
          f"(+/- {stats.train_loss[1]:.6f})")
    print(f"  Q2 (test)  = {stats.test_objective[0]:.4f} "
          f"(+/- {stats.test_objective[1]:.4f})  |  "
          f"Constraint = {stats.test_loss[0]:.6f} "
          f"(+/- {stats.test_loss[1]:.6f})")
    print("-" * 72)


def print_stats_eval(name, stats, elapsed_time=None):
    """Print single dataset evaluation statistics.
    Use this for evaluating on a hold-out test set or full dataset,
    """
    header = f"--- Evaluation: {name} "
    if elapsed_time is not None:
        header += f"(elapsed: {elapsed_time:.1f}s) "
    header += "-" * max(0, 72 - len(header))
    print(header)
    print(f"  R2 = {stats.test_objective[0]:.4f}  |  "
          f"Constraint = {stats.test_loss[0]:.6f}")
    print("-" * 72)


def ensure_result_dir(base_dir):
    """Create result directory if it doesn't exist."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_fold_histories(histories, result_dir, prefix):
    """
    Save per-fold training histories (loss, metrics) as CSVs.
    Each fold produces one CSV with columns: epoch, loss, val_loss, 
    and any other metrics recorded by Keras.
    """
    for fold_idx, history in enumerate(histories):
        df = pd.DataFrame(history.history)
        df.index.name = 'epoch'
        filepath = os.path.join(result_dir, f'{prefix}_fold{fold_idx}_history.csv')
        df.to_csv(filepath)
    print(f"  Saved {len(histories)} fold histories to {result_dir}/{prefix}_fold*_history.csv")


def save_fold_predictions(fold_records, result_dir, prefix):
    """
    Save per-fold out-of-sample predictions as CSVs.
    Each CSV contains columns: sample_idx, y_true_0..N, y_pred_0..N.
    """
    for fold_idx, record in enumerate(fold_records):
        indices = record['test_indices']
        y_true = record['y_true']
        y_pred = record['y_pred']

        n_cols_true = y_true.shape[1] if y_true.ndim > 1 else 1
        n_cols_pred = y_pred.shape[1] if y_pred.ndim > 1 else 1

        cols = ['sample_idx']
        cols += [f'y_true_{i}' for i in range(n_cols_true)]
        cols += [f'y_pred_{i}' for i in range(n_cols_pred)]

        data = np.column_stack([
            indices.reshape(-1, 1),
            y_true.reshape(len(indices), -1),
            y_pred.reshape(len(indices), -1)
        ])
        df = pd.DataFrame(data, columns=cols)
        df['sample_idx'] = df['sample_idx'].astype(int)
        filepath = os.path.join(result_dir, f'{prefix}_fold{fold_idx}_predictions.csv')
        df.to_csv(filepath, index=False)
    print(f"  Saved {len(fold_records)} fold predictions to {result_dir}/{prefix}_fold*_predictions.csv")


def save_fold_metrics(fold_records, result_dir, prefix):
    """
    Save per-fold summary metrics (R2, Q2, constraint loss) as a single CSV.
    """
    rows = []
    for record in fold_records:
        rows.append({
            'fold': record['fold'],
            'r2_train': record['r2_train'],
            'r2_test': record['r2_test'],
            'constraint_train': record['loss_train'],
            'constraint_test': record['loss_test'],
            'n_train': record['n_train'],
            'n_test': record['n_test'],
            'epochs_run': len(record['history'].history.get('loss', [])),
        })
    df = pd.DataFrame(rows)
    filepath = os.path.join(result_dir, f'{prefix}_fold_metrics.csv')
    df.to_csv(filepath, index=False)
    print(f"  Saved fold metrics to {filepath}")
    return df


def train_evaluate_with_logging(parameter, result_dir, prefix, 
                                 stratify=False, stratify_bins=10, verbose=True):
    """
    Custom cross-validation wrapper that:
    - Calls the library's train_model and evaluate_model per fold
    - Captures per-fold predictions, histories, and metrics
    - Saves everything to CSVs in result_dir
    - Prints clean progress (fold number, shapes, R2/Q2 per fold)
    
    Parameters
    ----------
    parameter : Neural_Model or RC_Model
        Model parameter object with X, Y, and all hyperparameters.
    result_dir : str
        Directory to save result CSVs.
    prefix : str
        Filename prefix for saved CSVs.
    stratify : bool
        If True, use StratifiedKFold based on binned Y values.
        This ensures each fold has similar distribution of target values.
    stratify_bins : int
        Number of quantile bins for stratification (default 10).
    verbose : bool or int
        Verbosity level. True prints fold info and shapes.
    
    Returns
    -------
    best_model : Neural_Model
        The model with highest Q2 on its test fold.
    stats : ReturnStats
        Aggregated cross-validation statistics.
    fold_records : list[dict]
        Per-fold detailed records.
    """
    param = copy.copy(parameter)
    X, Y = model_input(param, verbose=verbose)
    param.X, param.Y = X, Y

    # No cross-validation case
    if param.xfold < 2:
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
            train_model(param, X, Y, X, Y, verbose=verbose)
        stats = ReturnStats(otrain, 0, ltrain, 0, otest, 0, ltest, 0)
        return Net, ytrain, stats, []

    # Cross-validation
    Otrain, Otest, Ltrain, Ltest = [], [], [], []
    Omax, Netmax = -1.0e32, None
    Ypred = np.copy(Y)
    fold_records = []
    histories = []

    # Choose between stratified or regular K-Fold
    if stratify:
        kfold, y_binned = get_stratified_kfold(Y, n_splits=param.xfold, 
                                                n_bins=stratify_bins, shuffle=True)
        split_iterator = kfold.split(X, y_binned)
        if verbose:
            print(f"  Using StratifiedKFold with {param.xfold} folds and {stratify_bins} bins")
    else:
        kfold = KFold(n_splits=param.xfold, shuffle=True)
        split_iterator = kfold.split(X, Y)
    
    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator):
        print(f"\n  Fold {fold_idx + 1}/{param.xfold}: "
              f"train {X[train_idx].shape}, test {X[test_idx].shape}")

        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
            train_model(param, X[train_idx], Y[train_idx],
                        X[test_idx], Y[test_idx], verbose=verbose)

        # Collect metrics
        Otrain.append(otrain)
        Otest.append(otest)
        Ltrain.append(ltrain)
        Ltest.append(ltest)
        histories.append(history)

        # Collate out-of-fold predictions
        if Ypred.shape[1] != ytest.shape[1]:
            n, m = Y.shape[0], ytest.shape[1]
            Ypred = np.zeros(n * m).reshape(n, m)
        for i in range(len(test_idx)):
            Ypred[test_idx[i]] = ytest[i]

        # Track best model
        if otest > Omax:
            Omax, Netmax = otest, Net

        # Store per-fold record
        fold_records.append({
            'fold': fold_idx,
            'train_indices': train_idx,
            'test_indices': test_idx,
            'y_true': Y[test_idx],
            'y_pred': ytest,
            'r2_train': otrain,
            'r2_test': otest,
            'loss_train': ltrain,
            'loss_test': ltest,
            'history': history,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

        print(f"    R2={otrain:.4f}  Q2={otest:.4f}  "
              f"constraint_train={ltrain:.6f}  constraint_test={ltest:.6f}  "
              f"epochs={len(history.history.get('loss', []))}")

    # Final prediction using best model on whole dataset
    Pred, _ = evaluate_model(Netmax.model, X, Y, param, verbose=verbose)
    Ypred = Pred if param.niter > 0 else Ypred

    # Aggregated stats
    stats = ReturnStats(np.mean(Otrain), np.std(Otrain),
                        np.mean(Ltrain), np.std(Ltrain),
                        np.mean(Otest), np.std(Otest),
                        np.mean(Ltest), np.std(Ltest))

    # Save all results
    ensure_result_dir(result_dir)
    save_fold_metrics(fold_records, result_dir, prefix)
    save_fold_predictions(fold_records, result_dir, prefix)
    save_fold_histories(histories, result_dir, prefix)

    return Netmax, Ypred, stats, fold_records