#!/usr/bin/env python3
"""
Step 5: Train and evaluate the ensemble (Media Network + AMN Reservoir).

The ensemble chains two models:
1. Media Network (trainable): predicts uptake flux bounds from experimental
   media compositions.
2. AMN Reservoir (frozen, from step 4): predicts growth from those bounds.

Training is repeated for ``number_of_seeds`` seeds to get a trustworthy
performance estimate. The predicted uptake fluxes of every seed are saved so
they can be checked with COBRApy in step 6.

Usage:
    python scripts/train_eval_ensemble.py configs/ensemble_params.yaml
    python scripts/train_eval_ensemble.py --verbose --seeds 3 configs/ensemble_params.yaml
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import _common
from _common import DATA_INTERIM_DIR, MODELS_DIR, PROJECT_ROOT, banner, section

_common.silence_tensorflow()

from src.Build_Model import RC_Model
from src.Build_Dataset import read_XY
from src.Build_Model_Wrappers import (
    train_evaluate_with_logging,
    print_stats,
    ensure_result_dir,
)

REQUIRED_KEYS = [
    'number_of_seeds', 'trainname', 'reservoirname', 'n_input',
    'n_hidden_prior', 'hidden_dim_prior', 'activation_prior',
    'epochs', 'train_rate', 'batch_size', 'xfold', 'es',
]
DEFAULTS = {
    'stratify': False,
    'stratify_bins': 10,
    'verbose': True,
}


def load_reservoir_medium(reservoirfile):
    """
    Return the medium reaction IDs, in order, that the reservoir was trained on.

    The reservoir's ``_param.csv`` stores the path of its training ``.npz``.
    If that absolute path no longer exists (e.g. the project was moved), try
    successively shorter suffixes of it under ``PROJECT_ROOT``.

    Args:
        reservoirfile (str): Reservoir path without suffix.

    Returns:
        list: Medium exchange reaction IDs.
    """
    with open(f'{reservoirfile}_param.csv', 'r') as f:
        trainingfile = Path(f.readline().rstrip().split(',')[0])

    if not Path(f'{trainingfile}.npz').is_file():
        for n in range(1, len(trainingfile.parts)):
            candidate = PROJECT_ROOT.joinpath(*trainingfile.parts[n:])
            if Path(f'{candidate}.npz').is_file():
                trainingfile = candidate
                break

    npz = np.load(f'{trainingfile}.npz', allow_pickle=True)
    return list(npz['medium'])


def load_training_data(trainingfile, n_input, reservoirfile):
    """
    Load the experimental media/growth data in the reservoir's column order.

    The experimental CSV may order compounds differently from the reservoir's
    training set, so X columns are permuted to match, then log-transformed
    as ``log10(1 + X)``.

    Returns:
        tuple: ``(X, Y, reordered)`` where ``reordered`` says whether the
        column order changed.
    """
    X, Y = read_XY(trainingfile, n_input)
    exp_cols = pd.read_csv(trainingfile + '.csv', nrows=0).columns[:n_input].tolist()
    perm = [exp_cols.index(m) for m in load_reservoir_medium(reservoirfile)]
    X = np.log10(1 + X[:, perm])
    return X, Y, perm != list(range(len(perm)))


def save_seed_predictions(model, pred, seed, trainingfile, result_medium_dir, verbose=True):
    """
    Save the predicted uptake fluxes (Vin) and measured growth of one seed.

    The CSV uses the experimental file's header so it can be read back as an
    experimental dataset by ``scripts/evaluate_medium_pred.py``.

    ``pred`` columns are laid out as ``[obj, Loss_SV, Loss_Vin, Loss_Vpos,
    V (n_reactions), Vin (scaled)]``.

    Args:
        model (RC_Model): Trained ensemble model.
        pred (np.ndarray): Predictions for this seed.
        seed (int): Seed number (used in the filename).
        trainingfile (str): Experimental training file, without ``.csv``.
        result_medium_dir (Path): Output directory.
        verbose (bool): Print loss summary.

    Returns:
        Path: Path to the saved CSV.
    """
    n_reactions = model.S.shape[1]
    Loss_SV, Loss_Vin, Loss_Vpos = pred[:, 1], pred[:, 2], pred[:, 3]
    if verbose:
        print(f'\n Constraint losses (seed {seed}):')
        print(f'    Loss SV:   mean={np.mean(Loss_SV):.4f}  max={np.max(Loss_SV):.4f}')
        print(f'    Loss Vin:  mean={np.mean(Loss_Vin):.4f}  max={np.max(Loss_Vin):.4f}')
        print(f'    Loss Vpos: mean={np.mean(Loss_Vpos):.4f}  max={np.max(Loss_Vpos):.4f}')

    # Undo the reservoir's input scaling to get Vin in flux units
    Vin_scaled = pred[:, 4 + n_reactions:]
    if getattr(model.res, 'per_feature_norm', False):
        col_max = np.max(model.res.X, axis=0)
        col_max[col_max == 0] = 1.0
        # Only fixed (level == 1) compounds were max-normalised
        scaler = np.ones_like(col_max)
        for i, lev in enumerate(model.res.levmed):
            if lev == 1:
                scaler[i] = col_max[i]
        Vin = scaler * Vin_scaled
    else:
        Vin = model.res.scaler * Vin_scaled
    XY = np.concatenate((Vin, model.Y), axis=1)

    columns = pd.read_csv(trainingfile + '.csv', nrows=0).columns
    if XY.shape[1] != len(columns):
        print(f"  Warning: XY has {XY.shape[1]} columns but header has {len(columns)} columns.")

    out_path = Path(result_medium_dir) / f'pred_medium_seed{seed}.csv'
    np.savetxt(out_path, XY, delimiter=',', header=','.join(columns), comments='')
    if verbose:
        print(f'    Saved Cobra solution to {out_path}')
    return out_path


def train_eval_ensemble(config):
    """
    Train and evaluate the ensemble over several seeds.

    Outputs, under ``models/<reservoirname>/``:
        result_ensemble/   per-seed CV files, console log, and
                           ``<reservoirname>_<model_type>/`` with Q2 and
                           growth predictions across seeds
        result_medium/     ``pred_medium_seed<k>.csv`` for every seed

    Args:
        config (dict): Configuration (see ``configs/ensemble_params.yaml``).
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    verbose = config['verbose']
    reservoirname = config['reservoirname']
    number_of_seeds = config['number_of_seeds']

    trainingfile = str(DATA_INTERIM_DIR / config['trainname'])
    reservoirfile = str(MODELS_DIR / reservoirname / 'reservoir' / reservoirname)
    result_ensemble_dir = MODELS_DIR / reservoirname / 'result_ensemble'
    result_medium_dir = MODELS_DIR / reservoirname / 'result_medium'
    result_ensemble_dir.mkdir(parents=True, exist_ok=True)
    result_medium_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_ensemble_dir / f'console_log_{_common.timestamp()}.txt'

    with _common.tee_stdout(log_path):
        if verbose:
            banner("Step 5: Build Ensemble Model (Media Network + AMN Reservoir)")
            _common.print_config(config)
            print(f"Console log file: {log_path}\n")

        X, Y, reordered = load_training_data(trainingfile, config['n_input'], reservoirfile)
        if verbose:
            print(f"  Training data loaded: X shape = {X.shape}, Y shape = {Y.shape}")
            print(f"  Columns reordered to match reservoir: {reordered}")
            print()

        Q2, PRED = [], []
        model = None
        for seed in range(number_of_seeds):
            if verbose:
                section(f"Training with seed {seed}")
            np.random.seed(seed=seed)

            model = RC_Model(
                reservoirfile=reservoirfile,
                X=X, Y=Y,
                n_hidden_prior=config['n_hidden_prior'],
                hidden_dim_prior=config['hidden_dim_prior'],
                activation_prior=config['activation_prior'],
                epochs=config['epochs'],
                train_rate=config['train_rate'],
                batch_size=config['batch_size'],
                xfold=config['xfold'],
                stratify=config['stratify'],
                stratify_bins=config['stratify_bins'],
                es=config['es'],
                verbose=verbose,
            )
            if verbose:
                model.printout()

            start_time = time.time()
            _, pred, stats, _ = train_evaluate_with_logging(
                model, result_dir=str(result_ensemble_dir),
                prefix=f'seed{seed}_cv', verbose=verbose,
            )
            print_stats(f'{reservoirname} (seed={seed})', stats, time.time() - start_time)

            r2 = r2_score(model.Y, pred[:, 0], multioutput='variance_weighted')
            print(f'  Seed {seed}: Collated Q2 = {r2:.4f}')
            Q2.append(r2)
            PRED.append(pred)

        Q2, PRED = np.asarray(Q2), np.asarray(PRED)
        if verbose:
            print()
            banner("Aggregate Results Across Seeds")
            print(f'\nPredictions shape: {PRED.shape}')
            print(f'Averaged Q2 = {np.mean(Q2):.4f} (+/- {np.std(Q2):.4f})')

        final_dir = result_ensemble_dir / f'{reservoirname}_{model.model_type}'
        ensure_result_dir(str(final_dir))

        q2_path = final_dir / 'Q2_across_seeds.csv'
        np.savetxt(q2_path, Q2, delimiter=',')
        print(f'  Saved Q2 scores to {q2_path}')

        pred_path = final_dir / 'PRED_objective_across_seeds.csv'
        np.savetxt(pred_path, PRED[:, :, 0], delimiter=',')
        print(f'  Saved predictions to {pred_path}')

        if verbose:
            print()
            section("Saving Predictions for COBRApy (All Seeds)")
        for seed in range(number_of_seeds):
            save_seed_predictions(model, PRED[seed], seed, trainingfile,
                                  result_medium_dir, verbose=verbose)

        if verbose:
            print()
            banner("Ensemble training and evaluation complete!")
            print(f"Results saved to: {final_dir}/")
            print(f"Cobra predictions saved to: {result_medium_dir}/")
            print(f"Console log saved to: {log_path}")
            print()


def main():
    parser = _common.build_parser(
        description='Train and evaluate Ensemble model (Media Network + AMN Reservoir)',
        examples=[
            'python scripts/train_eval_ensemble.py configs/ensemble_params.yaml',
            'python scripts/train_eval_ensemble.py --verbose --seeds 3 configs/ensemble_params.yaml',
        ],
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=None,
        help='Enable verbose output. Overrides config file setting.',
    )
    parser.add_argument(
        '--seeds', '-n', type=int, default=None,
        help='Number of seeds to use. Overrides config file setting.',
    )
    args = parser.parse_args()

    config = _common.config_from_args(
        args, {'verbose': args.verbose, 'number_of_seeds': args.seeds}
    )
    _common.run_or_exit(train_eval_ensemble, config, 'training and evaluation')


if __name__ == '__main__':
    main()
