#!/usr/bin/env python3
"""
Step 4: Train and evaluate the Artificial Metabolic Network (AMN) Reservoir.

Trains a hybrid AMN on the pFBA-simulated dataset from step 3. Zero-biomass
samples are dropped, a ``ratio_test`` fraction is held out as an independent
test set, and k-fold cross-validation is run on the remainder.

Usage:
    python scripts/train_eval_reservoir.py configs/reservoir_params.yaml
    python scripts/train_eval_reservoir.py --verbose --seed 1 configs/reservoir_params.yaml
"""

import time

import numpy as np

import _common
from _common import DATA_PROCESSED_DIR, MODELS_DIR, banner, section

_common.silence_tensorflow()

from src.Build_Model import Neural_Model, model_input, evaluate_model
from src.Build_Model_Wrappers import (
    train_evaluate_with_logging,
    print_stats,
    print_stats_eval,
    ensure_result_dir,
)

REQUIRED_KEYS = [
    'seed', 'ratio_test', 'reservoirname', 'trainingfile', 'objective',
    'model_type', 'timestep', 'learn_rate', 'scaler', 'n_hidden', 'hidden_dim',
    'train_rate', 'dropout', 'epochs', 'xfold', 'batch_size', 'es',
]
DEFAULTS = {
    'y_scale': False,
    'per_feature_norm': False,
    'stratify': False,
    'stratify_bins': 10,
    'verbose': True,
}


def split_train_test(X, Y, ratio_test):
    """Randomly hold out ``ratio_test`` of the rows. Uses the global NumPy RNG."""
    test_idx = np.random.choice(X.shape[0], size=int(X.shape[0] * ratio_test), replace=False)
    return (np.delete(X, test_idx, axis=0), np.delete(Y, test_idx, axis=0),
            X[test_idx, :], Y[test_idx, :])


def train_eval_reservoir(config):
    """
    Train and evaluate the AMN Reservoir.

    Outputs, under ``models/<reservoirname>_<model_type>/``:
        reservoir/   saved model (``_model.keras``, ``_param.csv``)
        result/      per-fold CV histories/predictions/metrics, test-set
                     predictions and the console log

    Args:
        config (dict): Configuration (see ``configs/reservoir_params.yaml``).
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    verbose = config['verbose']
    np.random.seed(seed=config['seed'])

    reservoir_out = f"{config['reservoirname']}_{config['model_type']}"
    model_dir = MODELS_DIR / reservoir_out
    result_dir = model_dir / 'result'
    reservoir_dir = model_dir / 'reservoir'
    result_dir.mkdir(parents=True, exist_ok=True)
    reservoir_dir.mkdir(parents=True, exist_ok=True)
    reservoirfile = str(reservoir_dir / reservoir_out)
    log_path = result_dir / f'console_log_{_common.timestamp()}.txt'

    with _common.tee_stdout(log_path):
        if verbose:
            banner("Step 4: Build Artificial Metabolic Network (AMN) Reservoir")
            _common.print_config(config)
            print(f"Console log file: {log_path}\n")
            section("Step 4.1: Create Neural Model")

        model = Neural_Model(
            trainingfile=str(DATA_PROCESSED_DIR / config['trainingfile']),
            objective=config['objective'],
            model_type=config['model_type'],
            timestep=config['timestep'],
            learn_rate=config['learn_rate'],
            scaler=config['scaler'],
            n_hidden=config['n_hidden'],
            hidden_dim=config['hidden_dim'],
            train_rate=config['train_rate'],
            dropout=config['dropout'],
            epochs=config['epochs'],
            xfold=config['xfold'],
            batch_size=config['batch_size'],
            es=config['es'],
            y_scale=config['y_scale'],
            per_feature_norm=config['per_feature_norm'],
            verbose=verbose,
        )

        # Zero-growth samples add noise and distort the V scaler: drop them
        # from the whole dataset before splitting.
        nonzero = model.Y.flatten() > 0
        n_removed = int(np.sum(~nonzero))
        model.X, model.Y = model.X[nonzero], model.Y[nonzero]

        if verbose:
            section("Step 4.2: Split training and test sets")
        Xtrain, Ytrain, Xtest, Ytest = split_train_test(model.X, model.Y, config['ratio_test'])
        if verbose:
            print(f"  Original samples: {n_removed + model.X.shape[0]} "
                  f"(removed {n_removed} zero-biomass)")
            print(f"  Training samples: {Xtrain.shape[0]}")
            print(f"  Test samples: {Xtest.shape[0]}")
            print()
            model.printout()
            section("Step 4.3: Train and evaluate with cross-validation")
            print(f"  Reservoir name: {reservoir_out}")
            print(f"  Reservoir file: {reservoirfile}")
            print(f"  Result directory: {result_dir}")
            print()

        start_time = time.time()
        model.X, model.Y = Xtrain, Ytrain
        reservoir, _, stats, _ = train_evaluate_with_logging(
            model, result_dir=str(result_dir), prefix='cv',
            stratify=config['stratify'], stratify_bins=config['stratify_bins'],
            verbose=verbose,
        )
        print_stats(config['reservoirname'], stats, time.time() - start_time)
        reservoir.save(reservoirfile)

        if verbose:
            print()
            section("Step 4.4: Print reservoir model info")
            reservoir.printout()

        if len(Xtest) > 0:
            if verbose:
                print()
                section("Step 4.5: Evaluate on independent test set")
            start_time = time.time()
            reservoir.X, reservoir.Y = Xtest, Ytest
            X_eval, Y_eval = model_input(reservoir, verbose=verbose)
            pred_test, stats_test = evaluate_model(
                reservoir.model, X_eval, Y_eval, reservoir, verbose=verbose
            )
            print_stats_eval('Independent test set', stats_test, time.time() - start_time)

            ensure_result_dir(str(result_dir))
            test_pred_path = result_dir / 'test_set_predictions.csv'
            np.savetxt(test_pred_path, pred_test, delimiter=',')
            print(f"  Saved test predictions to {test_pred_path}")

        if verbose:
            print()
            banner("AMN Reservoir training and evaluation complete!")
            print(f"Reservoir saved to: {reservoirfile}_param.csv")
            print(f"Results saved to: {result_dir}/")
            print(f"Console log saved to: {log_path}")
            print()


def main():
    parser = _common.build_parser(
        description='Train and evaluate AMN Reservoir model',
        examples=[
            'python scripts/train_eval_reservoir.py configs/reservoir_params.yaml',
            'python scripts/train_eval_reservoir.py --verbose --seed 1 configs/reservoir_params.yaml',
        ],
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=None,
        help='Enable verbose output. Overrides config file setting.',
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=None,
        help='Random seed. Overrides config file setting.',
    )
    args = parser.parse_args()

    config = _common.config_from_args(args, {'verbose': args.verbose, 'seed': args.seed})
    _common.run_or_exit(train_eval_reservoir, config, 'training and evaluation')


if __name__ == '__main__':
    main()
