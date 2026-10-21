#!/usr/bin/env python3
"""
Step 6: Evaluate the Media Network's uptake-flux predictions with COBRApy.

Constrains the genome-scale model (GEM) with the uptake fluxes predicted by
the ensemble (step 5), runs FBA/pFBA for every condition, and reports the R2
between predicted and measured growth. Optionally the fluxes are multiplied
by several scaler values to find the best overall scaling.

Set ``use_experimental: true`` to run the same check on the raw experimental
concentrations instead (baseline without the Media Network).

Usage:
    python scripts/evaluate_medium_pred.py configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --verbose configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml
"""

import numpy as np
from sklearn.metrics import r2_score

import _common
from _common import DATA_INTERIM_DIR, DATA_RAW_DIR, MODELS_DIR, banner, section

from src.Build_Dataset import TrainingSet, run_cobra

REQUIRED_KEYS = ['cobraname', 'reservoirname', 'predfilename', 'mediumsize']
DEFAULTS = {
    'seed': 10,
    'mediumbound': 'UB',
    'method': 'EXP',
    'method_simulation': 'pFBA',
    'scaler_list': [1.0],
    'use_experimental': False,
    'expname': '',
    'verbose': True,
}

# Floor for near-zero uptake fluxes, avoids numerical issues in the solver
MIN_FLUX = 1.0e-4


def predict_growth(parameter, scaler, method, verbose):
    """
    Run COBRApy on every condition with medium fluxes multiplied by ``scaler``.

    Returns:
        list: Predicted objective value for each row of ``parameter.X``.
    """
    Y_pred = []
    for i in range(parameter.X.shape[0]):
        influx = {r.id: 0 for r in parameter.model.reactions}
        for j, rxn in enumerate(parameter.medium):
            value = parameter.X[i, j]
            if verbose:
                print(f"  Sample {i}, Compound {j}: {rxn}, Value: {value:.6f}")
            eps = MIN_FLUX if value < MIN_FLUX else 0
            influx[rxn] = scaler * value + eps

        _, y = run_cobra(parameter.model, parameter.objective, influx,
                         method=method, verbose=verbose)
        Y_pred.append(y)

        if verbose:
            print(f"  Sample {i}: Actual: {parameter.Y[i].item():.4f}, Pred: {y.item():.4f}")
    return Y_pred


def evaluate_medium_predictions(config):
    """
    Evaluate predicted media with COBRApy FBA/pFBA for each scaler value.

    Outputs, under ``models/<reservoirname>/result_evaluation/``:
        evaluation_results_<timestamp>.csv   R2 per scaler
        evaluation_log_<timestamp>.txt       console log

    Args:
        config (dict): Configuration (see ``configs/evaluate_medium_pred.yaml``).

    Returns:
        dict: ``{scaler: {'r2', 'predictions', 'actual'}}``.
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    verbose = config['verbose']
    np.random.seed(seed=config['seed'])

    result_dir = MODELS_DIR / config['reservoirname'] / 'result_evaluation'
    result_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = _common.timestamp()
    log_path = result_dir / f'evaluation_log_{run_stamp}.txt'

    with _common.tee_stdout(log_path):
        if verbose:
            banner("Step 6: Evaluate Medium Predictions with COBRApy")
            _common.print_config(config)
            print(f"Console log file: {log_path}\n")

        cobrafile = DATA_RAW_DIR / config['cobraname']  # .xml
        if config['use_experimental']:
            mediumfile = DATA_INTERIM_DIR / config['expname']
            source = 'experimental data'
        else:
            mediumfile = (MODELS_DIR / config['reservoirname'] / 'result_medium'
                          / config['predfilename'])
            source = 'prediction file'
        if verbose:
            print(f"  Using {source}: {mediumfile}")
            print("\n  Loading data and COBRApy model...")

        parameter = TrainingSet(
            cobraname=str(cobrafile),
            mediumname=str(mediumfile),
            mediumbound=config['mediumbound'],
            mediumsize=config['mediumsize'],
            method=config['method'],
            verbose=verbose,
        )
        if verbose:
            print(f"  Data loaded: X shape = {parameter.X.shape}, Y shape = {parameter.Y.shape}")
            print(f"  Number of medium compounds: {len(parameter.medium)}")
            print()

        results = {}
        for scaler in config['scaler_list']:
            if verbose:
                section(f"Evaluating with scaler = {scaler:.2f}")
            Y_pred = predict_growth(parameter, scaler, config['method_simulation'], verbose)
            r2 = r2_score(parameter.Y, Y_pred, multioutput='variance_weighted')
            results[scaler] = {
                'r2': r2,
                'predictions': Y_pred,
                'actual': parameter.Y.flatten().tolist(),
            }
            print(f'\n  Scaler: {scaler:.2f}, R2: {r2:.4f}')

        if verbose:
            print()
            banner("Summary of Results")
            print()
            print(f"{'Scaler':<10} {'R2':>10}")
            print("-" * 22)
            for scaler, result in results.items():
                print(f"{scaler:<10.2f} {result['r2']:>10.4f}")
            print()
            best = max(results, key=lambda s: results[s]['r2'])
            print(f"Best scaler: {best:.2f} with R2 = {results[best]['r2']:.4f}")

        results_csv_path = result_dir / f'evaluation_results_{run_stamp}.csv'
        with open(results_csv_path, 'w') as f:
            f.write("scaler,r2\n")
            for scaler, result in results.items():
                f.write(f"{scaler},{result['r2']}\n")

        if verbose:
            print(f"\n  Results saved to: {results_csv_path}")
            print(f"  Console log saved to: {log_path}")
            print()
            banner("Evaluation complete!")

    return results


def main():
    parser = _common.build_parser(
        description='Evaluate Medium Predictions with COBRApy FBA/pFBA',
        examples=[
            'python scripts/evaluate_medium_pred.py configs/evaluate_medium_pred.yaml',
            'python scripts/evaluate_medium_pred.py --verbose configs/evaluate_medium_pred.yaml',
            'python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml',
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
    parser.add_argument(
        '--scaler', type=float, nargs='+', default=None,
        help='Scaler values to evaluate. Overrides config file setting.',
    )
    parser.add_argument(
        '--method', type=str, choices=['FBA', 'pFBA', 'EXP'], default=None,
        help='Simulation method (FBA, pFBA, or EXP). Overrides config file setting.',
    )
    args = parser.parse_args()

    config = _common.config_from_args(args, {
        'verbose': args.verbose,
        'seed': args.seed,
        'scaler_list': args.scaler,
        'method_simulation': args.method,
    })
    _common.run_or_exit(evaluate_medium_predictions, config, 'evaluation')


if __name__ == '__main__':
    main()
