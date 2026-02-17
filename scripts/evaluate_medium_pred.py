#!/usr/bin/env python3
"""
Evaluate the predictions of the Media Network using COBRApy

This script runs COBRApy FBA/pFBA on the Uptake Flux predictions (from the ensemble model) 
and computes the R2 between the experimental and the predicted growth.

The goal is to validate that predicted medium fluxes lead to accurate growth rate estimates
when used to constrain a Genome-scale metabolic model (GEM).

Usage:
    python scripts/evaluate_medium_pred.py configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --verbose configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml
"""

import sys
import argparse
import os
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
from sklearn.metrics import r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define data directories
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
MODELS_DIR = PROJECT_ROOT / 'models'

from src.Build_Dataset import TrainingSet, run_cobra


class TeeOutput:
    """Class to redirect output to both console and file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_medium_predictions(config):
    """
    Evaluate medium predictions using COBRApy FBA/pFBA.
    
    This function:
    1. Loads a COBRApy model and medium/growth data
    2. Runs FBA/pFBA with the medium fluxes (optionally scaled)
    3. Computes R2 between predicted and actual growth rates
    
    Args:
        config (dict): Configuration dictionary with parameters
    """
    # Extract parameters from config
    seed = config.get('seed', 10)
    cobraname = config['cobraname']
    mediumbound = config.get('mediumbound', 'UB')
    reservoirname = config['reservoirname']
    predfilename = config['predfilename']
    mediumsize = config['mediumsize']
    method = config.get('method', 'EXP')
    verbose = config.get('verbose', True)
    method_simulation = config.get('method_simulation', 'pFBA')
    scaler_list = config.get('scaler_list', [1.0])
    use_experimental = config.get('use_experimental', False)
    expname = config.get('expname', '')
    
    # Set random seed
    np.random.seed(seed=seed)
    
    # Set up output directory and logging
    result_dir = MODELS_DIR / reservoirname / 'result_evaluation'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_log_{timestamp}.txt"
    log_filepath = os.path.join(str(result_dir), log_filename)
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    tee = TeeOutput(log_filepath)
    sys.stdout = tee
    
    try:
        if verbose:
            print("=" * 80)
            print("Step 6: Evaluate Medium Predictions with COBRApy")
            print("=" * 80)
            print(f"\nConfiguration:")
            print(f"  Random seed: {seed}")
            print(f"  COBRA model: {cobraname}")
            print(f"  Medium bound type: {mediumbound}")
            print(f"  Reservoir name: {reservoirname}")
            print(f"  Prediction file: {predfilename}")
            print(f"  Medium size: {mediumsize}")
            print(f"  Method: {method}")
            print(f"  Simulation method: {method_simulation}")
            print(f"  Scaler values: {scaler_list}")
            print(f"  Use experimental data: {use_experimental}")
            if use_experimental:
                print(f"  Experimental dataset: {expname}")
            print(f"  Console log file: {log_filepath}")
            print()
        
        # Build file paths
        cobrafile = DATA_RAW_DIR / cobraname  # .xml
        
        if use_experimental:
            # Use experimental dataset directly
            mediumfile = DATA_INTERIM_DIR / expname
            if verbose:
                print(f"  Using experimental data: {mediumfile}")
        else:
            # Use predictions from ensemble model
            predfile = MODELS_DIR / reservoirname / 'result_medium' / predfilename
            mediumfile = predfile
            if verbose:
                print(f"  Using prediction file: {predfile}")
        
        # Generate a parameter object with the data
        if verbose:
            print("\n  Loading data and COBRApy model...")
        
        parameter = TrainingSet(
            cobraname=str(cobrafile),
            mediumname=str(mediumfile),
            mediumbound=mediumbound,
            mediumsize=mediumsize,
            method=method,
            verbose=verbose
        )
        
        if verbose:
            print(f"  Data loaded: X shape = {parameter.X.shape}, Y shape = {parameter.Y.shape}")
            print(f"  Number of medium compounds: {len(parameter.medium)}")
            print()
        
        # Store results for each scaler
        results = {}
        
        # Run COBRApy predictions for each scaler value
        L = parameter.X.shape[0]
        
        for scaler in scaler_list:
            if verbose:
                print("-" * 80)
                print(f"Evaluating with scaler = {scaler:.2f}")
                print("-" * 80)
            
            Y_pred = {}
            
            for i in range(L):
                # Set up influx dictionary for all reactions
                inf = {r.id: 0 for r in parameter.model.reactions}
                
                # Set medium compound fluxes
                for j in range(len(parameter.medium)):
                    if verbose:
                        print(f"  Sample {i}, Compound {j}: {parameter.medium[j]}, "
                              f"Value: {parameter.X[i,j]:.6f}")
                    
                    # Add small epsilon to avoid numerical issues
                    eps = 1.0e-4 if parameter.X[i, j] < 1.0e-4 else 0
                    inf[parameter.medium[j]] = scaler * parameter.X[i, j] + eps
                
                # Run COBRApy simulation
                out, Y_pred[i] = run_cobra(
                    parameter.model, 
                    parameter.objective, 
                    inf, 
                    method=method_simulation, 
                    verbose=verbose
                )
                
                # Print prediction vs actual (implementing the TODO)
                if verbose:
                    actual = parameter.Y[i].item()
                    predicted = Y_pred[i].item()
                    print(f"  Sample {i}: Actual: {actual:.4f}, Pred: {predicted:.4f}")
            
            # Convert to list and compute R2
            Y_pred_list = list(Y_pred.values())
            r2 = r2_score(parameter.Y[0:L], Y_pred_list[0:L], multioutput='variance_weighted')
            
            results[scaler] = {
                'r2': r2,
                'predictions': Y_pred_list,
                'actual': parameter.Y[0:L].flatten().tolist()
            }
            
            print(f'\n  Scaler: {scaler:.2f}, R2: {r2:.4f}')
        
        # Summary of results
        if verbose:
            print()
            print("=" * 80)
            print("Summary of Results")
            print("=" * 80)
            print()
            print(f"{'Scaler':<10} {'R2':>10}")
            print("-" * 22)
            for scaler, result in results.items():
                print(f"{scaler:<10.2f} {result['r2']:>10.4f}")
            print()
            
            # Find best scaler
            best_scaler = max(results.keys(), key=lambda s: results[s]['r2'])
            best_r2 = results[best_scaler]['r2']
            print(f"Best scaler: {best_scaler:.2f} with R2 = {best_r2:.4f}")
        
        # Save results to CSV
        results_csv_path = result_dir / f"evaluation_results_{timestamp}.csv"
        with open(results_csv_path, 'w') as f:
            f.write("scaler,r2\n")
            for scaler, result in results.items():
                f.write(f"{scaler},{result['r2']}\n")
        
        if verbose:
            print(f"\n  Results saved to: {results_csv_path}")
            print(f"  Console log saved to: {log_filepath}")
            print()
            print("=" * 80)
            print("Evaluation complete!")
            print("=" * 80)
        
        return results
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Evaluate Medium Predictions with COBRApy FBA/pFBA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/evaluate_medium_pred.py configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --verbose configs/evaluate_medium_pred.yaml
    python scripts/evaluate_medium_pred.py --scaler 0.5 1.0 1.5 configs/evaluate_medium_pred.yaml
        """
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=None,
        help='Enable verbose output. Overrides config file setting.'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed. Overrides config file setting.'
    )
    parser.add_argument(
        '--scaler',
        type=float,
        nargs='+',
        default=None,
        help='Scaler values to evaluate. Overrides config file setting.'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['FBA', 'pFBA', 'EXP'],
        default=None,
        help='Simulation method (FBA, pFBA, or EXP). Overrides config file setting.'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    # CLI arguments override config file values
    if args.verbose is not None:
        config['verbose'] = args.verbose
    if args.seed is not None:
        config['seed'] = args.seed
    if args.scaler is not None:
        config['scaler_list'] = args.scaler
    if args.method is not None:
        config['method_simulation'] = args.method
    
    # Run the evaluation
    try:
        evaluate_medium_predictions(config)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
