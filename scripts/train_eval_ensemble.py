#!/usr/bin/env python3
"""
Train and evaluate ensemble of:
Artificial Metabolic Network (AMN) Reservoir model and Media Network

This script creates an ensemble model composed of 2 models:
1. Media Network, with trainable weights, learns to predict uptake flux bounds from experimental media compositions.
2. AMN Reservoir, with frozen weights, learns to predict growth rates from uptake flux bounds.

A stratified growth split is created to ensure that each fold contains a representative 
distribution of the target variable (growth rates) to prevent biases where one fold might only 
contain high or low growth rates.

Training process is repeated with different seeds to get a trustworthy performance estimate.

Usage:
    python scripts/train_eval_ensemble.py configs/ensemble_params.yaml
    python scripts/train_eval_ensemble.py --verbose configs/ensemble_params.yaml
"""

import sys
import argparse
import time
import os
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow/XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'

# Suppress absl logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define data directories
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
MODELS_DIR = PROJECT_ROOT / 'models'

from src.Build_Model import RC_Model
from src.Build_Dataset import read_XY
from src.Build_Model_Wrappers import (
    train_evaluate_with_logging,
    print_stats,
    ensure_result_dir,
)


# Configuration & main logic

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


def save_seed_predictions(model, pred, seed, trainingfile, result_medium_dir, verbose=True):
    """
    Save predictions for a single seed to be used as input for COBRApy.
    
    Args:
        model: RC_Model instance
        pred: Prediction array for this seed
        seed: Seed number
        trainingfile: Path to training file (for column headers)
        result_medium_dir: Directory to save output
        verbose: Print progress information
    """
    # Extract loss components
    obj = pred[:, 0]
    Loss_SV = pred[:, 1]
    Loss_Vin = pred[:, 2]
    Loss_Vpos = pred[:, 3]
    
    if verbose:
        print(f'\n Constraint losses (seed {seed}):')
        print(f'    Loss SV:   mean={np.mean(Loss_SV):.4f}  max={np.max(Loss_SV):.4f}')
        print(f'    Loss Vin:  mean={np.mean(Loss_Vin):.4f}  max={np.max(Loss_Vin):.4f}')
        print(f'    Loss Vpos: mean={np.mean(Loss_Vpos):.4f}  max={np.max(Loss_Vpos):.4f}')
    
    # Extract V and Vin
    V = pred[:, 4:4 + model.S.shape[1]]
    Vin_pre_scaling = pred[:, 4 + model.S.shape[1]:]
    Vin = model.res.scaler * Vin_pre_scaling  # rescale back for Cobra
    XY = np.concatenate((Vin, model.Y), axis=1)
    
    # Parse column names from training file (compounds + growth rate)
    df_cols = pd.read_csv(trainingfile, nrows=0).columns
    
    # Create a single string separated by commas for the CSV header
    header_string = ",".join(df_cols)
    
    # Check dimensions
    if XY.shape[1] != len(df_cols):
        print(f"  Warning: XY has {XY.shape[1]} columns but header has {len(df_cols)} columns.")
    
    # Save the file
    cobra_filename = f'pred_medium_seed{seed}.csv'
    cobra_path = os.path.join(result_medium_dir, cobra_filename)
    np.savetxt(cobra_path, XY, delimiter=',', header=header_string, comments='')
    
    if verbose:
        print(f'    Saved Cobra solution to {cobra_path}')
    
    return cobra_path


def train_eval_ensemble(config):
    """
    Train and evaluate ensemble model (Media Network + AMN Reservoir).
    
    Args:
        config (dict): Configuration dictionary with parameters
    """
    # Extract parameters from config
    number_of_seeds = config['number_of_seeds']
    trainname = config['trainname']
    reservoirname = config['reservoirname']
    n_input = config['n_input']
    verbose = config.get('verbose', True)
    
    # Model parameters
    n_hidden_prior = config['n_hidden_prior']
    hidden_dim_prior = config['hidden_dim_prior']
    activation_prior = config['activation_prior']
    epochs = config['epochs']
    train_rate = config['train_rate']
    batch_size = config['batch_size']
    xfold = config['xfold']
    es = config['es']
    
    # Stratified CV parameters (optional, default to False)
    stratify = config.get('stratify', False)
    stratify_bins = config.get('stratify_bins', 10)
    
    # Build file paths
    trainingfile = str(DATA_INTERIM_DIR / trainname)
    reservoirfile = str(MODELS_DIR / reservoirname / 'reservoir' / reservoirname)
    
    # Set up output directories
    result_ensemble_dir = MODELS_DIR / reservoirname / 'result_ensemble'
    result_medium_dir = MODELS_DIR / reservoirname / 'result_medium'
    result_ensemble_dir.mkdir(parents=True, exist_ok=True)
    result_medium_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up console output logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"console_log_{timestamp}.txt"
    log_filepath = os.path.join(str(result_ensemble_dir), log_filename)
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    tee = TeeOutput(log_filepath)
    sys.stdout = tee
    
    try:
        if verbose:
            print("=" * 80)
            print("Step 5: Build Ensemble Model (Media Network + AMN Reservoir)")
            print("=" * 80)
            print(f"\nConfiguration:")
            print(f"  Number of seeds: {number_of_seeds}")
            print(f"  Training file: {trainname}")
            print(f"  Reservoir name: {reservoirname}")
            print(f"  Number of inputs: {n_input}")
            print(f"  Prior hidden layers: {n_hidden_prior}")
            print(f"  Prior hidden dimension: {hidden_dim_prior}")
            print(f"  Prior activation: {activation_prior}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {train_rate}")
            print(f"  Batch size: {batch_size}")
            print(f"  Cross-validation folds: {xfold}")
            print(f"  Stratified CV: {stratify}")
            if stratify:
                print(f"  Stratify bins: {stratify_bins}")
            print(f"  Early stopping: {es}")
            print(f"  Console log file: {log_filepath}")
            print()
        
        # Load training data
        X, Y = read_XY(trainingfile, n_input)
        
        if verbose:
            print(f"  Training data loaded: X shape = {X.shape}, Y shape = {Y.shape}")
            print()
        
        # Initialize storage for results across seeds
        Q2, PRED = [], []
        model = None  # Will be set in the loop
        
        # Training loop over seeds
        for seed in range(number_of_seeds):
            if verbose:
                print("-" * 80)
                print(f"Training with seed {seed}")
                print("-" * 80)
            
            # Set random seed
            np.random.seed(seed=seed)
            
            # Create RC_Model
            model = RC_Model(
                reservoirfile=reservoirfile,
                X=X, Y=Y,
                n_hidden_prior=n_hidden_prior,
                hidden_dim_prior=hidden_dim_prior,
                activation_prior=activation_prior,
                epochs=epochs,
                train_rate=train_rate,
                batch_size=batch_size,
                xfold=xfold,
                stratify=stratify,
                stratify_bins=stratify_bins,
                es=es,
                verbose=verbose
            )
            
            if verbose:
                model.printout()
            
            # Train and evaluate with per-fold logging
            rc_result_dir = str(result_ensemble_dir)
            rc_prefix = f'seed{seed}_cv'
            start_time = time.time()
            reservoir, pred, stats, fold_records = train_evaluate_with_logging(
                model, result_dir=rc_result_dir, prefix=rc_prefix, verbose=verbose
            )
            delta_time = time.time() - start_time
            
            # Print cross-validation results
            print_stats(f'{reservoirname} (seed={seed})', stats, delta_time)
            r2 = r2_score(model.Y, pred[:, 0], multioutput='variance_weighted')
            print(f'  Seed {seed}: Collated Q2 = {r2:.4f}')
            Q2.append(r2)
            PRED.append(pred)
        
        # Aggregate results across seeds
        Q2 = np.asarray(Q2)
        PRED = np.asarray(PRED)
        
        if verbose:
            print()
            print("=" * 80)
            print("Aggregate Results Across Seeds")
            print("=" * 80)
            print(f'\nPredictions shape: {PRED.shape}')
            print(f'Averaged Q2 = {np.mean(Q2):.4f} (+/- {np.std(Q2):.4f})')
        
        # Save Q2 scores and predictions
        rc_result_final_dir = result_ensemble_dir / f'{reservoirname}_{model.model_type}'
        ensure_result_dir(str(rc_result_final_dir))
        
        q2_path = os.path.join(str(rc_result_final_dir), 'Q2_across_seeds.csv')
        np.savetxt(q2_path, Q2, delimiter=',')
        print(f'  Saved Q2 scores to {q2_path}')
        
        obj = PRED[:, :, 0]
        pred_path = os.path.join(str(rc_result_final_dir), 'PRED_objective_across_seeds.csv')
        np.savetxt(pred_path, obj, delimiter=',')
        print(f'  Saved predictions to {pred_path}')
        
        # Save predictions for ALL seeds (for COBRApy)
        if verbose:
            print()
            print("-" * 80)
            print("Saving Predictions for COBRApy (All Seeds)")
            print("-" * 80)
        
        for seed in range(number_of_seeds):
            save_seed_predictions(
                model=model,
                pred=PRED[seed],
                seed=seed,
                trainingfile=trainingfile,
                result_medium_dir=str(result_medium_dir),
                verbose=verbose
            )
        
        if verbose:
            print()
            print("=" * 80)
            print("Ensemble training and evaluation complete!")
            print("=" * 80)
            print(f"Results saved to: {rc_result_final_dir}/")
            print(f"Cobra predictions saved to: {result_medium_dir}/")
            print(f"Console log saved to: {log_filepath}")
            print()
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate Ensemble model (Media Network + AMN Reservoir)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/train_eval_ensemble.py configs/ensemble_params.yaml
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
        '--seeds', '-n',
        type=int,
        default=None,
        help='Number of seeds to use. Overrides config file setting.'
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
    if args.seeds is not None:
        config['number_of_seeds'] = args.seeds
    
    # Run the training and evaluation
    try:
        train_eval_ensemble(config)
    except Exception as e:
        print(f"Error during training and evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
