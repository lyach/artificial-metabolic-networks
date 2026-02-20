#!/usr/bin/env python3
"""
Train and evaluate Artificial Metabolic Network (AMN) Reservoir model

This script creates a hybrid AMN model for learning on FBA-generated datasets.
It performs 5-fold cross-validation on the training set and evaluates on an
independent test set.

Usage:
    python scripts/train_eval_reservoir.py configs/reservoir_params.yaml
    python scripts/train_eval_reservoir.py --verbose configs/reservoir_params.yaml
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define data directories
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

from src.Build_Model import Neural_Model, model_input, evaluate_model
from src.Build_Model_Wrappers import (
    train_evaluate_with_logging,
    print_stats,
    print_stats_eval,
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


def train_eval_reservoir(config):
    """
    Train and evaluate AMN Reservoir model.
    
    Args:
        config (dict): Configuration dictionary with parameters
    """
    # Extract parameters from config
    seed = config['seed']
    ratio_test = config['ratio_test']
    reservoirname = config['reservoirname']
    verbose = config.get('verbose', True)
    
    # Model parameters
    trainingfile_name = config['trainingfile']
    objective = config['objective']
    model_type = config['model_type']
    timestep = config['timestep']
    learn_rate = config['learn_rate']
    scaler = config['scaler']
    n_hidden = config['n_hidden']
    hidden_dim = config['hidden_dim']
    train_rate = config['train_rate']
    dropout = config['dropout']
    epochs = config['epochs']
    xfold = config['xfold']
    batch_size = config['batch_size']
    es = config['es']
    # Stratified CV parameters (optional, default to False)
    stratify = config.get('stratify', False)
    stratify_bins = config.get('stratify_bins', 10)
    
    # Set random seed
    np.random.seed(seed=seed)
    
    # Set up console output logging early (before any prints)
    # Create result directory first
    reservoir_out = reservoirname + '_' + model_type
    model_dir = MODELS_DIR / reservoir_out
    result_dir = model_dir / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"console_log_{timestamp}.txt"
    log_filepath = os.path.join(str(result_dir), log_filename)
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    tee = TeeOutput(log_filepath)
    sys.stdout = tee
    
    try:
        if verbose:
            print("=" * 80)
            print("Step 4: Build Artificial Metabolic Network (AMN) Reservoir")
            print("=" * 80)
            print(f"\nConfiguration:")
            print(f"  Seed: {seed}")
            print(f"  Test ratio: {ratio_test}")
            print(f"  Reservoir name: {reservoirname}")
            print(f"  Training file: {trainingfile_name}")
            print(f"  Objective: {objective}")
            print(f"  Model type: {model_type}")
            print(f"  Timestep: {timestep}")
            print(f"  Learning rate: {learn_rate}")
            print(f"  Scaler: {scaler}")
            print(f"  Hidden layers: {n_hidden}")
            print(f"  Hidden dimension: {hidden_dim}")
            print(f"  Train rate: {train_rate}")
            print(f"  Dropout: {dropout}")
            print(f"  Epochs: {epochs}")
            print(f"  Cross-validation folds: {xfold}")
            print(f"  Stratified CV: {stratify}")
            if stratify:
                print(f"  Stratify bins: {stratify_bins}")
            print(f"  Batch size: {batch_size}")
            print(f"  Early stopping: {es}")
            print(f"  Console log file: {log_filepath}")
            print()
        
        # Build training file path
        trainingfile = DATA_PROCESSED_DIR / trainingfile_name
        trainingfile = str(trainingfile)
        
        if verbose:
            print("-" * 80)
            print("Step 4.1: Create Neural Model")
            print("-" * 80)
        
        # Create model
        model = Neural_Model(
            trainingfile=trainingfile,
            objective=objective,
            model_type=model_type,
            timestep=timestep,
            learn_rate=learn_rate,
            scaler=scaler,
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            train_rate=train_rate,
            dropout=dropout,
            epochs=epochs,
            xfold=xfold,
            batch_size=batch_size,
            es=es,
            verbose=verbose,
        )
        
        # Split training and test set using ratio_test
        if verbose:
            print("-" * 80)
            print("Step 4.2: Split training and test sets")
            print("-" * 80)
        
        ID = np.random.choice(
            model.X.shape[0],
            size=int(model.X.shape[0] * ratio_test),
            replace=False
        )
        Xtest, Ytest = model.X[ID, :], model.Y[ID, :]
        Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
        
        if verbose:
            print(f"  Total samples: {model.X.shape[0]}")
            print(f"  Training samples: {Xtrain.shape[0]}")
            print(f"  Test samples: {Xtest.shape[0]}")
            print()
            model.printout()
        
        # Define remaining output paths
        reservoir_dir = model_dir / 'reservoir'
        reservoirfile = str(reservoir_dir / reservoir_out)
        
        # Ensure reservoir directory exists
        reservoir_dir.mkdir(parents=True, exist_ok=True)
        result_dir = str(result_dir)
        
        if verbose:
            print("-" * 80)
            print("Step 4.3: Train and evaluate with cross-validation")
            print("-" * 80)
            print(f"  Reservoir name: {reservoir_out}")
            print(f"  Reservoir file: {reservoirfile}")
            print(f"  Result directory: {result_dir}")
            print()
        
        # Train and evaluate with per-fold logging
        start_time = time.time()
        model.X, model.Y = Xtrain, Ytrain
        reservoir, pred, stats, fold_records = train_evaluate_with_logging(
            model, result_dir=result_dir, prefix='cv', 
            stratify=stratify, stratify_bins=stratify_bins, verbose=verbose
        )
        delta_time = time.time() - start_time
        
        # Print aggregated cross-validation results
        print_stats(reservoirname, stats, delta_time)
        
        # Save reservoir model
        reservoir.save(reservoirfile)
        
        if verbose:
            print()
            print("-" * 80)
            print("Step 4.4: Print reservoir model info")
            print("-" * 80)
            reservoir.printout()
        
        # Run independent test set
        if len(Xtest) > 0:
            if verbose:
                print()
                print("-" * 80)
                print("Step 4.5: Evaluate on independent test set")
                print("-" * 80)
            
            start_time = time.time()
            reservoir.X, reservoir.Y = Xtest, Ytest
            X_eval, Y_eval = model_input(reservoir, verbose=verbose)
            pred_test, stats_test = evaluate_model(
                reservoir.model, X_eval, Y_eval, reservoir, verbose=verbose
            )
            delta_time = time.time() - start_time
            print_stats_eval('Independent test set', stats_test, delta_time)
            
            # Save test set predictions
            ensure_result_dir(result_dir)
            test_pred_path = os.path.join(result_dir, 'test_set_predictions.csv')
            np.savetxt(test_pred_path, pred_test, delimiter=',')
            print(f"  Saved test predictions to {test_pred_path}")
        
        if verbose:
            print()
            print("=" * 80)
            print("AMN Reservoir training and evaluation complete!")
            print("=" * 80)
            print(f"Reservoir saved to: {reservoirfile}_param.csv")
            print(f"Results saved to: {result_dir}/")
            print(f"Console log saved to: {log_filepath}")
            print()
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate AMN Reservoir model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/train_eval_reservoir.py configs/reservoir_params.yaml
    python scripts/train_eval_reservoir.py --verbose configs/reservoir_params.yaml
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
    
    # Run the training and evaluation
    try:
        train_eval_reservoir(config)
    except Exception as e:
        print(f"Error during training and evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
