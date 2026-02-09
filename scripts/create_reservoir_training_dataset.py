#!/usr/bin/env python3
"""
Generate FBA-simulated training set for Reservoir AMN

This script generates an FBA-simulated dataset for the AMN Reservoir using experimental data.
The generation relies on the Aida & Ying Dataset, a CSV where rows = conditions and 
cols = exchanges + growth_value.

Usage:
    python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define data directories
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

from src.Build_Dataset import TrainingSet


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_reservoir_training_dataset(config):
    """
    Generate FBA-simulated training set for Reservoir AMN.
    
    Args:
        config (dict): Configuration dictionary with parameters
    """
    # Extract parameters from config
    seed = config['seed']
    cobraname = config['cobraname']
    mediumbound = config['mediumbound']
    expname = config['expname']
    size = config['size']
    size_i = config['size_i']
    reduce = config['reduce']
    mediumsize = config['mediumsize']
    verbose = config['verbose']
    mediumfile = config['mediumfile']
    output_prefix = config['output_prefix']
    
    # Set random seed
    np.random.seed(seed=seed)
    
    if verbose:
        print("=" * 80)
        print("Step 3: Generate FBA-simulated training set for Reservoir AMN")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Seed: {seed}")
        print(f"  COBRA model: {cobraname}")
        print(f"  Medium bound: {mediumbound}")
        print(f"  Experimental dataset: {expname}")
        print(f"  Medium size: {mediumsize}")
        print(f"  Training size per condition: {size_i}")
        print(f"  Reduce model: {reduce}")
        print(f"  Medium parameters file: {mediumfile}")
        print(f"  Output prefix: {output_prefix}")
        print()
    
    # Define file paths
    cobrafile = DATA_INTERIM_DIR / cobraname  # .xml
    expfile = DATA_INTERIM_DIR / expname  # .csv
    
    if verbose:
        print("-" * 80)
        print("Step 3.1: Retrieve compounds from experimental dataset")
        print("-" * 80)
    
    # Generate a parameter object with the experimental data
    parameter = TrainingSet(
        cobraname=str(cobrafile),
        mediumname=str(expfile),
        mediumbound=mediumbound,
        mediumsize=mediumsize,
        method='EXP',
        verbose=verbose
    )
    
    # Store the medium data (X matrix: rows=conditions, columns=compounds)
    X = parameter.X.copy()
    if verbose:
        print(f"\nX shape: {X.shape}")
        print(f"X:\n{X}\n")
    
    # Store the growth data (Y vector: growth rate per condition)
    Y = parameter.Y.copy()
    if verbose:
        print(f"Y shape: {Y.shape}")
        print(f"Y:\n{Y}\n")
    
    # Get parameter object for reservoir training set from medium parameters file
    mediumfile_path = DATA_PROCESSED_DIR / mediumfile  # .csv
    parameter = TrainingSet(
        cobraname=str(cobrafile),
        mediumname=str(mediumfile_path),
        mediumbound=mediumbound,
        method='FBA',
        verbose=verbose
    )
    
    # Create varmed: list of variable compound IDs based on experimental file (X)
    if verbose:
        print("-" * 80)
        print("Creating varmed (variable medium compounds per condition)")
        print("-" * 80)
    
    varmed = {}
    for i in range(X.shape[0]):
        varmed[i] = []
        for j in range(X.shape[1]):
            if parameter.levmed[j] > 1 and X[i, j] > 0:
                varmed[i].append(parameter.medium[j])
    
    varmed = list(varmed.values())
    
    if verbose:
        print(f"varmed length: {len(varmed)}")
        print(f"Example - varmed[0]: {varmed[0] if len(varmed) > 0 else 'N/A'}")
        print()
    
    # Generate FBA-simulated dataset
    if verbose:
        print("-" * 80)
        print("Step 3.2: Creating FBA-simulated dataset")
        print("-" * 80)
        print(f"Generating {size_i} samples per condition...")
        print()
    
    # Get a COBRApy-generated training set constrained by varmed
    successful_conditions = 0
    failed_conditions = 0
    
    for i in range(X.shape[0]):
        try:
            if verbose:
                print(f"Condition {i}/{X.shape[0]}: {varmed[i]}")
            parameter.get(sample_size=size_i, varmed=varmed[i], verbose=verbose)
            successful_conditions += 1
        except Exception as e:
            print(f"WARNING: Skipping condition {i} (varmed={varmed[i]}): {e}")
            failed_conditions += 1
            continue
    
    if verbose:
        print()
        print("-" * 80)
        print("Summary:")
        print(f"  Successful conditions: {successful_conditions}")
        print(f"  Failed conditions: {failed_conditions}")
        print("-" * 80)
        print()
    
    # Save the training set
    trainingfile = DATA_PROCESSED_DIR / f'{output_prefix}_{parameter.mediumbound}'
    
    if verbose:
        print(f"Saving training set to: {trainingfile}")
    
    parameter.save(trainingfile, reduce=reduce)
    
    if verbose:
        print()
        print("=" * 80)
        print("FBA-simulated training set generation complete!")
        print("=" * 80)
        print(f"Output file: {trainingfile}.npz")
        print()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate FBA-simulated training set for Reservoir AMN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
        """
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
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
    
    # Run the training set generation
    try:
        create_reservoir_training_dataset(config)
    except Exception as e:
        print(f"Error during training set generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
