#!/usr/bin/env python3
"""
Generate pFBA-simulated training set for Reservoir AMN

This script generates an pFBA-simulated dataset for the AMN Reservoir using experimental data.
The generation relies on the Aida & Ying Dataset, a CSV where rows = conditions and 
cols = exchanges + growth_value.

Supports multiprocessing to parallelize pFBA simulations across conditions within a single node.

Usage:
    python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
    python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml
"""

import sys
import argparse
import time
import uuid
from pathlib import Path
import yaml
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define data directories
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

from src.Build_Dataset import TrainingSet, get_io_cobra, get_index_from_id


# ---- Multiprocessing worker functions ----

# Per-worker global state (initialized once per worker process via pool initializer)
_worker_model = None
_worker_params = None


def _init_worker(cobrafile, objective, medium, mediumbound,
                 levmed, valmed, ratmed, method):
    """
    Pool initializer: each worker process loads its own COBRA model
    and caches shared parameters to avoid repeated serialization.

    This is called once when each worker process starts, so the model
    is loaded only num_workers times total (not once per condition).
    """
    global _worker_model, _worker_params
    import cobra
    import optlang
    optlang.glpk_interface.Configuration()
    _worker_model = cobra.io.read_sbml_model(cobrafile + '.xml')
    _worker_params = {
        'objective': list(objective),
        'medium': list(medium),
        'mediumbound': str(mediumbound),
        'levmed': list(levmed),
        'valmed': list(valmed),
        'ratmed': float(ratmed),
        'method': str(method),
    }


def _process_condition(args):
    """
    Worker function: generate `size_i` pFBA samples for a single condition.

    Each call runs size_i pFBA simulations using the worker's own COBRA model
    and a per-condition random seed for reproducibility.

    Args:
        args: tuple of (condition_index, varmed_list, sample_size, random_seed, progress_queue)

    Returns:
        tuple: (condition_index, X_array, Y_array, error_string_or_None)
    """
    condition_idx, varmed_i, size_i, condition_seed, progress_queue = args
    global _worker_model, _worker_params

    # Seed RNG per condition for reproducibility across runs
    np.random.seed(condition_seed)

    try:
        X, Y = {}, {}
        for i in range(size_i):
            X[i], Y[i] = get_io_cobra(
                _worker_model,
                _worker_params['objective'],
                _worker_params['medium'],
                _worker_params['mediumbound'],
                varmed_i,
                _worker_params['levmed'],
                _worker_params['valmed'],
                _worker_params['ratmed'],
                None,  # Pout (E parameter) — unused inside get_io_cobra
                method=_worker_params['method'],
                verbose=False,
            )
            # Report progress after each simulation
            if progress_queue is not None:
                progress_queue.put(1)
        
        X_arr = np.asarray(list(X.values()))
        Y_arr = np.asarray(list(Y.values()))
        return (condition_idx, X_arr, Y_arr, None)
    except Exception as e:
        # Report failed simulations to progress bar
        if progress_queue is not None:
            progress_queue.put(size_i)
        return (condition_idx, None, None, str(e))


# ── Configuration & main logic ────────────────────────────────────────────────

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_reservoir_training_dataset(config):
    """
    Generate pFBA-simulated training set for Reservoir AMN.
    
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
    num_workers = config.get('num_workers', 1)
    
    # Use run ID from config or generate a unique one
    run_id = config.get('run_id', uuid.uuid4().hex[:8])
    run_dir = DATA_PROCESSED_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed=seed)
    
    if verbose:
        print("=" * 80)
        print("Step 3: Generate pFBA-simulated training set for Reservoir AMN")
        print("=" * 80)
        print(f"  Run ID: {run_id}")
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
        print(f"  Parallel workers: {num_workers}")
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
    
    # Store the growth data (Y vector: growth rate per condition)
    Y = parameter.Y.copy()
    if verbose:
        print(f"Y shape: {Y.shape}")
    
    # Get parameter object for reservoir training set from medium parameters file
    mediumfile_path = DATA_PROCESSED_DIR / mediumfile  # .csv
    parameter = TrainingSet(
        cobraname=str(cobrafile),
        mediumname=str(mediumfile_path),
        mediumbound=mediumbound,
        method='pFBA',
        verbose=verbose
    )
    
    # Create varmed: list of variable compound IDs based on experimental file (X)
    if verbose:
        print("-" * 80)
        print("Creating varmed (variable medium compounds per condition)")
        print("-" * 80)
    
    varmed_dict = {}
    for i in range(X.shape[0]):
        varmed_dict[i] = []
        for j in range(X.shape[1]):
            if parameter.levmed[j] > 1 and X[i, j] > 0:
                varmed_dict[i].append(parameter.medium[j])
    
    # Keep only unique varmed items by converting
    # inner lists to tuples and deduplicate while preserving order
    varmed_all = list(varmed_dict.values())
    unique_tuples = dict.fromkeys(tuple(row) for row in varmed_all)
    varmed = [list(unique_item) for unique_item in unique_tuples]
    
    if verbose:
        print(f"Original varmed len: {len(varmed_all)}")
        print(f"Unique varmed len: {len(varmed)}")
        print(f"Example - varmed[0]: {varmed[0] if len(varmed) > 0 else 'N/A'}")
        print()
    
    # Generate pFBA-simulated dataset (parallel)
    n_conditions = len(varmed)
    total_simulations = size_i * n_conditions
    
    if verbose:
        print("-" * 80)
        print("Step 3.2: Creating pFBA-simulated dataset")
        print("-" * 80)
        print(f"Generating {size_i} samples x {n_conditions} unique medium combinations"
              f"= {total_simulations} total pFBA simulations")
        print(f"Using {num_workers} parallel workers")
        print()
    
    # Create a Manager queue for progress reporting from workers
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # Build argument list: each condition gets its own reproducible seed
    worker_args = [
        (i, varmed[i], size_i, seed + i, progress_queue)
        for i in range(n_conditions)
    ]
    
    successful_conditions = 0
    failed_conditions = 0
    results_by_idx = {}
    
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            str(cobrafile),
            parameter.objective,
            parameter.medium,
            parameter.mediumbound,
            parameter.levmed,
            parameter.valmed,
            parameter.ratmed,
            parameter.method,
        ),
    ) as pool:
        # Submit all conditions asynchronously
        async_result = pool.map_async(_process_condition, worker_args)
        
        # Poll the progress
        with tqdm(total=total_simulations, desc="pFBA simulations", unit="sim") as pbar:
            while not async_result.ready():
                while not progress_queue.empty():
                    try:
                        pbar.update(progress_queue.get_nowait())
                    except Exception:
                        break
                time.sleep(0.2)
            
            # Drain remaining progress updates after completion
            while not progress_queue.empty():
                try:
                    pbar.update(progress_queue.get_nowait())
                except Exception:
                    break
        
        # Collect results
        all_results = async_result.get()
        for cond_idx, X_cond, Y_cond, error in all_results:
            if error is not None:
                print(
                    f"WARNING: Skipping condition {cond_idx} "
                    f"(varmed={varmed[cond_idx]}): {error}"
                )
                failed_conditions += 1
            else:
                results_by_idx[cond_idx] = (X_cond, Y_cond)
                successful_conditions += 1
    
    # Aggregate results in original condition order
    if results_by_idx:
        ordered_X = []
        ordered_Y = []
        for idx in sorted(results_by_idx.keys()):
            X_cond, Y_cond = results_by_idx[idx]
            ordered_X.append(X_cond)
            ordered_Y.append(Y_cond)
        
        combined_X = np.concatenate(ordered_X, axis=0)
        combined_Y = np.concatenate(ordered_Y, axis=0)
        
        # Handle 'EB' mediumbound: replace X columns with corresponding Y columns
        if parameter.mediumbound == 'EB':
            col = 0
            for rid in parameter.medium:
                j = get_index_from_id(rid, parameter.model.reactions)
                combined_X[:, col] = combined_Y[:, j]
                col += 1
        
        parameter.X = combined_X
        parameter.Y = combined_Y
        parameter.size = parameter.X.shape[0]
    else:
        print("ERROR: All conditions failed. No training data generated.")
        sys.exit(1)
    
    if verbose:
        print()
        print("-" * 80)
        print("Summary:")
        print(f"  Successful conditions: {successful_conditions}")
        print(f"  Failed conditions: {failed_conditions}")
        print(f"  Total samples: {parameter.size}")
        print(f"  X shape: {parameter.X.shape}")
        print(f"  Y shape: {parameter.Y.shape}")
        print("-" * 80)
        print()
    
    # Save the training set
    trainingfile = run_dir / f'{output_prefix}_{parameter.mediumbound}'
    
    if verbose:
        print(f"Saving training set to: {trainingfile}")
    
    parameter.save(trainingfile, reduce=reduce)
    
    if verbose:
        print()
        print("=" * 80)
        print("pFBA-simulated training set generation complete!")
        print("=" * 80)
        print(f"Output file: {trainingfile}.npz")
        print()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate pFBA-simulated training set for Reservoir AMN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
    python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml
        """
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers.'
             'Defaults to value in config, or 1 if not specified.'
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
    
    # CLI --num-workers overrides config file value
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    
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
