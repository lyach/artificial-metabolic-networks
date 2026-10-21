#!/usr/bin/env python3
"""
Step 3: Generate the pFBA-simulated training set for the AMN Reservoir.

For every unique combination of variable medium compounds found in the
experimental dataset (Aida & Ying, rows = conditions, cols = exchanges +
growth), draw ``size_i`` random media and solve them with pFBA.

Simulations are parallelised with ``multiprocessing`` on a single node. Each
condition is split into chunks of ``chunk_size`` samples, handed out
round-robin across conditions, so a few slow conditions cannot leave most
workers idle at the end of the run.

Usage:
    python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
    python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml
"""

import math
import multiprocessing as mp
import time
import uuid
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import _common
from _common import DATA_INTERIM_DIR, DATA_PROCESSED_DIR, banner, section

from src.Build_Dataset import TrainingSet, get_io_cobra, get_index_from_id

REQUIRED_KEYS = [
    'seed', 'cobraname', 'mediumbound', 'expname', 'mediumsize',
    'mediumfile', 'size_i', 'output_prefix',
]
DEFAULTS = {
    'run_id': None,               # None -> random 8-char hex ID
    'num_workers': 1,
    'chunk_size': 10,             # samples per task; smaller = better load balance
    'cobra_min_objective': 0.01,
    'cap_varmed': False,
    'max_varmed_scale': 5,        # used only if cap_varmed
    'filter_zero_biomass': False,
    'biomass_threshold': 0.01,    # used only if filter_zero_biomass
    'reduce': False,
    'verbose': True,
}


# ── Multiprocessing workers ──────────────────────────────────────────────────

# Per-worker state, set once per process by the pool initializer
_worker_model = None
_worker_params = None


def _init_worker(cobrafile, objective, medium, mediumbound,
                 levmed, valmed, ratmed, method,
                 cobra_min_objective, max_varmed_scale):
    """
    Pool initializer: load one COBRA model per worker process.

    Runs once per worker, so the SBML file is parsed ``num_workers`` times in
    total rather than once per condition.
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
        'cobra_min_objective': float(cobra_min_objective),
        'max_varmed_scale': max_varmed_scale,
    }


def _process_chunk(args):
    """
    Generate ``n_samples`` pFBA samples for one chunk of one condition.

    Args:
        args (tuple): ``(condition_idx, chunk_idx, varmed_i, n_samples, seed)``.

    Returns:
        tuple: ``(condition_idx, chunk_idx, X, Y, error, seconds)`` where
        ``error`` is ``None`` on success, otherwise ``X`` and ``Y`` are ``None``.
    """
    condition_idx, chunk_idx, varmed_i, n_samples, chunk_seed = args
    p = _worker_params
    start = time.time()

    # Per-chunk seed keeps results reproducible regardless of scheduling
    np.random.seed(chunk_seed)

    try:
        X, Y = [], []
        for _ in range(n_samples):
            x, y = get_io_cobra(
                _worker_model,
                p['objective'],
                p['medium'],
                p['mediumbound'],
                varmed_i,
                p['levmed'],
                p['valmed'],
                p['ratmed'],
                None,  # E (Pout): unused inside get_io_cobra
                method=p['method'],
                verbose=False,
                cobra_min_objective=p['cobra_min_objective'],
                max_varmed_scale=p['max_varmed_scale'],
            )
            X.append(x)
            Y.append(y)
        return condition_idx, chunk_idx, np.asarray(X), np.asarray(Y), None, time.time() - start
    except Exception as e:
        return condition_idx, chunk_idx, None, None, str(e), time.time() - start


# ── Pipeline steps ───────────────────────────────────────────────────────────

def build_varmed(X, exp_medium, parameter):
    """
    List the variable compounds present in each experimental condition.

    A compound is variable if its ``level`` in the medium parameter file is
    > 1. Compounds are matched by name, because the experimental file and the
    parameter file list them in different orders. Duplicate combinations are
    removed, keeping first-seen order.

    Args:
        X (np.ndarray): Experimental media, shape (conditions, compounds).
        exp_medium (list): Compound IDs of the columns of ``X``.
        parameter (TrainingSet): Loaded from the medium parameter file.

    Returns:
        tuple: ``(varmed, n_conditions)`` where ``varmed`` is the list of unique
        compound-ID lists and ``n_conditions`` the number before deduplication.

    Raises:
        ValueError: If a variable compound is missing from the experimental file.
    """
    exp_col = {rid: j for j, rid in enumerate(exp_medium)}
    variable = [rid for rid, lev in zip(parameter.medium, parameter.levmed) if lev > 1]
    missing = [rid for rid in variable if rid not in exp_col]
    if missing:
        raise ValueError(f"Variable compounds missing from the experimental file: {missing}")

    varmed_all = [
        [rid for rid in variable if X[i, exp_col[rid]] > 0]
        for i in range(X.shape[0])
    ]
    unique = dict.fromkeys(tuple(row) for row in varmed_all)
    return [list(row) for row in unique], len(varmed_all)


def make_tasks(varmed, size_i, chunk_size, seed):
    """
    Split each condition into chunks and order them round-robin.

    Chunk ``k`` of every condition comes before chunk ``k + 1`` of any, so
    slow conditions are spread over the whole run instead of forming its tail.
    Each chunk's seed depends only on (condition, chunk), not on the order.
    """
    n_chunks = math.ceil(size_i / chunk_size)
    tasks = []
    for k in range(n_chunks):
        n_samples = min(chunk_size, size_i - k * chunk_size)
        for i, varmed_i in enumerate(varmed):
            tasks.append((i, k, varmed_i, n_samples, seed + i * n_chunks + k))
    return tasks


def report_slowest(seconds, varmed, size_i, n=5):
    """Print the conditions that used the most worker time."""
    print(f"  Slowest conditions (worker time, of {sum(seconds.values()) / 60:.0f} min total):")
    for i in sorted(seconds, key=seconds.get, reverse=True)[:n]:
        print(f"    condition {i:3d}: {seconds[i] / 60:6.1f} min "
              f"({seconds[i] / size_i:5.2f} s/sample, {len(varmed[i])} variable compounds)")


def simulate(varmed, parameter, cobrafile, config):
    """
    Run ``size_i`` pFBA simulations for every entry of ``varmed`` in parallel.

    Returns:
        tuple: ``(results, n_failed, seconds)`` where ``results`` maps
        condition index to ``(X, Y)`` (chunks concatenated in order),
        ``n_failed`` counts failed chunks and ``seconds`` maps condition index
        to total worker time.
    """
    size_i = config['size_i']
    max_varmed_scale = config['max_varmed_scale'] if config['cap_varmed'] else None
    tasks = make_tasks(varmed, size_i, config['chunk_size'], config['seed'])

    chunks = defaultdict(dict)       # condition -> {chunk: (X, Y)}
    seconds = defaultdict(float)     # condition -> worker seconds
    n_failed = 0

    with mp.Pool(
        processes=config['num_workers'],
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
            config['cobra_min_objective'],
            max_varmed_scale,
        ),
    ) as pool:
        # mininterval: fewer progress lines in batch logs
        with tqdm(total=size_i * len(varmed), desc='pFBA simulations', unit='sim',
                  mininterval=10) as pbar:
            for i, k, X_chunk, Y_chunk, error, secs in pool.imap_unordered(_process_chunk, tasks):
                seconds[i] += secs
                if error is not None:
                    tqdm.write(f"WARNING: Skipping chunk {k} of condition {i} "
                               f"(varmed={varmed[i]}): {error}")
                    n_failed += 1
                else:
                    chunks[i][k] = (X_chunk, Y_chunk)
                pbar.update(min(config['chunk_size'], size_i - k * config['chunk_size']))

    results = {
        i: (np.concatenate([chunks[i][k][0] for k in sorted(chunks[i])], axis=0),
            np.concatenate([chunks[i][k][1] for k in sorted(chunks[i])], axis=0))
        for i in sorted(chunks)
    }
    return results, n_failed, dict(seconds)


def filter_low_biomass(X, Y, parameter, threshold, verbose):
    """Drop samples whose objective (biomass) flux is below ``threshold``."""
    obj_col = get_index_from_id(parameter.objective[0], parameter.model.reactions)
    mask = np.abs(Y[:, obj_col]) >= threshold
    if verbose:
        print(f"\nPost-filtering: removed {int((~mask).sum())} samples "
              f"with biomass < {threshold}")
        print(f"  Samples before: {X.shape[0]}, after: {int(mask.sum())}")
    return X[mask], Y[mask]


def create_reservoir_training_dataset(config):
    """
    Generate the pFBA-simulated training set and save it as ``.npz``.

    Output: ``data/processed/<run_id>/<output_prefix>_<mediumbound>.npz``
    (plus the matching ``.xml`` model).

    Args:
        config (dict): Configuration (see ``configs/reservoir_training_dataset.yaml``).
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    if config['run_id'] is None:
        config['run_id'] = uuid.uuid4().hex[:8]
    verbose = config['verbose']

    run_dir = DATA_PROCESSED_DIR / config['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    cobrafile = DATA_INTERIM_DIR / config['cobraname']               # .xml
    expfile = DATA_INTERIM_DIR / config['expname']                   # .csv
    mediumfile = DATA_PROCESSED_DIR / config['mediumfile']           # .csv

    np.random.seed(seed=config['seed'])

    if verbose:
        banner("Step 3: Generate pFBA-simulated training set for Reservoir AMN")
        _common.print_config(config)
        section("Step 3.1: Retrieve compounds from experimental dataset")

    # Experimental media (X: conditions x compounds) and growth (Y)
    experimental = TrainingSet(
        cobraname=str(cobrafile),
        mediumname=str(expfile),
        mediumbound=config['mediumbound'],
        mediumsize=config['mediumsize'],
        method='EXP',
        verbose=verbose,
    )
    X_exp = experimental.X.copy()
    if verbose:
        print(f"\nX shape: {X_exp.shape}")
        print(f"Y shape: {experimental.Y.shape}")

    # Medium parameter file: levels/max values used to draw random media
    parameter = TrainingSet(
        cobraname=str(cobrafile),
        mediumname=str(mediumfile),
        mediumbound=config['mediumbound'],
        method='pFBA',
        verbose=verbose,
    )

    if verbose:
        section("Creating varmed (variable medium compounds per condition)")
    varmed, n_conditions_all = build_varmed(X_exp, experimental.medium, parameter)
    if verbose:
        print(f"Original varmed len: {n_conditions_all}")
        print(f"Unique varmed len: {len(varmed)}")
        print(f"Example - varmed[0]: {varmed[0] if varmed else 'N/A'}")
        print()
        section("Step 3.2: Creating pFBA-simulated dataset")
        print(f"Generating {config['size_i']} samples x {len(varmed)} unique medium "
              f"combinations = {config['size_i'] * len(varmed)} total pFBA simulations")
        print(f"Using {config['num_workers']} parallel workers, "
              f"chunks of {config['chunk_size']} samples")
        print()

    results, n_failed, seconds = simulate(varmed, parameter, cobrafile, config)
    if not results:
        raise RuntimeError("All conditions failed. No training data generated.")

    # Concatenate in original condition order
    X = np.concatenate([results[i][0] for i in sorted(results)], axis=0)
    Y = np.concatenate([results[i][1] for i in sorted(results)], axis=0)

    # Exact bounds: X is the realised uptake flux, not the drawn bound
    if parameter.mediumbound == 'EB':
        for col, rid in enumerate(parameter.medium):
            X[:, col] = Y[:, get_index_from_id(rid, parameter.model.reactions)]

    if config['filter_zero_biomass']:
        X, Y = filter_low_biomass(X, Y, parameter, config['biomass_threshold'], verbose)

    parameter.X, parameter.Y, parameter.size = X, Y, X.shape[0]

    if verbose:
        print()
        section("Summary:")
        print(f"  Conditions with data: {len(results)} / {len(varmed)}")
        print(f"  Failed chunks: {n_failed}")
        report_slowest(seconds, varmed, config['size_i'])
        print(f"  Total samples: {parameter.size}")
        print(f"  X shape: {parameter.X.shape}")
        print(f"  Y shape: {parameter.Y.shape}")
        print()

    trainingfile = run_dir / f"{config['output_prefix']}_{parameter.mediumbound}"
    if verbose:
        print(f"Saving training set to: {trainingfile}")
    parameter.save(trainingfile, reduce=config['reduce'])

    if verbose:
        print()
        banner("pFBA-simulated training set generation complete!")
        print(f"Output file: {trainingfile}.npz")
        print()


def main():
    parser = _common.build_parser(
        description='Generate pFBA-simulated training set for Reservoir AMN',
        examples=[
            'python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml',
            'python scripts/create_reservoir_training_dataset.py --num-workers 16 configs/reservoir_training_dataset.yaml',
        ],
    )
    parser.add_argument(
        '--num-workers', '-w', type=int, default=None,
        help='Number of parallel workers. Overrides config file setting (default 1).',
    )
    args = parser.parse_args()

    config = _common.config_from_args(args, {'num_workers': args.num_workers})
    _common.run_or_exit(create_reservoir_training_dataset, config, 'training set generation')


if __name__ == '__main__':
    main()
