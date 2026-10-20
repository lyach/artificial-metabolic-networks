"""
Shared helpers for the pipeline scripts in this folder.
"""

import argparse
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_EXTERNAL_DIR = PROJECT_ROOT / 'data' / 'external'
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

RULE_WIDTH = 80


# ── Environment ──────────────────────────────────────────────────────────────

def silence_tensorflow():
    """Suppress TensorFlow/XLA/absl log noise. Call before importing TensorFlow."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)


# ── Configuration ────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load a YAML configuration file into a dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_config(config, required, defaults):
    """
    Check required keys and fill in defaults for optional ones.

    Args:
        config (dict): Configuration loaded from YAML (plus CLI overrides).
        required (list): Keys that must be present in ``config``.
        defaults (dict): Values used for keys missing from ``config``.

    Returns:
        dict: New dict with ``defaults`` overlaid by ``config``.

    Raises:
        KeyError: If any required key is missing.
    """
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")
    return {**defaults, **config}


def build_parser(description, examples):
    """Create an argument parser with the positional ``config`` argument."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n' + '\n'.join(f'    {line}' for line in examples),
    )
    parser.add_argument('config', type=Path, help='Path to YAML configuration file')
    return parser


def config_from_args(args, overrides):
    """
    Load the config file named in ``args.config`` and apply CLI overrides.

    Exits with status 1 if the file is missing or cannot be parsed.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        overrides (dict): Mapping of config key -> CLI value. ``None`` values
            are ignored, so unset flags keep the value from the config file.

    Returns:
        dict: The configuration.
    """
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def run_or_exit(func, config, task):
    """Run ``func(config)``; on any exception print the traceback and exit 1."""
    try:
        return func(config)
    except Exception as e:
        print(f"Error during {task}: {e}")
        traceback.print_exc()
        sys.exit(1)


# ── Console output ───────────────────────────────────────────────────────────

def banner(title):
    """Print a top-level heading framed by '=' rules."""
    print('=' * RULE_WIDTH)
    print(title)
    print('=' * RULE_WIDTH)


def section(title):
    """Print a sub-heading framed by '-' rules."""
    print('-' * RULE_WIDTH)
    print(title)
    print('-' * RULE_WIDTH)


def print_config(config):
    """Print each ``key: value`` pair of the resolved configuration."""
    print('\nConfiguration:')
    for key, value in config.items():
        print(f'  {key}: {value}')
    print()


def timestamp():
    """Current time as ``YYYYmmdd_HHMMSS``, used in log and result filenames."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class _Tee:
    """File-like object that writes to both the console and a log file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # keep the log current if the job is killed

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


@contextmanager
def tee_stdout(log_path):
    """Copy everything printed inside the ``with`` block to ``log_path``."""
    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = tee.terminal
        tee.close()
