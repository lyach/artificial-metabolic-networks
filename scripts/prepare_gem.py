#!/usr/bin/env python3
"""
Step 1: Make the genome-scale model (GEM) AMN-compatible.

AMNs require every flux to be non-negative. Reversible internal reactions are
duplicated (one per direction) and exchange reactions are split into uptake
(``_i``) and secretion (``_o``) reactions. The duplicated model's medium is
then corrected so its uptake bounds match the original model's medium.

Usage:
    python scripts/prepare_gem.py configs/gem_preparation.yaml
    python scripts/prepare_gem.py --data-dir /path/to/data configs/gem_preparation.yaml
"""

from pathlib import Path

import cobra
import pandas as pd

import _common
from _common import PROJECT_ROOT, banner, section

from src.Duplicate_Model import screen_out_in, duplicate_model, correct_duplicated_med

REQUIRED_KEYS = ['cobraname', 'output_model', 'output_medium', 'io_dict', 'unsignificant_mols']
DEFAULTS = {
    'data_dir': PROJECT_ROOT / 'data',
}


def prepare_gem(config):
    """
    Duplicate the GEM and save it with its corrected medium.

    Outputs, in ``<data_dir>/interim/``:
        <output_model>.xml    duplicated (irreversible) model
        <output_medium>.csv   corrected medium (``rxn``, ``value``)

    Args:
        config (dict): Configuration (see ``configs/gem_preparation.yaml``).
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    data_dir = Path(config['data_dir'])
    model_path = data_dir / 'raw' / f"{config['cobraname']}.xml"
    interim_dir = data_dir / 'interim'
    interim_dir.mkdir(parents=True, exist_ok=True)

    # YAML gives [from, to] lists; screen_out_in compares (from, to) tuples
    io_dict = {suffix: [tuple(pair) for pair in pairs]
               for suffix, pairs in config['io_dict'].items()}

    banner("Step 1: Make the GEM AMN-compatible")
    _common.print_config(config)

    model = cobra.io.read_sbml_model(str(model_path))
    print(f"Total number of boundary reactions: {len(model.boundary)}")
    print(f"Total number of medium reactions: {len(model.medium)}")
    print()

    section("Step 1.1: Screen reaction directions")
    reac_id_to_io_count_and_way = screen_out_in(model, io_dict, config['unsignificant_mols'])
    print()

    section("Step 1.2: Duplicate reactions")
    new_model = duplicate_model(model, reac_id_to_io_count_and_way)

    # Uptake reactions absent from the default medium get a bound of 1e-300
    # (not 0) so that they still appear in the medium object.
    default_med = model.medium
    correct_med = correct_duplicated_med(default_med, new_model.medium)
    new_model.medium = correct_med
    print(f"Default medium: {len(default_med)} compounds")
    print(f"Corrected medium: {len(correct_med)} compounds")
    print()

    section("Step 1.3: Save")
    medium_path = interim_dir / f"{config['output_medium']}.csv"
    pd.DataFrame(list(correct_med.items()), columns=['rxn', 'value']) \
        .to_csv(medium_path, index=False)

    new_model.repair()
    new_model_path = interim_dir / f"{config['output_model']}.xml"
    cobra.io.write_sbml_model(new_model, str(new_model_path))

    print(f"Original model: {model_path}")
    print(f"Duplicated model: {new_model_path}")
    print(f"Corrected medium: {medium_path}")
    print()
    banner("GEM preparation complete!")


def main():
    parser = _common.build_parser(
        description='Make the GEM AMN-compatible (duplicate reversible and exchange reactions)',
        examples=[
            'python scripts/prepare_gem.py configs/gem_preparation.yaml',
            'python scripts/prepare_gem.py --data-dir /path/to/data configs/gem_preparation.yaml',
        ],
    )
    parser.add_argument(
        '--data-dir', type=Path, default=None,
        help='Root holding raw/ and interim/ (default: <project>/data).',
    )
    args = parser.parse_args()

    config = _common.config_from_args(args, {'data_dir': args.data_dir})
    _common.run_or_exit(prepare_gem, config, 'GEM preparation')


if __name__ == '__main__':
    main()
