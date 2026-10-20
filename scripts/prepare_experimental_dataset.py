#!/usr/bin/env python3
"""
Step 2: Pre-process the Aida & Ying dataset for the AMN pipeline.

2.1 Map Aida & Ying compounds onto GEM exchange reactions, average replicate
    growth rates and rename columns to the duplicated model's uptake (``_i``)
    reactions (``src/data_preprocess.py``).
2.2 Build the medium parameter file used to sample random media in step 3:
    Faure et al.'s environmental compounds + the Aida & Ying compounds.
2.3 Pad the experimental dataset with the unmeasured environmental compounds
    so that its columns match the parameter file.

Requires the corrected medium from step 1 (``scripts/prepare_gem.py``).

Usage:
    python scripts/prepare_experimental_dataset.py configs/experimental_dataset.yaml
    python scripts/prepare_experimental_dataset.py --data-dir /path/to/data configs/experimental_dataset.yaml
"""

from pathlib import Path

import cobra
import numpy as np
import pandas as pd

import _common
from _common import PROJECT_ROOT, banner, section

from src.data_preprocess import create_supplementary_csv, create_media_growth_csv, match_amns_media

REQUIRED_KEYS = [
    'cobraname', 'faure_parameters', 'output_parameters', 'faure_drop',
    'compound_classes', 'default_class', 'output_padded', 'padding',
]
DEFAULTS = {
    'drop_first_row': True,
    'data_dir': PROJECT_ROOT / 'data',
}

# Written by src/data_preprocess.py (file names are fixed there)
AMNS_CSV = 'aidaying_media_growth_amns.csv'
GROWTH_COLUMN = 'GR_AVG'


def map_compounds(model_path, external_dir, interim_dir):
    """
    Step 2.1: Map compounds to GEM reactions and write the experimental CSVs.

    Returns:
        Path: The ``aidaying_media_growth_amns.csv`` file.
    """
    model = cobra.io.read_sbml_model(str(model_path))
    mapping_csv = create_supplementary_csv(model, interim_dir)
    final_df = create_media_growth_csv(external_dir, mapping_csv, interim_dir)
    corrected_csv = match_amns_media(interim_dir, interim_dir, final_df)
    if corrected_csv is None:
        raise FileNotFoundError(
            f"{interim_dir / 'iML1515_medium_corrected.csv'} not found. Run scripts/prepare_gem.py first."
        )
    return corrected_csv


def build_parameter_file(aida_cols, faure_df, config):
    """
    Step 2.2: Combine Faure's environmental compounds with the Aida & Ying ones.

    Faure columns listed in ``faure_drop`` or shared with Aida & Ying are
    removed. Every Aida & Ying compound gets ``level``/``max_value`` from its
    class in ``compound_classes`` (``default_class`` if unlisted).

    Args:
        aida_cols (list): Aida & Ying compound columns, in dataset order.
        faure_df (pd.DataFrame): Faure parameter file (``name`` + compounds).
        config (dict): Resolved configuration.

    Returns:
        pd.DataFrame: Parameter file, rows ``level, max_value, ratio_drawing, comment``.
    """
    compound_to_class = {
        compound: cls
        for cls in config['compound_classes'].values()
        for compound in cls['compounds']
    }
    new_cols = {}
    for compound in aida_cols:
        cls = compound_to_class.get(compound, config['default_class'])
        new_cols[compound] = [cls['level'], cls['max_value'], np.nan, np.nan]

    faure_cols = faure_df.columns.drop('name')
    common = set(faure_cols) & set(aida_cols)
    print(f"Faure compounds: {len(faure_cols)}")
    print(f"Aida & Ying compounds: {len(aida_cols)}")
    print(f"Common compounds (Aida & Ying values used): {len(common)}")
    print(f"Faure-only compounds dropped: {len(config['faure_drop'])}")

    faure_df = faure_df.drop(columns=config['faure_drop']).drop(columns=list(common))
    params_df = pd.concat([faure_df, pd.DataFrame(new_cols)], axis=1)
    print(f"Parameter file shape: {params_df.shape}")
    return params_df


def pad_dataset(aida_df, padding, drop_first_row):
    """
    Step 2.3: Add constant columns for unmeasured compounds before ``GR_AVG``.

    Args:
        aida_df (pd.DataFrame): Experimental dataset (compounds + ``GR_AVG``).
        padding (dict): Compound -> constant value, in output column order.
        drop_first_row (bool): Drop the first condition.

    Returns:
        pd.DataFrame: Padded dataset.
    """
    print(f"Original shape: {aida_df.shape}")
    growth = aida_df.pop(GROWTH_COLUMN)
    aida_df = pd.concat([aida_df, pd.DataFrame(padding, index=aida_df.index)], axis=1)
    aida_df[GROWTH_COLUMN] = growth
    if drop_first_row:
        aida_df = aida_df.iloc[1:]
    print(f"Padded shape: {aida_df.shape}")
    return aida_df


def prepare_experimental_dataset(config):
    """
    Run steps 2.1–2.3.

    Outputs:
        <data_dir>/interim/aidaying_compound_mapping.csv
        <data_dir>/interim/aidaying_media_growth.csv
        <data_dir>/interim/aidaying_media_growth_amns.csv
        <data_dir>/interim/<output_padded>.csv
        <data_dir>/processed/<output_parameters>.csv

    Args:
        config (dict): Configuration (see ``configs/experimental_dataset.yaml``).
    """
    config = _common.resolve_config(config, REQUIRED_KEYS, DEFAULTS)
    data_dir = Path(config['data_dir'])
    external_dir = data_dir / 'external'
    interim_dir = data_dir / 'interim'
    processed_dir = data_dir / 'processed'
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    banner("Step 2: Pre-process the Aida & Ying dataset")
    _common.print_config(config)

    section("Step 2.1: Map Aida & Ying compounds to GEM reactions")
    amns_csv = map_compounds(data_dir / 'raw' / f"{config['cobraname']}.xml",
                             external_dir, interim_dir)
    print()

    # Read back from disk so values round-trip through CSV exactly as in the notebook
    aida_df = pd.read_csv(amns_csv)
    aida_cols = aida_df.columns.drop(GROWTH_COLUMN)

    section("Step 2.2: Build the medium parameter file")
    faure_df = pd.read_csv(external_dir / f"{config['faure_parameters']}.csv")
    params_df = build_parameter_file(aida_cols, faure_df, config)
    params_path = processed_dir / f"{config['output_parameters']}.csv"
    params_df.to_csv(params_path, index=False)
    print(f"Saved to: {params_path}")
    print()

    section("Step 2.3: Pad the experimental dataset")
    padded_df = pad_dataset(aida_df, config['padding'], config['drop_first_row'])
    padded_path = interim_dir / f"{config['output_padded']}.csv"
    padded_df.to_csv(padded_path, index=False)
    print(f"Saved to: {padded_path}")

    n_compounds = padded_df.shape[1] - 1
    if list(padded_df.columns[:-1].sort_values()) != sorted(params_df.columns.drop('name')):
        print("WARNING: padded dataset and parameter file have different compounds")
    print(f"Compounds: {n_compounds} (use as mediumsize / n_input in later configs)")
    print()
    banner("Experimental dataset preparation complete!")


def main():
    parser = _common.build_parser(
        description='Pre-process the Aida & Ying dataset (mapping, parameter file, padding)',
        examples=[
            'python scripts/prepare_experimental_dataset.py configs/experimental_dataset.yaml',
            'python scripts/prepare_experimental_dataset.py --data-dir /path/to/data configs/experimental_dataset.yaml',
        ],
    )
    parser.add_argument(
        '--data-dir', type=Path, default=None,
        help='Root holding raw/, external/, interim/ and processed/ (default: <project>/data).',
    )
    args = parser.parse_args()

    config = _common.config_from_args(args, {'data_dir': args.data_dir})
    _common.run_or_exit(prepare_experimental_dataset, config, 'experimental dataset preparation')


if __name__ == '__main__':
    main()
