"""
Dataset module for Differentiable FBA models.

This module provides dataset creation functions for end-to-end differentiable
Flux Balance Analysis (FBA) models.

Usage:
    python -m src.dataset --config configs/dataset.yaml

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cobra
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MetabolicModel:
    """
    Container for metabolic model matrices and metadata.
    
    Attributes:
        S: Stoichiometric matrix (m x n) - sparse format
        c: Objective vector (n,) 
        lb: Lower bounds (n,)
        ub: Upper bounds (n,) - default upper bounds
        reaction_ids: List of reaction IDs
        metabolite_ids: List of metabolite IDs
        medium_indices: Indices of uptake reactions in the flux vector
        medium_ids: IDs of uptake reactions
        objective_idx: Index of the objective reaction (biomass)
        objective_id: ID of the objective reaction
    """
    S: np.ndarray  # Will be converted to sparse
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    reaction_ids: list[str]
    metabolite_ids: list[str]
    medium_indices: np.ndarray
    medium_ids: list[str]
    objective_idx: int
    objective_id: str
    
    @property
    def n_reactions(self) -> int:
        return self.S.shape[1]
    
    @property
    def n_metabolites(self) -> int:
        return self.S.shape[0]
    
    @property
    def n_medium(self) -> int:
        return len(self.medium_indices)
    
    @property
    def S_sparse(self) -> sp.csr_matrix:
        """Return stoichiometric matrix as sparse CSR."""
        if sp.issparse(self.S):
            return self.S.tocsr()
        return sp.csr_matrix(self.S)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'S': self.S if isinstance(self.S, np.ndarray) else self.S.toarray(),
            'c': self.c,
            'lb': self.lb,
            'ub': self.ub,
            'reaction_ids': self.reaction_ids,
            'metabolite_ids': self.metabolite_ids,
            'medium_indices': self.medium_indices,
            'medium_ids': self.medium_ids,
            'objective_idx': self.objective_idx,
            'objective_id': self.objective_id,
        }
    
    def save(self, path: Path | str) -> None:
        """Save to compressed npz file."""
        path = Path(path)
        np.savez_compressed(
            path,
            **self.to_dict()
        )
        print(f"Saved MetabolicModel to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> 'MetabolicModel':
        """Load from npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        return cls(
            S=data['S'],
            c=data['c'],
            lb=data['lb'],
            ub=data['ub'],
            reaction_ids=list(data['reaction_ids']),
            metabolite_ids=list(data['metabolite_ids']),
            medium_indices=data['medium_indices'],
            medium_ids=list(data['medium_ids']),
            objective_idx=int(data['objective_idx']),
            objective_id=str(data['objective_id']),
        )


@dataclass
class ExperimentalData:
    """
    Container for experimental media compositions and growth rates.
    
    Attributes:
        X: Media compositions (n_samples, n_medium)
        Y: Growth rates (n_samples,)
        medium_ids: IDs of medium compounds (must match model)
        condition_ids: Optional condition identifiers
        feature_names: Names of media features
    """
    X: np.ndarray
    Y: np.ndarray
    medium_ids: list[str]
    condition_ids: Optional[list[str]] = None
    feature_names: Optional[list[str]] = None
    
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1]
    
    def __post_init__(self):
        """Validate data shapes."""
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of samples. "
                f"Got X: {self.X.shape[0]}, Y: {self.Y.shape[0]}"
            )
        if self.X.shape[1] != len(self.medium_ids):
            raise ValueError(
                f"X columns must match medium_ids. "
                f"Got X: {self.X.shape[1]}, medium_ids: {len(self.medium_ids)}"
            )
    
    def train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: Optional[int] = None
    ) -> tuple['ExperimentalData', 'ExperimentalData']:
        """Split into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(self.n_samples)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        train_data = ExperimentalData(
            X=self.X[train_idx],
            Y=self.Y[train_idx],
            medium_ids=self.medium_ids,
            condition_ids=[self.condition_ids[i] for i in train_idx] if self.condition_ids else None,
            feature_names=self.feature_names,
        )
        test_data = ExperimentalData(
            X=self.X[test_idx],
            Y=self.Y[test_idx],
            medium_ids=self.medium_ids,
            condition_ids=[self.condition_ids[i] for i in test_idx] if self.condition_ids else None,
            feature_names=self.feature_names,
        )
        return train_data, test_data


# =============================================================================
# Model Processing Functions
# =============================================================================

def get_reaction_index(model: cobra.Model, reaction_id: str) -> int:
    """Get the index of a reaction in the model."""
    for i, rxn in enumerate(model.reactions):
        if rxn.id == reaction_id:
            return i
    raise ValueError(f"Reaction {reaction_id} not found in model")


def get_objective_reaction(model: cobra.Model) -> tuple[str, int]:
    """
    Extract the objective reaction ID and index from a COBRA model.
    
    Returns:
        Tuple of (objective_id, objective_index)
    """
    obj_expr = str(model.objective.expression)
    # Parse "1.0*BIOMASS_... - 1.0*BIOMASS_..._reverse_xxxxx"
    parts = obj_expr.split()
    obj_id = parts[0].split('*')[1] if '*' in parts[0] else parts[0]
    obj_idx = get_reaction_index(model, obj_id)
    return obj_id, obj_idx


def extract_model_matrices(
    model: cobra.Model,
    medium_ids: Sequence[str],
) -> MetabolicModel:
    """
    Extract matrices and vectors from a COBRA model for differentiable FBA.
    
    Args:
        model: COBRA model (should be irreversible/duplicated)
        medium_ids: List of exchange reaction IDs that are in the medium
        
    Returns:
        MetabolicModel containing all necessary matrices
    """
    # Stoichiometric matrix
    S = cobra.util.array.create_stoichiometric_matrix(model)
    
    # Reaction and metabolite IDs
    reaction_ids = [r.id for r in model.reactions]
    metabolite_ids = [m.id for m in model.metabolites]
    
    # Bounds
    lb = np.array([r.lower_bound for r in model.reactions], dtype=np.float32)
    ub = np.array([r.upper_bound for r in model.reactions], dtype=np.float32)
    
    # Objective vector
    c = np.zeros(len(reaction_ids), dtype=np.float32)
    obj_id, obj_idx = get_objective_reaction(model)
    c[obj_idx] = 1.0  # Maximize biomass
    
    # Medium indices - map medium_ids to reaction indices
    medium_indices = []
    valid_medium_ids = []
    for mid in medium_ids:
        try:
            idx = get_reaction_index(model, mid)
            medium_indices.append(idx)
            valid_medium_ids.append(mid)
        except ValueError:
            warnings.warn(f"Medium reaction {mid} not found in model, skipping")
    
    medium_indices = np.array(medium_indices, dtype=np.int64)
    
    return MetabolicModel(
        S=S.astype(np.float32),
        c=c,
        lb=lb,
        ub=ub,
        reaction_ids=reaction_ids,
        metabolite_ids=metabolite_ids,
        medium_indices=medium_indices,
        medium_ids=valid_medium_ids,
        objective_idx=obj_idx,
        objective_id=obj_id,
    )


def duplicate_reversible_reactions(
    model: cobra.Model,
    io_dict: Optional[dict] = None,
    unsignificant_mols: Optional[list[str]] = None,
) -> cobra.Model:
    """
    Duplicate reversible reactions to make all fluxes positive.
    
    This is a simplified version of the Duplicate_Model functions.
    
    Args:
        model: Original COBRA model
        io_dict: Dictionary mapping suffixes to compartment transfers
        unsignificant_mols: Molecules to ignore in screening
        
    Returns:
        New model with duplicated reactions
    """
    from src.Duplicate_Model import screen_out_in, duplicate_model, correct_duplicated_med
    
    if io_dict is None:
        io_dict = {
            "_i": [(None, "e"), (None, "c"), ("e", "p"), ("p", "c"), ("e", "c"), ("c", "m"), ("p", "m")],
            "_o": [("c", None), ("e", None), ("p", "e"), ("c", "p"), ("c", "e"), ("m", "c"), ("m", "p")]
        }
    
    if unsignificant_mols is None:
        unsignificant_mols = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]
    
    # Screen reactions
    reac_dict = screen_out_in(model, io_dict, unsignificant_mols)
    
    # Duplicate model
    new_model = duplicate_model(model, reac_dict)
    
    # Correct medium
    default_med = model.medium
    new_med = new_model.medium
    correct_med = correct_duplicated_med(default_med, new_med)
    new_model.medium = correct_med
    
    # Repair model
    new_model.repair()
    
    return new_model


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_experimental_data(
    csv_path: Path | str,
    growth_column: str = 'GR_AVG',
    exclude_columns: Optional[list[str]] = None,
    min_growth: float = 0.0,
) -> ExperimentalData:
    """
    Load experimental media compositions and growth rates from CSV.
        
    Args:
        csv_path: Path to CSV file
        growth_column: Name of the column containing growth rates
        exclude_columns: Additional columns to exclude (e.g., 'Condition ID')
        min_growth: Minimum growth rate to include (filter out zero-growth)
        
    Returns:
        ExperimentalData object
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Identify columns to exclude
    exclude = {growth_column}
    if exclude_columns:
        exclude.update(exclude_columns)
    
    # Extract medium columns
    medium_cols = [c for c in df.columns if c not in exclude]
    
    # Extract data
    X = df[medium_cols].values.astype(np.float32)
    Y = df[growth_column].values.astype(np.float32)
    
    # Filter by minimum growth
    if min_growth > 0:
        mask = Y >= min_growth
        X = X[mask]
        Y = Y[mask]
        print(f"Filtered to {mask.sum()} samples with growth >= {min_growth}")
    
    # Handle condition IDs if present
    condition_ids = None
    if 'Condition ID' in df.columns:
        condition_ids = df['Condition ID'].tolist()
        if min_growth > 0:
            condition_ids = [c for c, m in zip(condition_ids, mask) if m]
    
    return ExperimentalData(
        X=X,
        Y=Y,
        medium_ids=medium_cols,
        condition_ids=condition_ids,
        feature_names=medium_cols,
    )


def normalize_features(
    data: ExperimentalData,
    method: str = 'max',
    reference_max: Optional[np.ndarray] = None,
) -> tuple[ExperimentalData, np.ndarray]:
    """
    Normalize media composition features.
    
    Args:
        data: ExperimentalData to normalize
        method: 'max' (divide by max), 'minmax', or 'zscore'
        reference_max: Pre-computed max values (for test set normalization)
        
    Returns:
        Tuple of (normalized ExperimentalData, normalization parameters)
    """
    X = data.X.copy()
    
    if method == 'max':
        if reference_max is None:
            reference_max = np.max(X, axis=0)
            reference_max[reference_max == 0] = 1.0  # Avoid division by zero
        X_norm = X / reference_max
        params = reference_max
        
    elif method == 'minmax':
        if reference_max is None:
            x_min = np.min(X, axis=0)
            x_max = np.max(X, axis=0)
            x_range = x_max - x_min
            x_range[x_range == 0] = 1.0
            reference_max = np.stack([x_min, x_range])
        x_min, x_range = reference_max[0], reference_max[1]
        X_norm = (X - x_min) / x_range
        params = reference_max
        
    elif method == 'zscore':
        if reference_max is None:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0)
            x_std[x_std == 0] = 1.0
            reference_max = np.stack([x_mean, x_std])
        x_mean, x_std = reference_max[0], reference_max[1]
        X_norm = (X - x_mean) / x_std
        params = reference_max
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_data = ExperimentalData(
        X=X_norm.astype(np.float32),
        Y=data.Y,
        medium_ids=data.medium_ids,
        condition_ids=data.condition_ids,
        feature_names=data.feature_names,
    )
    
    return normalized_data, params


# =============================================================================
# PyTorch Dataset
# =============================================================================

class DiffFBADataset(TorchDataset):
    """
    PyTorch Dataset for differentiable FBA models.
    
    This dataset provides:
        - Media compositions (X) as input
        - Growth rates (Y) as targets
        - Access to metabolic model matrices for the FBA layer
    """
    
    def __init__(
        self,
        experimental_data: ExperimentalData,
        metabolic_model: MetabolicModel,
        normalize: bool = True,
        normalization_method: str = 'max',
        normalization_params: Optional[np.ndarray] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            experimental_data: ExperimentalData object
            metabolic_model: MetabolicModel object
            normalize: Whether to normalize input features
            normalization_method: 'max', 'minmax', or 'zscore'
            normalization_params: Pre-computed normalization params (for test set)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DiffFBADataset")
        
        self.model = metabolic_model
        self.normalization_params = normalization_params
        
        # Validate medium alignment
        self._validate_medium_alignment(experimental_data, metabolic_model)
        
        # Normalize if requested
        if normalize:
            self.data, self.normalization_params = normalize_features(
                experimental_data, 
                method=normalization_method,
                reference_max=normalization_params,
            )
        else:
            self.data = experimental_data
        
        # Convert to tensors
        self.X = torch.tensor(self.data.X, dtype=torch.float32)
        self.Y = torch.tensor(self.data.Y, dtype=torch.float32)
    
    def _validate_medium_alignment(
        self, 
        data: ExperimentalData, 
        model: MetabolicModel
    ) -> None:
        """Check that experimental medium IDs match model medium IDs."""
        data_ids = set(data.medium_ids)
        model_ids = set(model.medium_ids)
        
        if data_ids != model_ids:
            missing_in_model = data_ids - model_ids
            missing_in_data = model_ids - data_ids
            
            if missing_in_model:
                warnings.warn(
                    f"Medium IDs in data but not in model: {missing_in_model}"
                )
            if missing_in_data:
                warnings.warn(
                    f"Medium IDs in model but not in data: {missing_in_data}"
                )
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
    
    @classmethod
    def from_files(
        cls,
        model_path: Path | str,
        experimental_csv: Path | str,
        growth_column: str = 'GR_AVG',
        duplicate_model: bool = True,
        min_growth: float = 0.0,
        normalize: bool = True,
        **kwargs,
    ) -> 'DiffFBADataset':
        """
        Create dataset directly from files.
        
        Args:
            model_path: Path to SBML model file
            experimental_csv: Path to experimental data CSV
            growth_column: Column name for growth rates
            duplicate_model: Whether to duplicate reversible reactions
            min_growth: Minimum growth rate filter
            normalize: Whether to normalize features
            **kwargs: Additional arguments for DiffFBADataset
            
        Returns:
            DiffFBADataset ready for training
        """
        model_path = Path(model_path)
        experimental_csv = Path(experimental_csv)
        
        # Load experimental data first to get medium IDs
        print(f"Loading experimental data from {experimental_csv}")
        exp_data = load_experimental_data(
            experimental_csv,
            growth_column=growth_column,
            min_growth=min_growth,
        )
        print(f"  Loaded {exp_data.n_samples} samples with {exp_data.n_features} features")
        
        # Load and process model
        print(f"Loading COBRA model from {model_path}")
        cobra_model = cobra.io.read_sbml_model(str(model_path))
        
        if duplicate_model:
            print("  Duplicating reversible reactions...")
            cobra_model = duplicate_reversible_reactions(cobra_model)
        
        # Extract matrices
        print("  Extracting model matrices...")
        metabolic_model = extract_model_matrices(cobra_model, exp_data.medium_ids)
        print(f"  Model: {metabolic_model.n_reactions} reactions, "
              f"{metabolic_model.n_metabolites} metabolites, "
              f"{metabolic_model.n_medium} medium compounds")
        
        return cls(
            experimental_data=exp_data,
            metabolic_model=metabolic_model,
            normalize=normalize,
            **kwargs,
        )
    
    def get_model_tensors(self) -> dict[str, torch.Tensor]:
        """
        Get metabolic model matrices as PyTorch tensors.
        
        Returns:
            Dictionary with 'S', 'c', 'lb', 'ub', 'medium_indices'
        """
        return {
            'S': torch.tensor(self.model.S, dtype=torch.float32),
            'c': torch.tensor(self.model.c, dtype=torch.float32),
            'lb': torch.tensor(self.model.lb, dtype=torch.float32),
            'ub': torch.tensor(self.model.ub, dtype=torch.float32),
            'medium_indices': torch.tensor(self.model.medium_indices, dtype=torch.long),
            'objective_idx': self.model.objective_idx,
        }
    
    def save(self, path: Path | str) -> None:
        """Save dataset to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path / 'model.npz')
        
        # Save data
        np.savez_compressed(
            path / 'data.npz',
            X=self.data.X,
            Y=self.data.Y,
            medium_ids=self.data.medium_ids,
            normalization_params=self.normalization_params,
        )
        
        print(f"Saved dataset to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> 'DiffFBADataset':
        """Load dataset from directory."""
        path = Path(path)
        
        # Load model
        model = MetabolicModel.load(path / 'model.npz')
        
        # Load data
        data_dict = np.load(path / 'data.npz', allow_pickle=True)
        exp_data = ExperimentalData(
            X=data_dict['X'],
            Y=data_dict['Y'],
            medium_ids=list(data_dict['medium_ids']),
        )
        
        return cls(
            experimental_data=exp_data,
            metabolic_model=model,
            normalize=False,  # Already normalized
            normalization_params=data_dict['normalization_params'],
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for dataset creation."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Create dataset for differentiable FBA models"
    )
    parser.add_argument(
        '--config', 
        type=Path,
        help="Path to YAML config file"
    )
    parser.add_argument(
        '--model', 
        type=Path,
        help="Path to SBML model file"
    )
    parser.add_argument(
        '--experimental', 
        type=Path,
        help="Path to experimental CSV file"
    )
    parser.add_argument(
        '--output', 
        type=Path,
        default=Path('data/processed/difffba_dataset'),
        help="Output directory"
    )
    parser.add_argument(
        '--growth-column',
        type=str,
        default='GR_AVG',
        help="Column name for growth rates"
    )
    parser.add_argument(
        '--no-duplicate',
        action='store_true',
        help="Skip model duplication"
    )
    parser.add_argument(
        '--min-growth',
        type=float,
        default=0.0,
        help="Minimum growth rate filter"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model_path = Path(config.get('model_path', args.model))
        experimental_csv = Path(config.get('experimental_csv', args.experimental))
        output_dir = Path(config.get('output_dir', args.output))
        growth_column = config.get('growth_column', args.growth_column)
        duplicate = config.get('duplicate_model', not args.no_duplicate)
        min_growth = config.get('min_growth', args.min_growth)
    else:
        if not args.model or not args.experimental:
            parser.error("Either --config or both --model and --experimental required")
        model_path = args.model
        experimental_csv = args.experimental
        output_dir = args.output
        growth_column = args.growth_column
        duplicate = not args.no_duplicate
        min_growth = args.min_growth
    
    # Create dataset
    print("=" * 60)
    print("Creating Differentiable FBA Dataset")
    print("=" * 60)
    
    dataset = DiffFBADataset.from_files(
        model_path=model_path,
        experimental_csv=experimental_csv,
        growth_column=growth_column,
        duplicate_model=duplicate,
        min_growth=min_growth,
    )
    
    # Save
    dataset.save(output_dir)
    
    print("=" * 60)
    print("Dataset Statistics:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Features: {dataset.data.n_features}")
    print(f"  Growth range: [{dataset.Y.min():.4f}, {dataset.Y.max():.4f}]")
    print(f"  Model reactions: {dataset.model.n_reactions}")
    print(f"  Model metabolites: {dataset.model.n_metabolites}")
    print("=" * 60)


if __name__ == "__main__":
    main()
