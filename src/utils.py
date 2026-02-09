###############################################################################
# DATASET UTILS
###############################################################################

def analyze_experimental_compounds(exp_df):
    """
    Analyze experimental compounds and return statistics for each.
    
    Parameters
    ----------
    exp_df : pd.DataFrame
        DataFrame with compounds as columns (last column assumed to be GR_AVG)
        
    Returns
    -------
    dict
        Dictionary where keys are compound names and values are stat dictionaries
    """
    compounds = exp_df.columns[:-1]  # assuming last col is GR_AVG
    
    # Initialize results dictionary
    exp_compounds_dict = {}
    
    for col in compounds:
        series = exp_df[col].dropna()
        
        # Basic Stats
        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()
        std_val = series.std()
        zero_count = int((series == 0).sum())
        distinct_count = int(series.nunique())
        total_count = len(series)
        
        # Check if it's a constant compound
        if mean_val > 0:
            is_constant = (std_val / mean_val < 0.1)
        else:
            is_constant = (max_val == min_val)

        # Histogram
        counts, bin_edges = np.histogram(series, bins=5)
        
        # Classify distribution type
        if (counts.max() / total_count) >= 0.90:
            distribution_type = 'skewed'
        else:
            distribution_type = 'uniform'
        
        # Build histogram bins info
        histogram_bins = []
        for i in range(len(counts)):
            histogram_bins.append({
                'lower': float(bin_edges[i]),
                'upper': float(bin_edges[i+1]),
                'count': int(counts[i]),
                'percentage': float((counts[i] / total_count) * 100)
            })
        
        # Store stats in dictionary
        exp_compounds_dict[col] = {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val),
            'zero_count': zero_count,
            'distinct_count': distinct_count,
            'total_count': total_count,
            'is_constant': bool(is_constant),
            'distribution_type': distribution_type,
            'histogram_bins': histogram_bins
        }
    
    return exp_compounds_dict


def create_parameter_file(exp_compounds_dict, 
                          output_path,
                          essential_compounds=None,
                          new_compounds=None):
    """
    Create a parameter CSV file from experimental compound statistics.
    
    Parameters
    ----------
    exp_compounds_dict : dict
        Nested dictionary where keys are compound names and values are stat dictionaries
    essential_compounds : list, optional
        List of compound IDs that should always have level=1
    new_compounds : dict, optional
        Dictionary of additional compounds to add that are not in the experimental data.
        Format: {compound_id: {'level': int, 'max_value': float}}
    output_path : str
        Path to save the output CSV file
        
    Returns
    -------
    pd.DataFrame
        The generated parameter DataFrame
    """
    if essential_compounds is None:
        essential_compounds = []
    if new_compounds is None:
        new_compounds = {}
    
    compounds = list(exp_compounds_dict.keys())
    
    # Build level values based on rules:
    # - 1 if in essential_compounds list (always ON)
    # - 1 if constant
    # - 1 if skewed (not constant)
    # - 100 otherwise (uniform)
    levels = []
    for comp in compounds:
        stats = exp_compounds_dict[comp]
        if comp in essential_compounds:
            levels.append(1)  # Essential compounds are always ON
        elif stats['is_constant']:
            levels.append(1)
        elif stats['distribution_type'] == 'skewed':
            levels.append(1)
        else:
            levels.append(100)
    
    # Build max_value row
    max_values = [exp_compounds_dict[comp]['max'] for comp in compounds]
    
    # Build ratio_drawing row (0 for first column, as it's an experimental medium)
    ratio_drawing = [0] + [''] * (len(compounds) - 1)
    
    # Build comment row
    comment_text = 'parameters generated from experimental medium distribution'
    comment = [comment_text] + [''] * (len(compounds) - 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'name': ['level', 'max_value', 'ratio_drawing', 'comment']
    })
    
    # Add compound columns from experimental data
    for i, comp in enumerate(compounds):
        df[comp] = [levels[i], max_values[i], ratio_drawing[i], comment[i]]
    
    # Add new compounds not in experimental data
    for comp, params in new_compounds.items():
        level = params.get('level', 1)
        max_value = params.get('max_value', 1.0)
        df[comp] = [level, max_value, '', '']
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

def print_compounds_summary(exp_compounds_dict):
    """
    Print details for each analysed compound
    """
    # Per compound details
    for compound, s in exp_compounds_dict.items():
        print(f"\n{'='*40}")
        print(f"Compound: {compound}")
        print(f"{'='*40}")
        print(f"Stats:")
        print(f"  min={s['min']:.4f}, max={s['max']:.4f}")
        print(f"  distinct={s['distinct_count']}, zeros={s['zero_count']}")
        print(f"  constant={s['is_constant']}")
        print(f"  distribution_type={s['distribution_type']}")
        print(f"Distribution:")
        for b in s['histogram_bins']:
            print(f"  [{b['lower']:.2f} - {b['upper']:.2f}): {b['count']} ({b['percentage']:.1f}%)")

    # Summary
    constant = [c for c, s in exp_compounds_dict.items() if s['is_constant']]
    skewed = [c for c, s in exp_compounds_dict.items() if not s['is_constant'] and s['distribution_type'] == 'skewed']
    uniform = [c for c, s in exp_compounds_dict.items() if not s['is_constant'] and s['distribution_type'] == 'uniform']

    print(f"\nMapped compounds: {len(exp_compounds_dict)}")
    print(f"Constant compounds: {len(constant)}")
    print(f"Skewed compounds: {len(skewed)}")
    print(f"Uniform compounds: {len(uniform)}")