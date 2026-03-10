import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

FONT_SIZE = 15

def plot_learning_curve(result_folder):
    """
    Plot learning and validation curves across epochs,
    averaging over 5 folds.
    
    Args:
        result_folder (str): Path to the result folder

    Returns:
        fig: matplotlib figure object
    """
    result_path = Path(result_folder)
    
    # Check if path exists
    if not result_path.exists():
        raise FileNotFoundError(f"Result folder not found: {result_folder}")
    
    # Read all fold history files
    fold_data = []
    history_files = sorted(result_path.glob('*history.csv'))
    
    print(f"Found {len(history_files)} history files.")
    for fold_file in history_files:
        df = pd.read_csv(fold_file)
        fold_data.append(df)
    
    # Stack all folds and compute statistics
    # Handle variable-length histories by truncating to minimum length
    n_folds = len(fold_data)
    fold_lengths = [len(df) for df in fold_data]
    n_epochs = min(fold_lengths)
    
    if len(set(fold_lengths)) > 1:
        print(f"Using first {n_epochs} epochs.")
    
    # Extract metrics for each fold, truncating to n_epochs
    loss_data = np.array([df['loss'].values[:n_epochs] for df in fold_data])
    my_r2_data = np.array([df['my_r2'].values[:n_epochs] for df in fold_data])
    val_loss_data = np.array([df['val_loss'].values[:n_epochs] for df in fold_data])
    val_my_r2_data = np.array([df['val_my_r2'].values[:n_epochs] for df in fold_data])
    
    # Calculate mean and std across folds
    loss_mean = np.mean(loss_data, axis=0)
    loss_std = np.std(loss_data, axis=0)
    my_r2_mean = np.mean(my_r2_data, axis=0)
    my_r2_std = np.std(my_r2_data, axis=0)
    val_loss_mean = np.mean(val_loss_data, axis=0)
    val_loss_std = np.std(val_loss_data, axis=0)
    val_my_r2_mean = np.mean(val_my_r2_data, axis=0)
    val_my_r2_std = np.std(val_my_r2_data, axis=0)
    
    epochs = np.arange(n_epochs)
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Training curves
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Plot training loss
    line1 = ax1.plot(epochs, loss_mean, 'b-', label='Training Loss', linewidth=2)
    ax1.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, 
                      color='b', alpha=0.2)
    
    # Plot training R2
    line2 = ax1_twin.plot(epochs, my_r2_mean, 'r-', label='Training R²', linewidth=2)
    ax1_twin.fill_between(epochs, my_r2_mean - my_r2_std, my_r2_mean + my_r2_std,
                           color='r', alpha=0.2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax1_twin.set_ylabel('R²', color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Panel 2: Validation curves
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Plot validation loss
    line3 = ax2.plot(epochs, val_loss_mean, 'b-', label='Validation Loss', linewidth=2)
    ax2.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std,
                      color='b', alpha=0.2)
    
    # Plot validation R2
    line4 = ax2_twin.plot(epochs, val_my_r2_mean, 'r-', label='Validation R²', linewidth=2)
    ax2_twin.fill_between(epochs, val_my_r2_mean - val_my_r2_std, val_my_r2_mean + val_my_r2_std,
                           color='r', alpha=0.2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', color='b', fontsize=12)
    ax2_twin.set_ylabel('R²', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Validation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    
    return fig


def plot_r2_constraint(result_file, font_size=FONT_SIZE):
    """
    Plot R2 and constraint loss across folds.
    Reads a CSV with columns [r2_train, r2_test, constraint_train, constraint_test]
    and creates a bar chart with stacked x-axis labels and dual y-axes.

    Args:
        result_file (str): Path to the result csv
        font_size (int): Font size for labels and ticks. Defaults to FONT_SIZE.
    Returns:
        fig: matplotlib figure object
    """
    # Read 
    df = pd.read_csv(result_file)
    # Ensure required columns exist
    required = ['r2_train', 'r2_test', 'constraint_train', 'constraint_test']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'")
    
    # Compute mean and std across folds (rows)
    r2_train_mean = df['r2_train'].mean()
    r2_train_std = df['r2_train'].std()
    r2_test_mean = df['r2_test'].mean()
    r2_test_std = df['r2_test'].std()
    constraint_train_mean = df['constraint_train'].mean()
    constraint_train_std = df['constraint_train'].std()
    constraint_test_mean = df['constraint_test'].mean()
    constraint_test_std = df['constraint_test'].std()
    
    # Get seaborn color palette
    colors = sns.color_palette(palette='PRGn', n_colors=2)
    r2_color = colors[0]
    constraint_color = colors[1]
    
    # Bar positions: [Train-R2, Train-Constraint, Test-R2, Test-Constraint]
    # Use positions 0, 1, 3, 4 to create a gap between Train and Test groups
    width = 0.6
    
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    
    # R2 bars (left y-axis) at positions 0 and 3
    r2_x = [0, 3]
    r2_means = [r2_train_mean, r2_test_mean]
    r2_stds = [r2_train_std, r2_test_std]
    ax1.bar(r2_x, r2_means, width, yerr=r2_stds, capsize=4,
            color=r2_color, label='R²', error_kw={'ecolor': 'gray'})
    
    # Constraint bars (right y-axis) at positions 1 and 4
    constraint_x = [1, 4]
    constraint_means = [constraint_train_mean, constraint_test_mean]
    constraint_stds = [constraint_train_std, constraint_test_std]
    ax2.bar(constraint_x, constraint_means, width, yerr=constraint_stds, capsize=4,
            color=constraint_color, label='Constraint Loss', error_kw={'ecolor': 'gray'})
    
    ax1.set_ylabel('R²', fontsize=font_size)
    ax2.set_ylabel('Constraint Loss', fontsize=font_size)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)
    
    # Stacked x-axis labels: second level (metric type)
    x_positions = [0, 1, 3, 4]
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(['R²', 'Constraint Loss', 'R²', 'Constraint Loss'], fontsize=font_size)
    ax1.set_xlim(-1, 5)
    
    # First-level labels (Train / Test) below the second-level labels
    fig.subplots_adjust(bottom=0.18)
    # Use blended transform: x in data coords, y in axes fraction
    trans = ax1.get_xaxis_transform()
    ax1.text(0.5, -0.12, 'Train', transform=trans, ha='center', fontsize=font_size, fontweight='bold')
    ax1.text(3.5, -0.12, 'Test', transform=trans, ha='center', fontsize=font_size, fontweight='bold')
    
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, None)
    ax1.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_r2(result_files, font_size=FONT_SIZE):
    """
    Plot R2 and Q2 bars aggregated from multiple result files.

    Args:
        result_files (list): List of paths to result csvs
        font_size (int): Font size for labels and ticks. Defaults to FONT_SIZE.
    Returns:
        fig: matplotlib figure object
    """
    # Collect all r2_train and r2_test values from all files
    all_r2_train = []
    all_r2_test = []
    
    for result_file in result_files:
        df = pd.read_csv(result_file)
        # Ensure required columns exist
        required = ['r2_train', 'r2_test']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'")
        
        # Collect all values (across all folds in this file)
        all_r2_train.extend(df['r2_train'].values)
        all_r2_test.extend(df['r2_test'].values)
    
    # Convert to numpy arrays and compute overall mean and std
    all_r2_train = np.array(all_r2_train)
    all_r2_test = np.array(all_r2_test)
    
    r2_train_mean = all_r2_train.mean()
    r2_train_std = all_r2_train.std()
    r2_test_mean = all_r2_test.mean()
    r2_test_std = all_r2_test.std()
    
    # Get seaborn color palette
    colors = sns.color_palette(palette='PRGn', n_colors=2)
    r2_color = colors[0]
    q2_color = colors[1]
    
    # Bar positions
    width = 0.6
    x_positions = [0, 1]
    means = [r2_train_mean, r2_test_mean]
    stds = [r2_train_std, r2_test_std]
    bar_colors = [r2_color, q2_color]
    
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    
    # Two bars at positions 0 and 1
    ax1.bar(x_positions, means, width, yerr=stds, capsize=4,
            color=bar_colors, error_kw={'ecolor': 'gray'})
    
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Single x-axis labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(['R²', 'Q²'], fontsize=font_size)
    ax1.set_xlim(-0.5, 1.5)
    
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig