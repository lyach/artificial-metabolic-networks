import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
