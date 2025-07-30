#!/usr/bin/env python3
"""
Visualization script for BCR-Net scattering inverse problem results.

This script loads the saved predictions from a trained model and creates
visualizations comparing inputs, predictions, and ground truth outputs.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

try:
    import cmocean
    has_cmocean = True
except ImportError:
    print("Warning: cmocean not found. Using matplotlib default colormaps.")
    has_cmocean = False

def load_results(data_file):
    """Load input, output, and prediction data from HDF5 file."""
    with h5py.File(data_file, 'r') as f:
        # Try to load unpadded versions first (original size)
        if 'input' in f and 'output' in f and 'pred' in f:
            inputs = f['input'][:]
            outputs = f['output'][:]
            preds = f['pred'][:]
            print(f"Loaded unpadded data (original size)")
            
            # Check if padded versions also exist
            if 'input_padded' in f:
                padded_size = f['input_padded'].shape[1:]
                original_size = inputs.shape[1:]
                print(f"Original size: {original_size}, Padded size: {padded_size}")
        else:
            # Fallback to older format or padded versions
            if 'input_padded' in f:
                inputs = f['input_padded'][:]
                outputs = f['output_padded'][:]
                preds = f['pred_padded'][:]
                print(f"Loaded padded data")
            else:
                raise KeyError("No valid input/output/pred datasets found in file")
    return inputs, outputs, preds

def setup_colormaps():
    """Set up appropriate colormaps."""
    if has_cmocean:
        # Use scientific colormaps from cmocean
        input_cmap = cmocean.cm.phase    # Good for phase/complex data
        output_cmap = cmocean.cm.matter  # Good for physical quantities
        diff_cmap = cmocean.cm.balance   # Good for differences (centered at 0)
    else:
        # Fallback to matplotlib colormaps
        input_cmap = 'viridis'
        output_cmap = 'plasma'
        diff_cmap = 'RdBu_r'
    
    return input_cmap, output_cmap, diff_cmap

def calculate_metrics(true, pred):
    """Calculate evaluation metrics."""
    mse = np.mean((true - pred) ** 2)
    mae = np.mean(np.abs(true - pred))
    
    # Calculate PSNR
    pixel_max = np.max(true) - np.min(true)
    psnr = -10 * np.log10(mse / (pixel_max ** 2)) if mse > 0 else float('inf')
    
    # Calculate relative error
    rel_error = np.linalg.norm(true - pred) / np.linalg.norm(true) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'PSNR': psnr,
        'Relative Error (%)': rel_error
    }

def plot_comparison(inputs, outputs, preds, sample_idx, save_path=None):
    """Create a comparison plot for a single sample."""
    input_cmap, output_cmap, diff_cmap = setup_colormaps()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input (scattering data)
    im1 = axes[0, 0].imshow(inputs[sample_idx], cmap=input_cmap, aspect='equal')
    axes[0, 0].set_title(f'Input (Scattering Data)\nSample {sample_idx}')
    axes[0, 0].set_xlabel('Detector Position')
    axes[0, 0].set_ylabel('Source Position')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Ground Truth
    im2 = axes[0, 1].imshow(outputs[sample_idx], cmap=output_cmap, aspect='equal')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Prediction
    im3 = axes[0, 2].imshow(preds[sample_idx], cmap=output_cmap, aspect='equal',
                           vmin=np.min(outputs[sample_idx]), vmax=np.max(outputs[sample_idx]))
    axes[0, 2].set_title('Prediction')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Difference map
    diff = preds[sample_idx] - outputs[sample_idx]
    max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
    im4 = axes[1, 0].imshow(diff, cmap=diff_cmap, aspect='equal', 
                           vmin=-max_diff, vmax=max_diff)
    axes[1, 0].set_title('Difference (Pred - True)')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    # Cross-section comparison (middle row)
    mid_row = outputs.shape[1] // 2
    axes[1, 1].plot(outputs[sample_idx, mid_row, :], 'b-', label='Ground Truth', linewidth=2)
    axes[1, 1].plot(preds[sample_idx, mid_row, :], 'r--', label='Prediction', linewidth=2)
    axes[1, 1].set_title(f'Cross-section (Row {mid_row})')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Metrics text
    metrics = calculate_metrics(outputs[sample_idx], preds[sample_idx])
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Metrics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig

def plot_overview(inputs, outputs, preds, num_samples=6, save_path=None):
    """Create an overview plot showing multiple samples."""
    input_cmap, output_cmap, diff_cmap = setup_colormaps()
    
    fig, axes = plt.subplots(4, num_samples, figsize=(3*num_samples, 12))
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(outputs[:num_samples], preds[:num_samples])
    
    for i in range(num_samples):
        # Input
        axes[0, i].imshow(inputs[i], cmap=input_cmap, aspect='equal')
        axes[0, i].set_title(f'Input {i}')
        axes[0, i].axis('off')
        
        # Ground Truth
        axes[1, i].imshow(outputs[i], cmap=output_cmap, aspect='equal')
        axes[1, i].set_title(f'True {i}')
        axes[1, i].axis('off')
        
        # Prediction
        axes[2, i].imshow(preds[i], cmap=output_cmap, aspect='equal',
                         vmin=np.min(outputs[i]), vmax=np.max(outputs[i]))
        axes[2, i].set_title(f'Pred {i}')
        axes[2, i].axis('off')
        
        # Difference
        diff = preds[i] - outputs[i]
        max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
        axes[3, i].imshow(diff, cmap=diff_cmap, aspect='equal', 
                         vmin=-max_diff, vmax=max_diff)
        axes[3, i].set_title(f'Diff {i}')
        axes[3, i].axis('off')
    
    # Add row labels
    row_labels = ['Scattering Data', 'Ground Truth', 'Prediction', 'Difference']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, size='large', labelpad=20)
    
    # Add overall metrics as title
    metrics_str = f"Overall - PSNR: {overall_metrics['PSNR']:.2f} dB, " + \
                 f"Rel. Error: {overall_metrics['Relative Error (%)']:.2f}%"
    fig.suptitle(metrics_str, fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview plot to: {save_path}")
    
    return fig

def plot_statistics(inputs, outputs, preds, save_path=None):
    """Plot statistical analysis of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error distribution
    errors = (preds - outputs).flatten()
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].set_xlabel('Prediction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(errors):.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR per sample
    psnr_values = []
    for i in range(len(outputs)):
        metrics = calculate_metrics(outputs[i], preds[i])
        psnr_values.append(metrics['PSNR'])
    
    axes[0, 1].plot(psnr_values, 'bo-', markersize=4)
    axes[0, 1].set_title(f'PSNR per Sample (Mean: {np.mean(psnr_values):.2f} dB)')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: True vs Predicted
    true_flat = outputs.flatten()
    pred_flat = preds.flatten()
    
    # Sample for visualization (too many points otherwise)
    n_sample = min(10000, len(true_flat))
    idx = np.random.choice(len(true_flat), n_sample, replace=False)
    
    axes[1, 0].scatter(true_flat[idx], pred_flat[idx], alpha=0.5, s=1)
    axes[1, 0].plot([true_flat.min(), true_flat.max()], 
                   [true_flat.min(), true_flat.max()], 'r--', lw=2)
    axes[1, 0].set_title('True vs Predicted Values')
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value distributions
    axes[1, 1].hist(true_flat, bins=50, alpha=0.5, label='True', density=True)
    axes[1, 1].hist(pred_flat, bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1, 1].set_title('Value Distributions')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistics plot to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize BCR-Net results')
    parser.add_argument('data_file', type=str, help='Path to HDF5 file with results')
    parser.add_argument('--sample', type=int, default=0, 
                       help='Sample index for detailed view (default: 0)')
    parser.add_argument('--overview-samples', type=int, default=6,
                       help='Number of samples for overview (default: 6)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for plots (default: visualizations)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format (default: png)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.data_file}")
    inputs, outputs, preds = load_results(args.data_file)
    
    print(f"Loaded {len(inputs)} samples")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Prediction shape: {preds.shape}")
    
    # Generate base filename for plots
    base_name = Path(args.data_file).stem
    
    # Create detailed comparison plot
    if args.sample < len(outputs):
        detail_path = os.path.join(args.output_dir, f"{base_name}_sample_{args.sample}.{args.format}")
        fig1 = plot_comparison(inputs, outputs, preds, args.sample, detail_path)
        
        if args.show:
            plt.show()
        else:
            plt.close(fig1)
    else:
        print(f"Warning: Sample {args.sample} not available (max: {len(outputs)-1})")
    
    # Create overview plot
    n_overview = min(args.overview_samples, len(outputs))
    overview_path = os.path.join(args.output_dir, f"{base_name}_overview.{args.format}")
    fig2 = plot_overview(inputs, outputs, preds, n_overview, overview_path)
    
    if args.show:
        plt.show()
    else:
        plt.close(fig2)
    
    # Create statistics plot
    stats_path = os.path.join(args.output_dir, f"{base_name}_statistics.{args.format}")
    fig3 = plot_statistics(inputs, outputs, preds, stats_path)
    
    if args.show:
        plt.show()
    else:
        plt.close(fig3)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    overall_metrics = calculate_metrics(outputs, preds)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print(f"\nVisualizations saved to: {args.output_dir}/")
    print(f"Files created:")
    print(f"  - {base_name}_sample_{args.sample}.{args.format}")
    print(f"  - {base_name}_overview.{args.format}")
    print(f"  - {base_name}_statistics.{args.format}")

if __name__ == '__main__':
    main()
