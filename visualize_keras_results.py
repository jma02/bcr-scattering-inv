#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for BCR-Net scattering inverse problem results
Uses TensorFlow/Keras models, H5py data files, Matplotlib, and cmocean
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import argparse
import glob
from pathlib import Path

# Import TensorFlow for compatibility
import tensorflow as tf
print(f'Using TensorFlow version: {tf.__version__}')

# Import cmocean for oceanographic colormaps
try:
    import cmocean
    print(f'cmocean available - using oceanographic colormaps')
    HAS_CMOCEAN = True
except ImportError:
    print('cmocean not available - using matplotlib colormaps')
    HAS_CMOCEAN = False

class BCRNetVisualizer:
    """Visualizer for BCR-Net scattering inverse problem results"""
    
    def __init__(self, data_file_path, output_dir='visualizations'):
        """
        Initialize the visualizer
        
        Args:
            data_file_path: Path to the .h5 data file containing input/output/pred
            output_dir: Directory to save visualization outputs
        """
        self.data_file_path = data_file_path
        self.output_dir = output_dir
        self.data = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Setup colormaps
        self.setup_colormaps()
        
    def load_data(self):
        """Load data from HDF5 file"""
        print(f'Loading data from: {self.data_file_path}')
        
        with h5py.File(self.data_file_path, 'r') as f:
            self.data = {
                'input': f['input'][:],      # Far-field scattering data
                'output': f['output'][:],    # Ground truth reconstruction
                'pred': f['pred'][:]         # Model predictions
            }
            
        print(f'Data loaded:')
        for key, val in self.data.items():
            print(f'  {key}: shape {val.shape}, range [{np.min(val):.4f}, {np.max(val):.4f}]')
            
        # Calculate statistics
        self.calculate_statistics()
        
    def calculate_statistics(self):
        """Calculate error statistics between predictions and ground truth"""
        gt = self.data['output']
        pred = self.data['pred']
        
        # Mean squared error
        self.mse = np.mean((pred - gt)**2, axis=(1, 2))
        
        # Mean absolute error
        self.mae = np.mean(np.abs(pred - gt), axis=(1, 2))
        
        # Relative error
        self.rel_error = np.linalg.norm(pred - gt, axis=(1, 2)) / np.linalg.norm(gt, axis=(1, 2))
        
        # PSNR (Peak Signal-to-Noise Ratio)
        pixel_max = np.max(gt) - np.min(gt)
        self.psnr = -10 * np.log10(np.maximum(self.mse / (pixel_max**2), 1e-10))
        
        print(f'\nError Statistics:')
        print(f'  MSE: mean={np.mean(self.mse):.6f}, std={np.std(self.mse):.6f}')
        print(f'  MAE: mean={np.mean(self.mae):.6f}, std={np.std(self.mae):.6f}')
        print(f'  Rel Error: mean={np.mean(self.rel_error):.6f}, std={np.std(self.rel_error):.6f}')
        print(f'  PSNR: mean={np.mean(self.psnr):.2f} dB, std={np.std(self.psnr):.2f} dB')
        
    def setup_colormaps(self):
        """Setup colormaps for different data types"""
        if HAS_CMOCEAN:
            self.colormaps = {
                'input': cmocean.cm.balance,     # For scattering data (can be positive/negative)
                'output': cmocean.cm.dense,     # For density/reconstruction
                'pred': cmocean.cm.dense,       # For predictions
                'error': cmocean.cm.diff,       # For error maps
                'phase': cmocean.cm.phase       # For phase information
            }
        else:
            self.colormaps = {
                'input': 'RdBu_r',
                'output': 'viridis',
                'pred': 'viridis', 
                'error': 'RdBu_r',
                'phase': 'hsv'
            }
    
    def plot_sample_comparison(self, sample_idx=0, save=True, show=True):
        """Plot comparison of input, ground truth, prediction, and error for a single sample"""
        
        input_data = self.data['input'][sample_idx]
        gt_data = self.data['output'][sample_idx]
        pred_data = self.data['pred'][sample_idx]
        error_data = pred_data - gt_data
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Input (far-field scattering data)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_data, cmap=self.colormaps['input'], aspect='equal')
        ax1.set_title(f'Input: Far-field Scattering Data\nRange: [{np.min(input_data):.3f}, {np.max(input_data):.3f}]')
        ax1.set_xlabel('Receiver Position')
        ax1.set_ylabel('Source Position')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Ground truth reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(gt_data, cmap=self.colormaps['output'], aspect='equal')
        ax2.set_title(f'Ground Truth Reconstruction\nRange: [{np.min(gt_data):.3f}, {np.max(gt_data):.3f}]')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Prediction
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(pred_data, cmap=self.colormaps['pred'], aspect='equal')
        ax3.set_title(f'Neural Network Prediction\nRange: [{np.min(pred_data):.3f}, {np.max(pred_data):.3f}]')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Error map
        ax4 = fig.add_subplot(gs[1, 1])
        error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
        im4 = ax4.imshow(error_data, cmap=self.colormaps['error'], 
                        vmin=-error_max, vmax=error_max, aspect='equal')
        ax4.set_title(f'Prediction Error (Pred - GT)\nMSE: {self.mse[sample_idx]:.6f}, PSNR: {self.psnr[sample_idx]:.2f} dB')
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        plt.suptitle(f'BCR-Net Scattering Inverse Problem - Sample {sample_idx}', fontsize=16)
        
        if save:
            filename = f'sample_{sample_idx:03d}_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'Saved: {filepath}')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_error_statistics(self, save=True, show=True):
        """Plot error statistics across all samples"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MSE distribution
        axes[0, 0].hist(self.mse, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(self.mse), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.mse):.6f}')
        axes[0, 0].set_xlabel('Mean Squared Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MSE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PSNR distribution
        axes[0, 1].hist(self.psnr, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.psnr), color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.psnr):.2f} dB')
        axes[0, 1].set_xlabel('PSNR (dB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('PSNR Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Relative error distribution
        axes[1, 0].hist(self.rel_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(self.rel_error), color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.rel_error):.6f}')
        axes[1, 0].set_xlabel('Relative Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Relative Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error correlation
        axes[1, 1].scatter(self.mse, self.psnr, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Mean Squared Error')
        axes[1, 1].set_ylabel('PSNR (dB)')
        axes[1, 1].set_title('MSE vs PSNR Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'error_statistics.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'Saved: {filepath}')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_best_worst_cases(self, save=True, show=True):
        """Plot best and worst prediction cases based on PSNR"""
        
        # Find best and worst cases
        best_idx = np.argmax(self.psnr)
        worst_idx = np.argmin(self.psnr)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for i, (idx, case_name) in enumerate([(best_idx, 'Best'), (worst_idx, 'Worst')]):
            input_data = self.data['input'][idx]
            gt_data = self.data['output'][idx]
            pred_data = self.data['pred'][idx]
            error_data = pred_data - gt_data
            
            # Input
            im1 = axes[i, 0].imshow(input_data, cmap=self.colormaps['input'], aspect='equal')
            axes[i, 0].set_title(f'{case_name} Case - Input')
            plt.colorbar(im1, ax=axes[i, 0], shrink=0.6)
            
            # Ground truth
            im2 = axes[i, 1].imshow(gt_data, cmap=self.colormaps['output'], aspect='equal')
            axes[i, 1].set_title(f'{case_name} Case - Ground Truth')
            plt.colorbar(im2, ax=axes[i, 1], shrink=0.6)
            
            # Prediction
            im3 = axes[i, 2].imshow(pred_data, cmap=self.colormaps['pred'], aspect='equal')
            axes[i, 2].set_title(f'{case_name} Case - Prediction')
            plt.colorbar(im3, ax=axes[i, 2], shrink=0.6)
            
            # Error
            error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
            im4 = axes[i, 3].imshow(error_data, cmap=self.colormaps['error'],
                                   vmin=-error_max, vmax=error_max, aspect='equal')
            axes[i, 3].set_title(f'{case_name} Case - Error\nPSNR: {self.psnr[idx]:.2f} dB')
            plt.colorbar(im4, ax=axes[i, 3], shrink=0.6)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'best_worst_cases.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'Saved: {filepath}')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_reconstruction_profiles(self, sample_idx=0, save=True, show=True):
        """Plot cross-sectional profiles of reconstruction"""
        
        gt_data = self.data['output'][sample_idx]
        pred_data = self.data['pred'][sample_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get image center for cross-sections
        center_y, center_x = gt_data.shape[0] // 2, gt_data.shape[1] // 2
        
        # Horizontal cross-section
        axes[0, 0].plot(gt_data[center_y, :], 'b-', linewidth=2, label='Ground Truth')
        axes[0, 0].plot(pred_data[center_y, :], 'r--', linewidth=2, label='Prediction')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'Horizontal Cross-section (Y={center_y})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vertical cross-section
        axes[0, 1].plot(gt_data[:, center_x], 'b-', linewidth=2, label='Ground Truth')
        axes[0, 1].plot(pred_data[:, center_x], 'r--', linewidth=2, label='Prediction')
        axes[0, 1].set_xlabel('Y Position')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title(f'Vertical Cross-section (X={center_x})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot of all pixel values
        gt_flat = gt_data.flatten()
        pred_flat = pred_data.flatten()
        
        axes[1, 0].scatter(gt_flat, pred_flat, alpha=0.5, s=1)
        axes[1, 0].plot([gt_flat.min(), gt_flat.max()], [gt_flat.min(), gt_flat.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Ground Truth')
        axes[1, 0].set_ylabel('Prediction')
        axes[1, 0].set_title('Pixel-wise Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error map
        error_data = pred_data - gt_data
        error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
        im = axes[1, 1].imshow(error_data, cmap=self.colormaps['error'],
                              vmin=-error_max, vmax=error_max, aspect='equal')
        axes[1, 1].set_title('Error Map')
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        plt.suptitle(f'Reconstruction Analysis - Sample {sample_idx}', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = f'sample_{sample_idx:03d}_profiles.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'Saved: {filepath}')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_summary_report(self, save=True):
        """Create a comprehensive summary report"""
        
        report_file = os.path.join(self.output_dir, 'summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("BCR-Net Scattering Inverse Problem - Results Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data file: {self.data_file_path}\n")
            f.write(f"Number of samples: {len(self.data['input'])}\n")
            f.write(f"Image dimensions: {self.data['output'].shape[1:]} pixels\n\n")
            
            f.write("Data Statistics:\n")
            f.write("-" * 20 + "\n")
            for key in ['input', 'output', 'pred']:
                data = self.data[key]
                f.write(f"{key.capitalize()}:\n")
                f.write(f"  Range: [{np.min(data):.6f}, {np.max(data):.6f}]\n")
                f.write(f"  Mean: {np.mean(data):.6f}\n")
                f.write(f"  Std: {np.std(data):.6f}\n\n")
            
            f.write("Error Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"MSE:  mean={np.mean(self.mse):.8f}, std={np.std(self.mse):.8f}\n")
            f.write(f"MAE:  mean={np.mean(self.mae):.8f}, std={np.std(self.mae):.8f}\n")
            f.write(f"RelErr: mean={np.mean(self.rel_error):.8f}, std={np.std(self.rel_error):.8f}\n")
            f.write(f"PSNR: mean={np.mean(self.psnr):.2f} dB, std={np.std(self.psnr):.2f} dB\n\n")
            
            f.write("Best/Worst Cases:\n")
            f.write("-" * 20 + "\n")
            best_idx = np.argmax(self.psnr)
            worst_idx = np.argmin(self.psnr)
            f.write(f"Best PSNR:  Sample {best_idx}, PSNR={self.psnr[best_idx]:.2f} dB\n")
            f.write(f"Worst PSNR: Sample {worst_idx}, PSNR={self.psnr[worst_idx]:.2f} dB\n")
        
        if save:
            print(f'Summary report saved: {report_file}')
    
    def generate_all_visualizations(self, num_samples=5):
        """Generate all visualizations"""
        print(f"\nGenerating visualizations in: {self.output_dir}")
        
        # Error statistics
        self.plot_error_statistics(show=False)
        
        # Best/worst cases
        self.plot_best_worst_cases(show=False)
        
        # Sample comparisons
        sample_indices = np.linspace(0, len(self.data['input'])-1, num_samples, dtype=int)
        for idx in sample_indices:
            self.plot_sample_comparison(idx, show=False)
            self.plot_reconstruction_profiles(idx, show=False)
        
        # Summary report
        self.create_summary_report()
        
        print(f"\nVisualization complete! Files saved in: {self.output_dir}")


def find_data_files(logs_dir='logs'):
    """Find all data .h5 files in the logs directory"""
    pattern = os.path.join(logs_dir, '*data.h5')
    data_files = glob.glob(pattern)
    return sorted(data_files)


def main():
    parser = argparse.ArgumentParser(description='Visualize BCR-Net scattering inverse problem results')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to specific data .h5 file (if not provided, will find automatically)')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing log files (default: logs)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations (default: visualizations)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of sample visualizations to generate (default: 5)')
    parser.add_argument('--show-plots', action='store_true',
                       help='Show plots interactively (default: save only)')
    
    args = parser.parse_args()
    
    # Find data files if not specified
    if args.data_file is None:
        data_files = find_data_files(args.logs_dir)
        if not data_files:
            print(f"No data files found in {args.logs_dir}")
            print("Looking for files matching pattern: *data.h5")
            return
        
        print(f"Found {len(data_files)} data files:")
        for i, f in enumerate(data_files):
            print(f"  {i}: {f}")
        
        # Use the first (or most recent) data file
        args.data_file = data_files[-1]  # Use the last one (likely most recent)
        print(f"\nUsing data file: {args.data_file}")
    
    # Create visualizer and generate plots
    try:
        visualizer = BCRNetVisualizer(args.data_file, args.output_dir)
        
        if args.show_plots:
            # Interactive mode - show a few key plots
            visualizer.plot_sample_comparison(0, save=True, show=True)
            visualizer.plot_error_statistics(save=True, show=True)
            visualizer.plot_best_worst_cases(save=True, show=True)
        else:
            # Batch mode - generate all visualizations
            visualizer.generate_all_visualizations(args.num_samples)
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
