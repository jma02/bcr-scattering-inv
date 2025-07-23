#!/usr/bin/env python3
"""
Simple visualization to show what the model is actually predicting.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def show_reality_check(filepath):
    """Show what the model is actually predicting vs targets."""
    
    with h5py.File(filepath, 'r') as f:
        targets = f['output'][:]
        predictions = f['pred'][:]
    
    # Show first few samples
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for i in range(3):
        # Target
        im1 = axes[i, 0].imshow(targets[i], cmap='viridis', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Target {i}')
        axes[i, 0].axis('off')
        
        # Prediction
        im2 = axes[i, 1].imshow(predictions[i], cmap='viridis', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Prediction {i}')
        axes[i, 1].axis('off')
        
        # Error
        error = predictions[i] - targets[i]
        im3 = axes[i, 2].imshow(error, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 2].set_title(f'Error {i}')
        axes[i, 2].axis('off')
        
        # Histograms
        axes[i, 3].hist(targets[i].flatten(), bins=50, alpha=0.7, label='Target', density=True)
        axes[i, 3].hist(predictions[i].flatten(), bins=50, alpha=0.7, label='Pred', density=True)
        axes[i, 3].set_title(f'Histograms {i}')
        axes[i, 3].legend()
        
        # Scatter plot
        axes[i, 4].scatter(targets[i].flatten()[::100], predictions[i].flatten()[::100], 
                          alpha=0.5, s=1)
        axes[i, 4].plot([-1, 1], [-1, 1], 'r--', label='Perfect')
        axes[i, 4].set_xlabel('Target')
        axes[i, 4].set_ylabel('Prediction')
        axes[i, 4].set_title(f'Scatter {i}')
        axes[i, 4].legend()
        
        # Stats
        target_std = np.std(targets[i])
        pred_std = np.std(predictions[i])
        corr = np.corrcoef(targets[i].flatten(), predictions[i].flatten())[0, 1]
        
        axes[i, 5].text(0.1, 0.8, f'Target std: {target_std:.3f}', transform=axes[i, 5].transAxes)
        axes[i, 5].text(0.1, 0.6, f'Pred std: {pred_std:.3f}', transform=axes[i, 5].transAxes)
        axes[i, 5].text(0.1, 0.4, f'Correlation: {corr:.3f}', transform=axes[i, 5].transAxes)
        axes[i, 5].set_title(f'Stats {i}')
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_reality_check.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Overall statistics
    print("=== REALITY CHECK ===")
    print(f"Overall target std: {np.std(targets):.6f}")
    print(f"Overall prediction std: {np.std(predictions):.6f}")
    print(f"Overall correlation: {np.corrcoef(targets.flatten(), predictions.flatten())[0, 1]:.6f}")
    
    # Check if predictions are essentially constant
    pred_range = np.max(predictions) - np.min(predictions)
    target_range = np.max(targets) - np.min(targets)
    print(f"Prediction range: {pred_range:.6f}")
    print(f"Target range: {target_range:.6f}")
    print(f"Range ratio (pred/target): {pred_range/target_range:.6f}")
    
    # Check mean and std per sample
    per_sample_pred_std = np.std(predictions, axis=(1, 2))
    per_sample_target_std = np.std(targets, axis=(1, 2))
    
    print(f"Average per-sample prediction std: {np.mean(per_sample_pred_std):.6f}")
    print(f"Average per-sample target std: {np.mean(per_sample_target_std):.6f}")
    
    print("\n*** CONCLUSION ***")
    if np.mean(per_sample_pred_std) < 0.1 * np.mean(per_sample_target_std):
        print("🚨 MODEL IS PREDICTING NEAR-CONSTANT VALUES!")
        print("The relative L2 error is misleading - the model hasn't learned the image structure.")
    else:
        print("✓ Model predictions show reasonable variance.")

if __name__ == "__main__":
    show_reality_check("models/BCRNet_datadata_cnn6_alpha40_noise0_test_rel_errors_predictions.h5")
