#!/usr/bin/env python3
"""
Diagnostic script to verify relative L2 error calculations and examine model performance.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_model_performance(filepath):
    """Analyze model performance and verify error calculations."""
    
    with h5py.File(filepath, 'r') as f:
        # Load unpadded data
        inputs = f['input'][:]
        targets = f['output'][:]
        predictions = f['pred'][:]
        
        print("=== Data Analysis ===")
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Prediction shape: {predictions.shape}")
        print()
        
        print("=== Target Statistics ===")
        print(f"Target mean: {np.mean(targets):.6f}")
        print(f"Target std: {np.std(targets):.6f}")
        print(f"Target min: {np.min(targets):.6f}")
        print(f"Target max: {np.max(targets):.6f}")
        print(f"Target range: {np.max(targets) - np.min(targets):.6f}")
        print()
        
        print("=== Prediction Statistics ===")
        print(f"Prediction mean: {np.mean(predictions):.6f}")
        print(f"Prediction std: {np.std(predictions):.6f}")
        print(f"Prediction min: {np.min(predictions):.6f}")
        print(f"Prediction max: {np.max(predictions):.6f}")
        print(f"Prediction range: {np.max(predictions) - np.min(predictions):.6f}")
        print()
        
        # Calculate errors
        error = predictions - targets
        abs_error = np.abs(error)
        squared_error = error ** 2
        
        print("=== Error Analysis ===")
        print(f"Error mean: {np.mean(error):.6f}")
        print(f"Error std: {np.std(error):.6f}")
        print(f"Max absolute error: {np.max(abs_error):.6f}")
        print()
        
        # Standard metrics
        mse = np.mean(squared_error)
        mae = np.mean(abs_error)
        rmse = np.sqrt(mse)
        
        print("=== Standard Metrics ===")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print()
        
        # Relative error calculations
        print("=== Relative Error Calculations ===")
        
        # Method 1: As we calculated (per sample, then average)
        rel_l2_per_sample = []
        rel_l1_per_sample = []
        
        for i in range(len(targets)):
            # L2 relative error for this sample
            l2_error = np.sqrt(np.sum((predictions[i] - targets[i]) ** 2))
            l2_target = np.sqrt(np.sum(targets[i] ** 2))
            rel_l2_sample = l2_error / max(l2_target, 1e-10)
            rel_l2_per_sample.append(rel_l2_sample)
            
            # L1 relative error for this sample
            l1_error = np.sum(np.abs(predictions[i] - targets[i]))
            l1_target = np.sum(np.abs(targets[i]))
            rel_l1_sample = l1_error / max(l1_target, 1e-10)
            rel_l1_per_sample.append(rel_l1_sample)
        
        avg_rel_l2 = np.mean(rel_l2_per_sample)
        avg_rel_l1 = np.mean(rel_l1_per_sample)
        
        print(f"Average Relative L2 Error (per-sample): {avg_rel_l2:.6f} ({avg_rel_l2*100:.2f}%)")
        print(f"Average Relative L1 Error (per-sample): {avg_rel_l1:.6f} ({avg_rel_l1*100:.2f}%)")
        print()
        
        # Method 2: Global calculation
        global_l2_error = np.sqrt(np.sum((predictions - targets) ** 2))
        global_l2_target = np.sqrt(np.sum(targets ** 2))
        global_rel_l2 = global_l2_error / global_l2_target
        
        global_l1_error = np.sum(np.abs(predictions - targets))
        global_l1_target = np.sum(np.abs(targets))
        global_rel_l1 = global_l1_error / global_l1_target
        
        print(f"Global Relative L2 Error: {global_rel_l2:.6f} ({global_rel_l2*100:.2f}%)")
        print(f"Global Relative L1 Error: {global_rel_l1:.6f} ({global_rel_l1*100:.2f}%)")
        print()
        
        # Check if the model is just predicting zeros or constants
        print("=== Model Behavior Analysis ===")
        pred_variance = np.var(predictions)
        target_variance = np.var(targets)
        
        print(f"Prediction variance: {pred_variance:.6f}")
        print(f"Target variance: {target_variance:.6f}")
        print(f"Variance ratio (pred/target): {pred_variance/target_variance:.6f}")
        print()
        
        # Check correlation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        print(f"Correlation coefficient: {correlation:.6f}")
        print()
        
        # Check if predictions are just the mean
        target_mean = np.mean(targets)
        mean_predictor_mse = np.mean((targets - target_mean) ** 2)
        improvement_ratio = mean_predictor_mse / mse
        
        print(f"MSE if predicting mean: {mean_predictor_mse:.6f}")
        print(f"Model improvement over mean predictor: {improvement_ratio:.2f}x")
        print()
        
        # Sample-wise analysis
        print("=== Sample-wise Analysis (first 10 samples) ===")
        for i in range(min(10, len(targets))):
            sample_mse = np.mean((predictions[i] - targets[i]) ** 2)
            sample_mae = np.mean(np.abs(predictions[i] - targets[i]))
            target_norm = np.sqrt(np.sum(targets[i] ** 2))
            pred_norm = np.sqrt(np.sum(predictions[i] ** 2))
            
            print(f"Sample {i}: MSE={sample_mse:.4f}, MAE={sample_mae:.4f}, "
                  f"||target||={target_norm:.4f}, ||pred||={pred_norm:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'avg_rel_l2': avg_rel_l2,
            'avg_rel_l1': avg_rel_l1,
            'global_rel_l2': global_rel_l2,
            'global_rel_l1': global_rel_l1,
            'correlation': correlation,
            'improvement_ratio': improvement_ratio
        }

if __name__ == "__main__":
    filepath = "models/BCRNet_datadata_cnn6_alpha40_noise0_test_rel_errors_predictions.h5"
    results = analyze_model_performance(filepath)
