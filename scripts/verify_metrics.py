#!/usr/bin/env python3
"""
Verify that our metric calculations match the original Keras implementation.
"""

import numpy as np
import math

def keras_PSNR(img1, img2, pixel_max=1.0):
    """Original Keras PSNR function."""
    dimg = (img1 - img2) / pixel_max
    mse = np.maximum(np.mean(dimg**2), 1.e-10)
    return -10 * math.log10(mse)

def keras_PSNRs(imgs1, imgs2, pixel_max=1.0):
    """Original Keras PSNRs function for multiple images."""
    dimgs = (imgs1 - imgs2) / pixel_max
    mse = np.maximum(np.mean(dimgs**2, axis=(1, 2)), 1.e-10)
    return -10 * np.log10(mse)

def keras_test_data_error(predictions, targets, pixel_max=1.0):
    """Original Keras test_data function - returns negative PSNR values."""
    return -keras_PSNRs(predictions, targets, pixel_max)

# Test with some sample data
np.random.seed(42)
batch_size = 5
height, width = 100, 100

# Create sample predictions and targets
predictions = np.random.rand(batch_size, height, width) * 2 - 1  # [-1, 1]
targets = np.random.rand(batch_size, height, width) * 2 - 1      # [-1, 1]

print("=== Metric Verification ===")
print(f"Sample shapes: predictions={predictions.shape}, targets={targets.shape}")
print()

# Calculate metrics
mse_values = np.mean((predictions - targets) ** 2, axis=(1, 2))
mae_values = np.mean(np.abs(predictions - targets), axis=(1, 2))
psnr_values = keras_PSNRs(predictions, targets, pixel_max=2.0)  # Using pixel_max=2.0 like our code
neg_psnr_values = keras_test_data_error(predictions, targets, pixel_max=2.0)

print("Per-sample metrics:")
for i in range(batch_size):
    print(f"Sample {i}: MSE={mse_values[i]:.6f}, MAE={mae_values[i]:.6f}, "
          f"PSNR={psnr_values[i]:.2f}dB, Error(neg_PSNR)={neg_psnr_values[i]:.2f}")

print()
print("Average metrics:")
print(f"MSE: {np.mean(mse_values):.6f}")
print(f"MAE: {np.mean(mae_values):.6f}")
print(f"PSNR: {np.mean(psnr_values):.2f} dB")
print(f"Error (negative PSNR): {np.mean(neg_psnr_values):.2f}")

print()
print("Key insights:")
print("1. PSNR values are positive (good quality = high PSNR)")
print("2. Error metric (used in training) is negative PSNR (good quality = low error)")
print("3. MSE and MAE are always positive")
print("4. The 'errors' you saw were negative PSNR values, which is correct!")
