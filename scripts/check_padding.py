#!/usr/bin/env python3
"""
Quick verification script to check padding removal functionality.
"""

import h5py
import numpy as np
import sys

def check_padding_data(filename):
    """Check the structure of the HDF5 file with padding information."""
    print(f"Checking file: {filename}")
    print("=" * 50)
    
    with h5py.File(filename, 'r') as f:
        print("Available datasets:")
        for key in f.keys():
            data = f[key]
            print(f"  {key}: {data.shape} {data.dtype}")
        
        print("\nAvailable attributes:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")
        
        print("\nData verification:")
        
        if 'input' in f and 'input_padded' in f:
            input_orig = f['input'][:]
            input_padded = f['input_padded'][:]
            print(f"  Original input shape: {input_orig.shape}")
            print(f"  Padded input shape: {input_padded.shape}")
            
            # Check if unpadding works correctly
            pad_width = f.attrs.get('pad_width', 14)
            input_unpadded = input_padded[:, pad_width:-pad_width, pad_width:-pad_width]
            print(f"  Unpadded from padded shape: {input_unpadded.shape}")
            
            # Check if they match (should be very close due to normalization)
            if input_orig.shape == input_unpadded.shape:
                diff = np.mean(np.abs(input_orig - input_unpadded))
                print(f"  Mean absolute difference: {diff:.6f}")
                print(f"  Padding removal: {'✓ SUCCESS' if diff < 1e-10 else '✗ MISMATCH'}")
            else:
                print(f"  Shape mismatch: {input_orig.shape} vs {input_unpadded.shape}")
        
        if 'output' in f and 'pred' in f:
            output = f['output'][:]
            pred = f['pred'][:]
            print(f"  Target shape: {output.shape}")
            print(f"  Prediction shape: {pred.shape}")
            
            # Calculate metrics on unpadded data (matching original Keras implementation)
            mse = np.mean((output - pred) ** 2)
            mae = np.mean(np.abs(output - pred))
            
            # PSNR calculation (matching original code)
            psnr = -10 * np.log10(np.maximum(mse, 1e-10))  # Positive PSNR
            neg_psnr = -psnr  # Negative PSNR (what the original code uses as "error")
            
            # Relative L2 and L1 errors
            rel_l2 = np.sqrt(np.sum((output - pred) ** 2)) / np.sqrt(np.sum(output ** 2))
            rel_l1 = np.sum(np.abs(output - pred)) / np.sum(np.abs(output))
            
            print(f"  MSE on unpadded data: {mse:.6f}")
            print(f"  MAE on unpadded data: {mae:.6f}")
            print(f"  PSNR on unpadded data: {psnr:.2f} dB")
            print(f"  Negative PSNR (error metric): {neg_psnr:.2f}")
            
            # Relative error calculation
            rel_error = mae / np.mean(np.abs(output)) * 100
            print(f"  Relative error: {rel_error:.2f}%")
            print(f"  Relative L2 error: {rel_l2:.6f} ({rel_l2*100:.2f}%)")
            print(f"  Relative L1 error: {rel_l1:.6f} ({rel_l1*100:.2f}%)")
            
        # Also show saved metrics if available
        if 'final_mse' in f.attrs:
            print(f"  Saved MSE: {f.attrs['final_mse']:.6f}")
            print(f"  Saved MAE: {f.attrs['final_mae']:.6f}")
            print(f"  Saved PSNR: {f.attrs['final_psnr']:.2f} dB")
            print(f"  Saved Relative Error: {f.attrs['final_rel_error']:.2f}%")
            if 'final_rel_l2' in f.attrs:
                print(f"  Saved Relative L2 Error: {f.attrs['final_rel_l2']:.6f} ({f.attrs['final_rel_l2']*100:.2f}%)")
            if 'final_rel_l1' in f.attrs:
                print(f"  Saved Relative L1 Error: {f.attrs['final_rel_l1']:.6f} ({f.attrs['final_rel_l1']*100:.2f}%)")
            if 'training_loss' in f.attrs:
                print(f"  Training Loss Function: {f.attrs['training_loss'].decode() if isinstance(f.attrs['training_loss'], bytes) else f.attrs['training_loss']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_padding.py <hdf5_file>")
        sys.exit(1)
    
    check_padding_data(sys.argv[1])
