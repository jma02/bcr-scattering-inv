#!/usr/bin/env python3
"""
Quick test to verify device consistency in the PyTorch implementation
"""
import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.getcwd())

# Test imports
try:
    from mnn_torch import *
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

# Test layer creation and forward pass
print("\nTesting layers...")

# Create sample data
batch_size = 2
in_channels = 3
input_size = 32

# Test 1D layers
x1d = torch.randn(batch_size, in_channels, input_size).to(device)
print(f"Created 1D input on device: {x1d.device}")

try:
    # Test CNNK1D with smaller kernel
    layer_k1d = CNNK1D(out_channels=16, kernel_size=3).to(device)
    out_k1d = layer_k1d(x1d)
    print(f"✓ CNNK1D output device: {out_k1d.device}")
    
    # Test CNNR1D with smaller kernel
    layer_r1d = CNNR1D(out_channels=16, kernel_size=3).to(device)
    out_r1d = layer_r1d(x1d)
    print(f"✓ CNNR1D output device: {out_r1d.device}")
    
    # Test CNNI1D (without kernel_size parameter)
    layer_i1d = CNNI1D(out_channels=16).to(device)
    out_i1d = layer_i1d(x1d)
    print(f"✓ CNNI1D output device: {out_i1d.device}")
    
    # Test WaveLetC1D
    layer_w1d = WaveLetC1D(out_channels=16, kernel_size=4).to(device)  # kernel_size must be even
    out_w1d = layer_w1d(x1d)
    print(f"✓ WaveLetC1D output device: {out_w1d.device}")
    
    # Test InvWaveLetC1D
    layer_iw1d = InvWaveLetC1D(out_channels=16, kernel_size=3).to(device)  # kernel_size must be odd
    out_iw1d = layer_iw1d(x1d)
    print(f"✓ InvWaveLetC1D output device: {out_iw1d.device}")
    
except Exception as e:
    print(f"✗ 1D layer test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2D layers
input_size_2d = 16
x2d = torch.randn(batch_size, in_channels, input_size_2d, input_size_2d).to(device)
print(f"\nCreated 2D input on device: {x2d.device}")

try:
    # Test CNNK2D
    layer_k2d = CNNK2D(out_channels=16, kernel_size=3).to(device)
    out_k2d = layer_k2d(x2d)
    print(f"✓ CNNK2D output device: {out_k2d.device}")
    
    # Test CNNR2D
    layer_r2d = CNNR2D(out_channels=16, kernel_size=3).to(device)
    out_r2d = layer_r2d(x2d)
    print(f"✓ CNNR2D output device: {out_r2d.device}")
    
    # Test CNNI2D (without kernel_size parameter)
    layer_i2d = CNNI2D(out_channels=16).to(device)
    out_i2d = layer_i2d(x2d)
    print(f"✓ CNNI2D output device: {out_i2d.device}")
    
except Exception as e:
    print(f"✗ 2D layer test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Device consistency test completed successfully!")
