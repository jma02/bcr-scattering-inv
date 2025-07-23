# coding=utf-8
# vim: sw=4 et tw=100
"""
PyTorch implementation of scattering 2D inverse problem: BCR-Net

Modern PyTorch implementation converted from the original Keras version.
Uses modern PyTorch features and best practices.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import math
from shutil import copyfile
from typing import Tuple, Optional, Callable, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Add the parent directory to Python path to import mnn_torch
sys.path.append(os.path.abspath('.'))
# Import the new PyTorch MNN modules
from mnn_torch.layers import CNNK1D, CNNR1D, CNNI1D, WaveLetC1D, InvWaveLetC1D, CNNK2D
from mnn_torch.callback import SaveBestModel, train_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScatteringDataset(Dataset):
    """Dataset for scattering inverse problem."""
    
    def __init__(self, input_data: np.ndarray, output_data: np.ndarray):
        """Initialize dataset.
        
        Args:
            input_data: Input scattering data of shape (N, Ns, Nd)  
            output_data: Output images of shape (N, Nt, Nr)
        """
        self.input_data = torch.from_numpy(input_data).float()
        self.output_data = torch.from_numpy(output_data).float()
        
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.output_data[idx]


class BCRNet(nn.Module):
    """BCR-Net architecture for scattering inverse problem.
    
    This implements the same architecture as the original Keras version
    but using modern PyTorch modules.
    """
    
    def __init__(self, 
                 n_input: Tuple[int, int],
                 n_output: Tuple[int, int],
                 L: int,
                 alpha: int,
                 N_cnn: int,
                 N_cnn3: int,
                 w: int = 6,
                 m: int = 7,
                 n_b: int = 5,
                 w_comp: int = 1,
                 bc: str = 'zero'):
        """Initialize BCR-Net.
        
        Args:
            n_input: Input dimensions (Ns, Nd)
            n_output: Output dimensions (Nt, Nr)  
            L: Number of wavelet levels
            alpha: Number of channels for depth
            N_cnn: Number of CNN layers
            N_cnn3: Number of final 2D CNN layers
            w: Wavelet support size
            m: Coarse grid size
            n_b: Band size of matrix
            w_comp: Window size of compression
            bc: Boundary condition ('zero' or 'period')
        """
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.L = L
        self.alpha = alpha
        self.N_cnn = N_cnn
        self.N_cnn3 = N_cnn3
        self.w = w
        self.m = m
        self.n_b = n_b
        self.w_comp = w_comp
        self.bc = bc
        
        Ns, Nd = n_input
        Nt, Nr = n_output
        
        # Initial convolution layer
        self.input_conv = CNNK1D(alpha, w_comp, activation='linear', bc_padding=bc)
        
        # Forward wavelet transform layers
        self.wavelet_layers = nn.ModuleList()
        for ll in range(1, L + 1):
            self.wavelet_layers.append(
                WaveLetC1D(2 * alpha, w, activation='linear', bias=False)
            )
        
        # Core CNN layers on coarse grid
        self.core_layers = nn.ModuleList()
        for k in range(N_cnn):
            self.core_layers.append(
                CNNK1D(alpha, m, activation='relu', bc_padding='period')
            )
        
        # Detail processing layers for each level
        self.detail_layers = nn.ModuleDict()
        for ll in range(L, 0, -1):
            detail_cnn = nn.ModuleList()
            for k in range(N_cnn):
                detail_cnn.append(
                    CNNK1D(2 * alpha, n_b, activation='relu', bc_padding='period')
                )
            self.detail_layers[f'level_{ll}'] = detail_cnn
        
        # Inverse wavelet transform layers
        self.inv_wavelet_layers = nn.ModuleList()
        for ll in range(L, 0, -1):
            output_size = Nt // (2**(ll-1)) if ll > 1 else Nt
            self.inv_wavelet_layers.append(
                InvWaveLetC1D(2 * alpha, w // 2, Nout=output_size, 
                             activation='linear', bias=False)
            )
        
        # Final reconstruction layers
        self.final_conv = CNNK1D(Nr, w_comp, activation='linear', bc_padding=bc)
        
        # 2D refinement layers
        self.conv2d_layers = nn.ModuleList()
        for k in range(N_cnn3 - 1):
            if k == 0:
                self.conv2d_layers.append(
                    nn.Conv2d(1, 4, kernel_size=3, padding=0)  # No automatic padding
                )
            else:
                self.conv2d_layers.append(
                    nn.Conv2d(4, 4, kernel_size=3, padding=0)  # No automatic padding
                )
        
        # Final 2D layer
        self.final_2d = nn.Conv2d(4, 1, kernel_size=3, padding=0)  # No automatic padding
        
    def _padding_x(self, x: torch.Tensor, s: int) -> torch.Tensor:
        """Apply periodic padding in the first spatial dimension."""
        if s == 0:
            return x
        left_pad = x[:, -s:, ...]
        right_pad = x[:, :s, ...]
        return torch.cat([left_pad, x, right_pad], dim=1)
    
    def _split_scaling_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract scaling coefficients."""
        return x[:, :, self.alpha:2*self.alpha]
    
    def _split_wavelet_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract wavelet coefficients."""
        return x[:, :, :self.alpha]
    
    def _triangle_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Triangle addition operation."""
        scaling_part = self._split_scaling_1d(x) + y
        wavelet_part = self._split_wavelet_1d(x)
        return torch.cat([wavelet_part, scaling_part], dim=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BCR-Net.
        
        Args:
            x: Input tensor of shape (batch, Ns, Nd)
            
        Returns:
            Output tensor of shape (batch, Nt, Nr)
        """
        batch_size = x.size(0)
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Forward wavelet transforms
        bt_list = [None] * (self.L + 1)  # Store detail coefficients
        b = x
        
        for ll in range(1, self.L + 1):
            bt = self.wavelet_layers[ll - 1](b)
            bt_list[ll] = bt
            b = self._split_scaling_1d(bt)
        
        # Process coarse scale
        d = b
        for layer in self.core_layers:
            d = layer(d)
        
        # Inverse wavelet transforms with detail processing
        for ll in range(self.L, 0, -1):
            # Process details at this level
            d1 = bt_list[ll]
            for layer in self.detail_layers[f'level_{ll}']:
                d1 = layer(d1)
            
            # Triangle addition
            d = self._triangle_add(d1, d)
            
            # Inverse wavelet transform
            d = self.inv_wavelet_layers[self.L - ll](d)
        
        # Store intermediate result
        img_c = d
        
        # Final 1D convolution
        img = self.final_conv(img_c)
        
        # Reshape for 2D processing
        img_2d = img.view(batch_size, self.n_output[0], self.n_output[1], 1)
        img_2d = img_2d.permute(0, 3, 1, 2)  # (batch, 1, Nt, Nr)
        
        # 2D refinement layers
        img_p = img_2d
        for k, layer in enumerate(self.conv2d_layers):
            # Apply periodic padding preserving spatial dimensions
            img_p = self._periodic_pad_2d(img_p, 1)
            img_p = F.relu(layer(img_p))
        
        # Final 2D layer
        img_p = self._periodic_pad_2d(img_p, 1)
        img_p = self.final_2d(img_p)
        
        # Convert back to (batch, Nt, Nr) format
        img_p = img_p.squeeze(1)  # Remove channel dimension
        
        # Add residual connection
        img_flat = img.view(batch_size, self.n_output[0], self.n_output[1])
        output = img_flat + img_p
        
        return output
    
    def _padding_2d_height(self, x: torch.Tensor, s: int) -> torch.Tensor:
        """Apply periodic padding in height dimension."""
        if s == 0:
            return x
        top_pad = x[:, :, -s:, :]
        bottom_pad = x[:, :, :s, :]
        return torch.cat([top_pad, x, bottom_pad], dim=2)
    
    def _periodic_pad_2d(self, x: torch.Tensor, pad_size: int) -> torch.Tensor:
        """Apply periodic padding in both dimensions for conv2d with kernel_size=3.
        
        This ensures the output has the same spatial dimensions as the input.
        """
        if pad_size == 0:
            return x
        
        # Periodic padding in height dimension
        top_pad = x[:, :, -pad_size:, :]
        bottom_pad = x[:, :, :pad_size, :]
        x_h_padded = torch.cat([top_pad, x, bottom_pad], dim=2)
        
        # Periodic padding in width dimension  
        left_pad = x_h_padded[:, :, :, -pad_size:]
        right_pad = x_h_padded[:, :, :, :pad_size]
        x_padded = torch.cat([left_pad, x_h_padded, right_pad], dim=3)
        
        return x_padded


def PSNR(img1: torch.Tensor, img2: torch.Tensor, pixel_max: float = 1.0) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2))
    mse = torch.clamp(mse, min=1e-10)
    return -10 * torch.log10(mse / (pixel_max ** 2))


def MSE(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error."""
    return torch.mean((img1 - img2) ** 2, dim=(1, 2))


def MAE(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(img1 - img2), dim=(1, 2))


def relative_L2_error(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate relative L2 error: ||pred - true||_2 / ||true||_2."""
    numerator = torch.sqrt(torch.sum((img1 - img2) ** 2, dim=(1, 2)))
    denominator = torch.sqrt(torch.sum(img2 ** 2, dim=(1, 2)))
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-10)
    return numerator / denominator


def relative_L1_error(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate relative L1 error: ||pred - true||_1 / ||true||_1."""
    numerator = torch.sum(torch.abs(img1 - img2), dim=(1, 2))
    denominator = torch.sum(torch.abs(img2), dim=(1, 2))
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-10)
    return numerator / denominator


def test_data_fn(model: nn.Module, 
                 data_loader: DataLoader, 
                 device: torch.device, 
                 pixel_max: float) -> np.ndarray:
    """Test function for model evaluation - returns negative PSNR (like original Keras code)."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Return negative PSNR to match original Keras implementation
            psnr_values = -PSNR(outputs, targets, pixel_max)
            errors.extend(psnr_values.cpu().numpy())
    
    return np.array(errors)


def calculate_metrics(model: nn.Module, 
                     data_loader: DataLoader, 
                     device: torch.device) -> dict:
    """Calculate comprehensive metrics (MSE, MAE, PSNR, relative errors) for model evaluation."""
    model.eval()
    mse_values = []
    mae_values = []
    psnr_values = []
    rel_l2_values = []
    rel_l1_values = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            mse_batch = MSE(outputs, targets)
            mae_batch = MAE(outputs, targets)
            psnr_batch = PSNR(outputs, targets, pixel_max=1.0)
            rel_l2_batch = relative_L2_error(outputs, targets)
            rel_l1_batch = relative_L1_error(outputs, targets)
            
            mse_values.extend(mse_batch.cpu().numpy())
            mae_values.extend(mae_batch.cpu().numpy())
            psnr_values.extend(psnr_batch.cpu().numpy())
            rel_l2_values.extend(rel_l2_batch.cpu().numpy())
            rel_l1_values.extend(rel_l1_batch.cpu().numpy())
    
    return {
        'mse': np.array(mse_values),
        'mae': np.array(mae_values),
        'psnr': np.array(psnr_values),
        'rel_l2': np.array(rel_l2_values),
        'rel_l1': np.array(rel_l1_values)
    }


def remove_padding(data: np.ndarray, pad_width: int) -> np.ndarray:
    """Remove padding from predictions to restore original size.
    
    Args:
        data: Padded data of shape (N, H_padded, W_padded) or (H_padded, W_padded)
        pad_width: Width of padding on each side
    
    Returns:
        Unpadded data with original dimensions
    """
    if data.ndim == 3:
        # Multiple samples: (N, H, W)
        return data[:, pad_width:-pad_width, pad_width:-pad_width]
    elif data.ndim == 2:
        # Single sample: (H, W)
        return data[pad_width:-pad_width, pad_width:-pad_width]
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def create_data_loaders(X_train: np.ndarray, Y_train: np.ndarray,
                       X_test: np.ndarray, Y_test: np.ndarray,
                       batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders."""
    train_dataset = ScatteringDataset(X_train, Y_train)
    test_dataset = ScatteringDataset(X_test, Y_test)
    
    # Use fewer workers to avoid device issues
    num_workers = 2 if torch.cuda.is_available() else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Scattering 2D - PyTorch')
    parser.add_argument('--epoch', type=int, default=40, metavar='N',
                        help='# epochs for training (default: %(default)s)')
    parser.add_argument('--input-prefix', type=str, default='merged_data', metavar='N',
                        help='prefix of input data filename (default: %(default)s)')
    parser.add_argument('--alpha', type=int, default=40, metavar='N',
                        help='number of channels for depth (default: %(default)s)')
    parser.add_argument('--n-cnn', type=int, default=6, metavar='N',
                        help='number CNN layers (default: %(default)s)')
    parser.add_argument('--n-cnn3', type=int, default=5, metavar='N',
                        help='number 2D CNN layers (default: %(default)s)')
    parser.add_argument('--noise', type=float, default=0, metavar='noise',
                        help='noise on the measure data (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size (default: %(default)s)')
    parser.add_argument('--grad-clip', type=float, default=1.0, metavar='CLIP',
                        help='gradient clipping threshold (default: %(default)s)')
    parser.add_argument('--verbose', type=int, default=2, metavar='N',
                        help='verbosity level (default: %(default)s)')
    parser.add_argument('--output-suffix', type=str, default=None, metavar='N',
                        help='suffix output filename')
    parser.add_argument('--percent', type=float, default=4./5., metavar='percent',
                        help='percentage of training data (default: %(default)s)')
    parser.add_argument('--initialvalue', type=str, default=None, metavar='filename',
                        help='filename storing the weights of the model')
    parser.add_argument('--w-comp', type=int, default=1, metavar='N',
                        help='window size of the compress (default: %(default)s)')
    parser.add_argument('--data-path', type=str, default='data', metavar='string',
                        help='data path (default: %(default)s)')
    parser.add_argument('--log-path', type=str, default='logs', metavar='string',
                        help='log path (default: %(default)s)')
    parser.add_argument('--model-path', type=str, default='models', metavar='string',
                        help='model save path (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training (default: auto)')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directories
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Generate more sensible output filenames
    base_name = f"BCRNet_data{args.input_prefix[7:]}_cnn{args.n_cnn}_alpha{args.alpha}_noise{args.noise}"
    if args.output_suffix:
        base_name += f"_{args.output_suffix}"
    else:
        base_name += f"_{os.getpid()}"
    
    model_filename = os.path.join(args.model_path, f"{base_name}.pth")
    log_filename = os.path.join(args.log_path, f"{base_name}.txt")
    
    # Set up logging
    log_file = open(log_filename, "w+")
    
    def output(obj):
        print(obj)
        log_file.write(str(obj) + '\n')
        log_file.flush()
    
    output(f'Output filename: {log_filename}')
    output(f'Model filename: {model_filename}')
    
    # Load data
    data_file = os.path.join(args.data_path, f'{args.input_prefix}.hdf5')
    output(f'Reading data from: {data_file}')
    
    with h5py.File(data_file, 'r') as fin:
        input_array = fin['farfield.real'][:]
        output_array = fin['image'][:]
    
    # Process data
    Nsamples = input_array.shape[1]
    input_array = np.array(input_array).T.reshape(Nsamples, 100, 100)
    output_array = np.array(output_array).T.reshape(Nsamples, 100, 100)
    
    # Store original dimensions for padding removal
    original_size = (100, 100)
    padded_size = (128, 128)
    pad_width = 14  # padding on each side
    
    # Zero pad to 128x128
    input_array = np.pad(input_array, ((0, 0), (14, 14), (14, 14)), 
                        mode='constant', constant_values=0)
    output_array = np.pad(output_array, ((0, 0), (14, 14), (14, 14)),
                         mode='constant', constant_values=0)
    
    output(f'Original size: {original_size}, Padded size: {padded_size}, Pad width: {pad_width}')
    
    Nsamples, Ns, Nd = input_array.shape
    _, Nt, Nr = output_array.shape
    
    # Process input array (frequency domain shift and concatenation)
    Nd *= 2
    tmp = input_array
    tmp2 = np.concatenate([tmp[:, Ns//2:Ns, :], tmp[:, 0:Ns//2, :]], axis=1)
    input_array = np.concatenate([tmp, tmp2], axis=2)
    input_array = input_array[:, :, Nd//4:3*Nd//4]
    
    Ns, Nd = input_array.shape[1], input_array.shape[2]
    n_input = (Ns, Nd)
    n_output = (Nt, Nr)
    
    output(f"Input shape: {input_array.shape}")
    output(f"Output shape: {output_array.shape}")
    output(f"(Ns, Nd) = {n_input}")
    output(f"(Nt, Nr) = {n_output}")
    
    # Normalize input data to [-1, 1] using min-max normalization
    input_min = np.amin(input_array)
    input_max = np.amax(input_array)
    input_array = 2.0 * (input_array - input_min) / (input_max - input_min) - 1.0
    output(f'Input data normalized from ({input_min:.6f}, {input_max:.6f}) to ({np.amin(input_array):.6f}, {np.amax(input_array):.6f})')
    
    # Normalize output data to [-1, 1] using min-max normalization
    output_min = np.amin(output_array)
    output_max = np.amax(output_array)
    output_array = 2.0 * (output_array - output_min) / (output_max - output_min) - 1.0
    pixel_max = 2.0  # Range is now [-1, 1]
    
    output(f'Output data normalized from ({output_min:.6f}, {output_max:.6f}) to ({np.amin(output_array):.6f}, {np.amax(output_array):.6f})')
    output(f'Pixel max for PSNR calculation: {pixel_max}')
    
    # Split data
    n_train = int(Nsamples * args.percent)
    n_test = min(max(n_train, 5000), Nsamples - n_train)
    n_valid = 1024
    
    X_train = input_array[:n_train]
    Y_train = output_array[:n_train]
    X_test = input_array[n_train:n_train+n_test]
    Y_test = output_array[n_train:n_train+n_test]
    
    output(f"Training samples: {n_train}")
    output(f"Test samples: {n_test}")
    output(f"Validation samples: {n_valid}")
    
    # Add noise
    if args.noise > 0:
        noise_rate = args.noise / 100.0
        noise_train = np.random.randn(*X_train.shape) * noise_rate
        X_train = X_train * (1 + noise_train)
        noise_test = np.random.randn(*X_test.shape) * noise_rate
        X_test = X_test * (1 + noise_test)
        output(f"Added {args.noise}% noise to input data")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, Y_train, X_test, Y_test, args.batch_size
    )

    # Model parameters
    L = math.floor(math.log2(Ns)) - 2  # number of levels
    m = Ns // 2**L  # size of coarse grid
    m = 2 * ((m + 1) // 2) - 1
    w = 2 * 3  # wavelet support
    n_b = 5  # band size
    
    output(f"Wavelet levels L: {L}")
    output(f"Coarse grid size m: {m}")
    
    # Create model
    model = BCRNet(
        n_input=n_input,
        n_output=n_output, 
        L=L,
        alpha=args.alpha,
        N_cnn=args.n_cnn,
        N_cnn3=args.n_cnn3,
        w=w,
        m=m,
        n_b=n_b,
        w_comp=args.w_comp
    )
    
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    output(f'Number of parameters: {num_params}')
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=10)
    
    # Create callback for saving best model
    def check_result_fn(model):
        """Check function for callback."""
        # Evaluate on subset of data for speed
        train_subset_indices = np.random.choice(len(train_loader.dataset), 
                                              min(n_valid, len(train_loader.dataset)), 
                                              replace=False)
        test_subset_indices = np.random.choice(len(test_loader.dataset),
                                             min(n_valid, len(test_loader.dataset)), 
                                             replace=False)
        
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_subset_indices)
        test_subset = torch.utils.data.Subset(test_loader.dataset, test_subset_indices)
        
        train_subset_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
        test_subset_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        
        train_errors = test_data_fn(model, train_subset_loader, device, pixel_max)
        test_errors = test_data_fn(model, test_subset_loader, device, pixel_max)
        
        return train_errors, test_errors
    
    callback = SaveBestModel(
        filename=model_filename,
        check_result=check_result_fn,
        patience=20,
        output=output,
        test_weight=1.0,
        verbose=args.verbose,
        reduce_lr=True,
        min_lr=args.lr / 100,
        patience_lr=10
    )
    
    # Load initial weights if provided
    if args.initialvalue:
        output(f'Loading initial weights from: {args.initialvalue}')
        model.load_state_dict(torch.load(args.initialvalue, map_location=device))
    
    # Train model
    output(f'Starting training for {args.epoch} epochs...')
    if args.grad_clip > 0:
        output(f'Using gradient clipping with threshold: {args.grad_clip}')
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epoch,
        device=device,
        callback=callback,
        scheduler=scheduler,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None
    )
    
    # Save final results
    best_model_path = model_filename.replace('.pth', '_best.pth')
    data_save_path = model_filename.replace('.pth', '_predictions.h5')
    
    if os.path.exists(model_filename):
        if os.path.exists(best_model_path):
            # Compare final model with best model
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            output('Loaded best model for final evaluation')
        
        copyfile(model_filename, best_model_path)
        output(f'Saved best model to: {best_model_path}')
        
        # Save sample predictions
        model.eval()
        with torch.no_grad():
            sample_inputs = torch.from_numpy(X_test[:100]).float().to(device)
            sample_targets = torch.from_numpy(Y_test[:100]).float()
            sample_preds = model(sample_inputs).cpu()
        
        # Remove padding from predictions to restore original size
        sample_inputs_unpadded = remove_padding(X_test[:100], pad_width)
        sample_targets_unpadded = remove_padding(Y_test[:100], pad_width)
        sample_preds_unpadded = remove_padding(sample_preds.numpy(), pad_width)
        
        output(f'Removed padding: {sample_preds.shape} -> {sample_preds_unpadded.shape}')
        
        # Calculate final metrics on unpadded data
        final_mse = np.mean((sample_targets_unpadded - sample_preds_unpadded) ** 2)
        final_mae = np.mean(np.abs(sample_targets_unpadded - sample_preds_unpadded))
        final_psnr = -10 * np.log10(np.maximum(final_mse, 1e-10))
        final_rel_error = final_mae / np.mean(np.abs(sample_targets_unpadded)) * 100
        
        # Calculate relative L2 and L1 errors
        final_rel_l2 = np.sqrt(np.sum((sample_targets_unpadded - sample_preds_unpadded) ** 2)) / np.sqrt(np.sum(sample_targets_unpadded ** 2))
        final_rel_l1 = np.sum(np.abs(sample_targets_unpadded - sample_preds_unpadded)) / np.sum(np.abs(sample_targets_unpadded))
        
        output('=== Final Evaluation on Unpadded Data ===')
        output(f'Training Loss: MSE (Mean Squared Error)')
        output(f'MSE: {final_mse:.6f}')
        output(f'MAE: {final_mae:.6f}')
        output(f'PSNR: {final_psnr:.2f} dB')
        output(f'Relative Error: {final_rel_error:.2f}%')
        output(f'Relative L2 Error: {final_rel_l2:.6f} ({final_rel_l2*100:.2f}%)')
        output(f'Relative L1 Error: {final_rel_l1:.6f} ({final_rel_l1*100:.2f}%)')
        
        with h5py.File(data_save_path, 'w') as hf:
            # Save both padded and unpadded versions
            hf.create_dataset('input_padded', data=X_test[:100])
            hf.create_dataset('output_padded', data=Y_test[:100])
            hf.create_dataset('pred_padded', data=sample_preds.numpy())
            
            # Save unpadded versions (original size)
            hf.create_dataset('input', data=sample_inputs_unpadded)
            hf.create_dataset('output', data=sample_targets_unpadded)
            hf.create_dataset('pred', data=sample_preds_unpadded)
            
            # Save metadata
            hf.attrs['original_size'] = original_size
            hf.attrs['padded_size'] = padded_size
            hf.attrs['pad_width'] = pad_width
            
            # Save final metrics
            hf.attrs['final_mse'] = final_mse
            hf.attrs['final_mae'] = final_mae
            hf.attrs['final_psnr'] = final_psnr
            hf.attrs['final_rel_error'] = final_rel_error
            hf.attrs['final_rel_l2'] = final_rel_l2
            hf.attrs['final_rel_l1'] = final_rel_l1
            hf.attrs['training_loss'] = 'MSE'
        
        output(f'Saved sample predictions to: {data_save_path}')
        output(f'Saved both padded ({padded_size}) and unpadded ({original_size}) versions')
    
    # Save error history
    if hasattr(callback, 'err_history') and callback.err_history:
        err_filename = log_filename.replace('.txt', '_err.txt')
        with open(err_filename, 'w') as err_file:
            for epoch, train_err, test_err in callback.err_history:
                err_file.write(f'{epoch}\t{train_err:.6e}\t{test_err:.6e}\n')
        output(f'Saved error history to: {err_filename}')
    
    output('Training completed!')
    log_file.close()


if __name__ == '__main__':
    main()
