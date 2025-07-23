# MNN-Torch: Modern PyTorch Implementation

This is a modern PyTorch implementation of the Multiscale Neural Network (MNN) library, specifically designed for scattering inverse problems using the BCR-Net architecture.

## Overview

This implementation converts the original Keras-based MNN library to modern PyTorch, providing:

- **Modern PyTorch architecture** with proper `nn.Module` design
- **Type hints** for better code maintainability
- **Efficient data loading** with PyTorch `DataLoader`
- **Advanced training features** including:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Mixed precision training support (ready)
- **GPU acceleration** with automatic device detection
- **Better error handling** and logging

## Structure

```
mnn_torch/
├── __init__.py          # Package initialization
├── backend.py           # Backend utilities (padding, reshaping)
├── layers.py           # Custom PyTorch layers (CNN, Wavelet, etc.)
├── callback.py         # Training callbacks and utilities
└── utils.py            # Model building utilities
```

## Key Components

### Layers (`layers.py`)
- **CNNK1D/2D**: Convolution with boundary condition padding
- **CNNR1D/2D**: Restriction operators
- **CNNI1D/2D**: Interpolation operators  
- **WaveLetC1D**: Forward wavelet transform
- **InvWaveLetC1D**: Inverse wavelet transform
- **PeriodPadding1D/2D/3D**: Periodic boundary padding
- **ReshapeM/T**: Matrix/tensor reshaping operations

### Training Utilities (`callback.py`)
- **SaveBestModel**: Advanced model checkpointing with early stopping
- **train_model**: Complete training loop with callbacks
- Learning rate scheduling and reduction on plateau

### Architecture (`scattering_inv_torch.py`)
- **BCRNet**: Complete PyTorch implementation of the BCR-Net architecture
- **ScatteringDataset**: Efficient data loading for scattering problems
- Modern training pipeline with GPU support

## Usage

### Basic Training
```bash
python scattering_inv_torch.py --epoch 100 --alpha 40 --batch-size 32 --lr 0.001
```

### Advanced Options
```bash
python scattering_inv_torch.py \
    --epoch 200 \
    --alpha 64 \
    --n-cnn 8 \
    --batch-size 64 \
    --lr 0.001 \
    --noise 5.0 \
    --device cuda \
    --data-path data \
    --log-path logs_torch
```

### Key Arguments
- `--epoch`: Number of training epochs
- `--alpha`: Number of channels for depth
- `--n-cnn`: Number of CNN layers in the core
- `--n-cnn3`: Number of 2D refinement layers
- `--batch-size`: Training batch size
- `--lr`: Learning rate
- `--noise`: Noise level (percentage)
- `--device`: Device to use ('cuda' or 'cpu')
- `--data-path`: Path to training data
- `--log-path`: Path for output logs and models

## Installation

1. Install PyTorch (visit pytorch.org for your specific setup)
2. Install other dependencies:
```bash
pip install -r requirements_torch.txt
```

## Key Improvements over Original Keras Version

### Modern PyTorch Features
- **Dynamic input channel handling**: Layers automatically adapt to input dimensions
- **Proper GPU memory management**: Efficient tensor operations
- **Mixed precision ready**: Easy to add AMP for faster training
- **Better numerical stability**: Modern initialization and regularization

### Training Improvements
- **Advanced callbacks**: Better model saving and early stopping
- **Learning rate scheduling**: Automatic LR reduction on plateau  
- **Progress monitoring**: Detailed logging and error tracking
- **Checkpointing**: Robust model saving and loading
- **Data pipeline**: Efficient data loading with multiple workers

### Code Quality
- **Type hints**: Better IDE support and error catching
- **Documentation**: Comprehensive docstrings
- **Error handling**: Robust error reporting
- **Modular design**: Clean separation of concerns

## Architecture Details

The BCR-Net architecture implements a multiscale approach:

1. **Input Processing**: Initial convolution and wavelet decomposition
2. **Multiscale Processing**: 
   - Coarse scale processing with CNN layers
   - Detail processing at each wavelet level
3. **Reconstruction**: 
   - Inverse wavelet transforms
   - 2D refinement layers
   - Residual connections

This design is particularly effective for scattering inverse problems where the solution requires capturing both coarse-scale structures and fine-scale details.

## Performance Considerations

- **GPU Utilization**: Optimized for modern GPUs with efficient memory usage
- **Batch Processing**: Vectorized operations for better throughput
- **Data Loading**: Multi-threaded data loading with pin memory
- **Memory Efficiency**: Gradient checkpointing ready for large models

## Migration from Keras Version

The PyTorch implementation maintains compatibility with the original Keras model while providing modern features:

- **Same architecture**: Identical mathematical operations
- **Compatible data format**: Works with existing HDF5 datasets
- **Similar API**: Familiar argument structure
- **Enhanced functionality**: Additional features without breaking changes

## Contributing

When extending this implementation:

1. **Follow PyTorch conventions**: Use `nn.Module` for all layers
2. **Add type hints**: Maintain type safety
3. **Document thoroughly**: Include docstrings and comments
4. **Test thoroughly**: Verify mathematical equivalence with original
5. **Consider efficiency**: Optimize for modern GPU architectures
