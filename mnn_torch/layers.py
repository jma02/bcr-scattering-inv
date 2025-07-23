# coding=utf-8
# vim: sw=4 et tw=100
"""
PyTorch Layers for Multiscale Neural Network (MNN)

Converted from Keras implementation to modern PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Callable
from .backend import (_convert2tuple, _convert2tuple_of_tuple, 
                     _PeriodPadding1D, _PeriodPadding2D, _PeriodPadding3D,
                     _reshapeM2D, _reshapeT2D, _reshapeM3D, _reshapeT3D)


class PeriodPadding1D(nn.Module):
    """Period-padding layer for 1D input.
    
    Args:
        padding: int or tuple of ints for (left, right) padding
        
    Input shape: 3D tensor with shape (batch_size, length, features)
    Output shape: 3D tensor with shape (batch_size, length+padding, features)
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        self.padding = _convert2tuple(padding, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) >= max(self.padding[0], self.padding[1])
        return _PeriodPadding1D(x, self.padding)


class PeriodPadding2D(nn.Module):
    """Period-padding layer for 2D input."""
    
    def __init__(self, padding: Union[int, Tuple]):
        super().__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _PeriodPadding2D(x, self.padding)


class PeriodPadding3D(nn.Module):
    """Period-padding layer for 3D input."""
    
    def __init__(self, padding: Union[int, Tuple]):
        super().__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _PeriodPadding3D(x, self.padding)


class ReshapeM1D(nn.Module):
    """Reshape a tensor to matrix by blocks.
    
    Args:
        w: Window size
        
    Input shape: 3D tensor with shape (batch_size, length, features)
    Output shape: 3D tensor with shape (batch_size, length*w, features//w)
    """
    
    def __init__(self, w: Union[int, Tuple[int]]):
        super().__init__()
        self.w = w[0] if isinstance(w, tuple) else w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, features = x.shape
        assert features % self.w == 0
        return x.view(batch, length * self.w, features // self.w)


class ReshapeT1D(nn.Module):
    """Reshape a matrix to tensor by blocks, inverse of ReshapeM1D.
    
    Args:
        w: Window size
        
    Input shape: 3D tensor with shape (batch_size, length, features)
    Output shape: 3D tensor with shape (batch_size, length//w, features*w)
    """
    
    def __init__(self, w: Union[int, Tuple[int]]):
        super().__init__()
        self.w = w[0] if isinstance(w, tuple) else w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, features = x.shape
        assert length % self.w == 0
        return x.view(batch, length // self.w, features * self.w)


class ReshapeM2D(nn.Module):
    """Reshape tensor to matrix by blocks in 2D."""
    
    def __init__(self, w: Union[int, Tuple[int, int]]):
        super().__init__()
        self.w = w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _reshapeM2D(x, self.w)


class ReshapeT2D(nn.Module):
    """Reshape matrix to tensor by blocks in 2D."""
    
    def __init__(self, w: Union[int, Tuple[int, int]]):
        super().__init__()
        self.w = w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _reshapeT2D(x, self.w)


class ReshapeM3D(nn.Module):
    """Reshape tensor to matrix by blocks in 3D."""
    
    def __init__(self, w: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.w = w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _reshapeM3D(x, self.w)


class ReshapeT3D(nn.Module):
    """Reshape matrix to tensor by blocks in 3D."""
    
    def __init__(self, w: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.w = w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _reshapeT3D(x, self.w)


class CNNR1D(nn.Module):
    """Restriction operator implemented by Conv1D with stride = kernel_size.
    
    Restricts a vector with size Nx to Nx//kernel_size.
    """
    
    def __init__(self, 
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 activation: Optional[str] = 'linear',
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,  # Will be adjusted in forward
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias
        )
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: Optional[str]) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear' or activation is None:
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        # Convert to (batch, channels, length) for conv1d
        x = x.transpose(1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), self.conv.kernel_size[0], 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, length, channels)
        return out.transpose(1, 2)


class CNNK1D(nn.Module):
    """Multiplication of a block band matrix with a vector.
    
    Implemented by padding and Conv1D with stride=1.
    
    Args:
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (should be odd)
        bc_padding: Boundary condition padding ('period' or 'zero')
        activation: Activation function
        bias: Whether to use bias
    """
    
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 bc_padding: str = 'period',
                 dilation: int = 1,
                 activation: Optional[str] = None,
                 bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.bc_padding = bc_padding
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias
        )
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: Optional[str]) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear' or activation is None:
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        
        # Apply padding
        if self.bc_padding == 'period':
            x = _PeriodPadding1D(x, self.padding_size)
        elif self.bc_padding == 'zero':
            x = F.pad(x, (0, 0, self.padding_size, self.padding_size))
        else:
            raise ValueError('Only "period" and "zero" padding are supported')
        
        # Convert to (batch, channels, length) for conv1d
        x = x.transpose(1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), self.conv.kernel_size[0], 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, length, channels)
        return out.transpose(1, 2)


class CNNI1D(nn.Module):
    """Interpolation solution from coarse grid to fine grid.
    
    Implemented by Conv1D with kernel_size=1 and stride=1.
    If Nout is given, it reshapes the output.
    """
    
    def __init__(self,
                 out_channels: int,
                 Nout: Optional[int] = None,
                 activation: str = 'linear',
                 bias: bool = True):
        super().__init__()
        self.Nout = Nout
        self.conv = nn.Conv1d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        
        # Convert to (batch, channels, length) for conv1d
        x = x.transpose(1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), 1, 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, length, channels)
        out = out.transpose(1, 2)
        
        # Reshape if Nout is specified
        if self.Nout is not None:
            batch, length, channels = out.shape
            assert length * channels % self.Nout == 0
            new_channels = length * channels // self.Nout
            out = out.reshape(batch, self.Nout, new_channels)
        
        return out


class WaveLetC1D(nn.Module):
    """Wavelet transformation implemented by Conv1D with stride=2.
    
    Args:
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (should be even)
        bc_padding: Boundary condition padding ('period' or 'zero')
        activation: Activation function
        bias: Whether to use bias
    """
    
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 activation: str = 'linear',
                 bc_padding: str = 'period',
                 bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 0, "Kernel size must be even"
        self.bc_padding = bc_padding
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2 - 1 if kernel_size > 2 else 0
        
        self.conv = nn.Conv1d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
            dilation=dilation,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        
        # Apply padding if kernel size > 2
        if self.kernel_size > 2:
            if self.bc_padding == 'period':
                x = _PeriodPadding1D(x, self.padding_size)
            elif self.bc_padding == 'zero':
                x = F.pad(x, (0, 0, self.padding_size, self.padding_size))
            else:
                raise ValueError('Only "period" and "zero" padding are supported')
        
        # Convert to (batch, channels, length) for conv1d
        x = x.transpose(1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), self.conv.kernel_size[0], 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, length, channels)
        return out.transpose(1, 2)


class InvWaveLetC1D(nn.Module):
    """Inverse wavelet transformation implemented by Conv1D with stride=1.
    
    If Nout is given, it reshapes the output.
    
    Args:
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (should be odd)
        Nout: Optional output size for reshaping
        bc_padding: Boundary condition padding ('period' or 'zero')
        activation: Activation function
        bias: Whether to use bias
    """
    
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 Nout: Optional[int] = None,
                 dilation: int = 1,
                 activation: str = 'linear',
                 bc_padding: str = 'period',
                 bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.bc_padding = bc_padding
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2
        self.Nout = Nout
        
        self.conv = nn.Conv1d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, length, channels)
        
        # Apply padding
        if self.bc_padding == 'period':
            x = _PeriodPadding1D(x, self.padding_size)
        elif self.bc_padding == 'zero':
            x = F.pad(x, (0, 0, self.padding_size, self.padding_size))
        else:
            raise ValueError('Only "period" and "zero" padding are supported')
        
        # Convert to (batch, channels, length) for conv1d
        x = x.transpose(1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), self.conv.kernel_size[0], 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, length, channels)
        out = out.transpose(1, 2)
        
        # Reshape if Nout is specified
        if self.Nout is not None:
            batch, length, channels = out.shape
            assert length * channels % self.Nout == 0
            new_channels = length * channels // self.Nout
            out = out.reshape(batch, self.Nout, new_channels)
        
        return out


# 2D Layers
class CNNR2D(nn.Module):
    """Restriction operator for 2D inputs."""
    
    def __init__(self, 
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 activation: str = 'linear',
                 bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.conv = nn.Conv2d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, height, width, channels)
        # Convert to (batch, channels, height, width) for conv2d
        x = x.permute(0, 3, 1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                kh, kw = self.conv.kernel_size
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), kh, kw, 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, height, width, channels)
        return out.permute(0, 2, 3, 1)


class CNNK2D(nn.Module):
    """2D convolution with boundary condition padding."""
    
    def __init__(self,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 bc_padding: str = 'period',
                 activation: Optional[str] = None,
                 bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        assert all(k % 2 == 1 for k in kernel_size), "Kernel sizes must be odd"
        self.bc_padding = bc_padding
        self.kernel_size = kernel_size
        self.padding_size = tuple(k // 2 for k in kernel_size)
        
        self.conv = nn.Conv2d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: Optional[str]) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear' or activation is None:
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, height, width, channels)
        
        # Apply padding
        if self.bc_padding == 'period':
            x = _PeriodPadding2D(x, self.padding_size)
        elif self.bc_padding == 'zero':
            pad_h, pad_w = self.padding_size
            x = F.pad(x, (0, 0, pad_w, pad_w, pad_h, pad_h))
        else:
            raise ValueError('Only "period" and "zero" padding are supported')
        
        # Convert to (batch, channels, height, width) for conv2d
        x = x.permute(0, 3, 1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                kh, kw = self.conv.kernel_size
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), kh, kw, 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, height, width, channels)
        return out.permute(0, 2, 3, 1)


class CNNI2D(nn.Module):
    """2D interpolation layer."""
    
    def __init__(self,
                 out_channels: int,
                 Nout: Optional[Tuple[int, int]] = None,
                 activation: str = 'linear',
                 bias: bool = True):
        super().__init__()
        self.Nout = Nout
        self.conv = nn.Conv2d(
            in_channels=1,  # Will be set dynamically
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[Callable]:
        if activation == 'relu':
            return F.relu
        elif activation == 'linear':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, height, width, channels)
        
        # Convert to (batch, channels, height, width) for conv2d
        x = x.permute(0, 3, 1, 2)
        
        # Update conv layer input channels if needed
        if self.conv.in_channels != x.size(1):
            self.conv.in_channels = x.size(1)
            # Reinitialize weight with correct input channels on the same device
            device = x.device
            with torch.no_grad():
                self.conv.weight = nn.Parameter(
                    torch.randn(self.conv.out_channels, x.size(1), 1, 1, 
                               device=device, dtype=x.dtype) * 0.1
                )
        
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        
        # Convert back to (batch, height, width, channels)
        out = out.permute(0, 2, 3, 1)
        
        # Reshape if Nout is specified
        if self.Nout is not None:
            batch, height, width, channels = out.shape
            Nout_h, Nout_w = self.Nout
            assert height * width * channels % (Nout_h * Nout_w) == 0
            new_channels = height * width * channels // (Nout_h * Nout_w)
            out = out.view(batch, Nout_h, Nout_w, new_channels)
        
        return out


# Placeholder classes for locally connected layers (can be implemented if needed)
class LCR1D(nn.Module):
    """Locally connected restriction for 1D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")

class LCK1D(nn.Module):
    """Locally connected kernel for 1D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")

class LCI1D(nn.Module):
    """Locally connected interpolation for 1D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")

class LCR2D(nn.Module):
    """Locally connected restriction for 2D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")

class LCK2D(nn.Module):
    """Locally connected kernel for 2D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")

class LCI2D(nn.Module):
    """Locally connected interpolation for 2D - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Locally connected layers not yet implemented")


# 3D Layers (placeholders - can be implemented following same pattern)
class CNNR3D(nn.Module):
    """3D restriction - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("3D layers not yet implemented")

class CNNK3D(nn.Module):
    """3D convolution - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("3D layers not yet implemented")

class CNNI3D(nn.Module):
    """3D interpolation - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("3D layers not yet implemented")

class WaveLetC2D(nn.Module):
    """2D wavelet - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("2D/3D wavelet layers not yet implemented")

class WaveLetC3D(nn.Module):
    """3D wavelet - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("2D/3D wavelet layers not yet implemented")

class InvWaveLetC2D(nn.Module):
    """2D inverse wavelet - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("2D/3D wavelet layers not yet implemented")

class InvWaveLetC3D(nn.Module):
    """3D inverse wavelet - placeholder for future implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("2D/3D wavelet layers not yet implemented")
