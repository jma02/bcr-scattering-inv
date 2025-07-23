# coding=utf-8
# vim: sw=4 et tw=100
"""
PyTorch backend utilities for MNN

Converted from Keras backend to PyTorch operations.
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Sequence


def _convert2tuple(s: Union[int, Tuple], n: int) -> Tuple:
    """Convert int or tuple to tuple of specified length."""
    assert isinstance(n, int) and n >= 1
    if isinstance(s, int):
        return (s,) * n
    elif isinstance(s, (tuple, list)):
        s = tuple(s)
        assert len(s) <= n
        return s + (s[-1],) * (n - len(s))
    else:
        raise ValueError('Input must be an int or tuple')


def _convert2tuple_of_tuple(s: Union[int, Tuple], n1: int, n2: int) -> Tuple[Tuple]:
    """Convert input to tuple of tuples with specified dimensions.
    
    Example: s=1, n1=3, n2=2: ((1,1), (1,1), (1,1))
    """
    assert isinstance(n1, int) and isinstance(n2, int)
    assert (n1 >= 1) and (n2 >= 1)
    
    if isinstance(s, int):
        return ((s,) * n2,) * n1
    elif isinstance(s, (tuple, list)) and len(s) > 0 and isinstance(s[0], int):
        s_tmp = tuple((x,) * n2 for x in s)
        return _convert2tuple(s_tmp, n1)
    elif isinstance(s, (tuple, list)) and len(s) > 0 and isinstance(s[0], (tuple, list)):
        s = tuple(tuple(x) for x in s)
        assert all(len(x) == n2 for x in s)
        return _convert2tuple(s, n1)
    else:
        raise ValueError('Input must be an int, tuple, or tuple of tuples')


def _PeriodPadding1D(x: torch.Tensor, s: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Apply periodic padding to 1D tensor.
    
    Args:
        x: Input tensor of shape (batch, length, channels)
        s: Padding size (left, right) or single int for symmetric padding
        
    Returns:
        Padded tensor
    """
    s = _convert2tuple(s, 2)
    left_pad = x[:, -s[0]:, :] if s[0] > 0 else torch.empty(x.size(0), 0, x.size(2), device=x.device)
    right_pad = x[:, :s[1], :] if s[1] > 0 else torch.empty(x.size(0), 0, x.size(2), device=x.device)
    return torch.cat([left_pad, x, right_pad], dim=1)


def _PeriodPadding2D(x: torch.Tensor, s: Union[int, Tuple]) -> torch.Tensor:
    """Apply periodic padding to 2D tensor.
    
    Args:
        x: Input tensor of shape (batch, height, width, channels)
        s: Padding size specification
        
    Returns:
        Padded tensor
    """
    sx, sy = _convert2tuple_of_tuple(s, 2, 2)
    
    # Pad in height dimension (axis 1)
    if sx[0] > 0 or sx[1] > 0:
        top_pad = x[:, -sx[0]:, :, :] if sx[0] > 0 else torch.empty(x.size(0), 0, x.size(2), x.size(3), device=x.device)
        bottom_pad = x[:, :sx[1], :, :] if sx[1] > 0 else torch.empty(x.size(0), 0, x.size(2), x.size(3), device=x.device)
        x = torch.cat([top_pad, x, bottom_pad], dim=1)
    
    # Pad in width dimension (axis 2)
    if sy[0] > 0 or sy[1] > 0:
        left_pad = x[:, :, -sy[0]:, :] if sy[0] > 0 else torch.empty(x.size(0), x.size(1), 0, x.size(3), device=x.device)
        right_pad = x[:, :, :sy[1], :] if sy[1] > 0 else torch.empty(x.size(0), x.size(1), 0, x.size(3), device=x.device)
        x = torch.cat([left_pad, x, right_pad], dim=2)
    
    return x


def _PeriodPadding3D(x: torch.Tensor, s: Union[int, Tuple]) -> torch.Tensor:
    """Apply periodic padding to 3D tensor.
    
    Args:
        x: Input tensor of shape (batch, depth, height, width, channels)
        s: Padding size specification
        
    Returns:
        Padded tensor
    """
    sx, sy, sz = _convert2tuple_of_tuple(s, 3, 2)
    
    # Pad in depth dimension (axis 1)
    if sx[0] > 0 or sx[1] > 0:
        front_pad = x[:, -sx[0]:, :, :, :] if sx[0] > 0 else torch.empty(x.size(0), 0, x.size(2), x.size(3), x.size(4), device=x.device)
        back_pad = x[:, :sx[1], :, :, :] if sx[1] > 0 else torch.empty(x.size(0), 0, x.size(2), x.size(3), x.size(4), device=x.device)
        x = torch.cat([front_pad, x, back_pad], dim=1)
    
    # Pad in height dimension (axis 2)
    if sy[0] > 0 or sy[1] > 0:
        top_pad = x[:, :, -sy[0]:, :, :] if sy[0] > 0 else torch.empty(x.size(0), x.size(1), 0, x.size(3), x.size(4), device=x.device)
        bottom_pad = x[:, :, :sy[1], :, :] if sy[1] > 0 else torch.empty(x.size(0), x.size(1), 0, x.size(3), x.size(4), device=x.device)
        x = torch.cat([top_pad, x, bottom_pad], dim=2)
    
    # Pad in width dimension (axis 3)
    if sz[0] > 0 or sz[1] > 0:
        left_pad = x[:, :, :, -sz[0]:, :] if sz[0] > 0 else torch.empty(x.size(0), x.size(1), x.size(2), 0, x.size(4), device=x.device)
        right_pad = x[:, :, :, :sz[1], :] if sz[1] > 0 else torch.empty(x.size(0), x.size(1), x.size(2), 0, x.size(4), device=x.device)
        x = torch.cat([left_pad, x, right_pad], dim=3)
    
    return x


def _reshapeM2D(x: torch.Tensor, w: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Reshape tensor to matrix by blocks.
    
    Args:
        x: Input tensor of shape (batch, height, width, channels)
        w: Window size (wx, wy)
        
    Returns:
        Reshaped tensor
    """
    wx, wy = _convert2tuple(w, 2)
    batch, nx, ny, nw = x.shape
    nc = nw // (wx * wy)
    assert nc >= 1 and nw % (wx * wy) == 0
    
    # Reshape to (batch, nx, ny, nc, wx, wy)
    y = x.view(batch, nx, ny, nc, wx, wy)
    # Permute to (batch, nx, wx, ny, wy, nc)
    z = y.permute(0, 1, 4, 2, 5, 3)
    # Final reshape
    return z.contiguous().view(batch, wx * nx, wy * ny, nc)


def _reshapeT2D(x: torch.Tensor, w: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Reshape matrix to tensor by blocks, inverse of _reshapeM2D.
    
    Args:
        x: Input tensor of shape (batch, height, width, channels)
        w: Window size (wx, wy)
        
    Returns:
        Reshaped tensor
    """
    wx, wy = _convert2tuple(w, 2)
    batch, nx, ny, nw = x.shape
    assert nx % wx == 0 and ny % wy == 0
    
    # Reshape to (batch, nx//wx, wx, ny//wy, wy, nw)
    y = x.view(batch, nx // wx, wx, ny // wy, wy, nw)
    # Permute to (batch, nx//wx, ny//wy, wx, wy, nw)
    z = y.permute(0, 1, 3, 2, 4, 5)
    # Final reshape
    return z.contiguous().view(batch, nx // wx, ny // wy, nw * wx * wy)


def _reshapeM3D(x: torch.Tensor, w: Union[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Reshape tensor to matrix by blocks in 3D.
    
    Args:
        x: Input tensor of shape (batch, depth, height, width, channels)
        w: Window size (wx, wy, wz)
        
    Returns:
        Reshaped tensor
    """
    wx, wy, wz = _convert2tuple(w, 3)
    batch, nx, ny, nz, nw = x.shape
    assert nw % (wx * wy * wz) == 0
    nc = nw // (wx * wy * wz)
    assert nc >= 1
    
    # Reshape to (batch, nx, ny, nz, nc, wx, wy, wz)
    y = x.view(batch, nx, ny, nz, nc, wx, wy, wz)
    # Permute to (batch, nx, wx, ny, wy, nz, wz, nc)
    z = y.permute(0, 1, 5, 2, 6, 3, 7, 4)
    # Final reshape
    return z.contiguous().view(batch, wx * nx, wy * ny, wz * nz, nc)


def _reshapeT3D(x: torch.Tensor, w: Union[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Reshape matrix to tensor by blocks in 3D, inverse of _reshapeM3D.
    
    Args:
        x: Input tensor of shape (batch, depth, height, width, channels)
        w: Window size (wx, wy, wz)
        
    Returns:
        Reshaped tensor
    """
    wx, wy, wz = _convert2tuple(w, 3)
    batch, nx, ny, nz, nw = x.shape
    assert nx % wx == 0 and ny % wy == 0 and nz % wz == 0
    
    # Reshape to (batch, nx//wx, wx, ny//wy, wy, nz//wz, wz, nw)
    y = x.view(batch, nx // wx, wx, ny // wy, wy, nz // wz, wz, nw)
    # Permute to (batch, nx//wx, ny//wy, nz//wz, wx, wy, wz, nw)
    z = y.permute(0, 1, 3, 5, 2, 4, 6, 7)
    # Final reshape
    return z.contiguous().view(batch, nx // wx, ny // wy, nz // wz, nw * wx * wy * wz)
