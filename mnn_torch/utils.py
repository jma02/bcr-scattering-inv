# coding=utf-8
# vim: sw=4 et tw=100
"""
PyTorch utility functions for generating MNN models

Converted from Keras implementation to PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Sequence
from .layers import (CNNR1D, CNNK1D, CNNI1D, LCR1D, LCK1D, LCI1D,
                     CNNR2D, CNNK2D, CNNI2D, LCR2D, LCK2D, LCI2D,
                     CNNR3D, CNNK3D, CNNI3D,
                     ReshapeT1D, ReshapeM1D,
                     ReshapeT2D, ReshapeM2D,
                     ReshapeT3D, ReshapeM3D)


def MNNHmodel(input_shape: Tuple[int, ...],
              Dim: int,
              L: int, 
              n_cnn: int,
              alpha: int,
              alpha_out: Optional[int] = None,
              w_b: Tuple[int, int, int] = (3, 5, 7),
              activation: str = 'relu',
              layer: str = 'CNN',
              bc_padding: str = 'period') -> nn.Module:
    """Return a MNN-H model in PyTorch.
    
    Args:
        input_shape: Shape of input tensor (without batch dimension)
        Dim: Spatial dimension (1, 2, or 3)
        L: Number of levels, input size must satisfy Nx[d] % 2**L == 0
        n_cnn: Number of CNN/LC layers in the kernel part
        alpha: Number of filters
        alpha_out: Number of output filters (default: 1)
        w_b: Tuple of kernel sizes for different parts
        activation: Activation function for nonlinear part
        layer: 'CNN' or 'LC' for layer type
        bc_padding: Boundary condition padding type
        
    Returns:
        PyTorch model implementing MNN-H architecture
        
    Example:
        >>> from mnn_torch.utils import MNNHmodel
        >>> model = MNNHmodel((320, 1), Dim=1, L=6, n_cnn=5, alpha=6)
    """
    if layer.upper() in ('CNN', 'CONV'):
        if Dim not in (1, 2, 3):
            raise ValueError('For CNN, dimension must be 1, 2 or 3')
    elif layer.upper() == 'LC':
        if Dim not in (1, 2):
            raise ValueError('For LC, dimension must be 1 or 2')
    else:
        raise ValueError('layer can be either "CNN/Conv" or "LC"')
    
    if len(w_b) < 3:
        raise ValueError('w_b must have at least 3 elements')
    
    if alpha_out is None:
        alpha_out = 1
    
    # Get layer classes based on dimension and type
    CR = globals()[f'{layer.upper()}R{Dim}D']
    CK = globals()[f'{layer.upper()}K{Dim}D'] 
    CI = globals()[f'{layer.upper()}I{Dim}D']
    ReshapeT = globals()[f'ReshapeT{Dim}D']
    ReshapeM = globals()[f'ReshapeM{Dim}D']
    
    # Calculate dimensions
    if Dim == 1:
        Nx = (input_shape[0],)
    elif Dim == 2:
        Nx = input_shape[:2]
    elif Dim == 3:
        Nx = input_shape[:3]
    
    m = tuple(n // (2**L) for n in Nx)
    m_total = int(np.prod(m)) * alpha_out
    
    w_b_ad = (w_b[0],) * Dim
    w_b_2 = (w_b[1],) * Dim  
    w_b_l = (w_b[2],) * Dim
    
    class MNNHNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.Dim = Dim
            self.L = L
            self.n_cnn = n_cnn
            self.alpha = alpha
            self.alpha_out = alpha_out
            self.m = m
            self.m_total = m_total
            self.Nx = Nx
            
            # Adjacent part layers
            self.reshape_t_ad = ReshapeT(m)
            self.cnn_ad_layers = nn.ModuleList()
            for i in range(n_cnn - 1):
                self.cnn_ad_layers.append(
                    CK(m_total, w_b_ad[0], activation=activation, bc_padding=bc_padding)
                )
            self.cnn_ad_final = CK(m_total, w_b_ad[0], activation='linear', bc_padding=bc_padding)
            self.reshape_m_ad = ReshapeM(m)
            
            # Far field part layers for each level
            self.far_field_layers = nn.ModuleDict()
            for k in range(2, L + 1):
                w = tuple(n * 2**(L-k) for n in m)
                wk = w_b_2[0] if k == 2 else w_b_l[0]
                w_total = int(np.prod(w)) * alpha_out
                
                # Restriction layer
                cr_layer = CR(alpha, w, activation='linear')
                
                # CNN layers for this level
                ck_layers = nn.ModuleList()
                for i in range(n_cnn):
                    ck_layers.append(
                        CK(alpha, wk, activation=activation, bc_padding=bc_padding)
                    )
                
                # Interpolation layer
                if Dim == 1:
                    ci_layer = CI(w_total, Nout=Nx[0], activation='linear')
                elif Dim == 2:
                    ci_layer = CI(w_total, Nout=Nx, activation='linear')
                else:  # Dim == 3
                    ci_layer = CI(w_total, Nout=Nx, activation='linear')
                
                self.far_field_layers[f'level_{k}'] = nn.ModuleDict({
                    'cr': cr_layer,
                    'ck': ck_layers, 
                    'ci': ci_layer
                })
        
        def forward(self, x):
            # Ensure input has channel dimension
            if len(x.shape) == self.Dim + 1:  # No channel dim
                x = x.unsqueeze(-1)
            elif len(x.shape) != self.Dim + 2:  # Wrong dimensions
                raise ValueError(f'Input must have {self.Dim + 1} or {self.Dim + 2} dimensions')
            
            u_list = []
            
            # Adjacent part
            u_ad = self.reshape_t_ad(x)
            for layer in self.cnn_ad_layers:
                u_ad = layer(u_ad)
            u_ad = self.cnn_ad_final(u_ad)
            u_ad = self.reshape_m_ad(u_ad)
            u_list.append(u_ad)
            
            # Far field part
            for k in range(2, self.L + 1):
                level_layers = self.far_field_layers[f'level_{k}']
                
                # Restriction
                Vv = level_layers['cr'](x)
                
                # CNN processing
                MVv = Vv
                for ck_layer in level_layers['ck']:
                    MVv = ck_layer(MVv)
                
                # Interpolation
                u_l = level_layers['ci'](MVv)
                u_list.append(u_l)
            
            # Combine all components
            result = u_list[0]
            for u in u_list[1:]:
                result = result + u
            
            return result
    
    return MNNHNet()


def MNNHmodel1D(*args, **kwargs) -> nn.Module:
    """1D version of MNNHmodel."""
    return MNNHmodel(*args, Dim=1, **kwargs)


def MNNHmodel2D(*args, **kwargs) -> nn.Module:
    """2D version of MNNHmodel.""" 
    return MNNHmodel(*args, Dim=2, **kwargs)


def MNNHmodel3D(*args, **kwargs) -> nn.Module:
    """3D version of MNNHmodel."""
    return MNNHmodel(*args, Dim=3, **kwargs)


def MNNH2model(*args, **kwargs) -> nn.Module:
    """MNNH2 model - simplified version of MNNHmodel."""
    # This is a placeholder - implement if needed for your specific use case
    return MNNHmodel(*args, **kwargs)


def MNNH2model1D(*args, **kwargs) -> nn.Module:
    """1D version of MNNH2model."""
    return MNNH2model(*args, Dim=1, **kwargs)


def MNNH2model2D(*args, **kwargs) -> nn.Module:
    """2D version of MNNH2model."""
    return MNNH2model(*args, Dim=2, **kwargs)


def MNNH2model3D(*args, **kwargs) -> nn.Module:
    """3D version of MNNH2model."""
    return MNNH2model(*args, Dim=3, **kwargs)
