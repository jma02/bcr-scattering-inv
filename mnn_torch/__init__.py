"""PyTorch implementation of Multiscale Neural Network (MNN)

Modern PyTorch implementation of the original Keras-based MNN library
for scattering inverse problems using BCR-Net architecture.
"""

from .layers import (
    PeriodPadding1D, ReshapeM1D, ReshapeT1D,
    CNNR1D, CNNK1D, CNNI1D, LCR1D, LCK1D, LCI1D,
    PeriodPadding2D, ReshapeM2D, ReshapeT2D,
    CNNR2D, CNNK2D, CNNI2D, LCR2D, LCK2D, LCI2D,
    PeriodPadding3D, ReshapeM3D, ReshapeT3D,
    CNNR3D, CNNK3D, CNNI3D,
    WaveLetC1D, WaveLetC2D, WaveLetC3D,
    InvWaveLetC1D, InvWaveLetC2D, InvWaveLetC3D
)

from .callback import SaveBestModel

from .utils import (
    MNNHmodel, MNNHmodel1D, MNNHmodel2D, MNNHmodel3D,
    MNNH2model, MNNH2model1D, MNNH2model2D, MNNH2model3D
)

__version__ = "2.0.0"
__author__ = "PyTorch Implementation"
