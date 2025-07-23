# coding=utf-8
# vim: sw=4 et tw=100
""" Functions to generate mnn models"""

import numpy as np
from keras.layers import Add, Reshape
from keras import backend as K
from .layers import CNNR1D, CNNK1D, CNNI1D, LCR1D, LCK1D, LCI1D
from .layers import CNNR2D, CNNK2D, CNNI2D, LCR2D, LCK2D, LCI2D
from .layers import CNNR3D, CNNK3D, CNNI3D
from .layers import ReshapeT1D, ReshapeM1D
from .layers import ReshapeT2D, ReshapeM2D
from .layers import ReshapeT3D, ReshapeM3D

def MNNHmodel(Ipt, Dim, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
              activation='relu', layer='CNN', bc_padding='period'):
    """ Return Opt = MNN-H(Ipt)
    # Arguments:
        Dim: dimension
        Ipt: a (Dim+2)-tensor with shape (batch_size,) + Nx (1,)
        L: integer and Nx[d] % 2**L == 0, d=0,...,Dim-1
        n_cnn: integer, number of CNN/LC layers in the kernel part
        alpha: number of filters
        activation: for the nonlinear part
        layer: CNN / LC used in MNN-H
        bc_padding: padding on the boundary condition

    # Example:
    ```python
        >>> from .utils import MNNHmodel
        >>> from keras.layers import Input
        >>> from keras.models import Model
        >>> Nx = 320
        >>> Ipt = Input((Nx, 1))
        >>> Opt = MNNHmodel(Ipt, Dim=1, 6, 5, 6)
        >>> model = Model(Ipt, Opt)
    ```
    """
    if layer == 'CNN' or layer == 'Conv':  # {
        if Dim not in (1, 2, 3):  # {
            raise ImportError('For CNN, dimension must be 1, 2 or 3')
        # }
    elif layer == 'LC':
        if Dim not in (1, 2):  # {
            raise ImportError('For LC, dimension must be 1 or 2')
        # }
    else:
        raise ImportError('layer can be either "CNN/Conv" or "LC"')
    # }
    assert isinstance(w_b, tuple)
    assert len(w_b) >= 3
    if alpha_out is not None:
        assert alpha_out > 0

    al_out = alpha_out or 1

    CR = eval(layer+'R'+str(Dim)+'D')
    CK = eval(layer+'K'+str(Dim)+'D')
    CI = eval(layer+'I'+str(Dim)+'D')
    ReshapeT = eval('ReshapeT'+str(Dim)+'D')
    ReshapeM = eval('ReshapeM'+str(Dim)+'D')

    if K.ndim(Ipt) == Dim + 1:
        sp = K.int_shape(Ipt)
        Ipt = Reshape(sp[1:] + (1,))(Ipt)
    elif K.ndim(Ipt) != Dim + 2:
        raise ImportError('Dimension of the Input layer must be Dim+1 or Dim+2')

    w_b_ad = (w_b[0],) * Dim
    w_b_2 = (w_b[1],) * Dim
    w_b_l = (w_b[2],) * Dim
    n_input = K.int_shape(Ipt)
    assert n_input[-1] == 1
    Nx = n_input[1:(Dim+1)]
    m = tuple(n // (2**L) for n in Nx)
    m_total = np.prod(m) * al_out

    u_list = []  # list of u_l and u_ad
    # === adjacent part
    u_ad = ReshapeT(m)(Ipt)
    for i in range(0, n_cnn-1):
        u_ad = CK(m_total, w_b_ad, activation=activation, bc_padding=bc_padding)(u_ad)

    u_ad = CK(m_total, w_b_ad, activation='linear', bc_padding=bc_padding)(u_ad)
    u_ad = ReshapeM(m)(u_ad)
    u_list.append(u_ad)

    # === far field part
    for k in range(2, L+1):
        w = tuple(n * 2**(L-k) for n in m)
        wk = w_b_2 if k == 2 else w_b_l
        Vv = CR(alpha, w, activation='linear')(Ipt)
        MVv = Vv
        for i in range(0, n_cnn):
            MVv = CK(alpha, wk, activation=activation, bc_padding=bc_padding)(MVv)

        w_total = int(np.prod(w)) * al_out
        u_l = CI(w_total, Nx, activation='linear')(MVv)
        u_list.append(u_l)

    Opt = Add()(u_list)
    if alpha_out is None:
        Opt = Reshape(Nx)(Opt)

    return Opt


def MNNH2model(Ipt, Dim, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
              activation='relu', layer='CNN', bc_padding='period'):
    """ Return Opt = MNN-H2(Ipt)
    # Arguments:
        Dim: dimension
        Ipt: a (Dim+2)-tensor with shape (batch_size,) + Nx (1,)
        L: integer and Nx[d] % 2**L == 0, d=0,...,Dim-1
        n_cnn: integer, number of CNN/LC layers in the kernel part
        alpha: number of filters
        activation: for the nonlinear part
        layer: CNN / LC used in MNN-H2
        bc_padding: padding on the boundary condition

    # Example:
    ```python
        >>> from .utils import MNNH2model
        >>> from keras.layers import Input
        >>> from keras.models import Model
        >>> Nx = 320
        >>> Ipt = Input((Nx, 1))
        >>> Opt = MNNH2model(Ipt, 1, 6, 5, 6)
        >>> model = Model(Ipt, Opt)
    ```
    """
    if layer == 'CNN':  # {
        if Dim not in (1, 2, 3):  # {
            raise ImportError('For CNN, dimension must be 1, 2 or 3')
        # }
    elif layer == 'LC':
        if Dim not in (1, 2):  # {
            raise ImportError('For LC, dimension must be 1 or 2')
        # }
    else:
        raise ImportError('layer can be either "CNN" or "LC"')
    # }
    assert isinstance(w_b, tuple)
    assert len(w_b) >= 3
    if alpha_out is not None:
        assert alpha_out > 0

    al_out = alpha_out or 1

    CR = eval(layer+'R'+str(Dim)+'D')
    CK = eval(layer+'K'+str(Dim)+'D')
    CI = eval(layer+'I'+str(Dim)+'D')
    ReshapeT = eval('ReshapeT'+str(Dim)+'D')
    ReshapeM = eval('ReshapeM'+str(Dim)+'D')

    if K.ndim(Ipt) == Dim + 1:
        sp = K.int_shape(Ipt)
        Ipt = Reshape(sp[1:] + (1,))(Ipt)
    elif K.ndim(Ipt) != Dim + 2:
        raise ImportError('Dimension of the Input layer must be Dim+1 or Dim+2')

    w_b_ad = (w_b[0],) * Dim
    w_b_2 = (w_b[1],) * Dim
    w_b_l = (w_b[2],) * Dim
    n_input = K.int_shape(Ipt)
    Nx = n_input[1:(Dim+1)]
    m = tuple(n // (2**L) for n in Nx)
    m_total = np.prod(m) * al_out

    # === adjacent part
    uad = ReshapeT(m)(Ipt)
    for i in range(0, n_cnn-1):
        uad = CK(m_total, w_b_ad, activation=activation, bc_padding=bc_padding)(uad)

    uad = CK(m_total, w_b_ad, activation='linear', bc_padding=bc_padding)(uad)
    uad = ReshapeM(m)(uad)

    # === far field part
    Vv_list = []
    Vv = CR(alpha, m, activation='linear')(Ipt)
    Vv_list.insert(0, Vv)
    for ll in range(L-1, 1, -1):
        Vv = CR(alpha, (2,)*Dim, activation='linear')(Vv)
        Vv_list.insert(0, Vv)

    MVv_list = []
    for ll in range(2, L+1):
        MVv = Vv_list[ll-2]
        w = w_b_2 if ll == 2 else w_b_l
        for k in range(0, n_cnn):
            MVv = CK(alpha, w, activation=activation, bc_padding=bc_padding)(MVv)

        MVv_list.append(MVv)

    for ll in range(2, L):
        if ll == 2:
            chi = MVv_list[ll-2]
        else:
            chi = Add()([chi, MVv_list[ll-2]])

        chi = CI((2**Dim)*alpha, (2**(ll+1),)*Dim, activation='linear')(chi)

    chi = Add()([chi, MVv_list[L-2]])
    chi = CI(m_total, Nx, activation='linear')(chi)
    chi = ReshapeM((1,)*Dim)(chi)

    # === addition of far field and adjacent part
    Opt = Add()([chi, uad])
    if alpha_out is None:
        Opt = Reshape(Nx)(Opt)

    return Opt


def MNNHmodel1D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                activation='relu', layer='CNN', bc_padding='period'):
    return MNNHmodel(Ipt, 1, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                     activation=activation, layer=layer, bc_padding=bc_padding)


def MNNHmodel2D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                activation='relu', layer='CNN', bc_padding='period'):
    return MNNHmodel(Ipt, 2, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                     activation=activation, layer=layer, bc_padding=bc_padding)


def MNNHmodel3D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                activation='relu', layer='CNN', bc_padding='period'):
    return MNNHmodel(Ipt, 3, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                     activation=activation, layer=layer, bc_padding=bc_padding)


def MNNH2model1D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                 activation='relu', layer='CNN', bc_padding='period'):
    return MNNH2model(Ipt, 1, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                      activation=activation, layer=layer, bc_padding=bc_padding)


def MNNH2model2D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                 activation='relu', layer='CNN', bc_padding='period'):
    return MNNH2model(Ipt, 2, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                      activation=activation, layer=layer, bc_padding=bc_padding)


def MNNH2model3D(Ipt, L, n_cnn, alpha, alpha_out=None, w_b=(3, 5, 7),
                 activation='relu', layer='CNN', bc_padding='period'):
    return MNNH2model(Ipt, 3, L, n_cnn, alpha, alpha_out=alpha_out, w_b=w_b,
                      activation=activation, layer=layer, bc_padding=bc_padding)
