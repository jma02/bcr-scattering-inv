# coding=utf-8
# vim: sw=4 et tw=100
"""
backend of mnn

written by Yuwei Fan (ywfan@stanford.edu)
"""
from keras import backend as K

def _convert2tuple(s, n):
    assert isinstance(n, int)
    assert n >= 1
    if isinstance(s, int):
        return (s,) * n
    elif isinstance(s, tuple):
        assert len(s) <= n
        return s + (s[-1],) * (n-len(s))
    else:
        raise ImportError('input must be an int or tuple')


def _convert2tuple_of_tuple(s, n1, n2):
    """ s = 1, n1 = 3, n2 = 2: ((1,1), (1,1), (1,1))"""
    assert isinstance(n1, int)
    assert isinstance(n2, int)
    assert (n1 >= 1) & (n2 >= 1)
    if isinstance(s, int):
        return ((s,)*n2,) * n1
    elif isinstance(s[0], int):
        assert all(isinstance(x, int) for x in s)
        s_tmp = tuple((x,)*n2 for x in s)
        return _convert2tuple(s_tmp, n1)
    elif isinstance(s[0], tuple):
        assert all(isinstance(x, tuple) for x in s)
        assert all(len(x) == n2 for x in s)
        return _convert2tuple(s, n1)
    else:
        raise ImportError('input must be an int or tuple, or tuple of tuple')


def _PeriodPadding1D(x, s):
    s = _convert2tuple(s, 2)
    return K.concatenate([x[:, x.shape[1]-s[0]:x.shape[1], :], x, x[:, 0:s[1], :]], axis=1)


def _PeriodPadding2D(x, s):
    sx, sy = _convert2tuple_of_tuple(s, 2, 2)
    nx = x.shape[1]
    ny = x.shape[2]
    # x direction
    y = K.concatenate([x[:, nx-sx[0]:nx, :, :], x, x[:, 0:sx[1], :, :]], axis=1)
    # y direction
    z = K.concatenate([y[:, :, ny-sy[0]:ny, :], y, y[:, :, 0:sy[1], :]], axis=2)
    return z


def _reshapeM2D(x, w):
    wx, wy = _convert2tuple(w, 2)
    nx, ny, nw = x.shape[1:4]
    nc = nw // (wx*wy)
    assert nc >= 1
    assert nw % (wx*wy) == 0
    y = K.reshape(x, (-1, nx, ny, nc, wx, wy))
    z = K.permute_dimensions(y, (0, 1, 4, 2, 5, 3))
    return K.reshape(z, (-1, wx*nx, wy*ny, nc))


def _reshapeT2D(x, w):
    wx, wy = _convert2tuple(w, 2)
    nx, ny, nw = x.shape[1:4]
    assert nx % wx == 0
    assert ny % wy == 0
    y = K.reshape(x, (-1, nx//wx, wx, ny//wy, wy, nw))
    z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
    return K.reshape(z, (-1, nx//wx, ny//wy, nw * wx * wy))


def _PeriodPadding3D(x, s):
    sx, sy, sz = _convert2tuple_of_tuple(s, 3, 2)
    nx, ny, nz = x.shape[1:4]
    # x direction
    y = K.concatenate([x[:, nx-sx[0]:nx, :, :, :], x, x[:, 0:sx[1], :, :, :]], axis=1)
    # y direction
    z = K.concatenate([y[:, :, ny-sy[0]:ny, :, :], y, y[:, :, 0:sy[1], :, :]], axis=2)
    # z direction
    w = K.concatenate([z[:, :, :, nz-sz[0]:nz, :], z, z[:, :, :, 0:sz[1], :]], axis=3)
    return w


def _reshapeM3D(x, w):
    wx, wy, wz = _convert2tuple(w, 3)
    nx, ny, nz, nw = x.shape[1:5]
    assert nw % (wx*wy*wz) == 0
    nc = nw // (wx*wy*wz)
    assert nc >= 1
    y = K.reshape(x, (-1, nx, ny, nz, nc, wx, wy, wz))
    z = K.permute_dimensions(y, (0, 1, 5, 2, 6, 3, 7, 4))
    return K.reshape(z, (-1, wx*nx, wy*ny, wz*nz, nc))


def _reshapeT3D(x, w):
    wx, wy, wz = _convert2tuple(w, 3)
    nx, ny, nz, nw = x.shape[1:5]
    assert (nx % wx, ny % wy, nz % wz) == (0, 0, 0)
    y = K.reshape(x, (-1, nx//wx, wx, ny//wy, wy, nz//wz, wz, nw))
    z = K.permute_dimensions(y, (0, 1, 3, 5, 2, 4, 6, 7))
    return K.reshape(z, (-1, nx//wx, ny//wy, nz//wz, nw * wx * wy * wz))
