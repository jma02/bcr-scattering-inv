# coding=utf-8
# vim: sw=4 et tw=100
""" Layers and function for MNN.

written by Yuwei Fan (ywfan@stanford.edu)
"""

from keras.engine.topology import Layer
from keras.layers import Conv1D, LocallyConnected1D, ZeroPadding1D
from keras.layers import Conv2D, LocallyConnected2D, ZeroPadding2D
from keras.layers import Conv3D, ZeroPadding3D
from keras import backend as K
from .backend import _PeriodPadding1D, _PeriodPadding2D, _reshapeM2D, _reshapeT2D
from .backend import _PeriodPadding3D, _reshapeM3D, _reshapeT3D
from .backend import _convert2tuple, _convert2tuple_of_tuple

class PeriodPadding1D(Layer):
    """Period-padding layer for 1D input

    # Arguments
        padding: int

    # Input shape: 3D tensor with shape `(batch_size, Nx, features)`

    # Output shape: 3D tensor with shape `(batch_size, Nx+2*size, features)`
    """

    def __init__(self, size, **kwargs):
        self.size = _convert2tuple(size, 2)
        super(PeriodPadding1D, self).__init__(**kwargs)

    def call(self, x):
        assert x.shape[1] >= max(self.size[0], self.size[1])
        return _PeriodPadding1D(x, self.size)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] + self.size[0] + self.size[1], input_shapes[2])


class ReshapeM1D(Layer):
    """ Reshape a tensor to matrix by blocks

    # Arguments
        w: int or tuple of int (length 1)

    # Input shape: 3D tensor with shape (batch_size, Nx, features)

    # Output shape: 3D tensor with shape (batch_size, Nx*w, features//w)
    """

    def __init__(self, w, **kwargs):
        self.w = w[0] if isinstance(w, tuple) else w
        super(ReshapeM1D, self).__init__(**kwargs)

    def call(self, x):
        assert x.shape[2] % self.w == 0
        return K.reshape(x, (-1, x.shape[1] * self.w, x.shape[2] // self.w))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] * self.w, input_shapes[2] // self.w)


class ReshapeT1D(Layer):
    """ Reshape a matrix to tensor by blocks, inverse of ReshapeM1D

    # Arguments
        w: int or tuple of int (length 1)

    # Input shape: 3D tensor with shape (batch_size, Nx, features)

    # Output shape: 3D tensor with shape (batch_size, Nx//w, features*w)
    """

    def __init__(self, w, **kwargs):
        self.w = w[0] if isinstance(w, tuple) else w
        super(ReshapeT1D, self).__init__(**kwargs)

    def call(self, x):
        assert x.shape[1] % self.w == 0
        return K.reshape(x, (-1, x.shape[1] // self.w, x.shape[2] * self.w))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // self.w, input_shapes[2] * self.w)


class CNNR1D(Conv1D):
    """ Restriction operator implemented by Conv1D with `strides = kernel_size`
    , restrict a vector with size `Nx` to `Nx//2`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CNNR1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class CNNK1D(Conv1D):
    """ Multiplication of a block band matrix with a vector, implemented by a padding and `Conv1D`
    with strides = 1

    # Arguments
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding1D` and `ZeroPadding1D`
    """

    def __init__(self, filters,
                 kernel_size,
                 bc_padding='period',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(CNNK1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(CNNK1D, self).build((input_shape[0],
                                   input_shape[1] + self.kernel_size[0] - 1, input_shape[2]))

    def call(self, inputs):
        assert self.kernel_size[0] % 2 == 1
        if self.bc_padding == 'period':
            x = PeriodPadding1D(self.kernel_size[0] // 2)(inputs)
        elif self.bc_padding == 'zero':
            x = ZeroPadding1D(self.kernel_size[0] // 2)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        return super(CNNK1D, self).call(x)

    def compute_output_shape(self, input_shapes):
        shape = (input_shapes[0], input_shapes[1] + self.kernel_size[0] - 1, input_shapes[2])
        return super(CNNK1D, self).compute_output_shape(shape)

    def get_config(self):
        config = super(CNNK1D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config

class CNNI1D(Conv1D):
    """ Interpolation solution from coarse grid to fine grid,
    implemented by `Conv1D` with `kernel_size = 1` and `strides = 1`.
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)
    """

    def __init__(self, filters,
                 Nout=None,
                 kernel_size=1,
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert kernel_size == 1
        self.Nout = Nout[0] if isinstance(Nout, tuple) else Nout
        super(CNNI1D, self).__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = super(CNNI1D, self).call(inputs)
        if self.Nout is not None:
            sp = outputs.shape
            assert int(sp[1]) * self.filters % self.Nout == 0
            nc = int(sp[1]) * self.filters // self.Nout
            return K.reshape(outputs, (-1, self.Nout, nc))
        return outputs

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return super(CNNI1D, self).compute_output_shape(input_shapes)
        return (input_shapes[0], self.Nout, input_shapes[1] * self.filters // self.Nout)


class LCR1D(LocallyConnected1D):
    """ Restriction operator implemented by LocallyConnected1D with `strides = kernel_size`
    , restrict a vector with size `Nx` to `Nx//2`
    """

    def __init__(self, filters,
                 kernel_size,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LCR1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class LCK1D(LocallyConnected1D):
    """ Multiplication of a block band matrix with a vector,
    implemented by a padding and `LocallyConnected1D` with strides = 1

    # Arguments
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding1D` and `ZeroPadding1D`
    """

    def __init__(self, filters,
                 kernel_size,
                 bc_padding='period',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(LCK1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(LCK1D, self).build((input_shape[0],
                                  input_shape[1] + self.kernel_size[0] - 1, input_shape[2]))

    def call(self, inputs):
        assert self.kernel_size[0] % 2 == 1
        if self.bc_padding == 'period':
            x = PeriodPadding1D(self.kernel_size[0] // 2)(inputs)
        elif self.bc_padding == 'zero':
            x = ZeroPadding1D(self.kernel_size[0] // 2)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        return super(LCK1D, self).call(x)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.filters)

    def get_config(self):
        config = super(LCK1D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class LCI1D(LocallyConnected1D):
    """ Interpolation solution from coarse grid to fine grid,
    implemented by `LocallyConnected1D` with `kernel_size = 1` and `strides = 1`.
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)
    """

    def __init__(self, filters,
                 Nout=None,
                 kernel_size=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert kernel_size == 1
        self.Nout = Nout[0] if isinstance(Nout, tuple) else Nout[0]
        super(LCI1D, self).__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = super(LCI1D, self).call(inputs)
        if self.Nout is not None:
            sp = outputs.shape
            assert int(sp[1]) * self.filters % self.Nout == 0
            nc = int(sp[1]) * self.filters // self.Nout
            return K.reshape(outputs, (-1, self.Nout, nc))
        return outputs

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return super(LCI1D, self).compute_output_shape(input_shapes)
        return (input_shapes[0], self.Nout, input_shapes[1] * self.filters // self.Nout)


class PeriodPadding2D(Layer):
    """Period-padding layer for 2D input

    # Arguments
        padding: tuple of int (length 2)

    # Input shape: 4D tensor with shape `(batch_size, Nx, Ny, features)`

    # Output shape: 4D tensor with shape `(batch_size, Nx+2*size[0], Ny+2*size[1], features)`
    """

    def __init__(self, size, **kwargs):
        self.size = _convert2tuple_of_tuple(size, 2, 2)
        super(PeriodPadding2D, self).__init__(**kwargs)

    def call(self, x):
        assert (x.shape[1], x.shape[2]) >= tuple(max(x) for x in self.size)
        return _PeriodPadding2D(x, self.size)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] + sum(self.size[0]),
                input_shapes[2] + sum(self.size[1]), input_shapes[3])


class ReshapeM2D(Layer):
    """ Reshape a tensor to matrix by blocks

    # Arguments
        w: tuple of int (length 2)

    # Input shape: 4D tensor with shape (batch_size, Nx, Ny, features)

    # Output shape: 4D tensor with shape (batch_size, Nx*w[0], Ny*w[1], features//(w[0]*w[1]))
    """

    def __init__(self, w, **kwargs):
        self.w = w
        super(ReshapeM2D, self).__init__(**kwargs)

    def call(self, x):
        assert x.shape[3] % (self.w[0] * self.w[1]) == 0
        return _reshapeM2D(x, self.w)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] * self.w[0],
                input_shapes[2] * self.w[1], input_shapes[3] // (self.w[0] * self.w[1]))


class ReshapeT2D(Layer):
    """ Reshape a matrix to tensor by blocks, inverse of ReshapeM2D

    # Arguments
        w: tuple of int (length 2)

    # Input shape: 4D tensor with shape (batch_size, Nx, Ny, features)

    # Output shape: 4D tensor with shape (batch_size, Nx//w[0], Ny//w[1], features*w[0]*w[1])
    """

    def __init__(self, w, **kwargs):
        self.w = w
        super(ReshapeT2D, self).__init__(**kwargs)

    def call(self, x):
        assert (x.shape[1] % self.w[0], x.shape[2] % self.w[1]) == (0, 0)
        return _reshapeT2D(x, self.w)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // self.w[0],
                input_shapes[2] // self.w[1], input_shapes[3] * self.w[0] * self.w[1])


class CNNR2D(Conv2D):
    """ Restriction operator implemented by `Conv2D` with `strides = kernel_size`
    , restrict a vector with size `(Nx, Ny)` to `(Nx//2, Ny//2)`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CNNR2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class CNNK2D(Conv2D):
    """ Multiplication of a block band matrix with a vector, implemented by a padding and `Conv2D`
    with strides = 1

    # Arguments
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding2D` and `ZeroPadding2D`
    """

    def __init__(self, filters,
                 kernel_size,
                 bc_padding='period',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(CNNK2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(CNNK2D, self).build((input_shape[0],
                                   input_shape[1] + self.kernel_size[0] - 1,
                                   input_shape[2] + self.kernel_size[1] - 1,
                                   input_shape[3]))

    def call(self, inputs):
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        if self.bc_padding == 'period':
            x = PeriodPadding2D((self.kernel_size[0] // 2, self.kernel_size[1] // 2))(inputs)
        elif self.bc_padding == 'zero':
            x = ZeroPadding2D((self.kernel_size[0] // 2, self.kernel_size[1] // 2))(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        return super(CNNK2D, self).call(x)

    def compute_output_shape(self, input_shapes):
        shape = (input_shapes[0], input_shapes[1] + self.kernel_size[0] - 1,
                 input_shapes[2] + self.kernel_size[1] - 1, input_shapes[3])
        return super(CNNK2D, self).compute_output_shape(shape)

    def get_config(self):
        config = super(CNNK2D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class CNNI2D(Conv2D):
    """ Interpolation solution from coarse grid to fine grid,
    implemented by `Conv2D` with `kernel_size = 1` and `strides = 1`.
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)
    """

    def __init__(self, filters,
                 Nout=None,
                 kernel_size=(1, 1),
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert kernel_size == (1, 1)
        self.Nout = Nout
        super(CNNI2D, self).__init__(
            filters=filters,
            kernel_size=(1, 1),
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = super(CNNI2D, self).call(inputs)
        if self.Nout is not None:
            sp = outputs.shape
            nc = int(sp[1]) * int(sp[2]) * self.filters / (self.Nout[0] * self.Nout[1])
            w = (self.Nout[0] // int(sp[1]), self.Nout[1] // int(sp[2]))
            assert (w[0] * int(sp[1]), w[1] * int(sp[2])) == self.Nout
            assert w[0] * w[1] * nc == self.filters
            return _reshapeM2D(outputs, w)
        return outputs

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return super(CNNI2D, self).compute_output_shape(input_shapes)
        return (input_shapes[0], self.Nout[0], self.Nout[1],
                input_shapes[1] * input_shapes[2] * self.filters // (self.Nout[0] * self.Nout[1]))


class LCR2D(LocallyConnected2D):
    """ Restriction operator implemented by `LocallyConnected2D` with `strides = kernel_size`
    , restrict a vector with size `(Nx, Ny)` to `(Nx//2, Ny//2)`
    """

    def __init__(self, filters,
                 kernel_size,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LCR2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class LCK2D(LocallyConnected2D):
    """ Multiplication of a block band matrix with a vector,
    implemented by a padding and `LocallyConnected2D` with strides = 1

    # Arguments
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding2D` and `ZeroPadding2D`
    """

    def __init__(self, filters,
                 kernel_size,
                 bc_padding='period',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(LCK2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(LCK2D, self).build((input_shape[0], input_shape[1] + self.kernel_size[0] - 1,
                                  input_shape[2] + self.kernel_size[1] - 1, input_shape[3]))

    def call(self, inputs):
        assert (self.kernel_size[0] % 2, self.kernel_size[0] % 2) == (1, 1)
        if self.bc_padding == 'period':
            x = PeriodPadding2D((self.kernel_size[0] // 2, self.kernel_size[1] // 2))(inputs)
        elif self.bc_padding == 'zero':
            x = ZeroPadding2D((self.kernel_size[0] // 2, self.kernel_size[1] // 2))(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')
        return super(LCK2D, self).call(x)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], input_shapes[2], self.filters)

    def get_config(self):
        config = super(LCK2D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config

class LCI2D(LocallyConnected2D):
    """ Interpolation solution from coarse grid to fine grid,
    implemented by `LocallyConnected2D` with `kernel_size = 1` and `strides = 1`.
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)
    """

    def __init__(self, filters,
                 Nout=None,
                 kernel_size=(1, 1),
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert kernel_size == (1, 1)
        self.Nout = Nout
        super(LCI2D, self).__init__(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = super(LCI2D, self).call(inputs)
        if self.Nout is not None:
            sp = outputs.shape
            nc = int(sp[1]) * int(sp[2]) * self.filters / (self.Nout[0] * self.Nout[1])
            w = (self.Nout[0] // int(sp[1]), self.Nout[1] // int(sp[2]))
            assert (w[0] * int(sp[1]), w[1] * int(sp[2])) == self.Nout
            assert w[0] * w[1] * nc == self.filters
            return _reshapeM2D(outputs, w)
        return outputs

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return super(LCI2D, self).compute_output_shape(input_shapes)
        return (input_shapes[0], self.Nout[0], self.Nout[1],
                input_shapes[1] * input_shapes[2] * self.filters // (self.Nout[0] * self.Nout[1]))


class PeriodPadding3D(Layer):
    """Period-padding layer for 3D input

    # Arguments
        padding: tuple of int (length 3)

    # Input shape: 5D tensor with shape `(batch_size, Nx, Ny, Nz, features)`

    # Output shape:
        5D tensor with shape `(batch_size, Nx+2*size[0], Ny+2*size[1], Nz+2*size[2], features)`
    """

    def __init__(self, size, **kwargs):
        self.size = _convert2tuple_of_tuple(size, 3, 2)
        super(PeriodPadding3D, self).__init__(**kwargs)

    def call(self, x):
        assert (x.shape[1], x.shape[2], x.shape[3]) >= tuple(max(x) for x in self.size)
        return _PeriodPadding3D(x, self.size)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] + sum(self.size[0]),
                input_shapes[2] + sum(self.size[1]),
                input_shapes[3] + sum(self.size[2]), input_shapes[4])


class ReshapeM3D(Layer):
    """ Reshape a tensor to matrix by blocks

    # Arguments
        w: tuple of int (length 3)

    # Input shape: 5D tensor with shape (batch_size, Nx, Ny, Nz, features)

    # Output shape:
        5D tensor with shape (batch_size, Nx*w[0], Ny*w[1], Nz*w[2], features//(w[0]*w[1]*w[2]))
    """

    def __init__(self, w, **kwargs):
        self.w = w
        super(ReshapeM3D, self).__init__(**kwargs)

    def call(self, x):
        assert x.shape[4] % (self.w[0] * self.w[1] * self.w[2]) == 0
        return _reshapeM3D(x, self.w)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] * self.w[0],
                input_shapes[2] * self.w[1], input_shapes[3] * self.w[2],
                input_shapes[4] // (self.w[0] * self.w[1] * self.w[2]))


class ReshapeT3D(Layer):
    """ Reshape a matrix to tensor by blocks, inverse of ReshapeM3D

    # Arguments
        w: tuple of int (length 3)

    # Input shape: 5D tensor with shape (batch_size, Nx, Ny, Nz, features)

    # Output shape:
        5D tensor with shape (batch_size, Nx//w[0], Ny//w[1], Nz//w[2], features*w[0]*w[1]*w[2])
    """

    def __init__(self, w, **kwargs):
        self.w = w
        super(ReshapeT3D, self).__init__(**kwargs)

    def call(self, x):
        assert (x.shape[1] % self.w[0], x.shape[2] % self.w[1], x.shape[3] % self.w[2]) == (0, 0, 0)
        return _reshapeT3D(x, self.w)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // self.w[0],
                input_shapes[2] // self.w[1], input_shapes[3] // self.w[2],
                input_shapes[3] * self.w[0] * self.w[1] * self.w[2])


class CNNR3D(Conv3D):
    """ Restriction operator implemented by `Conv3D` with `strides = kernel_size`
    , restrict a vector with size `(Nx, Ny, Nz)` to `(Nx//2, Ny//2, Nz//2)`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CNNR3D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class CNNK3D(Conv3D):
    """ Multiplication of a block band matrix with a vector, implemented by a padding and `Conv3D`
    with strides = 1

    # Arguments
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding3D` and `ZeroPadding3D`
    """

    def __init__(self, filters,
                 kernel_size,
                 bc_padding='period',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(CNNK3D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(CNNK3D, self).build((input_shape[0],
                                   input_shape[1] + self.kernel_size[0] - 1,
                                   input_shape[2] + self.kernel_size[1] - 1,
                                   input_shape[3] + self.kernel_size[2] - 1,
                                   input_shape[4]))

    def call(self, inputs):
        assert tuple(x % 2 for x in self.kernel_size) == tuple(1 for x in self.kernel_size)
        ww = tuple(x // 2 for x in self.kernel_size)
        if self.bc_padding == 'period':
            x = PeriodPadding3D(ww)(inputs)
        elif self.bc_padding == 'zero':
            x = ZeroPadding3D(ww)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        return super(CNNK3D, self).call(x)

    def compute_output_shape(self, input_shapes):
        shape = (input_shapes[0], input_shapes[1] + self.kernel_size[0] - 1,
                 input_shapes[2] + self.kernel_size[1] - 1,
                 input_shapes[3] + self.kernel_size[2] - 1, input_shapes[4])
        return super(CNNK3D, self).compute_output_shape(shape)

    def get_config(self):
        config = super(CNNK3D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class CNNI3D(Conv3D):
    """ Interpolation solution from coarse grid to fine grid,
    implemented by `Conv3D` with `kernel_size = 1` and `strides = 1`.
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)
    """

    def __init__(self, filters,
                 Nout=None,
                 kernel_size=(1, 1, 1),
                 dilation_rate=1,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert kernel_size == (1, 1, 1)
        self.Nout = Nout
        super(CNNI3D, self).__init__(
            filters=filters,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = super(CNNI3D, self).call(inputs)
        if self.Nout is not None:
            sp = outputs.shape
            nc = int(sp[1]) * int(sp[2]) * int(sp[3]) \
                * self.filters / (self.Nout[0] * self.Nout[1] * self.Nout[2])
            wx = self.Nout[0] // int(sp[1])
            wy = self.Nout[1] // int(sp[2])
            wz = self.Nout[2] // int(sp[3])
            assert (wx * int(sp[1]), wy * int(sp[2]), wz * int(sp[3])) == self.Nout
            assert wx * wy * wz * nc == self.filters
            return _reshapeM3D(outputs, (wx, wy, wz))
        return outputs

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return super(CNNI3D, self).compute_output_shape(input_shapes)
        n_total = input_shapes[1] * input_shapes[2] * input_shapes[3] * self.filters
        return (input_shapes[0], self.Nout[0], self.Nout[1], self.Nout[2],
                n_total // (self.Nout[0] * self.Nout[1] * self.Nout[2]))


class WaveLetC1D(Conv1D):
    """Wavelet transformation implemented by `Conv1D` with `strides=2`

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding1D` and `ZeroPadding1D`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(WaveLetC1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(WaveLetC1D, self).build((input_shape[0],
                                       input_shape[1] + self.kernel_size[0] - 2, input_shape[2]))

    def call(self, inputs):
        assert self.kernel_size[0] % 2 == 0
        if self.kernel_size[0] > 2:
            if self.bc_padding == 'period':
                inputs = PeriodPadding1D(self.kernel_size[0] // 2 - 1)(inputs)
            elif self.bc_padding == 'zero':
                inputs = ZeroPadding1D(self.kernel_size[0] // 2 - 1)(inputs)
            else:
                raise ImportError('Only "period" and "zero" padding are provided')

        return super(WaveLetC1D, self).call(inputs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // 2, self.filters)

    def get_config(self):
        config = super(WaveLetC1D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class InvWaveLetC1D(Conv1D):
    """Wavelet transformation implemented by `Conv1D` with `strides=1`
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding1D` and `ZeroPadding1D`
    """

    def __init__(self, filters,
                 kernel_size,
                 Nout=None,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        self.Nout = Nout
        super(InvWaveLetC1D, self).__init__(
            filters=filters,  # 1D case, (wavelet, scaling)
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(InvWaveLetC1D, self).build((input_shape[0],
                                          input_shape[1] + self.kernel_size[0] - 1, input_shape[2]))

    def call(self, inputs):
        assert self.kernel_size[0] % 2 == 1
        if self.bc_padding == 'period':
            inputs = PeriodPadding1D(self.kernel_size[0] // 2)(inputs)
        elif self.bc_padding == 'zero':
            inputs = ZeroPadding1D(self.kernel_size[0] // 2)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        opt = super(InvWaveLetC1D, self).call(inputs)
        if self.Nout is not None:
            assert opt.shape[1] * opt.shape[2] % self.Nout == 0
            nc = opt.shape[1] * opt.shape[2] // self.Nout
            return K.reshape(opt, (-1, self.Nout, nc))
        return opt

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return input_shapes[0:2] + (self.filters, )
        return (input_shapes[0], self.Nout, input_shapes[1] * self.filters // self.Nout)

    def get_config(self):
        config = super(InvWaveLetC1D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class WaveLetC2D(Conv2D):
    """Wavelet transformation implemented by `Conv2D` with `strides=2`

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding2D` and `ZeroPadding2D`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(WaveLetC2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(WaveLetC2D, self).build((input_shape[0],
                                       input_shape[1] + self.kernel_size[0] - 2,
                                       input_shape[2] + self.kernel_size[1] - 2, input_shape[3]))

    def call(self, inputs):
        assert (self.kernel_size[0] % 2, self.kernel_size[1] % 2) == (0, 0)
        if not (self.kernel_size[0] <= 2 and self.kernel_size[1] <= 2):
            ww = (self.kernel_size[0] // 2 - 1, self.kernel_size[1] // 2 - 1)
            if self.bc_padding == 'period':
                inputs = PeriodPadding2D(ww)(inputs)
            elif self.bc_padding == 'zero':
                inputs = ZeroPadding2D(ww)(inputs)
            else:
                raise ImportError('Only "period" and "zero" padding are provided')

        return super(WaveLetC2D, self).call(inputs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // 2, input_shapes[2] // 2, self.filters)

    def get_config(self):
        config = super(WaveLetC2D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class InvWaveLetC2D(Conv2D):
    """Wavelet transformation implemented by `Conv2D` with `strides=1`
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding2D` and `ZeroPadding2D`
    """

    def __init__(self, filters,
                 kernel_size,
                 Nout=None,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        self.Nout = Nout
        super(InvWaveLetC2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(InvWaveLetC2D, self).build((input_shape[0],
                                          input_shape[1] + self.kernel_size[0] - 1,
                                          input_shape[2] + self.kernel_size[1] - 1, input_shape[3]))

    def call(self, inputs):
        assert (self.kernel_size[0] % 2, self.kernel_size[1] % 2) == (1, 1)
        ww = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        if self.bc_padding == 'period':
            inputs = PeriodPadding2D(ww)(inputs)
        elif self.bc_padding == 'zero':
            inputs = ZeroPadding2D(ww)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        opt = super(InvWaveLetC2D, self).call(inputs)
        if self.Nout is not None:
            sp = opt.shape
            assert (self.Nout[0] % sp[1], self.Nout[1] % sp[2]) == (0, 0)
            assert (sp[1] * sp[2] * sp[3]) % (self.Nout[0] * self.Nout[1]) == 0
            ww = (self.Nout[0] // sp[1], self.Nout[1] // sp[2])
            return ReshapeM2D(ww)(opt)
        return opt

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return input_shapes[0:3] + (self.filters, )
        return (input_shapes[0], self.Nout[0], self.Nout[1],
                input_shapes[1] * input_shapes[2] * self.filters // (self.Nout[0] * self.Nout[1]))

    def get_config(self):
        config = super(InvWaveLetC2D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class WaveLetC3D(Conv3D):
    """Wavelet transformation implemented by `Conv3D` with `strides=2`

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding3D` and `ZeroPadding3D`
    """

    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        super(WaveLetC3D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=(2, 2, 2),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(WaveLetC3D, self).build((input_shape[0],
                                       input_shape[1] + self.kernel_size[0] - 2,
                                       input_shape[2] + self.kernel_size[1] - 2,
                                       input_shape[3] + self.kernel_size[2] - 2, input_shape[4]))

    def call(self, inputs):
        assert tuple(x % 2 for x in self.kernel_size) == tuple(0 for x in self.kernel_size)
        if not self.kernel_size <= tuple(2 for x in self.kernel_size):
            ww = tuple(x // 2 - 1 for x in self.kernel_size)
            if self.bc_padding == 'period':
                inputs = PeriodPadding3D(ww)(inputs)
            elif self.bc_padding == 'zero':
                inputs = ZeroPadding3D(ww)(inputs)
            else:
                raise ImportError('Only "period" and "zero" padding are provided')

        return super(WaveLetC3D, self).call(inputs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1] // 2, input_shapes[2] // 2,
                input_shapes[3] // 2, self.filters)

    def get_config(self):
        config = super(WaveLetC3D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


class InvWaveLetC3D(Conv3D):
    """Wavelet transformation implemented by `Conv3D` with `strides=1`
    If `Nout` is given, it reshape the output with shape (batch_size,)+Nout+(features,)

    # Arguments:
        bc_padding: 'period' or 'zero' corresponds to `PeriodPadding3D` and `ZeroPadding3D`
    """

    def __init__(self, filters,
                 kernel_size,
                 Nout=None,
                 dilation_rate=1,
                 activation='linear',
                 bc_padding='period',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.bc_padding = bc_padding
        self.Nout = Nout
        super(InvWaveLetC3D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(InvWaveLetC3D, self).build((input_shape[0],
                                          input_shape[1] + self.kernel_size[0] - 1,
                                          input_shape[2] + self.kernel_size[0] - 1,
                                          input_shape[3] + self.kernel_size[0] - 1, input_shape[4]))

    def call(self, inputs):
        assert tuple(x % 2 for x in self.kernel_size) == tuple(1 for x in self.kernel_size)
        ww = tuple(x // 2 for x in self.kernel_size)
        if self.bc_padding == 'period':
            inputs = PeriodPadding3D(ww)(inputs)
        elif self.bc_padding == 'zero':
            inputs = ZeroPadding3D(ww)(inputs)
        else:
            raise ImportError('Only "period" and "zero" padding are provided')

        opt = super(InvWaveLetC3D, self).call(inputs)
        if self.Nout is not None:
            sp = opt.shape
            assert (self.Nout[0] % sp[1], self.Nout[1] % sp[2],
                    self.Nout[2] % sp[3]) == (0, 0, 0)
            assert (sp[1] * sp[2] * sp[3] * sp[4]) \
                % (self.Nout[0] * self.Nout[1] * self.Nout[2]) == 0
            ww = (self.Nout[0] // sp[1], self.Nout[1] // sp[2], self.Nout[2] // sp[3])
            return ReshapeM3D(ww)(opt)
        return opt

    def compute_output_shape(self, input_shapes):
        if self.Nout is None:
            return input_shapes[0:4] + (self.filters, )
        n_total = input_shapes[1] * input_shapes[2] * input_shapes[3] * self.filters
        return (input_shapes[0], self.Nout[0], self.Nout[1], self.Nout[2],
                n_total // (self.Nout[0] * self.Nout[1] * self.Nout[2]))

    def get_config(self):
        config = super(InvWaveLetC3D, self).get_config()
        config['bc_padding'] = self.bc_padding
        return config


ConvR1D = CNNR1D
ConvK1D = CNNK1D
ConvI1D = CNNI1D
ConvR2D = CNNR2D
ConvK2D = CNNK2D
ConvI2D = CNNI2D
ConvR3D = CNNR3D
ConvK3D = CNNK3D
ConvI3D = CNNI3D

WaveLet1D = WaveLetC1D
WaveLet2D = WaveLetC2D
WaveLet3D = WaveLetC3D
InvWaveLet1D = InvWaveLetC1D
InvWaveLet2D = InvWaveLetC2D
InvWaveLet3D = InvWaveLetC3D
