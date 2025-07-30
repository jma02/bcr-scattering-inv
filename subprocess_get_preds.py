# coding=utf-8
# vim: sw=4 et tw=100
"""
code for scattering 2D Inverse problem: BCR-Net
"""
from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force TensorFlow to use FP32
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '0'

import os.path
from shutil import copyfile
import sys
sys.path.append(os.path.abspath('../../'))
# ----------------- import keras tools ----------------------
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Reshape, Lambda, Concatenate, ZeroPadding2D, Activation, BatchNormalization
from keras import backend as K
# from keras import backend as K
# K.set_floatx('float32')
import tensorflow as tf
import matplotlib.pyplot as plt

# Force CPU usage unless on DARWIN
config = tf.ConfigProto(device_count={'GPU': 0})
config.allow_soft_placement = True
sess = tf.Session(config=config)
K.set_session(sess)
print("TensorFlow configured to use CPU only")

# we need gradient clipping - jma
# also, im swapping to Adam
from keras.optimizers import Nadam, Adam

# from keras.utils import plot_model

from mnn.layers import CNNK1D, CNNR1D, CNNI1D, WaveLetC1D, InvWaveLetC1D
from mnn.layers import CNNK2D
from mnn.callback import SaveBestModel
# ---------------- import python packages --------------------
import argparse
import h5py
import numpy as np
import math

# ---- define input parameters and set their default values ---
parser = argparse.ArgumentParser(description='Scattering Inference -- 2D')
parser.add_argument('--input-prefix', type=str, default='merged_data', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--data-path', type=str, default='data', metavar='string',
                    help='data path (default: )')
parser.add_argument('--percent', type=float, default=.3/.4, metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')
parser.add_argument('--initialvalue', type=str, required=True, metavar='string',
                    help='path to the model weights file (.h5)')
parser.add_argument('--w-comp', type=int, default=1, metavar='N',
                    help='window size of the compress(default: %(default)s)')
args = parser.parse_args()

input_prefix = args.input_prefix
data_path = args.data_path + '/'



delta_noise = 0.0 # this isn't used to noise the data in this file. data is already noised
outputfilename = f'BCRpredictions-delta{str(delta_noise)}.txt'
log_os = open(outputfilename, "w+")
def output(obj):
    print(obj)
    log_os.write(str(obj)+'\n')

def outputnewline():
    log_os.write('\n')
    log_os.flush()


# ---------- prepare the train and test data -------------------
filenameIpt = data_path + input_prefix + '.hdf5'
print('Reading data...')
fin = h5py.File(filenameIpt, 'r')
InputArray = fin['farfield.real'][:]

Nsamples = InputArray.shape[1]
InputArray = np.array(InputArray)
# InputArray = np.array(InputArray).T.reshape(Nsamples, 100, 100)

Nsamples, Ns, Nd = InputArray.shape
print('Reading data finished')
Nsamples, Ns, Nd = InputArray.shape
print(f'Input shape is {InputArray.shape}')
Nr, Nt = 100, 100

# normalize output data -- these were the training constants we used
max_out = 0.8
min_out = 0
pixel_max = max_out - min_out

n_input  = (Ns, Nd)

n_output = (100,100)
output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output("(Ns, Nd)                = (%d, %d)" % (Ns, Nd))
output("(Nt, Nr)                = (%d, %d)" % (Nt, Nr))
output("Nsamples                = %d" % Nsamples)
outputnewline()


mean_out = 0
"""
max / min of the output data are (0.80, 0.00)
max / min of the output data are (2.00, 0.00)
"""


X_test  = InputArray


def splitScaling1D(X, alpha):
    return Lambda(lambda x: x[:, :, alpha:2*alpha])(X)


def splitWavelet1D(X, alpha):
    return Lambda(lambda x: x[:, :, 0:alpha])(X)

def Padding_x(x, s):
    return K.concatenate([x[:, x.shape[1]-s:x.shape[1], ...], x, x[:, 0:s, ...]], axis=1)

def __TriangleAdd(X, Y, alpha):
    return K.concatenate([X[:, :, 0:alpha], X[:, :, alpha:2*alpha] + Y], axis=2)

def TriangleAdd(X, Y, alpha):
    return Lambda(lambda x: __TriangleAdd(x[0], x[1], alpha))([X, Y])


# ---------- architecture of W -------------------
bc = 'period'
w_comp = args.w_comp
w_interp = w_comp
L = math.floor(math.log2(Ns)) - 4 # number of levels
m = Ns // 2**L     # size of the coarse grid
m = 2 * ((m+1)//2) - 1
w = 2 * 3    # support of the wavelet function
n_b = 5      # bandsize of the matrix
output("(L, m) = (%d, %d)" % (L, m))

alpha = 128
N_cnn=6
N_cnn3=5

Ipt = Input(shape=n_input)
Ipt_c = CNNK1D(alpha, w_comp, activation='linear', bc_padding=bc)(Ipt)

bt_list = (L+1) * [None]
b = Ipt_c
for ll in range(1, L+1):
    bt = WaveLetC1D(2*alpha, w, activation='linear', use_bias=False)(b)
    bt_list[ll] = bt
    b = splitScaling1D(bt, alpha)

# (b,t) --> d
# d^L = A^L * b^L
d = b
for k in range(0, N_cnn):
    d = CNNK1D(alpha, m, activation='relu', bc_padding='period')(d)

# d = T^* * (D tb + (0,d))
for ll in range(L, 0, -1):
    d1 = bt_list[ll]
    for k in range(0, N_cnn):
        d1 = CNNK1D(2*alpha, n_b, activation='relu', bc_padding='period')(d1)

#     d11 = splitWavelet1D(d1, alpha)
#     d12 = splitScaling1D(d1, alpha)
#     d12 = Add()([d12, d])
#     d = Concatenate(axis=-1)([d11, d12])
#     d = Lambda(lambda x: TriangleAdd(x[0], x[1], alpha))([d1, d])
    d = TriangleAdd(d1, d, alpha)
    d = InvWaveLetC1D(2*alpha, w//2, Nout=Nt//(2**(ll-1)), activation='linear', use_bias=False)(d)

Img_c = d

Img = CNNK1D(Nr, w_interp, activation='linear', bc_padding=bc)(Img_c)
Img_p = Reshape(n_output+(1,))(Img)
for k in range(0, N_cnn3-1):
    Img_p = Lambda(lambda x: Padding_x(x, 1))(Img_p)
    Img_p = ZeroPadding2D((0, 1))(Img_p)
    Img_p = Conv2D(4, 3, activation='relu')(Img_p)
    # Img_p = CNNK2D(4, 3, activation='relu', bc_padding=bc)(Img_p)

Img_p = Lambda(lambda x: Padding_x(x, 1))(Img_p)
Img_p = ZeroPadding2D((0, 1))(Img_p)
Img_p = Conv2D(1, 3, activation='linear')(Img_p)
# Img_p = CNNK2D(1, 3, activation='linear', bc_padding=bc)(Img_p)
Opt = Reshape(n_output)(Img_p)
Opt = Add()([Img, Opt])



model = Model(inputs=Ipt, outputs=Opt)
model.compile(loss='mean_squared_error', optimizer='Nadam')

model.optimizer.schedule_decay = (0.004)
output('number of params = %d' % model.count_params())

# Load model weights with error checking
try:
    output(f'Loading model weights from: {args.initialvalue}')
    
    # Check if the weights file exists
    if not os.path.exists(args.initialvalue):
        raise FileNotFoundError(f"Model weights file not found: {args.initialvalue}")
    
    # Load weights
    model.load_weights(args.initialvalue, by_name=False)
    output("Model weights loaded successfully")
    
    # Verify model is ready for inference
    output(f'Model input shape: {model.input_shape}')
    output(f'Model output shape: {model.output_shape}')
    
except Exception as e:
    output(f'Error loading model weights: {str(e)}')
    print(f'Error loading model weights: {str(e)}')
    raise e

# Run inference
output('Starting inference on test data...')
Yhat = model.predict(X_test)
xx_test = X_test
yyhat = Yhat
# unnormalize output
"""
max / min of the output data are (0.80, 0.00)
max / min of the output data are (2.00, 0.00)
"""


yyhat = yyhat * (0.5 * pixel_max)
# Save predictions for external access
predictions_file = f"BCRpredictions-delta{str(delta_noise)}.hdf5"
with h5py.File(predictions_file, 'w') as f:
    f.create_dataset('predictions', data=yyhat)
    f.create_dataset('input', data=xx_test)
    
    f.attrs['n_samples'] = len(yyhat)

output(f'Predictions saved to: {predictions_file}')
print(f'PREDICTIONS_FILE:{predictions_file}')  