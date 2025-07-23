# coding=utf-8
# vim: sw=4 et tw=100
"""
code for scattering 2D Inverse problem: BCR-Net
"""
from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os.path
from shutil import copyfile
import sys
sys.path.append(os.path.abspath('../../'))
# ----------------- import keras tools ----------------------
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Reshape, Lambda, Concatenate, ZeroPadding2D
from keras import backend as K
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
parser = argparse.ArgumentParser(description='Scattering -- 2D')
parser.add_argument('--epoch', type=int, default=40, metavar='N',
                    help='# epochs for training in the each round (default: %(default)s)')
parser.add_argument('--input-prefix', type=str, default='merged_data', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--alpha', type=int, default=40, metavar='N',
                    help='number of channels for the depth for training (default: %(default)s)')
parser.add_argument('--n-cnn', type=int, default=6, metavar='N',
                    help='number CNN layers (default: %(default)s)')
parser.add_argument('--n-cnn3', type=int, default=5, metavar='N',
                    help='number CNN layers (default: %(default)s)')
parser.add_argument('--noise', type=float, default=0, metavar='noise',
                    help='noise on the measure data (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate for the first round (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size (default: %(default)s)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: %(default)s)')
parser.add_argument('--output-suffix', type=str, default=None, metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=4./5., metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')
parser.add_argument('--initialvalue', type=str, default=None, metavar='filename',
                    help='filename storing the weights of the model (default: '')')
parser.add_argument('--w-comp', type=int, default=1, metavar='N',
                    help='window size of the compress(default: %(default)s)')
parser.add_argument('--data-path', type=str, default='data', metavar='string',
                    help='data path (default: )')
parser.add_argument('--log-path', type=str, default='logs', metavar='string',
                    help='log path (default: )')
args = parser.parse_args()

N_epoch = args.epoch
alpha = args.alpha
N_cnn = args.n_cnn
N_cnn3 = args.n_cnn3
lr = args.lr
percent = args.percent
batch_size = args.batch_size
noise = args.noise
noise_rate = noise / 100.
input_prefix = args.input_prefix
output_suffix = args.output_suffix
data_path = args.data_path + '/'
log_path = args.log_path + '/'
print(f'N_epoch = {N_epoch}\t alpha = {alpha}\t (N_cnn, N_cnn3) = ({N_cnn}, {N_cnn3})\t\
      batch size = {batch_size}')
print(f'lr = {lr:.2e}\t percent = {percent}\t noise = {noise}')
print(f'input_prefix = {input_prefix}\t output suffix = {output_suffix}')


if not os.path.exists(log_path):
    os.mkdir(log_path)

outputfilename  = log_path + 'S2d' + input_prefix[7:]     + 'Nc' + str(N_cnn) + 'Al' + str(alpha)
if abs(int(noise) - noise) < 1.e-6:
    outputfilename += "Ns" + str(int(noise))
else:
    outputfilename += "Ns" + str(noise)
outputfilename += output_suffix or str(os.getpid())
modelfilename   = outputfilename + '.h5'
outputfilename += '.txt'
log_os          = open(outputfilename, "w+")

def output(obj):
    print(obj)
    log_os.write(str(obj)+'\n')

def outputnewline():
    log_os.write('\n')
    log_os.flush()


output(f'output filename is {outputfilename}')

# ---------- prepare the train and test data -------------------
filenameIpt = data_path + input_prefix + '.hdf5'
print('Reading data...')
fin = h5py.File(filenameIpt, 'r')
InputArray = fin['farfield.real'][:]

Nsamples = InputArray.shape[1]
InputArray = np.array(InputArray).T.reshape(Nsamples, 100, 100)
OutputArray = fin['image'][:]
OutputArray = np.array(OutputArray).T.reshape(Nsamples, 100, 100)

# zero pad images to 128x128

InputArray= np.pad(InputArray, ((0, 0), (14, 14), (14, 14)), mode='constant', constant_values=0)
OutputArray= np.pad(OutputArray, ((0, 0), (14, 14), (14, 14)), mode='constant', constant_values=0)

Nsamples, Ns, Nd = InputArray.shape
assert OutputArray.shape[0] == Nsamples
Nsamples, Nt, Nr = OutputArray.shape

print('Reading data finished')
print(f'Input shape is {InputArray.shape}')
print(f'Output shape is {OutputArray.shape}')


output(args)
output('alpha                   = %d\t' % alpha)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output("(Ns, Nd)                = (%d, %d)" % (Ns, Nd))
output("(Nt, Nr)                = (%d, %d)" % (Nt, Nr))
output("Nsamples                = %d" % Nsamples)
outputnewline()


mean_out = 0
max_out = np.amax(OutputArray)
min_out = np.amin(OutputArray)
pixel_max = max_out - min_out
OutputArray /= 0.5 * pixel_max
output(f'max / min of the output data are ({max_out:0.2f}, {min_out:0.2f})')
max_out = np.amax(OutputArray)
min_out = np.amin(OutputArray)
pixel_max = max_out - min_out
output(f'max / min of the output data are ({max_out:0.2f}, {min_out:0.2f})')

n_train = int(Nsamples * percent)
n_test  = min(max(n_train, 5000), Nsamples - n_train)
BATCH_SIZE = batch_size
n_valid = 1024


n_input  = (Ns, Nd)
n_output = (Nt, Nr)
output("[n_input, n_output] = [(%d,%d),  (%d,%d)]" % (n_input + n_output))
output("[n_train, n_test, n_valid]   = [%d, %d, %d]" % (n_train, n_test, n_valid))
output("batch size = %d" % BATCH_SIZE)
output("noise rate = %.2e" % noise_rate)


X_train = InputArray[0:n_train, :, :]
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[n_train:(n_train+n_test), :, :]
Y_test  = OutputArray[n_train:(n_train+n_test), :, :]

# ---------- add noise on the input data ----------------------
noiseTrain = np.random.randn(n_train, Ns, Nd) * noise_rate
X_train = X_train * (1 + noiseTrain)
noiseTest = np.random.randn(n_test, Ns, Nd) * noise_rate
X_test = X_test * (1 + noiseTest)


# ----------------- functions used in NN --------------------
weight_pixel = np.arange(1, 2*Nr+1, 2)
def PSNR(img1, img2, pixel_max=1.0):
    dimg = (img1 - img2) / pixel_max
    mse = np.maximum(np.mean(dimg**2), 1.e-10)
    return -10 * math.log10(mse)

def PSNRs(imgs1, imgs2, pixel_max=1.0):
    dimgs = (imgs1 - imgs2) / pixel_max
    mse = np.maximum(np.mean(dimgs**2, axis=(1, 2)), 1.e-10)
    return -10 * np.log10(mse)

def test_data(model, X, Y):
    Yhat = model.predict(X, n_valid)
    if 'V1' in input_prefix:
        errs = np.linalg.norm((Yhat - Y) * weight_pixel, axis=(1, 2)) \
            / np.linalg.norm((Y+mean_out) * weight_pixel, axis=(1, 2))
        return errs
    else:
        return -PSNRs(Yhat, Y, pixel_max)

def check_result(model):
    return (test_data(model, X_train[0:n_valid, ...], Y_train[0:n_valid, ...]),
            test_data(model, X_test[0:n_valid, ...], Y_test[0:n_valid, ...]))

def test_data_mh(model_mh, X, Y):
    Yhat = model_mh.predict(X, n_valid)
    if 'V1' in input_prefix:
        errs1 = np.linalg.norm((Yhat[0] - Y) * weight_pixel, axis=(1, 2)) \
            / np.linalg.norm((Y+mean_out)*weight_pixel, axis=(1, 2))
        errs2 = np.linalg.norm((Yhat[1] - Y) * weight_pixel, axis=(1, 2)) \
            / np.linalg.norm((Y+mean_out)*weight_pixel, axis=(1, 2))
        return (errs1, errs2)
    else:
        return (-PSNRs(Yhat[0], Y, pixel_max), -PSNRs(Yhat[1], Y, pixel_max))

def check_result_mh(model_mh):
    return test_data_mh(model_mh, X_test[0:n_valid, ...], Y_test[0:n_valid, ...])

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
bc = 'zero'
w_comp = args.w_comp
w_interp = w_comp
L = math.floor(math.log2(Ns)) - 2  # number of levels
m = Ns // 2**L     # size of the coarse grid
m = 2 * ((m+1)//2) - 1
w = 2 * 3   # support of the wavelet function
n_b = 5      # bandsize of the matrix
output("(L, m) = (%d, %d)" % (L, m))


print(f"Initial input shape: {n_input}")  # (128, 128)

Ipt = Input(shape=n_input)
print(f"Input layer shape: {Ipt.shape}")
Ipt_c = CNNK1D(alpha, w_comp, activation='linear', bc_padding=bc)(Ipt)
print(f"After first CNNK1D layer shape: {Ipt_c.shape}")  # (128, 128, alpha)

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


lr_bs = []
for bs in range(0, 5):
    lr_bs.append([BATCH_SIZE * 2**bs, lr])

for ll in range(1, 5):
    lr_bs.append([BATCH_SIZE * 2**bs, lr * math.sqrt(0.1)**ll])

print(lr_bs)


lr_bs2 = []
for bs in range(3, 5):
    lr_bs2.append([BATCH_SIZE * 2**bs, lr])

for ll in range(1, 3):
    lr_bs2.append([BATCH_SIZE * 2**bs, lr * 0.1**ll])

print(lr_bs2)

# model: final model
model = Model(inputs=Ipt, outputs=Opt)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params = %d' % model.count_params())


if args.initialvalue is None:  # model is not pre-trained, use multiple outputs to train it first
    model_multihead = Model(inputs=Ipt, outputs=[Opt, Img])
    model_multihead.compile(loss='mean_squared_error', optimizer='Nadam', loss_weights=[1., 1.])
    model_multihead.optimizer.schedule_decay = (0.004)
    output('number of params = %d' % model_multihead.count_params())
    save_best_model_mh = SaveBestModel(modelfilename, check_result=check_result_mh, period=1,
                                       patience=10, output=output, test_weight=0., verbose=2)
    n_epochs_pre = 0
    N_e = n_epochs_pre + 2 * N_epoch
    for b_s, l_r in lr_bs:
        print(f'batch_size = {b_s} and learning rate = {l_r:.2e}')
        model_multihead.optimizer.lr = (l_r)
        model_multihead.stop_training = False
        model_multihead.fit(X_train, [Y_train, Y_train], batch_size=b_s, epochs=N_e,
                            initial_epoch=n_epochs_pre, verbose=2, callbacks=[save_best_model_mh])
        n_epochs_pre = N_e
        N_e += N_epoch
        model_multihead.load_weights(modelfilename, by_name=False)  # re-load the best model
        save_best_model_mh.best_epoch_update = n_epochs_pre

    save_best_model = SaveBestModel(modelfilename, check_result=check_result, period=1,
                                    patience=10, output=output, test_weight=1., verbose=2)
    save_best_model.start = save_best_model_mh.start
    save_best_model = save_best_model_mh
    save_best_model.check_result = check_result
    save_best_model.test_weight = 1.
    N_e = n_epochs_pre + 2 * N_epoch
    for b_s, l_r in lr_bs2:
        print(f'batch_size = {b_s} and learning rate = {l_r:.2e}')
        model.optimizer.lr = (l_r)
        model.stop_training = False
        model.fit(X_train, Y_train, batch_size=b_s, epochs=N_e,
                  initial_epoch=n_epochs_pre, verbose=2, callbacks=[save_best_model])
        n_epochs_pre = N_e
        N_e += N_epoch
        model.load_weights(modelfilename, by_name=False)
        save_best_model.best_epoch_update = n_epochs_pre
else:
    lr = lr / 10.
    output('initial the network by %s\n' % args.initialvalue)
    model.load_weights(args.initialvalue, by_name=False)
    n_epochs_pre = 0
    BATCH_SIZE = 256

    save_best_model = SaveBestModel(modelfilename, check_result=check_result, period=1,
                                    output=output, patience=max(N_epoch//40, 40), test_weight=1.,
                                    reduceLR=True, min_lr=lr/10., patience_lr=20,
                                    factor=np.sqrt(0.1), verbose=args.verbose)
    model.optimizer.lr = (lr)
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epoch, verbose=args.verbose,
              initial_epoch=n_epochs_pre, callbacks=[save_best_model])


bestmodel = modelfilename[0:-5] + 'best.h5'
if os.path.isfile(bestmodel):
    best_err_test = save_best_model.best_err_test
    model.load_weights(bestmodel, by_name=False)
    err_test = test_data(model, X_test[0:n_valid, ...], Y_test[0:n_valid, ...])
    be_test = np.mean(err_test)

    if(best_err_test < be_test):
        output('The current model is the best model')
        copyfile(modelfilename, bestmodel)
        output('copy the model %s as the best model %s' % (modelfilename, bestmodel))
        is_best = 'Y'
    else:
        output('The current model is not the best one')
        is_best = 'N'
else:
    is_best = 'Y'
    output('The best model does not exist')
    copyfile(modelfilename, bestmodel)
    output('copy the model %s as the best model %s' % (modelfilename, bestmodel))

if is_best == 'Y':
    Yhat = model.predict(X_test[0:100], 100)
    xx_test = X_test[0:100, ...]
    yy_test = Y_test[0:100, ...]
    yyhat = Yhat[0:100, ...]
    hf = h5py.File(bestmodel[0:-3] + 'data.h5', 'w')
    hf.create_dataset('input', data=xx_test)
    hf.create_dataset('output', data=yy_test)
    hf.create_dataset('pred', data=yyhat)

log_os.close()

with open('tr2dInvNoise.txt', "a") as tr_os:
    tr_os.write('%s\t%s\t%d\t%d\t%d\t' % (args.input_prefix,
                                          modelfilename, alpha, N_cnn, N_cnn3))
    tr_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
    tr_os.write('%d\t%d\t' % (N_epoch, save_best_model.stop_epoch))
    tr_os.write('%s\t' % is_best)
    tr_os.write('%.3e\t%.3e\t' % (save_best_model.best_err_train, save_best_model.best_err_test))
    tr_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' %
                (save_best_model.best_err_train_max,
                 save_best_model.best_err_test_max,
                 save_best_model.best_err_var_train,
                 save_best_model.best_err_var_test))
    tr_os.write('\n')

outputfilename2 = outputfilename[0:-4] + '_err.txt'
with open(outputfilename2, "w+") as err_os:
    for dat in save_best_model.err_history:
        err_os.write('%d\t%.3e\t%.3e\n' % (dat[0], dat[1], dat[2]))
