# coding=utf-8
# vim: sw=4 et tw=100
"""
code for scattering 2D Inverse problem: BCR-Net
"""
from __future__ import absolute_import
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
parser.add_argument('--noise', type=float, default=0, metavar='noise',
                    help='noise on the measure data (default: %(default)s)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: %(default)s)')
parser.add_argument('--output-suffix', type=str, default=None, metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--data-path', type=str, default='data', metavar='string',
                    help='data path (default: )')
parser.add_argument('--log-path', type=str, default='logs', metavar='string',
                    help='log path (default: )')
parser.add_argument('--percent', type=float, default=.3/.4, metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')
parser.add_argument('--initialvalue', type=str, required=True, metavar='string',
                    help='path to the model weights file (.h5)')
parser.add_argument('--w-comp', type=int, default=1, metavar='N',
                    help='window size of the compress(default: %(default)s)')
args = parser.parse_args()

noise = args.noise
noise_rate = noise / 100.
percent = args.percent
input_prefix = args.input_prefix
output_suffix = args.output_suffix
data_path = args.data_path + '/'
print(f'noise = {noise}')
print(f'input_prefix = {input_prefix}\t output suffix = {output_suffix}')



# Create plots directory
plots_path = 'plots/'
if not os.path.exists(plots_path):
    os.mkdir(plots_path)
    print(f'Created plots directory: {plots_path}')

outputfilename  = plots_path + 'Inference' + input_prefix[7:]
if abs(int(noise) - noise) < 1.e-6:
    outputfilename += "Noises" + str(int(noise))
else:
    outputfilename += "Noises" + str(noise)
outputfilename += output_suffix or str(os.getpid())
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

Nsamples, Ns, Nd = InputArray.shape
assert OutputArray.shape[0] == Nsamples
Nsamples, Nt, Nr = OutputArray.shape
# Nd *= 2
# tmp = InputArray
# tmp2 = np.concatenate([tmp[:, Ns//2:Ns, :], tmp[:, 0:Ns//2, :]], axis=1)
# InputArray = np.concatenate([tmp, tmp2], axis=2)
# InputArray = InputArray[:, :, Nd//4:3*Nd//4]
print('Reading data finished')
Nsamples, Ns, Nd = InputArray.shape
print(f'Input shape is {InputArray.shape}')
print(f'Output shape is {OutputArray.shape}')


# normalize output data -- we'll use these constants to invert
max_out = np.amax(OutputArray)
min_out = np.amin(OutputArray)
pixel_max = max_out - min_out

n_input  = (Ns, Nd)
n_output = (Nt, Nr)
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


n_train = int(Nsamples * percent)
n_test  = min(max(n_train, 5000), Nsamples - n_train)

X_train = InputArray[0:n_train, :, :]
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[n_train:(n_train+n_test), :, :]
Y_test  = OutputArray[n_train:(n_train+n_test), :, :]


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


# model: final model
output("noise rate = %.2e" % noise_rate)

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
Yhat = model.predict(X_test[0:100], 100)
xx_test = X_test[0:100, ...]
yy_test = Y_test[0:100, ...]
yyhat = Yhat[0:100, ...]
# unnormalize output
"""
max / min of the output data are (0.80, 0.00)
max / min of the output data are (2.00, 0.00)
"""


yyhat = yyhat * (0.5 * pixel_max)

# Setup colormaps matching visualize_keras_results.py
try:
    import cmocean
    HAS_CMOCEAN = True
    colormaps = {
        'input': cmocean.cm.balance,     # For scattering data (can be positive/negative)
        'output': cmocean.cm.dense,     # For density/reconstruction
        'pred': cmocean.cm.dense,       # For predictions
        'error': cmocean.cm.diff,       # For error maps
        'phase': cmocean.cm.phase       # For phase information
    }
except ImportError:
    HAS_CMOCEAN = False
    colormaps = {
        'input': 'RdBu_r',
        'output': 'viridis',
        'pred': 'viridis', 
        'error': 'RdBu_r',
        'phase': 'hsv'
    }

# Calculate comprehensive statistics
all_errors = []
all_mse = []
all_mae = []
all_psnr = []

pixel_max = max_out - min_out

for i in range(len(xx_test)):
    # Mean squared error
    mse = np.mean((yyhat[i, :, :] - yy_test[i, :, :]) ** 2)
    all_mse.append(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(yyhat[i, :, :] - yy_test[i, :, :]))
    all_mae.append(mae)
    
    # Relative error
    rel_error = np.linalg.norm(yyhat[i, :, :] - yy_test[i, :, :]) / np.linalg.norm(yy_test[i, :, :])
    all_errors.append(rel_error)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = -10 * np.log10(max(mse / (pixel_max**2), 1e-10))
    all_psnr.append(psnr)

all_errors = np.array(all_errors)
all_mse = np.array(all_mse)
all_mae = np.array(all_mae)
all_psnr = np.array(all_psnr)

print(f'\nComprehensive Error Statistics:')
print(f'  MSE: mean={np.mean(all_mse):.8f}, std={np.std(all_mse):.8f}')
print(f'  MAE: mean={np.mean(all_mae):.8f}, std={np.std(all_mae):.8f}')
print(f'  Rel Error: mean={np.mean(all_errors):.8f}, std={np.std(all_errors):.8f}')
print(f'  PSNR: mean={np.mean(all_psnr):.2f} dB, std={np.std(all_psnr):.2f} dB')

# Find best and worst cases based on PSNR (higher PSNR is better)
best_idx = np.argmax(all_psnr)
worst_idx = np.argmin(all_psnr)
median_idx = np.argsort(all_psnr)[len(all_psnr)//2]

print(f'Best case: Sample {best_idx} (PSNR: {all_psnr[best_idx]:.2f} dB)')
print(f'Worst case: Sample {worst_idx} (PSNR: {all_psnr[worst_idx]:.2f} dB)')

# Create individual comparison plots for each sample (BCR-Net style)
for i in range(min(100, len(xx_test))):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Input (far-field scattering data)
    im1 = axes[0, 0].imshow(xx_test[i, :, :], cmap=colormaps['input'], aspect='equal')
    axes[0, 0].set_title(f'Input: Far-field Scattering Data\nRange: [{np.min(xx_test[i, :, :]):.3f}, {np.max(xx_test[i, :, :]):.3f}]')
    axes[0, 0].set_xlabel('Receiver Position')
    axes[0, 0].set_ylabel('Source Position')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Ground Truth
    im2 = axes[0, 1].imshow(yy_test[i, :, :], cmap=colormaps['output'], aspect='equal')
    axes[0, 1].set_title(f'Ground Truth Reconstruction\nRange: [{np.min(yy_test[i, :, :]):.3f}, {np.max(yy_test[i, :, :]):.3f}]')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Prediction
    im3 = axes[1, 0].imshow(yyhat[i, :, :], cmap=colormaps['pred'], aspect='equal')
    axes[1, 0].set_title(f'Neural Network Prediction\nRange: [{np.min(yyhat[i, :, :]):.3f}, {np.max(yyhat[i, :, :]):.3f}]')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Error map
    error_data = yyhat[i, :, :] - yy_test[i, :, :]
    error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
    im4 = axes[1, 1].imshow(error_data, cmap=colormaps['error'], 
                           vmin=-error_max, vmax=error_max, aspect='equal')
    axes[1, 1].set_title(f'Prediction Error (Pred - GT)\nMSE: {all_mse[i]:.6f}, PSNR: {all_psnr[i]:.2f} dB')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.suptitle(f'BCR-Net Scattering Inverse Problem - Sample {i}', fontsize=16)
    plt.tight_layout()
    
    # Save individual plot
    plot_filename = plots_path + outputfilename.split('/')[-1].replace('.txt', f'_sample_{i:03d}_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    if i % 10 == 0:
        print(f"Saved plot for sample {i}")

# Create best/worst cases comparison (visualize_keras_results style)
print("Creating best/worst cases plot...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, (idx, case_name) in enumerate([(best_idx, 'Best'), (worst_idx, 'Worst')]):
    input_data = xx_test[idx, :, :]
    gt_data = yy_test[idx, :, :]
    pred_data = yyhat[idx, :, :]
    error_data = pred_data - gt_data
    
    # Input
    im1 = axes[i, 0].imshow(input_data, cmap=colormaps['input'], aspect='equal')
    axes[i, 0].set_title(f'{case_name} Case - Input')
    plt.colorbar(im1, ax=axes[i, 0], shrink=0.6)
    
    # Ground truth
    im2 = axes[i, 1].imshow(gt_data, cmap=colormaps['output'], aspect='equal')
    axes[i, 1].set_title(f'{case_name} Case - Ground Truth')
    plt.colorbar(im2, ax=axes[i, 1], shrink=0.6)
    
    # Prediction
    im3 = axes[i, 2].imshow(pred_data, cmap=colormaps['pred'], aspect='equal')
    axes[i, 2].set_title(f'{case_name} Case - Prediction')
    plt.colorbar(im3, ax=axes[i, 2], shrink=0.6)
    
    # Error
    error_max = max(abs(np.min(error_data)), abs(np.max(error_data)))
    im4 = axes[i, 3].imshow(error_data, cmap=colormaps['error'],
                           vmin=-error_max, vmax=error_max, aspect='equal')
    axes[i, 3].set_title(f'{case_name} Case - Error\nPSNR: {all_psnr[idx]:.2f} dB')
    plt.colorbar(im4, ax=axes[i, 3], shrink=0.6)

plt.tight_layout()
best_worst_filename = plots_path + outputfilename.split('/')[-1].replace('.txt', '_best_worst_cases.png')
plt.savefig(best_worst_filename, dpi=300, bbox_inches='tight')
plt.close()

# Create error statistics plot (visualize_keras_results style)
print("Creating error statistics plot...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# MSE distribution
axes[0, 0].hist(all_mse, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(np.mean(all_mse), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_mse):.6f}')
axes[0, 0].set_xlabel('Mean Squared Error')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('MSE Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PSNR distribution
axes[0, 1].hist(all_psnr, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].axvline(np.mean(all_psnr), color='red', linestyle='--',
                  label=f'Mean: {np.mean(all_psnr):.2f} dB')
axes[0, 1].set_xlabel('PSNR (dB)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('PSNR Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Relative error distribution
axes[1, 0].hist(all_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].axvline(np.mean(all_errors), color='red', linestyle='--',
                  label=f'Mean: {np.mean(all_errors):.6f}')
axes[1, 0].set_xlabel('Relative Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Relative Error Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Error correlation
axes[1, 1].scatter(all_mse, all_psnr, alpha=0.6, s=20)
axes[1, 1].set_xlabel('Mean Squared Error')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('MSE vs PSNR Correlation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
stats_filename = plots_path + outputfilename.split('/')[-1].replace('.txt', '_error_statistics.png')
plt.savefig(stats_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"All plots saved!")
print(f"Best/Worst cases: {best_worst_filename}")
print(f"Error statistics: {stats_filename}")
print(f"Individual plots: {plots_path + outputfilename.split('/')[-1].replace('.txt', '_sample_XXX_comparison.png')}")

output(f'Generated {len(xx_test)} individual plots and 2 summary plots')
output(f'Mean relative error: {np.mean(all_errors):.6f} ± {np.std(all_errors):.6f}')
output(f'Best case: Sample {best_idx} (PSNR: {all_psnr[best_idx]:.2f} dB)')
output(f'Worst case: Sample {worst_idx} (PSNR: {all_psnr[worst_idx]:.2f} dB)')


