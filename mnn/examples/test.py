# coding=utf-8
# vim: sw=4 et tw=100
"""
test mnn
"""
from __future__ import absolute_import
import os
import sys
import argparse
import h5py
import numpy as np
sys.path.append(os.path.abspath('../../'))
# ----------------- import keras tools ----------------------
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

# from mnn.layers import CNNR1D, CNNK1D, CNNI1D
from mnn.callback import SaveBestModel
from mnn.utils import MNNH2model, MNNHmodel, MNNHmodel1D, MNNH2model1D

# ---- define input parameters and set their default values ---
parser = argparse.ArgumentParser(description='NLSE - MNN-H2')
parser.add_argument('--epoch', type=int, default=4000, metavar='N',
                    help='input number of epochs for training (default: %(default)s)')
parser.add_argument('--input-prefix', type=str, default='nlse2v2', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='number of channels for training (default: %(default)s)')
parser.add_argument('--k-grid', type=int, default=7, metavar='N',
                    help='number of grids (L+1, N=2^L*m) (default: %(default)s)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='number CNN layers (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                    help='batch size (default: #train samples/50)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: %(default)s)')
parser.add_argument('--output-suffix', type=str, default=None, metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=0.5, metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')
parser.add_argument('--initialvalue', type=str, default='', metavar='filename',
                    help='filename storing the weights of the model (default: '')')
args = parser.parse_args()
# === setup: parameters
N_epochs = args.epoch
alpha = args.alpha
k_multigrid = args.k_grid
L = k_multigrid - 1
N_cnn = args.n_cnn
lr = args.lr

# ---------------- code for logs ---------------
data_path = '../../../NNPDE/code/NLSE/data/'
log_path = 'logs/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

outputfilename = log_path + 'tH2L' + str(k_multigrid) + 'Nc' + str(N_cnn) + 'Al' + str(alpha)
outputfilename += args.output_suffix or str(os.getpid())
modelfilename = outputfilename + '.h5'
outputfilename += '.txt'
log_os = open(outputfilename, "w+")

def output(obj):
    print(obj)
    log_os.write(str(obj)+'\n')

def outputnewline():
    log_os.write('\n')
    log_os.flush()


# ---------- prepare the training and test data sets ----------
filenameIpt = data_path + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_path + 'Output_' + args.input_prefix + '.h5'
# === import data with size: Nsamples * Nx
fInput      = h5py.File(filenameIpt, 'r')
InputArray  = fInput['Input'][:]
fOutput     = h5py.File(filenameOpt, 'r')
OutputArray = fOutput['Output'][:]

[Nsamples, Nx] = InputArray.shape
assert OutputArray.shape == (Nsamples, Nx)

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("Nx                      = %d" % Nx)
output("Nsamples                = %d" % Nsamples)
outputnewline()

# === training and test data
n_train = int(Nsamples * args.percent)
n_test  = min(max(n_train, 5000), Nsamples - n_train)

BATCH_SIZE = args.batch_size or n_train // 50

# === pre-treat the data
mean_out = np.mean(OutputArray[0:n_train, :])
mean_in  = np.mean(InputArray[0:n_train, :])
InputArray /= mean_in * 2
InputArray -= 0.5
OutputArray -= mean_out
output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))

X_train = InputArray[0:n_train, :]
Y_train = OutputArray[0:n_train, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :]

# X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
# X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1],  1])

# --------------- prepare parameters for MNN ------------------
n_input = Nx
n_output = Nx
output("[n_input, n_output] = [%d, %d]" % (n_input, n_output))
output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))

# parameters: Nx = m*2^L
m = Nx // (2**L)
output('m = %d' % m)

# ----------------- functions used in MNN --------------------
# === calculate the relative error of the training/test data sets
def test_data(model, X, Y):
    Yhat = model.predict(X, batch_size=max(BATCH_SIZE, 1000))
    errs = np.linalg.norm(Y - Yhat, axis=1) / np.linalg.norm(Y+mean_out, axis=1)
    return errs

def check_result(model):
    return (test_data(model, X_train, Y_train), test_data(model, X_test, Y_test))


# ---------- architecture of MNN-H2 -------------------
# read it with the Algorithm 4 of arXiv:1807.01883
Ipt = Input(shape=(n_input, ))  # Ipt = v
Opt = MNNH2model1D(Ipt, L, N_cnn, alpha, layer='CNN', bc_padding='period')

# === model
model = Model(inputs=Ipt, outputs=Opt)
plot_model(model, to_file='mnnH2.png', show_shapes=True)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

model.optimizer.lr = (lr)

# if args.initialvalue is given, read weights from the file
if len(args.initialvalue) > 3:
    model.load_weights(args.initialvalue, by_name=False)
    model.optimizer.lr = (lr/10)
    output('initial the network by %s\n' % args.initialvalue)
    (err_train, err_test) = check_result(model)
    output("ave/max error of train/test data:\t %.1e %.1e \t %.1e %.1e " %
           (np.mean(err_train), np.mean(err_test),
            np.amax(err_train), np.amax(err_test)))

save_best_model = SaveBestModel(modelfilename, check_result=check_result, period=10)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epochs, verbose=args.verbose,
          callbacks=[save_best_model])
log_os.close()

# === save summary of results
tr_os = open('trainresultH2.txt', "a")
tr_os.write('%s\t%s\t%d\t%d\t%d\t' % (args.input_prefix, modelfilename, alpha, k_multigrid, N_cnn))
tr_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
tr_os.write('%.3e\t%.3e\t' % (save_best_model.best_err_train, save_best_model.best_err_test))
tr_os.write('%.3e\t%.3e\t%.3e\t%.3e' % (save_best_model.best_err_train_max,
                                        save_best_model.best_err_test_max,
                                        save_best_model.best_err_var_train,
                                        save_best_model.best_err_var_test))
tr_os.write('\n')
tr_os.close()
