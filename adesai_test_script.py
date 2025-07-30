from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pandas as pd
import h5py
from scipy.sparse.linalg import lsqr
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm  # For progress bars
import cmocean as cmo
import sys
from torch.serialization import add_safe_globals
import tempfile
import subprocess
import os
import matplotlib 
matplotlib.use('Agg') 

device = torch.device('cuda')
print("Using device:", device)

# Add the src directory so both core/ and utils/ are visible
# sys.path.append("/home/adesai/scattering/nns/nio_jma/src")
sys.path.append("/home/johnma/nio-jma/src")

# Import and allowlist SNOHelmConv (and optionally NIOHelmPermInv)
import core.nio.helmholtz
add_safe_globals([
    core.nio.helmholtz.SNOHelmConv,
    core.nio.helmholtz.NIOHelmPermInv  # include this to avoid another safe_globals error later
])

# My GPU doesn't have latex.
# plt.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

delta_noise = 0.0
task = "Shepp"
folder = Path(f"task{str(delta_noise)}")
folder.mkdir(parents=True, exist_ok=True)

# Load data
farf_file_path = "data/shepp.hdf5"
try :
    with h5py.File(farf_file_path, "r") as hdf_file:
        image = hdf_file["image"][:]
        farfield_real = hdf_file["farfield.real"][:]
        farfield_imag = hdf_file["farfield.imag"][:]
except Exception as e:
    print(f"Error reading HDF5 file: {e}")
    sys.exit(1)


# Params
N_TEST_SAMPLES = image.shape[1]
image_Ngrid = 100
Ngrid = 100
nfar = 100

# farfield
farfield_real = farfield_real.T.reshape(N_TEST_SAMPLES, nfar, nfar)
farfield_imag = farfield_imag.T.reshape(N_TEST_SAMPLES, nfar, nfar)
total_farf = farfield_real + 1J*farfield_imag
total_farf = torch.tensor(total_farf, dtype=torch.complex64, device=device)

# Noise, Z ~ CN(0,1)
Z_r = torch.randn(total_farf.shape, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
Z_i = torch.randn(total_farf.shape, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
Z = Z_r + 1j * Z_i
farf_noise = (1 + delta_noise * Z) * total_farf
farf_real = farf_noise.real.cpu().numpy()
farf_imag = farf_noise.imag.cpu().numpy()


with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_filename = temp_file.name
    try:
    # Save data to HDF5 file in the same format as your original data
        with h5py.File(temp_filename, 'w') as f:
            # Transpose back to original format (features, samples)
            f.create_dataset('farfield.real', data=farf_real)

        # Copy to data directory with a known name
        bcr_data_file = '/home/johnma/bcr-scattering-inv/data/temp_inference_data.hdf5'
        os.system(f'cp {temp_filename} {bcr_data_file}')
        
        # Run BCR inference
        print("Executing subprocess command::run_bcr.sh")
        result = subprocess.run(['bash', 'run_bcr.sh'], capture_output=True, text=True)

        print("BCR Output:", result.stdout)
        if result.stderr:
            print("BCR Errors:", result.stderr)
            
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(bcr_data_file):
            os.remove(bcr_data_file)

with h5py.File(f"BCRpredictions-delta{str(delta_noise)}.hdf5", "r") as hdf_file:
    BCR_opt_preds = hdf_file["predictions"][:]


# CNN, BCNN, CNNB
farfield = np.stack([farf_real, farf_imag], axis=1)
farfield_tensor = torch.tensor(farfield, dtype=torch.float32, device=device)
print(farfield_tensor.shape)
test_loader = DataLoader(TensorDataset(farfield_tensor), batch_size=32, shuffle=False)

# NIO Farfield (normalized)
farf_real = torch.tensor(farf_real)
farf_imag = torch.tensor(farf_imag)
"""
these should be fixed from training.

min_data_real: -3.505291223526001, max_data_real: 2.7653729915618896
min_data_imag: -2.815202236175537, max_data_imag: 4.5707292556762695
min_model: 0.0, max_model: 0.7999973297119141
"""
min_data_real = -3.505291223526001
max_data_real = 2.7653729915618896 
min_data_imag = -2.815202236175537
max_data_imag = 4.5707292556762695 
farf_real_NIO = 2 * (farf_real - min_data_real) / (max_data_real - min_data_real) - 1.
farf_imag_NIO = 2 * (farf_imag - min_data_imag) / (max_data_imag - min_data_imag) - 1.
farfieldNIO = torch.stack([farf_real_NIO, farf_imag_NIO], dim=-1)
farfieldNIO = farfieldNIO.view(-1, 2, 100, 100)  # Reshape to (N_TEST_SAMPLES, 2, 100, 100)
print(farfieldNIO.shape)
farfieldNIO = farfieldNIO.to(device)
farfieldNIO = (1+torch.randn(farfieldNIO.shape).to(device)*delta_noise)*farfieldNIO  # Add mult noise
test_loader_NIO = DataLoader(TensorDataset(farfieldNIO), batch_size=32, shuffle=False)

# Ground truth
farfields = farfield_real + 1J*farfield_imag
images = image.T.reshape(N_TEST_SAMPLES, image_Ngrid, image_Ngrid)

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNModel, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = 2

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.flatten_size = self._compute_flatten_size(input_shape)
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, 100 * 100)
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)

        x = x.view(x.size(0), -1)

        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)
        x = self.output_layer(x)
        x = x.view(x.size(0), 100, 100)
        return x

class BCNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(BCNNModel, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        self.flatten_size = self._compute_flatten_size(input_shape)
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, output_shape[0] * output_shape[1])
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)
        x = x.view(x.size(0), -1)
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)
        x = self.output_layer(x)

        x = x.view(x.size(0), 100, 100)
        return x

class CNNBModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNBModel, self).__init__()

        self.output_shape = output_shape
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        self.flatten_size = self._compute_flatten_size(input_shape)

        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        self.output_layer = nn.Linear(input_fc_dim, int(np.prod(output_shape)))

        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)
            x = self.activation_fn(x)
            x = self.cnn_layers[i + 1](x)
            x = self.cnn_layers[i + 2](x)
        x = x.view(x.size(0), -1)
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)
            x = self.activation_fn(x)
            x = self.fc_layers[i + 1](x)

        x = self.output_layer(x)
        x = x.view(-1, *self.output_shape)

        return x

input_shape1 = (2, nfar, nfar)
input_shape2 = (1, Ngrid, Ngrid)

output_shape1 = (2, nfar, nfar)
output_shape2 = (Ngrid, Ngrid)

num_cnn_layers = 4
channels_per_layer = [125, 358, 426, 221]
num_fc_layers = 1
fc_units = [576]
activation_function = nn.GELU()
dropout_rate = 0

CNNB_opt = CNNBModel(input_shape1, output_shape1, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

num_cnn_layers = 4
channels_per_layer = [335, 33, 195, 65]
num_fc_layers = 1
fc_units = [971]
activation_function = nn.GELU()
dropout_rate = 0

BCNN_opt = BCNNModel(input_shape1, output_shape2, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

num_cnn_layers = 4
channels_per_layer = [296, 211, 152, 61]
num_fc_layers = 3
fc_units = [537, 465, 419]
activation_function = nn.GELU()
dropout_rate = 0

CNN_opt = CNNModel(input_shape1, output_shape2, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_function, dropout_rate)

CNNB_opt = CNNB_opt.to(device)
BCNN_opt = BCNN_opt.to(device)
CNN_opt = CNN_opt.to(device)

CNNB_opt.load_state_dict(torch.load('models/CNNB_tuned_model_kvalue1.pt'))
BCNN_opt.load_state_dict(torch.load('models/BCNN_tuned_model_kvalue1.pt'))
CNN_opt.load_state_dict(torch.load('models/CNN_tuned_model_kvalue1.pt'))

CNNB_opt.eval()
BCNN_opt.eval()
CNN_opt.eval()

CNNB_opt_preds = []
BCNN_opt_preds = []
CNN_opt_preds = []

print("CNNB Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = CNNB_opt(inputs)
        CNNB_opt_preds.append(outputs.cpu().numpy())
CNNB_opt_preds = np.concatenate(CNNB_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, 2, nfar, nfar)

print("BCNN Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = BCNN_opt(inputs)
        BCNN_opt_preds.append(outputs.cpu().numpy())
BCNN_opt_preds = np.concatenate(BCNN_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, nfar, nfar)

print("CNN Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = CNN_opt(inputs)
        CNN_opt_preds.append(outputs.cpu().numpy())
CNN_opt_preds = np.concatenate(CNN_opt_preds, axis=0)  # Shape: (N_TEST_SAMPLES, Ngrid, Ngrid)

# Make grid for NIO's DeepONet
grid_x = np.tile(np.linspace(0, 1, 100), (100, 1))
grid_y = np.tile(np.linspace(0, 1, 100), (100, 1)).T
grid = torch.tensor(np.stack([grid_y, grid_x], axis=-1)).type(torch.float32).to(device)

NIO_opt = torch.load("models/nio-model.pkl", map_location=device, weights_only=False)
NIO_opt.eval()
NIO_opt_preds = []

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters for CNNB model: {count_parameters(CNNB_opt)}")
print(f"Number of trainable parameters for CNNB model: {count_trainable_parameters(CNNB_opt)}")
print(f"Number of parameters for BCNN model {count_parameters(BCNN_opt)}")
print(f"Number of trainable parameters for BCNN model: {count_trainable_parameters(BCNN_opt)}")
print(f"Number of parameters for CNN model: {count_parameters(CNN_opt)}")
print(f"Number of trainable parameters for CNN model: {count_trainable_parameters(CNN_opt)}")
print(f"Number of parameters for NIO model: {count_parameters(NIO_opt)}")
print(f"Number of trainable parameters for NIO model: {count_trainable_parameters(NIO_opt)}")


print("NIO Prediction")
with torch.no_grad():
    for (inputs,) in tqdm(test_loader_NIO, desc="Testing", leave=False):
        inputs = inputs.to(device)
        outputs = NIO_opt(inputs, grid)
        # unnormalize, these constants are fixed from training
        min_model = 0.0 
        max_model = 0.7999973297119141
        outputs = 0.5 * (outputs + 1) * (max_model - min_model) + min_model
        NIO_opt_preds.append(outputs.cpu().numpy())

NIO_opt_preds = np.concatenate(NIO_opt_preds, axis=0)  # shape: (N_TEST_SAMPLES, 100, 100)

# Build Born Matrix
def discretize_born(k, xlim, phi, Ngrid, theta):
    vert_step = 2 * xlim / Ngrid
    hor_step = 2 * xlim / Ngrid
    Cfac = vert_step * hor_step * np.exp(1j * np.pi / 4) * np.sqrt(k**3 / (np.pi * 8))
    y1 = np.linspace(-xlim, xlim, Ngrid)
    y2 = np.linspace(-xlim, xlim, Ngrid)
    Y1, Y2 = np.meshgrid(y1, y2)
    grid_points = np.column_stack((Y1.ravel(), Y2.ravel()))
    xhat = np.array([np.cos(theta), np.sin(theta)]).T
    d = np.array([np.cos(phi), np.sin(phi)])
    diff = xhat - d
    dot_products = np.dot(diff, grid_points.T)
    Exp = np.exp(1j * k * dot_products)
    A = Cfac * Exp
    return A

def build_born(incp, farp, kappa, xlim, Ngrid):
    phi=np.zeros(incp["n"])
    center=incp["cent"]
    app=incp["app"]
    for ip in range(0,incp["n"]):
        if incp["n"]==1:
            phi[0]=0.0
        else:
            phi[ip]=(center-app/2)+app*ip/(incp["n"]-1)
    ntheta=farp["n"]
    ctheta=farp["cent"]
    apptheta=farp["app"]
    theta=np.zeros(ntheta)
    for jp in range(0,ntheta):
        theta[jp]=(ctheta-apptheta/2)+apptheta*jp/(ntheta-1)
    born_operator_list = [discretize_born(kappa, xlim, inc_field, Ngrid, theta) for inc_field in phi]
    operator_combined = np.vstack(born_operator_list)
    return operator_combined

ninc = 100
nfar = 100
incp = {"n": ninc, "app": 2*np.pi, "cent":0}
farp = {"n": nfar, "app": 2*np.pi, "cent":0}
kappa = 16
Ngrid = 100
xlim = 1
born = build_born(incp, farp, kappa, xlim, Ngrid)

ticks = [0, 50, 99]
tick_labels = [r"$-1$", r"$0$", r"$1$"]

l2error = {"Born1": [], "NIO": [], "CNNB": [], "BCNN": [], "CNN": [], 'BCR': []}
l1error = {"Born1": [], "NIO": [], "CNNB": [], "BCNN": [], "CNN": [], 'BCR': []}

for i in tqdm(range(N_TEST_SAMPLES), desc="Calculating errors", leave=False):
    ground_truth = images[i].flatten()
    true_farfield =  farfields[i].reshape(nfar*nfar,1)

    NIO_opt_pred = NIO_opt_preds[i]
    BCR_opt_pred = BCR_opt_preds[i]
    CNNB_opt_pred = (CNNB_opt_preds[i][0]+1J*CNNB_opt_preds[i][1]).reshape(nfar*nfar,1)
    BCNN_opt_pred = BCNN_opt_preds[i].reshape(Ngrid,Ngrid)
    CNN_opt_pred = CNN_opt_preds[i].reshape(Ngrid,Ngrid)

    born1 = np.real(lsqr(born, true_farfield, damp=1e0)[0]).flatten()

    nio = NIO_opt_pred.flatten()
    nn1 = np.real(lsqr(born, CNNB_opt_pred, damp=1e0)[0]).flatten()
    nn2 = BCNN_opt_pred.flatten()+born1
    nn3 = CNN_opt_pred.flatten()
    bcr = BCR_opt_pred.flatten()

    l2gt = np.linalg.norm(ground_truth, ord=2)

    l2error["Born1"].append(np.linalg.norm(ground_truth - born1, ord=2)/l2gt)
    l2error["NIO"].append(np.linalg.norm(ground_truth - nio, ord=2)/l2gt)
    l2error["CNNB"].append(np.linalg.norm(ground_truth - nn1, ord=2)/l2gt)
    l2error["BCNN"].append(np.linalg.norm(ground_truth - nn2, ord=2)/l2gt)
    l2error["CNN"].append(np.linalg.norm(ground_truth - nn3, ord=2)/l2gt)
    l2error["BCR"].append(np.linalg.norm(ground_truth - bcr, ord=2)/l2gt)

    l1gt = np.linalg.norm(ground_truth, ord=1)
    l1error["Born1"].append(np.linalg.norm(ground_truth - born1, ord=1)/l1gt)
    l1error["NIO"].append(np.linalg.norm(ground_truth - nio, ord=1)/l1gt)
    l1error["CNNB"].append(np.linalg.norm(ground_truth - nn1, ord=1)/l1gt)
    l1error["BCNN"].append(np.linalg.norm(ground_truth - nn2, ord=1)/l1gt)
    l1error["CNN"].append(np.linalg.norm(ground_truth - nn3, ord=1)/l1gt)
    l1error["BCR"].append(np.linalg.norm(ground_truth - bcr, ord=1)/l1gt)

    nio = nio.reshape(100, 100)
    nn1 = nn1.reshape(100, 100)
    nn2 = nn2.reshape(100, 100)
    nn3 = nn3.reshape(100, 100)
    born1 = born1.reshape(100, 100)
    ground_truth = ground_truth.reshape(100, 100)
    bcr = bcr.reshape(100, 100)

    baseline = [ground_truth+1, born1+1, nio+1, bcr+1]
    baseline_titles = ["(I.) Ground Truth", r"(II.) Born ($\gamma=1$)", r"(III.) NIO", r"(IV.) BCR"]
    nndata = [nn1+1,nn2+1,nn3+1]
    nndata_titles = ["(V.) CNNB", "(VI.) BCNN", "(VII.) CNN"]

    vmin = .8
    vmax = 2

    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Top row: 4 plots (full width)
    axes_top = []
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        axes_top.append(ax)
        
        im1 = ax.imshow(baseline[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
        ax.set_xlabel(baseline_titles[j], fontsize=12)
        ax.set_xticks([0, 50, 99])
        ax.set_xticklabels(tick_labels)
        ax.set_yticks([0, 50, 99])
        ax.set_yticklabels(tick_labels)
    
    # Bottom row: 3 plots (centered) - start from column 0.5 to center them
    axes_bottom = []
    for j in range(3):
        # Position the 3 plots centered: use columns 0.5, 1.5, 2.5 (centered in the 4-column grid)
        ax = fig.add_subplot(gs[1, j:j+1])  # Each plot takes 1 column
        axes_bottom.append(ax)
        
        im2 = ax.imshow(nndata[j], origin="lower", cmap='cmo.dense', vmin=vmin, vmax=vmax)
        ax.set_xlabel(nndata_titles[j], fontsize=12)
        ax.set_xticks([0, 50, 99])
        ax.set_xticklabels(tick_labels)
        ax.set_yticks([0, 50, 99])
        ax.set_yticklabels(tick_labels)
    
    # Adjust the bottom row to be centered
    # Get the position of the bottom row plots and shift them
    for j, ax in enumerate(axes_bottom):
        pos = ax.get_position()
        # Shift right by 0.125 (which is 0.5/4 of the total width) to center the 3 plots
        new_pos = [pos.x0 + 0.125, pos.y0, pos.width, pos.height]
        ax.set_position(new_pos)
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes_top + axes_bottom, orientation="vertical", 
                       fraction=0.02, pad=0.04, aspect=30)
    
    plt.savefig(folder / f'test_result_{i}.png', bbox_inches='tight', dpi=150)
    plt.close(fig) 

l2error["BCNN_mean"] = np.mean(l2error["BCNN"])
l1error["BCNN_mean"] = np.mean(l1error["BCNN"])
l2error["CNNB_mean"] = np.mean(l2error["CNNB"])
l1error["CNNB_mean"] = np.mean(l1error["CNNB"])
l2error["NIO_mean"] = np.mean(l2error["NIO"])
l1error["NIO_mean"] = np.mean(l1error["NIO"])
l2error["Born1_mean"] = np.mean(l2error["Born1"])
l1error["Born1_mean"] = np.mean(l1error["Born1"])
l2error["CNN_mean"] = np.mean(l2error["CNN"])
l1error["CNN_mean"] = np.mean(l1error["CNN"])
l2error["BCR_mean"] = np.mean(l2error["BCR"])
l1error["BCR_mean"] = np.mean(l1error["BCR"])
dfl2 = pd.DataFrame(l2error)
dfl1 = pd.DataFrame(l1error)

print(dfl2)
print(dfl1)

dfl2.to_csv(f"l2_{task}-delta{str(delta_noise)}.csv", index=False)
dfl1.to_csv(f"l1_{task}-delta{str(delta_noise)}.csv", index=False)
