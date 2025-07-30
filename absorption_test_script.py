import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import h5py
from scipy.sparse.linalg import lsqr
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import cmocean as cmo
import sys
import matplotlib.colors as mcolor
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

file_path = "data/absorption.hdf5"
with h5py.File(file_path, "r") as hdf_file:
    image = hdf_file["image"][:, :]
    farfield_real = hdf_file["farfield.real"][:, :, :]
    farfield_imag = hdf_file["farfield.imag"][:, :, :]

N_TEST_SAMPLES = farfield_real.shape[2]
Ngrid = int(np.sqrt(image.shape[0]))
nfar = int(np.sqrt(farfield_real.shape[0]))

images = image.T.reshape(N_TEST_SAMPLES, Ngrid, Ngrid)

# Split farfields into 4 corruption levels (real + imag)
def build_farfield_tensors(level_idx):
    real = farfield_real[:, level_idx, :].T.reshape(N_TEST_SAMPLES, nfar, nfar)
    imag = farfield_imag[:, level_idx, :].T.reshape(N_TEST_SAMPLES, nfar, nfar)
    complex_ff = real + 1j * imag
    stacked = np.stack([real, imag], axis=1)  # (N_TEST_SAMPLES, 2, nfar, nfar)
    tensor = torch.tensor(stacked, dtype=torch.float32, device=device)
    return complex_ff, tensor

farfields_all, farfield_tensors = [], []
for idx in range(4):
    ff, t = build_farfield_tensors(idx)
    farfields_all.append(ff)
    farfield_tensors.append(t)

test_loaders = [DataLoader(TensorDataset(ft), batch_size=32, shuffle=False) for ft in farfield_tensors]

class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNModel, self).__init__()

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 2

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with linear activation
        self.output_layer = nn.Linear(input_fc_dim, 100 * 100)

        # Activation function saved
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Apply activation function only here
            x = self.cnn_layers[i + 1](x)  # MP
            x = self.cnn_layers[i + 2](x)  # DP

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fc layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dps

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(x.size(0), 100, 100)
        return x

class BCNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(BCNNModel, self).__init__()

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with linear activation
        self.output_layer = nn.Linear(input_fc_dim, output_shape[0] * output_shape[1])

        # Activation function saved
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Apply activation function only here
            x = self.cnn_layers[i + 1](x)  # MP
            x = self.cnn_layers[i + 2](x)  # DP

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fc layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dps

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(x.size(0), 100, 100)
        return x

class CNNBModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_cnn_layers, channels_per_layer, num_fc_layers, fc_units, activation_fn, dropout_rate):
        super(CNNBModel, self).__init__()

        self.output_shape = output_shape  # Store output shape

        # Initialize CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_cnn_layers):
            out_channels = channels_per_layer[i]
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.cnn_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Compute the flatten size dynamically after CNN layers
        self.flatten_size = self._compute_flatten_size(input_shape)

        # Initialize FC layers
        self.fc_layers = nn.ModuleList()
        input_fc_dim = self.flatten_size

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_fc_dim, fc_units[i]))
            self.fc_layers.append(nn.Dropout(dropout_rate))
            input_fc_dim = fc_units[i]

        # Final output layer with flattened output shape
        self.output_layer = nn.Linear(input_fc_dim, int(np.prod(output_shape)))

        # Save activation function
        self.activation_fn = activation_fn

    def _compute_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.cnn_layers:
                dummy_input = layer(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        # Pass through CNN layers
        for i in range(0, len(self.cnn_layers), 3):
            x = self.cnn_layers[i](x)  # Conv
            x = self.activation_fn(x)  # Activation
            x = self.cnn_layers[i + 1](x)  # MaxPool
            x = self.cnn_layers[i + 2](x)  # Dropout

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i in range(0, len(self.fc_layers), 2):
            x = self.fc_layers[i](x)  # Fully connected layer
            x = self.activation_fn(x)  # Activation
            x = self.fc_layers[i + 1](x)  # Dropout

        # Final linear output layer
        x = self.output_layer(x)

        # Reshape to output shape
        x = x.view(-1, *self.output_shape)

        return x

input_shape1 = (2, nfar, nfar)
input_shape2 = (1, Ngrid, Ngrid)
output_shape1 = (2, nfar, nfar)
output_shape2 = (Ngrid, Ngrid)

CNNB_opt = CNNBModel(input_shape1, output_shape1, 4, [125, 358, 426, 221], 1, [576], nn.GELU(), 0)
BCNN_opt = BCNNModel(input_shape1, output_shape2, 4, [335, 33, 195, 65], 1, [971], nn.GELU(), 0)
CNN_opt = CNNModel(input_shape1, output_shape2, 4, [296, 211, 152, 61], 3, [537, 465, 419], nn.GELU(), 0)

CNNB_opt.to(device).eval()
BCNN_opt.to(device).eval()
CNN_opt.to(device).eval()

CNNB_opt.load_state_dict(torch.load('models/CNNB_tuned_model_kvalue1.pt'))
BCNN_opt.load_state_dict(torch.load('models/BCNN_tuned_model_kvalue1.pt'))
CNN_opt.load_state_dict(torch.load('models/CNN_tuned_model_kvalue1.pt'))

# NIO
import sys
sys.path.append("/home/johnma/nio-jma/src")
import core.nio.helmholtz
from torch.serialization import add_safe_globals
add_safe_globals([core.nio.helmholtz.SNOHelmConv, core.nio.helmholtz.NIOHelmPermInv])

NIO_opt = torch.load("models/nio-model.pkl", map_location=device, weights_only=False)
NIO_opt.eval()

def prepare_NIO_farfield(ff_real, ff_imag):
    fr, fi = torch.tensor(ff_real), torch.tensor(ff_imag)
    min_data_real = -3.505291223526001
    max_data_real = 2.7653729915618896 
    min_data_imag = -2.815202236175537
    max_data_imag = 4.5707292556762695
    fr_n = 2 * (fr - min_data_real) / (max_data_real - min_data_real) - 1.
    fi_n = 2 * (fi - min_data_imag) / (max_data_imag - min_data_imag) - 1.
    f = torch.stack([fr_n, fi_n], dim=-1).view(-1, 2, nfar, nfar).to(device).float()
    return f

farfieldNIO_levels = [prepare_NIO_farfield(farfield_real[:, i, :].T.reshape(N_TEST_SAMPLES, nfar, nfar),
                                           farfield_imag[:, i, :].T.reshape(N_TEST_SAMPLES, nfar, nfar))
                      for i in range(4)]
test_loaders_NIO = [DataLoader(TensorDataset(f), batch_size=32, shuffle=False) for f in farfieldNIO_levels]

min_model = 0.0 
max_model = 0.7999973297119141
def unnormalize_NIO(out):
    return 0.5 * (out + 1) * (max_model - min_model) + min_model

grid_x = np.tile(np.linspace(0, 1, Ngrid), (Ngrid, 1))
grid_y = np.tile(np.linspace(0, 1, Ngrid), (Ngrid, 1)).T
grid = torch.tensor(np.stack([grid_y, grid_x], axis=-1), dtype=torch.float32).to(device)

def discretize_born(k, xlim, phi, Ngrid, theta):
    vert_step = 2 * xlim / Ngrid
    hor_step = 2 * xlim / Ngrid
    Cfac = vert_step * hor_step * np.exp(1j * np.pi / 4) * np.sqrt(k ** 3 / (np.pi * 8))
    y1 = np.linspace(-xlim, xlim, Ngrid)
    y2 = np.linspace(-xlim, xlim, Ngrid)
    Y1, Y2 = np.meshgrid(y1, y2)
    grid_points = np.column_stack((Y1.ravel(), Y2.ravel()))
    xhat = np.array([np.cos(theta), np.sin(theta)]).T
    d = np.array([np.cos(phi), np.sin(phi)])
    diff = xhat - d
    dot_products = np.dot(diff, grid_points.T)
    return Cfac * np.exp(1j * k * dot_products)

def build_born(incp, farp, kappa, xlim, Ngrid):
    phi = np.linspace(incp["cent"] - incp["app"] / 2, incp["cent"] + incp["app"] / 2, incp["n"])
    theta = np.linspace(farp["cent"] - farp["app"] / 2, farp["cent"] + farp["app"] / 2, farp["n"])
    return np.vstack([discretize_born(kappa, xlim, p, Ngrid, theta) for p in phi])

born = build_born({"n": 100, "app": 2*np.pi, "cent": 0}, {"n": 100, "app": 2*np.pi, "cent": 0}, 16, 1, Ngrid)

methods = ["Born1", "Born2", "CNN", "BCR", "NIO", "CNNB", "BCNN"]
results = {lvl: {m: {"L2": [], "L1": []} for m in methods} for lvl in ["alpha0", "alpha0.1", "alpha1", "alpha10"]}
titles = [r"$\alpha=0$", r"$\alpha=1/10$", r"$\alpha=1$", r"$\alpha=10$"]

# for i in range(N_TEST_SAMPLES):
for i in range(12, 13):
    # Make keys match later usage (Born1 not LaTeX string)
    sample_store = {"Born1": [],  "NIO": [], "BCR": [],"CNN": [], "CNNB": [], "BCNN": []}

    for level_idx, (ff_complex, loader, loader_NIO) in enumerate(zip(farfields_all, test_loaders, test_loaders_NIO)):

       # extract real part of dataset
        farf_real = []
        for (inputs,) in loader:
            # extract the real part of inputs
            inputs = inputs.to(device)
            farf_real.append(inputs[:, 0, :, :].cpu().numpy())
        farf_real = np.concatenate(farf_real, axis=0)
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
                # print("Executing subprocess command::run_bcr.sh")
                result = subprocess.run(['bash', 'run_bcr.sh'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # print("BCR Output:", result.stdout)
                if result.stderr:
                    print("BCR Errors:", result.stderr)
                    
                
            finally:
            # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                if os.path.exists(bcr_data_file):
                    os.remove(bcr_data_file) 

        delta_noise = 0.0
        with h5py.File(f"BCRpredictions-delta{str(delta_noise)}.hdf5", "r") as hdf_file:
            BCR_preds = hdf_file["predictions"][:]
        # BCR_preds = np.concatenate(BCR_preds, axis=0)

        # NIO inference
        NIO_preds = []
        with torch.no_grad():
            for (inputs,) in loader_NIO:
                inputs = inputs.float()
                outputs = NIO_opt(inputs, grid)
                NIO_preds.append(unnormalize_NIO(outputs).cpu().numpy())
        NIO_preds = np.concatenate(NIO_preds, axis=0)

        # CNN, BCNN, CNNB inference
        CNNB_preds, BCNN_preds, CNN_preds = [], [], []
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(device)
                CNNB_preds.append(CNNB_opt(inputs).cpu().numpy())
                BCNN_preds.append(BCNN_opt(inputs).cpu().numpy())
                CNN_preds.append(CNN_opt(inputs).cpu().numpy())
        CNNB_preds = np.concatenate(CNNB_preds, axis=0)
        BCNN_preds = np.concatenate(BCNN_preds, axis=0)
        CNN_preds = np.concatenate(CNN_preds, axis=0)

        # Reconstructions
        gt = images[i].flatten()
        true_ff = ff_complex[i].reshape(nfar * nfar, 1)

        born1 = np.real(lsqr(born, true_ff, damp=1e0)[0]).flatten()
        born2 = np.real(lsqr(born, true_ff, damp=1e-1)[0]).flatten()

        nn1 = np.real(lsqr(born, (CNNB_preds[i][0] + 1j * CNNB_preds[i][1]).reshape(nfar * nfar, 1), damp=1e0)[0]).flatten()
        nn2 = BCNN_preds[i].flatten() + born1
        nn3 = CNN_preds[i].flatten()
        nio = NIO_preds[i].flatten()
        bcr = BCR_preds[i].flatten()

        # Save for plotting (+1 for refractive index)
        sample_store["Born1"].append(born1.reshape(Ngrid, Ngrid) + 1)
        sample_store["CNN"].append(nn3.reshape(Ngrid, Ngrid) + 1)
        sample_store["NIO"].append(nio.reshape(Ngrid, Ngrid) + 1)
        sample_store["CNNB"].append(nn1.reshape(Ngrid, Ngrid) + 1)
        sample_store["BCNN"].append(nn2.reshape(Ngrid, Ngrid) + 1)
        sample_store["BCR"].append(bcr.reshape(Ngrid, Ngrid) + 1)

        # Relative errors
        norm2, norm1 = np.linalg.norm(gt, 2), np.linalg.norm(gt, 1)
        for name, pred in [("Born1", born1), ("Born2", born2),
                          ("NIO", nio), ("BCR", bcr), ("CNN", nn3), ("CNNB", nn1), ("BCNN", nn2)]:
            results[["alpha0", "alpha0.1", "alpha1", "alpha10"][level_idx]][name]["L2"].append(
                np.linalg.norm(gt - pred, 2) / norm2)
            results[["alpha0", "alpha0.1", "alpha1", "alpha10"][level_idx]][name]["L1"].append(
                np.linalg.norm(gt - pred, 1) / norm1)

    # Plot after all 4 α-levels collected
    fig, axes = plt.subplots(6, 4, figsize=(14, 15), gridspec_kw={'hspace': 0.15}, constrained_layout=False)
    row_labels = [r"Born ($\gamma=1$)", "NIO", "BCR", "CNN", "CNNB", "BCNN"]
    for r, label in enumerate(row_labels):
        axes[r, 0].annotate(label, xy=(0, 0.5), xytext=(-0.2, 0.5),
                            textcoords="axes fraction", ha="center", va="center", fontsize=12, rotation=90)

    vmin = 0.8
    vmax = max(arr.max() for arr in sample_store["NIO"])
    ticks, tick_labels = [0, 50, 99], [r"$-1$", r"$0$", r"$1$"]

    for col, alpha_label in enumerate(titles):
        for r, key in enumerate(sample_store.keys()):
            im = axes[r, col].imshow(sample_store[key][col], origin="lower", cmap='cmo.dense',
                                     vmin=vmin, vmax=vmax)
            axes[r, col].set_xticks(ticks)
            axes[r, col].set_xticklabels(tick_labels)
            axes[r, col].set_yticks(ticks)
            axes[r, col].set_yticklabels(tick_labels)
        axes[0, col].set_title(alpha_label, fontsize=12)

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,  # Make colorbar thicker
        pad=0.05,       # Move colorbar further from the axes
        aspect=25,      # Make colorbar longer
        location="top"  # Place colorbar at the top
    )
    folder = Path("Absorption_Plots_With_BCR_NIO")
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / f"sample_{i}.png", bbox_inches="tight")
    plt.close(fig)

# Save error results per method and alpha-level
# for lvl in results:
#     for method in results[lvl]:
#         pd.DataFrame({
#             "RelL2": results[lvl][method]["L2"],
#             "RelL1": results[lvl][method]["L1"],
#             "RelL2_mean": np.mean(results[lvl][method]["L2"]),
#             "RelL1_mean": np.mean(results[lvl][method]["L1"]),
#         }).to_csv(f"errors_{lvl}_{method}.csv", index=False)