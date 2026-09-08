from datetime import datetime
import os 
from pathlib import Path
import xarray as xr
import warnings
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model classes
class BaseParametrization:
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices):
        super().__init__()

        # Parametrization params
        self.latent_dims = latent_dims
        self.past_timesteps = past_timesteps
        self.memory_indices = memory_indices
        if self.memory_indices:
            self.n_memory_features = len(self.memory_indices)
        else:
            self.n_memory_features = 0
        self.n_features = n_features
        self.model_name = None
        
        # Training data params
        self.mean = None
        self.std = None
        self.region = None
        self.surface = None
        self.precip_threshold = None
        self.rescale_precip = None
        self.tstart = None
        self.train_ind = None
        self.val_ind = None
        self.train_share = None
        self.val_share = None

        # training params
        self.weight_decay = None
        self.learning_rate = None
        self.alpha = None
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.R2 = []
        self.precip_val_loss = []
        self.recon_val_loss = []

        # interpretability metrics
        self.linear_predictability = None
        self.nonlinear_predictability = None

        # other
        self.save_path = None

    def set_metadata(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f'Warning: BaseParametrization or Subclass has no attribute {key}, but got value {value} in kwargs. This key-value pair will be ignored.')

    def save(self, additional_info=None, save_path=None):
        # Save the model
        if hasattr(self, 'X'):
            self.X = None  # remove data before saving
        
        if save_path:
            torch.save(self, save_path)
        else:
            save_dir = Path(f'networks/{self.model_name}/n_features={self.n_features}_memory_indices={self.memory_indices}_latent_dims={self.latent_dims}_past_timesteps={self.past_timesteps}')
            if not save_dir.exists(): 
                os.makedirs(save_dir) 
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            if additional_info:
                save_file = f'{timestamp}_{additional_info}.pkl'
            else:
                save_file = f'{timestamp}.pkl'
            save_path = f'{save_dir}/{save_file}'
            torch.save(self, save_path)
            
        return save_path

class ODE_Z(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, path_ODE, path_AE, path_NN, model_name, K=8, dt=.001):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps, latent_dims=latent_dims, n_neighbours=n_neighbours)
        '''
        This class is a wrapper that allows the interplay between the existing L96 implementation and a fitted pysindy model
        path: Path to pysindy model
        '''
        with open(path_ODE, 'rb') as file:
            # self.model = pickle.load(file)
            self.coefs = np.load(file)
            self.coefs = torch.tensor(self.coefs, dtype=torch.float32).T  
        
        with open(path_AE, 'rb') as file:
            self.AE = torch.load(file, map_location='cpu', weights_only=False)
        
        with open(path_NN, 'rb') as file:
            self.NN = torch.load(file, map_location='cpu', weights_only=False)
    
        self.model_name = model_name
        self.latent_dims = latent_dims
        self.past_timesteps = past_timesteps
        self.memory_cutoff = m
        self.tau = tau
        self.dt = torch.tensor(dt, dtype=torch.float32)
        self.K = K
        self.one = torch.ones((self.K, 1), dtype=torch.float32)

    def _rhs_Z_dt(self, z, x):
        # return self.model.predict(x=z, u=x)
        # return self.dt * torch.sum(self.coefs[:, 0] + z * self.coefs[:, 1: -1] + self.coefs[:, -1] * x, dim=1)
        zx = torch.cat((self.one, z, x.reshape(self.K, 1)), dim=1)
        return self.dt * zx @ self.coefs

    def forward(self, z, x):
        k1 = self._rhs_Z_dt(z, x)
        k2 = self._rhs_Z_dt(z + k1 / 2, x)
        k3 = self._rhs_Z_dt(z + k2 / 2, x)
        k4 = self._rhs_Z_dt(z + k3, x)

        return z + (1 /6) * (k1 + 2*k2 + 2*k3 + k4)
    
    
class NNpAEpD(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices, X, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices,)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN+AE+D'
        self.X = X
        self.X = self.X.to(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.past_timesteps * self.n_memory_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.latent_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.past_timesteps * self.n_memory_features)
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + self.n_features, self.nodes_per_layer), 
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )

    def forward(self, x_memory, x_present):
        latent_space = self.encoder(x_memory)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


class ConvNNpAEpD(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices, kernel_size, X, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices)
        
        self.kernel_size = kernel_size
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'ConvNN+AE+D'
        self.X = X
        self.X = self.X.to(device)

        # Length after convolution
        self.conv_out_len = self.past_timesteps - self.kernel_size + 1
        self.conv_out_features = self.past_timesteps * self.n_memory_features

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_memory_features, self.conv_out_features, self.kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_out_features * self.conv_out_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.latent_dims)
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.conv_out_features * self.conv_out_len),
            nn.ReLU(),
            nn.Unflatten(1, (self.conv_out_features, self.conv_out_len)),
            nn.ConvTranspose1d(
                in_channels=self.conv_out_features,
                out_channels=self.n_memory_features,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0
            ),
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + self.n_features, self.nodes_per_layer), 
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )

    def forward(self, x_memory, x_present):
        latent_space = self.encoder(x_memory)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


class NN_ODE(nn.Module, BaseParametrization):
    def __init__(self, ODE, dt, path_to_AE_model, neural_net=None):
        model_AE = torch.load(path_to_AE_model, map_location=device, weights_only=False)
        past_timesteps, latent_dims =  model_AE.past_timesteps, model_AE.latent_dims
        memory_indices = model_AE.memory_indices
        n_features = model_AE.n_features #len(memory_indices)
        nodes_per_layer = model_AE.nodes_per_layer
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices,)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN_ODE'
        self.ODE = ODE
        self.dt = dt
        self.inputs = ['qv2m', 'Prw', 'T2m', 'Ts', 'Fs', 'Fl', 'is_land', 'Precip'] + [f'z{i+1}' for i in range(latent_dims)]
        self.n_forcings = self.n_features + 1  # +1 to keep precip

        if not neural_net:  # No NN was passed so we use the NN from the NN+AE model
            self.neural_net = model_AE.neural_net
        else:
            self.neural_net = neural_net
            warnings.warn('Custom NN was passed as parametrization. If nodes_per_layer was changed you need to set this attribute manauly. Otherwise nodes_per_layer from the AE is used.')

        # Evaluation
        self.loss = []
        self.train_R2 = []
        self.test_R2 = []
        self.N_diverged_coords = []

    def forward(self, coords, XP, Z, t_prediction, t_rollout):
        Z_ED = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.latent_dims), device=device)  # Z_ED saves Z variables at t0 + 1 up to t_prediction
        XP_batch = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.n_forcings), device=device)
        t0 = t_prediction - t_rollout
        z = Z[:, t0, :].reshape(-1, self.latent_dims)  # dims = (coords, prediction_points, laten_dims)
        for delta_t in range(t_rollout): 
            x = XP[:,t0 + delta_t].reshape(-1, self.n_forcings)
            zx = torch.cat((z, x), dim=-1)
            symbols = dict(zip(self.inputs, zx.T))  # Note that we pass precip here, but sympy equations never use the precip symbol
            z = z + self.ODE.forward(**symbols) * self.dt  # Euler Step
            Z_ED[delta_t] = z
            XP_batch[delta_t] = zx[..., self.latent_dims:]
            
        Z_ED_batch = Z_ED.reshape(-1, self.latent_dims)
        XP_batch = XP_batch.reshape(-1, self.n_forcings)  # Last forcing dim carries precip
        P_batch = XP_batch[:, -1].reshape(-1, 1)
        P_pred_batch = self.neural_net(torch.cat((XP_batch[:, :-1], Z_ED_batch), dim=1))  # Predict precip from input except 
        return P_batch, P_pred_batch  # Return precip and predicted precip to compute parametrization loss


class NNbase(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps=0, latent_dims=0, memory_indices=None, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NNbase'

        self.neural_net = nn.Sequential(
            nn.Linear(self.n_features, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )
    
    def forward(self, x_present):
        return self.neural_net(x_present)


class NNpast(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NNpast'

        self.neural_net = nn.Sequential(
            nn.Linear(self.past_timesteps * self.n_memory_features + self.n_features, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )
    
    def forward(self, x):
        return self.neural_net(x)


# Deprecated functions for loading data and generating dataloaders. Current function where moved to training_gpu and data_utils
class TimeSeriesDataset(Dataset):
    def __init__(self, X, past_timesteps, time_series_length):
        """
        X: torch.Tensor of shape (Nfeatures, NT, num_coords)
        Y: torch.Tensor of shape (NT, num_coords)
        past_timesteps: int, number of past timesteps per sample
        """
        self.X = X
        self.past_timesteps = past_timesteps
        self.Nfeatures, self.NT, self.num_coords = X.shape
        self.time_series_length = time_series_length
        self.num_samples = int((self.NT - self.past_timesteps) / self.time_series_length)  # include present
        print((self.NT - self.past_timesteps) / self.time_series_length)

    def __len__(self):
        return self.num_coords * self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x_window: shape (time_series_length + past_timesteps,)
            b_target: scalar
        """
        coord = idx // self.num_samples       # which coordinate
        time_idx = idx % self.num_samples     # which time step

        tstart =  time_idx * self.time_series_length 
        tend = tstart + self.time_series_length + self.past_timesteps
        x_window = self.X[:, tstart: tend, coord]
        # y_target = self.Y[tstart: tend, coord]  # present value

        return x_window #, y_target


class InputOutputDataset(Dataset):
    def __init__(self, X, mask, past_timesteps):
        self.X = X
        self.past_timesteps = past_timesteps
        self.Nfeatures, self.NT, self.num_coords = X.shape
        self.mask = mask
        self.times, self.coords = np.where(mask)
        self.num_samples = len(self.times)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        tstart = self.times[index] - self.past_timesteps
        tend = self.times[index]  
        coord = self.coords[index]
        x = self.X[:-1, tstart:tend + 1, coord]
        y = self.X[-1, tend, coord]
    
        return x, y  # return X, Y (last variable is precip)


def train_test_split(X, train_ind):
    return X[:,:train_ind], X[:,train_ind:]


def generate_dataloaders(X, model, time_series_length=100, train_share=.8, batch_size=64, num_workers=1):
    train_ind = int(X.shape[1] * train_share)
    X_train, X_test = X[:,:train_ind], X[:,train_ind:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, model.past_timesteps, time_series_length=time_series_length)
    test_dataset = TimeSeriesDataset(X_test, model.past_timesteps, time_series_length=time_series_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader


def get_mask(grid, ds, region, surface, precip_threshold):
    region_coords_dict = {'tropical': (-23, 23), 'global': (-180, 180)}
    surface_mask_values_dict = {'ocean': -2, 'ocean_land': None}
    region_mask = (grid.clat <= region_coords_dict[region][1]) & (grid.clat >= region_coords_dict[region][0])
    if surface_mask_values_dict[surface]:
        surface_mask = grid.cell_sea_land_mask == surface_mask_values_dict[surface]
    else: 
        surface_mask = np.ones_like(grid.cell_sea_land_mask, dtype=bool)
    
    mask = (ds.precip >= precip_threshold) & surface_mask & region_mask
    return mask


def generate_masked_dataloaders(X, model, mask, train_share=.8, val_share=.2, batch_size=64, num_workers=1):
    # Split into train, val, test
    Nt = X.shape[1]
    train_ind = int(Nt * train_share)
    val_ind = int(Nt * (train_share + val_share))
    X_train, X_val, X_test = X[:,:train_ind], X[:,train_ind:val_ind], X[:,val_ind:]
    mask_train, mask_val, mask_test = mask[:train_ind], mask[train_ind:val_ind], mask[val_ind:]
    mask_train[:model.past_timesteps] = False  # ensure that time steps with insofficient history are not used for training
    mask_val[:model.past_timesteps] = False  # ensure that time steps with insofficient history are not used for training
    mask_test[:model.past_timesteps] = False  # ensure that time steps with insofficient history are not used for training

    # Create datasets
    train_dataset = InputOutputDataset(X_train, mask_train, model.past_timesteps)
    val_dataset = InputOutputDataset(X_val, mask_val, model.past_timesteps)
    test_dataset = InputOutputDataset(X_test, mask_test, model.past_timesteps)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


# Depracated Model training
def reconstruction_loss(x_batch, x_present, y_batch, model, criterion, reconstruction_criterion, alpha):
    b_pred, x_recon = model(x_batch, x_present)
    loss = alpha * criterion(b_pred, y_batch) + (1 - alpha) * reconstruction_criterion(x_recon, x_batch)
    return loss

def prediction_loss(_, x_batch, y_batch, model, criterion):
    b_pred = model(x_batch)
    loss = criterion(b_pred, y_batch)
    return loss

def process_past_timesteps(x, n_features, past_timesteps, mask_memory, mask_present):
    x = x.to(device)
    X_lag = x.permute(2, 0, 1)  # change dimension order to (time, batch, variable)
    X_lag = X_lag.unfold(0, past_timesteps + 1, 1)  # create lagged features, exclude present timestep to align with X_present
    X_lag = X_lag.reshape(-1, (n_features + 1) * (past_timesteps + 1))
    X_present = X_lag[:, mask_present]
    Y = X_present[:, -1].reshape(-1, 1)  # Precip is safed as the last variable
    X_present = X_present[:, :-1]  # Drop target from features
    X_lag = X_lag[:, mask_memory]
    return X_lag.to(device), X_present.to(device), Y.to(device)    

def process_pt(x_batch, y_batch):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    x_past = x_batch[:, :, :-1]  # all but last variable
    # x_past = x_past.reshape(-1, x_past.shape[1] * x_past.shape[2])  # flatten time and feature dimensions
    x_present = x_batch[:, :, -1]  # last variable
    y_batch = y_batch.reshape(-1, 1)
    return x_past, x_present, y_batch

def process_NN(x_batch, y_batch):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    x_present = x_batch[:, :, -1]  # last variable
    y_batch = y_batch.reshape(-1, 1)
    return None, x_present, y_batch

def process_NN_past(x_batch, y_batch):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    x = x_batch.reshape(x_batch.shape[0], -1)  # all variables, flatten time and feature dimesnsions
    y_batch = y_batch.reshape(-1, 1)
    return None, x, y_batch

def train_model_deprecated(train_loader, test_loader, model, num_epochs=5, weight_decay=1.e-6, lr=0.0005):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.learning_rate = lr
    model.weight_decay = weight_decay

    pbar = trange(num_epochs, desc="Training", ncols=150)
    # Set loss function
    if model.model_name in ['ConvNN+AE+D', 'NN+AE+D']:
        reconstruction_criterion = nn.MSELoss()
        loss_func = reconstruction_loss
        alpha = 1.0
        loss_args = (reconstruction_criterion, alpha)
    elif model.model_name in ['NNbase', 'NNpast']:
        loss_func = prediction_loss
        loss_args = ()
    else:
        raise ValueError(f'No loss function set for {model.model_name}')
    
    # Set preprocessing function
    if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
        process_func = process_pt
        process_args = ()
        # mask_memory = torch.zeros((model.n_features + 1, model.past_timesteps + 1), dtype=torch.bool, device='cpu')
        # mask_present = torch.zeros((model.n_features + 1, model.past_timesteps + 1), dtype=torch.bool, device='cpu')
        # mask_present[:, -1] = True  # only get last value
        # mask_present = mask_present.flatten()
        # mask_memory[model.memory_indices] = True  # only get variables from desired indices
        # mask_memory[:, -1] = False  # discard present value
        # mask_memory = mask_memory.flatten()
        # process_args = (model.n_features, model.past_timesteps, mask_memory, mask_present)
    elif model.model_name in ['NNbase']:
        process_func = process_NN
        process_args = ()
    elif model.model_name in ['NNpast']:
        process_func = process_NN_past
        process_args = ()
    else:
        raise ValueError(f'No pre-processing function set for {model.model_name}')
    
    for epoch in pbar:
        model.train()
        total_train_loss = 0.0

        # Iterate over batches
        for x_batch, y_batch in train_loader:
            # Preprocessing
            x_batch, x_present, y = process_func(x_batch, y_batch, *process_args)

            # Compute loss
            loss = loss_func(x_batch, x_present, y, model, criterion, *loss_args)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                # Preprocessing
                x_batch, x_present, y = process_func(x_batch, y_batch, *process_args)

                # Compute loss
                loss = loss_func(x_batch, x_present, y, model, criterion, *loss_args)
                total_test_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        model.train_loss.append(avg_train_loss)
        model.test_loss.append(avg_test_loss)

        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Test Loss": f"{avg_test_loss:.4f}"
            })

    return model


if __name__=='__main__':
    pass
    
