import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from util import transpose_if_1d, return_lagged_input_vector_and_present_k, return_lagged_input_vector_opt
from torchmetrics.regression import R2Score
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm, trange
import pickle
import itertools
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime


# Load initial conditions for L96 model
initX, initY = np.load('./initX.npy'), np.load('./initY.npy')
np.random.seed(123)

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model classes
class BaseParametrization:
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours):
        super().__init__()
        # L96 params
        self.m = m
        self.tau = tau
        self.path_to_training_data = None

        # Parametrization params
        self.latent_dims = latent_dims
        self.past_timesteps = past_timesteps
        self.n_neighbours = n_neighbours
        self.model_name = None
        
        # training params
        self.weight_decay = None
        self.learning_rate = None
        self.train_loss = []
        self.test_loss = []
        self.R2 = []
    
    def save(self, additional_info=None):
        # Save the model
        save_dir = Path(f'networks/{self.model_name}/latent_dims={self.latent_dims}_past_timesteps={self.past_timesteps}_n_neighbours={self.n_neighbours}')
        if not save_dir.exists(): 
            os.makedirs(save_dir) 
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if additional_info:
            save_file = f'm={self.m}_tau={self.tau}_{additional_info}_{timestamp}.pkl'
        else:
            save_file = f'm={self.m}_tau={self.tau}_{timestamp}.pkl'
        save_path = f'{save_dir}/{save_file}'
        torch.save(self, save_path)
        
        return save_path


class NNpAE(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps,
                         latent_dims=latent_dims, n_neighbours=n_neighbours)

        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN+AE'

        self.encoder = nn.Sequential(
            nn.Linear(self.past_timesteps, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.latent_dims)
        )
        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + 1, self.nodes_per_layer),
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
        x_past = x[:, :-1]
        x_present = torch.unsqueeze(x[:, -1], 1)

        latent_space = self.encoder(x_past)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        return y_pred
   

class NNpast(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps,
                                     latent_dims=latent_dims, n_neighbours=n_neighbours)
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'FCNN'
        # Layers
        self.linear1 = nn.Linear(past_timesteps + 1, self.nodes_per_layer)  # number of past time steps the model receives as input
        self.linear2 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear3 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear4 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear5 = nn.Linear(self.nodes_per_layer, 1)  

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        return x
    

class NN(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, model_name, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps,
                                     latent_dims=latent_dims, n_neighbours=n_neighbours)

        self.nodes_per_layer = nodes_per_layer
        self.model_name = model_name

        self.neural_net = nn.Sequential(
            nn.Linear(self.past_timesteps + self.n_neighbours + 1, self.nodes_per_layer),  #n_neighbours+1 as X_k is always passed on top
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
        y_pred = self.neural_net(x)
        return y_pred


class NNpODE(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps,
                                     latent_dims=latent_dims, n_neighbours=n_neighbours)

        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN+ODE'

        self.neural_net = nn.Sequential(
            nn.Linear(self.latent_dims + 1, self.nodes_per_layer),
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
        y_pred = self.neural_net(x)
        return y_pred


class NNpAEpD(nn.Module, BaseParametrization):
    def __init__(self, m, tau, past_timesteps, latent_dims, n_neighbours, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, m=m, tau=tau, past_timesteps=past_timesteps,
                         latent_dims=latent_dims, n_neighbours=n_neighbours)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN+AE+D'

        self.encoder = nn.Sequential(
            nn.Linear(self.past_timesteps, 32),
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
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, past_timesteps)
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + 1, self.nodes_per_layer),  # + 1 to account for present X_k value
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
        x_past = x[:, :self.past_timesteps]
        x_present = x[:, self.past_timesteps:]
        latent_space = self.encoder(x_past)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


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


class TimeSeriesDataset(Dataset):
    def __init__(self, X, B, past_timesteps, time_series_length):
        """
        X: torch.Tensor of shape (NT, num_coords)
        B: torch.Tensor of shape (NT, num_coords)
        past_timesteps: int, number of past timesteps per sample
        """
        self.X = X
        self.B = B
        self.past_timesteps = past_timesteps
        self.NT, self.num_coords = X.shape
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
        x_window = self.X[tstart: tend, coord]
        b_target = self.B[tstart: tend, coord]  # present value

        return x_window, b_target


def get_circular_neighbours(n_neighbours, num_coords):
    # Make lookup array for circular neighbours
    if n_neighbours >= num_coords:
        raise ValueError("n_neighbours must be smaller then K")
    circular_neighbours = []
    for k in range(num_coords):
        ks_subset = [k]
        for i in range(1, 1 + n_neighbours // 2):
            ks_subset.append(k + i)
            ks_subset.append(k - i)
        if n_neighbours % 2 != 0:
            ks_subset.append(k + i + 1)
        circular_neighbours.append(ks_subset)

    circular_neighbours = np.mod(circular_neighbours, num_coords)
    return circular_neighbours


class ShiftedCoordinatesDataset(Dataset):
    def __init__(self, X, B, n_neighbours, time_series_length):
        """
        X: torch.Tensor of shape (NT, num_coords)
        B: torch.Tensor of shape (NT, num_coords)
        past_timesteps: int, number of past timesteps per sample
        """
        self.X = X
        self.B = B
        self.n_neighbours = n_neighbours
        self.NT, self.num_coords = X.shape
        print(self.num_coords, self.n_neighbours)
        self.time_series_length = time_series_length
        self.num_samples = int(self.NT / self.time_series_length)  
        print((self.NT) / self.time_series_length)
        self.circular_neighbours = get_circular_neighbours(self.n_neighbours, self.num_coords)

    def __len__(self):
        return self.num_coords * self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x_window: shape (time_series_length + past_timesteps,)
            b_target: scalar
        """
        coord = idx // self.num_samples       # which coordinate
        coords = self.circular_neighbours[coord]  # coordinates from neighbours
        time_idx = idx % self.num_samples     # which time step
        
        tstart =  time_idx * self.time_series_length 
        tend = tstart + self.time_series_length 
        x_window = self.X[tstart: tend, coords]
        b_target = self.B[tstart: tend, coord]  # present value, only for xk

        return x_window, b_target


class LatentDimsCoordinatesDataset(Dataset):
    def __init__(self, ZX, B):
        """
        X: torch.Tensor of shape (NT, num_coords)
        B: torch.Tensor of shape (NT, num_coords)
        past_timesteps: int, number of past timesteps per sample
        """
        self.ZX = ZX
        self.B = B
        self.num_samples = self.ZX.shape[0]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x_window: shape (time_series_length + past_timesteps,)
            b_target: scalar
        """
        zx_window = self.ZX[idx]
        b_target = self.B[idx]  # present value, only for xk
        return zx_window, b_target


def train_test_split(X, train_ind):
    return torch.tensor(X[:train_ind], dtype=torch.float32), torch.tensor(X[train_ind:], dtype=torch.float32)


def generate_dataloaders(L96, model, time_series_length=10_000, train_share=.8, batch_size=64, num_workers=1):
    X = L96.X.values.astype(np.float32)[2000:]  # discard first two MTU
    B = L96.B.values.astype(np.float32)[2000:]  # discard first two MTU
    train_ind = int(len(X) * train_share)
    X_train, X_test = train_test_split(X, train_ind)
    B_train, B_test = train_test_split(B, train_ind)
    

    # Create datasets
    if model.model_name in ['NN+AE', 'NN+AE+D', 'NNpast']:
        train_dataset = TimeSeriesDataset(X_train, B_train, model.past_timesteps, time_series_length=time_series_length)
        test_dataset = TimeSeriesDataset(X_test, B_test, model.past_timesteps, time_series_length=time_series_length)
    elif model.model_name in ['NN']:
        train_dataset = ShiftedCoordinatesDataset(X_train, B_train, model.n_neighbours, time_series_length)
        test_dataset = ShiftedCoordinatesDataset(X_test, B_test, model.n_neighbours, time_series_length)
    elif model.model_name in ['NN+ODE']:
        Z = L96.Z_ODE.values.astype(np.float32)[2000:]
        Z_train, Z_test = train_test_split(Z.reshape(-1, 8), train_ind)
        X_train, X_test = train_test_split(X.reshape(-1, 1), train_ind)
        B_train, B_test = train_test_split(B.reshape(-1, 1), train_ind)
        ZX_train = torch.hstack((Z_train, X_train))
        ZX_test = torch.hstack((Z_test, X_test))
        train_dataset = LatentDimsCoordinatesDataset(ZX_train, B_train)
        test_dataset = LatentDimsCoordinatesDataset(ZX_test, B_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader


# Model training
def reconstruction_loss(x_batch, b_batch, model, criterion, reconstruction_criterion, alpha):
    b_pred, x_recon = model(x_batch)
    loss = alpha * criterion(b_pred, b_batch) + (1 - alpha) * reconstruction_criterion(x_recon, x_batch[:, :-1])
    return loss, b_pred  # b_pred is important for computing R^2 on the test data. Can be ignored for training data


def prediction_loss(x_batch, b_batch, model, criterion):
    b_pred = model(x_batch)
    loss = criterion(b_pred, b_batch)
    return loss, b_pred  # b_pred is important for computing R^2 on the test data. Can be ignored for training data
    

def process_past_timesteps(x_batch, b_batch, past_timesteps):
    x_batch, b_batch = x_batch.to(device), b_batch.to(device)
    x_batch = x_batch.unfold(1, past_timesteps + 1, 1).reshape(-1, past_timesteps + 1)  # create lagged time series on gpu
    b_batch = b_batch[:, past_timesteps:].reshape(-1, 1)
    return x_batch, b_batch


def process_neighbours(x_batch, b_batch, n_neighbours):
    x_batch, b_batch = x_batch.to(device), b_batch.to(device)
    x_batch = x_batch.reshape(-1, n_neighbours + 1)  # n_neigbours +1 as x_k is included additionally
    b_batch = b_batch.reshape(-1, 1)
    return x_batch, b_batch


def process_nothing(x_batch, b_batch):
    return x_batch.to(device), b_batch.to(device)

def train_model(train_loader, test_loader, model, num_epochs=5, weight_decay=0.0):
    model = model.to(device)
    criterion = nn.MSELoss()
    lr = 0.007
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.learning_rate = lr
    model.weight_decay = weight_decay

    best_R2 = float('-inf')
    best_R2_epoch = 0
    pbar = trange(num_epochs, desc="Training", ncols=150)
    
    # Set loss function
    if model.model_name == 'NN+AE+D':
        reconstruction_criterion = nn.MSELoss()
        loss_func = reconstruction_loss
        alpha = 0.5
        loss_args = (reconstruction_criterion, alpha)
    else:
        loss_func = prediction_loss
        loss_args = ()
    
    # Set preprocessing function
    if model.model_name in ['NN']:
        process_func = process_neighbours
        process_args = (model.n_neighbours,)

    elif model.model_name in ['NN+AE+D', 'NNpast', 'NN+AE']:
        process_func = process_past_timesteps
        process_args = (model.past_timesteps,)
    elif model.model_name in ['NN+ODE']:
        process_func = process_nothing
        process_args = ()

    for epoch in pbar:
        model.train()
        total_train_loss = 0.0

        # Iterate over batches
        for x_batch, b_batch in train_loader:
            # Preprocessing
            x_batch, b_batch = process_func(x_batch, b_batch, *process_args)

            # Compute loss
            loss, _ = loss_func(x_batch, b_batch, model, criterion, *loss_args)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_test_loss = 0.0
        total_R2 = 0.0
        with torch.no_grad():
            for x_batch, b_batch in test_loader:
                # Preprocessing
                x_batch, b_batch = process_func(x_batch, b_batch, *process_args)

                # Compute loss
                loss, b_pred = loss_func(x_batch, b_batch, model, criterion, *loss_args)
                total_test_loss += loss.item()
                total_R2 += R2Score().to(device)(b_pred, b_batch)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        avg_R2 = total_R2 / len(test_loader)

        model.train_loss.append(avg_train_loss)
        model.test_loss.append(avg_test_loss)
        model.R2.append(avg_R2)

        if avg_R2 > best_R2:
            best_R2 = avg_R2
            best_R2_epoch = epoch + 1

        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Test Loss": f"{avg_test_loss:.4f}",
            "R²": f"{avg_R2:.4f}",
            "Best R²": f"{best_R2:.4f} @ {best_R2_epoch}"})

    return model


def train_model_NN(x, x_test, dZdt, dZdt_test, model, num_epochs=5):
    # Send to gpu if available
    model = model.to(device)
    criterion = nn.MSELoss()
    lr = .007
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Make progress bar
    best_R2 = float('-inf')
    best_R2_epoch = 0
    pbar = trange(num_epochs, desc="Training", ncols=150)

    # Training
    for epoch in pbar:
        model.train()
        test_loss, train_loss, R2 = 0, 0, 0

        # Transpose 1-dimensional data
        dZdt_pred = model.forward(x)
        loss = criterion(dZdt_pred, dZdt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            # Transpose 1-dimensional data
            dZdt_pred = model.forward(x_test)
            loss = criterion(dZdt_pred, dZdt_test)
            R2 += R2Score().to(device)(dZdt_pred, dZdt_test)
            test_loss += loss.item()
        
        train_loss = train_loss 
        test_loss = test_loss 
        model.train_loss.append(train_loss)
        model.test_loss.append(test_loss)
        model.R2.append(R2)

        # Update best R²
        if R2 > best_R2:
            best_R2 = R2
            best_R2_epoch = epoch + 1

        # Inline update
        pbar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Test Loss": f"{test_loss:.4f}",
            "R²": f"{R2:.4f}",
            "Best R²": f"{best_R2:.4f} @ {best_R2_epoch}"})
    
    return model


def set_model_metadata(model, past_timesteps, latent_dims, model_name, m, tau):
    model.model_name = model_name
    model.past_timesteps = past_timesteps
    model.latent_dims = latent_dims
    model.memory_cutoff = m
    model.tau = tau
    return model


# Model evaluation
def sensitivity_experiment(pt, models=['baseline_nn', 'nn', 'NN+AE'], latent_dims=5, diag=True):
    nodes_per_layer = 140
    nodes_per_layer_AE = 16
    simulation_time = 100
    num_epochs = 2_000
    print('Latent Dims = ', latent_dims)

    ms = np.logspace(-3, 0, 10)
    taus = np.logspace(-3, 2, 10)
    i=0
    if diag:
        param_pairs = list(zip(ms, taus))
    else:
        param_pairs = list(itertools.product(ms, taus))

    for m, tau in param_pairs:
    # for m, tau in zip(ms, taus):
        ts = time.time()
        print(f'Run experiment for m={m} and tau={tau}')
        # Simulate L96
        # L96 = L96TwoLevelMemory(X_init=initX, Y_init=initY, save_dt=.001, memory_cutoff=m, memory_tau=tau, memory_activation_func=None)
        # L96 = L962LvlMem(X_init=None, Y_init=None, save_dt=.001, m=m, tau=tau, memory_activation_func=None)
        # L96.iterate(simulation_time)
        files = os.listdir('online_runs/NO_PARAMETRIZATION')
        files.sort()
        path = f'online_runs/NO_PARAMETRIZATION/input_lagg=0/time=10000MTU_m={m}_tau={tau}.nc'
        print(path)
        with open(path, 'rb') as file:
                L96 = xr.open_dataset(file)
        Nt = 1000 * 1000 
        L96 = L96.isel(time=slice(0, Nt))

        if 'baseline_nn' in models:
            # Train Baseline:
            print(f'Baseline')
            X_train, X_test, B_train, B_test = generate_data(L96, past_timesteps=0)
            nn_model = FCNN(past_timesteps=0, nodes_per_layer=nodes_per_layer)
            nn_model = train_model(X_train, X_test, B_train, B_test, nn_model, num_epochs=num_epochs)
            nn_model = set_model_metadata(nn_model, past_timesteps=0, latent_dims=None, model_name='baseline_nn', m=m, tau=tau)
            save_model(nn_model)
        
        if 'NN' in models:
            print('new baseline')
            past_timesteps = 0
            Ntrain = int(Nt*.8)
            X_train, X_test = L96.history.X.values[:Ntrain], L96.history.X.values[Ntrain:]
            B_train, B_test = L96.history.B.values[:Ntrain], L96.history.B.values[Ntrain:]
            X_train = torch.tensor(return_lagged_input_vector_and_present_k(X_train.T, past_timesteps), dtype=torch.float32, device=device)
            X_test = torch.tensor(return_lagged_input_vector_and_present_k(X_test.T, past_timesteps), dtype=torch.float32, device=device)
            B_train = torch.tensor(B_train[past_timesteps:].T.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
            B_test = torch.tensor(B_test[past_timesteps:].T.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
            model = NN(past_timesteps=0, n_neighbours=8)
            model = train_model_NN(X_train, X_test, B_train, B_test, model, num_epochs=num_epochs)
            model = set_model_metadata(model, past_timesteps, latent_dims=None, model_name='NN', m=m, tau=tau)
            save_model(model)


        for past_timesteps in [pt]:
            print(past_timesteps)
            if 'nn' in models:
                # Train NN with acces to past
                print(f'NN')
                X_train, X_test, B_train, B_test = generate_data(L96, past_timesteps=past_timesteps)
                nn_model = FCNN(past_timesteps=past_timesteps, nodes_per_layer=nodes_per_layer)
                nn_model = train_model(X_train, X_test, B_train, B_test, nn_model, num_epochs=num_epochs)
                nn_model = set_model_metadata(nn_model, past_timesteps, latent_dims=None, model_name='nn', m=m, tau=tau)
                save_model(nn_model)
            
            if 'NN+AE' in models:
                # Train NN+AE together (without decoder part)
                print('NN+AE')
                if 'nn' in models:
                    pass
                else:  # Data was already generated previously
                    print('generate data')
                    X_train, X_test, B_train, B_test = generate_data(L96, past_timesteps=past_timesteps)
                
                print('init model')
                nnpae_model = NNpAE(past_timesteps=past_timesteps, latent_dims=latent_dims)
                nnpae_model = train_model(X_train, X_test, B_train, B_test, nnpae_model, num_epochs=num_epochs)
                nnpae_model = set_model_metadata(nnpae_model, past_timesteps, latent_dims, 'NN+AE', m, tau)
                #save_model(nnpae_model)
            

        print('Time for experiment [min]: ', (time.time() - ts) / 60)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run L96 sensitivity experiment')
    parser.add_argument('--model_type', type=str, default='nn', help='Model type to use (e.g., nn, rf, svm)')
    parser.add_argument('--past_timesteps', type=int, default=1000, help='Number of past timesteps to consider')
    parser.add_argument('--latent_dims', type=int, default=5, help='Number of past timesteps to consider')

    args = parser.parse_args()
    
    
    m, tau, id = .001, .001, 20250903172407
    # m, tau, id = 1.0, 100.0, 20250903220204
    past_timesteps, latent_dims, n_neighbours = 1000, 0, 0
    
    for m, tau, id in zip([.001, 1.0], [.001, 100.0], [20250903172407, 20250903220204]):
        print('m, tau:', m, tau)
        L96 = xr.open_dataset(f'online_runs/NO_PARAMETRIZATION/m={m}_tau={tau}_t=10000MTU_{id}.nc')
        for past_timesteps, model_name, n_neighbours in zip([1000, 0, 0], ['NNpast', 'NN', 'NN'], [0, 0, 7]):
            for w in [0.0, 1.e-6, 1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.]:        
                model = NN(m, tau, past_timesteps, latent_dims, n_neighbours, model_name)
                print(w, model.past_timesteps, model.model_name)
                dataloader_train, dataloader_test = generate_dataloaders(L96, model, train_share=.5)
                model = train_model(dataloader_train, dataloader_test, model, num_epochs=100, weight_decay=w)
                model.save(additional_info=f'w={w}')



    #print(args.model_type, args.past_timesteps, args.latent_dims)
    #sensitivity_experiment(args.past_timesteps, models=['baseline_nn', 'nn', 'NN+AE'])
    #sensitivity_experiment(args.past_timesteps, models=[args.model_type], latent_dims=None)
    #sensitivity_experiment(1000, ['NN+AE'], latent_dims=1)


    # m, tau = 0.001, 0.001
    # past_timesteps = 1000    
    # with open(f'online_runs/NO_PARAMETRIZATION/time=10000000_m={m}_tau={tau}.pkl', 'rb') as file:
    #     L96 = pickle.load(file)
    # L96_subset = L962LvlMem(m=m, tau=tau)
    # L96_subset._history_X = L96._history_X[:500_000]
    # L96_subset._history_B = L96._history_B[:500_000]
    # L96 = L96_subset

    # with open(f'online_runs/NO_PARAMETRIZATION/time=1000MTU_m={m}_tau={tau}.pkl', 'rb') as file:
    #     L96 = pickle.load(file)

    # train_loader, test_loader = generate_simple_dataloaders(L96, past_timesteps, time_series_length=10_000, batch_size=10)
    # for w in [1.e-3]:
    #     print(w)
    #     model = NNpAEpD(n_neighbours=1, past_timesteps=past_timesteps, latent_dims=8)
    #     model.memory_cutoff = m
    #     model.tau = tau
    #     model = train_model_NNpAEpD_simple_dataloader(train_loader, test_loader, model, past_timesteps, num_epochs=100, weight_decay=w)
    #     save_model(model, w=w)

    # path_ODE = 'ODEs/dim=8_deg=1_lambda=0.0_finitedifference_more_data.npy'
    # path_NN = 'networks/NN+ODE/input_lagg=1000/m=0.001_tau=0.001_NN+ODE_faster.pkl'
    # path_AE = 'networks/NN+AE+D/input_lagg=1000/m=0.001_tau=0.001_NN+AE+D_w=0.001_for_ODE_more_data.pkl'
    # ode = ODE_Z(path_ODE, path_AE, path_NN, model_name='ODE_Z_online', K=8, latent_dims=8, past_timesteps=1000, dt=0.001)
    # save_model(ode)


    #create_latent_space_trainig_data(m=.001, tau=.001, past_timesteps=1000, latent_dims=5, num_samples=1_000)
    # create_latent_space_trainig_data_multi_trajectory(m=.001, tau=.001, past_timesteps=1000, latent_dims=1)

    # train_loader, test_loader = generate_simple_dataloaders(L96, past_timesteps, batch_size=10_000)
    # X, B = torch.tensor(L96.history.X.values.T, dtype=torch.float32), torch.tensor(L96.history.B.values.T, dtype=torch.float32)
    # train_loader, test_loader = create_train_test_loaders(X, B, past_timesteps, train_share=0.8, batch_size=4096, num_workers=0
    # model = train_model_NNpAEpD_simple_dataloader(train_loader, test_loader, model, past_timesteps, num_epochs=30, weight_decay=w)

    # Data
    '''
    L96 = L96TwoLevelMemory(X_init=initX, Y_init=initY, save_dt=.001, memory_cutoff=.1, memory_tau=.001, memory_activation_func=None)
    L96.iterate(10)
    past_timesteps = 1000
    latent_dims = 5

    # # AE
    # X_train, X_test = generate_data_autoencoder(L96, past_timesteps=past_timesteps)
    # ae_model = Autoencoder(past_timesteps=past_timesteps, latent_dims=latent_dims)
    # ts = time.time()
    # ae_model = train_model(X_train, X_test, X_train, X_test, ae_model, num_epochs=200)
    # print('t:', time.time() - ts)

    # # Model
    # X_train, X_test, B_train, B_test = generate_data_nn_autoencoder(L96, past_timesteps=past_timesteps, model_autoencoder=ae_model)
    # nn_model = FCNN(past_timesteps=latent_dims, nodes_per_layer=50)

    # # Training
    # ts = time.time()
    # train_model(X_train, X_test, B_train, B_test, nn_model, num_epochs=200)
    # print('t:', time.time() - ts)

    # Model
    X_train, X_test, B_train, B_test = generate_data(L96, past_timesteps=past_timesteps)
    nn_model = FCNN(past_timesteps=past_timesteps, nodes_per_layer=140)

    # # Training
    ts = time.time()
    train_model(X_train, X_test, B_train, B_test, nn_model, num_epochs=200)
    print('t:', time.time() - ts)'''

