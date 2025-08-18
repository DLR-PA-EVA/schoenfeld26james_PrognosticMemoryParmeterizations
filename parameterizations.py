import numpy as np
import torch
import torch.nn as nn
from util import return_lagged_input_vector, transpose_if_1d, return_lagged_input_vector_and_present_k, return_lagged_input_vector_opt
from torchmetrics.regression import R2Score
import torch.utils.data as Data
from L96 import L96TwoLevelMemory, L962LvlMem
import os
from pathlib import Path
import argparse
from tqdm import tqdm, trange
import pickle
import itertools


# Load initial conditions for L96 model
initX, initY = np.load('./initX.npy'), np.load('./initY.npy')
np.random.seed(123)

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model classes
class Autoencoder(nn.Module):
    def __init__(self, past_timesteps, latent_dims):
        super().__init__()

        self.past_timesteps = past_timesteps
        self.latent_dims = latent_dims

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
            nn.Linear(self.latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.past_timesteps)
        )
        
        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'autoencoder'

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class NNpAE(nn.Module):
    def __init__(self, past_timesteps, latent_dims, nodes_per_layer=16):
        super().__init__()

        self.past_timesteps = past_timesteps
        self.latent_dims = latent_dims
        self.nodes_per_layer = nodes_per_layer

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

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'NN+AE'

    def forward(self, x):
        x_past = x[:, :-1]
        x_present = torch.unsqueeze(x[:, -1], 1)
        # x_present = x[:, -1]

        latent_space = self.encoder(x_past)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        return y_pred
   

class FCNN(nn.Module):
    def __init__(self, past_timesteps=0, nodes_per_layer=16):
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        # Layers
        self.linear1 = nn.Linear(past_timesteps + 1, self.nodes_per_layer)  # number of past time steps the model receives as input
        self.linear2 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear3 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear4 = nn.Linear(self.nodes_per_layer, self.nodes_per_layer)
        self.linear5 = nn.Linear(self.nodes_per_layer, 1)  

        # Activation
        self.relu = nn.ReLU()

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.past_timesteps = past_timesteps
        self.model_name = None
        self.latent_dims = None

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        return x
    

class NN(nn.Module):
    def __init__(self,past_timesteps, n_neighbours, nodes_per_layer=16):
        super().__init__()
        self.past_timesteps = past_timesteps
        self.n_neighbours = n_neighbours
        self.nodes_per_layer = nodes_per_layer

        self.neural_net = nn.Sequential(
            nn.Linear(self.past_timesteps + self.n_neighbours, self.nodes_per_layer),
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

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'NN'
        self.latent_dims = 0

    def forward(self, x):
        y_pred = self.neural_net(x)
        return y_pred


class NNpAEpD(nn.Module):
    def __init__(self, past_timesteps, n_neighbours, latent_dims, nodes_per_layer=16):
        super().__init__()

        self.past_timesteps = past_timesteps
        self.n_neighbours = n_neighbours
        self.latent_dims = latent_dims
        self.nodes_per_layer = nodes_per_layer

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
            nn.Linear(latent_dims + n_neighbours, self.nodes_per_layer),
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

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'NN+AE+D'

    def forward(self, x):
        x_past = x[:, :self.past_timesteps]
        x_present = x[:, self.past_timesteps:]
        latent_space = self.encoder(x_past)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


def save_model(model, w):
    # Save the model
    if model.model_name == 'NN+AE':
        model_name = f'NN+AE_latent_dims={model.latent_dims}'
    else:
        model_name = model.model_name

    save_dir = Path('networks') / model_name / f'input_lagg={model.past_timesteps}'
    if not save_dir.exists(): 
        os.makedirs(save_dir) 
    torch.save(model, f'{save_dir}/m={model.memory_cutoff}_tau={model.tau}_{model.model_name}_w={w}.pkl')


# Feature creation
def generate_data(L96, past_timesteps, BATCH_SIZE=3000, train_share=.8):
    # Get data
    X = L96.history.X.values.astype(np.float32).T
    B = L96.history.B.values.astype(np.float32).T

    X = return_lagged_input_vector(X, past_timesteps)
    X = torch.from_numpy(X.copy())  # Hade some stride problems here, hence the copy workaround
    train_ind = int(len(X) * train_share)
    X_train = X[:train_ind]
    X_test = X[train_ind:]
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    print('finished X')
    
    B = return_lagged_input_vector(B, past_timesteps)  
    B = B[:, -1].reshape(-1, 1)  # B is always B(t) -> past_timesteps = x but only the last value is taken
    B = torch.from_numpy(B)
    B_train = B[:train_ind]
    B_test = B[train_ind:]
    B_train = B_train.to(device)
    B_test = B_test.to(device)

    # Make DataLoader
    # dataset_train = Data.TensorDataset(X_train, B_train)
    # dataset_test = Data.TensorDataset(X_test, B_test)
    # dataloader_train = Data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
    # dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

    return X_train, X_test, B_train, B_test


def generate_data_autoencoder(L96, past_timesteps, BATCH_SIZE=3000, train_share=.8):
    # Get data
    X = L96.history.X.values.astype(np.float32).T
    X_lagged = return_lagged_input_vector(X, past_timesteps)
    X_lagged = torch.from_numpy(X_lagged.copy())

    # Remove X(t)
    X_lagged = X_lagged[:, :-1]

    # train test split
    train_ind = int(len(X_lagged) * train_share)
    X_train = X_lagged[:train_ind]
    X_test = X_lagged[train_ind:]

    # Send data to device
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    # Make DataLoader
    # dataset_train = Data.TensorDataset(X_train, X_train)
    # dataset_test = Data.TensorDataset(X_test, X_test)
    # dataloader_train = Data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
    # dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

    return X_train, X_test


def generate_data_nn_autoencoder(L96, past_timesteps, model_autoencoder, BATCH_SIZE=3000, train_share=.8):
    # Get data
    X = L96.history.X.values.astype(np.float32).T
    B = L96.history.B.values.astype(np.float32).T

    B = return_lagged_input_vector(B, past_timesteps)  
    B = B[:, -1]  # B is always B(t) -> past_timesteps = x but only the last value is taken
    X_lagged = return_lagged_input_vector(X, past_timesteps)
    X_lagged, B = torch.from_numpy(X_lagged.copy()),torch.from_numpy(B)

    # Compute Z
    X_past = X_lagged[:, :-1]
    X_past = X_past.to(device)
    with torch.no_grad():
        model_autoencoder.eval()
        Z = model_autoencoder.encoder(X_past)
    ZX = torch.zeros((Z.shape[0], Z.shape[1] + 1))
    ZX[:, :Z.shape[1]] = Z
    ZX[:, Z.shape[1]] = X_lagged[:, -1]  # Present x-state

    # train test split
    train_ind = int(len(X_lagged) * train_share)
    ZX_train = ZX[:train_ind]
    B_train = B[:train_ind]
    ZX_test = ZX[train_ind:]
    B_test = B[train_ind:]

    # Send data to device
    ZX_train = ZX_train.to(device)
    ZX_test = ZX_test.to(device)
    B_train = B_train.to(device)
    B_test = B_test.to(device)

    # Make DataLoader
    # dataset_train = Data.TensorDataset(ZX_train, B_train)
    # dataset_test = Data.TensorDataset(ZX_test, B_test)
    # dataloader_train = Data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
    # dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

    return ZX_train, ZX_test, B_train, B_test


# Model training
def train_model(x, x_test, b, b_test, model, num_epochs=5, weight_decay=0.0):
    # Send to gpu if available
    model = model.to(device)

    criterion = nn.MSELoss()
    lr = .007
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Make progress bar
    best_R2 = float('-inf')
    best_R2_epoch = 0
    pbar = trange(num_epochs, desc="Training", ncols=150)

    for epoch in pbar:
        model.train()
        test_loss, train_loss, R2 = 0, 0, 0

        # Transpose 1-dimensional data
        xT, bT = transpose_if_1d(x), transpose_if_1d(b)
        b_pred = model.forward(xT)
        loss = criterion(b_pred, bT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            # Transpose 1-dimensional data
            xT, bT = transpose_if_1d(x_test), transpose_if_1d(b_test)
            b_pred = model.forward(xT)
            loss = criterion(b_pred, bT)
            R2 += R2Score().to(device)(b_pred, bT)
            test_loss += loss.item()
        
        train_loss = train_loss #/ len(dataloader_train)
        test_loss = test_loss #/ len(dataloader_test)
        model.train_loss.append(train_loss)
        model.test_loss.append(test_loss)
        R2 = R2 #/ len(dataloader_test)
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


def train_model_NNpAEpD(x, x_test, b, b_test, model, num_epochs=5, weight_decay=0.0):
    # Send to gpu if available
    model = model.to(device)

    criterion = nn.MSELoss()
    reconstruction_criterion = nn.MSELoss()
    lr = .007
    alpha = .5  # weigth of param loss, reconstruct loss will be 1-alpha
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Make progress bar
    best_R2 = float('-inf')
    best_R2_epoch = 0
    pbar = trange(num_epochs, desc="Training", ncols=150)

    for epoch in pbar:
        model.train()
        test_loss, train_loss, R2 = 0, 0, 0

        # Transpose 1-dimensional data
        xT, bT = transpose_if_1d(x), transpose_if_1d(b)
        b_pred, x_recon = model.forward(xT)
        loss = alpha * criterion(b_pred, bT) + (1-alpha) * reconstruction_criterion(x_recon, xT[:,:-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            # Transpose 1-dimensional test data
            xT, bT = transpose_if_1d(x_test), transpose_if_1d(b_test)
            b_pred, x_recon = model.forward(xT)
            loss = alpha * criterion(b_pred, bT) + (1 - alpha) * reconstruction_criterion(x_recon, xT[:,:-1])
            test_loss += loss.item()
            R2 += R2Score().to(device)(b_pred, bT)
        
        train_loss = train_loss #/ len(dataloader_train)
        test_loss = test_loss #/ len(dataloader_test)
        model.train_loss.append(train_loss)
        model.test_loss.append(test_loss)
        R2 = R2 #/ len(dataloader_test)
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
        path = f'online_runs/NO_PARAMETRIZATION/' + files[i]
        print(path)
        i += 1
        with open(path, 'rb') as file:
                L96 = pickle.load(file)
        Nt = 700 * 1000 

        L96_subset = L962LvlMem(m=m, tau=tau)
        L96_subset._history_X = L96._history_X[:Nt]
        L96_subset._history_B = L96._history_B[:Nt]
        print(m, tau, L96.m, L96.tau)
        L96 = L96_subset

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

            if ('autoencoder' in models) or ('nn+autoencoder' in models):
                # Train NN+AE
                print('NN+AE')
                # AE
                X_train, X_test = generate_data_autoencoder(L96, past_timesteps=past_timesteps)
                ae_model = Autoencoder(past_timesteps=past_timesteps, latent_dims=latent_dims)
                ae_model = train_model(X_train, X_test, X_train, X_test, ae_model, num_epochs=num_epochs)
                ae_model = set_model_metadata(ae_model, past_timesteps, latent_dims=None, model_name='autoencoder', m=m, tau=tau)
                save_model(ae_model)

                # NN
                X_train, X_test, B_train, B_test = generate_data_nn_autoencoder(L96, past_timesteps=past_timesteps, model_autoencoder=ae_model)
                nn_model = FCNN(past_timesteps=latent_dims, nodes_per_layer=nodes_per_layer_AE)
                nn_model = train_model(X_train, X_test, B_train, B_test, nn_model, num_epochs=num_epochs)
                nn_model = set_model_metadata(nn_model, past_timesteps, latent_dims=None, model_name='nn+autoencoder', m=m, tau=tau)
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


def create_latent_space_trainig_data(m, tau, past_timesteps=1000, latent_dims=5, num_samples=1_000):
    criterion = nn.MSELoss()
    # Load Autoencoder model
    model_name = f'networks/NN+AE_latent_dims={latent_dims}/input_lagg={past_timesteps}/m={m}_tau={tau}_NN+AE.pkl'
    with open(model_name, 'rb') as file:
        model = torch.load(file, weights_only=False, map_location='cpu')
    
    # Load L96 data
    print('Loading L96 data...')
    with open(f'online_runs/NO_PARAMETRIZATION/time=10000000_m={m}_tau={tau}.pkl', 'rb') as file:
         L96 = pickle.load(file)
    
    Nt = 5000  # in MTU
    tstep = 50

    # Prepare save directory
    save_dir = Path(f'latent_space_data/dim={latent_dims}_t={Nt}MTU_pt={past_timesteps}_m={m}_tau={tau}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder variables to be initialized later
    Z_file = X_file = B_file = None
    Z_shape, X_shape, B_shape = None, None, None
    total_samples = 0

    for t in range(tstep, Nt, tstep):
        print(t)
        L96_subset = L962LvlMem(m=m, tau=tau)
        L96_subset._history_X = L96._history_X[(t - tstep) * 1000:t * 1000]
        L96_subset._history_B = L96._history_B[(t - tstep) * 1000:t * 1000]

        X_train, X_test, B_train, B_test = generate_data(L96_subset, past_timesteps, BATCH_SIZE=3000, train_share=1.0)
        X_present = X_train[:, -1]  # save only the present timestep

        with torch.no_grad():
            xT, bT = transpose_if_1d(X_train), transpose_if_1d(B_train)
            b_pred = model.forward(xT)
            loss = criterion(b_pred, bT)
            R2 = R2Score()(b_pred, bT)

            x_past = xT[:, :-1]
            Z_new = model.encoder(x_past).cpu().numpy()  # (N, latent_dims)
            X_new = X_present.cpu().numpy()
            B_new = B_train.cpu().numpy()

            print('Loss on data:', loss.item())
            print('R2 on data:', R2.item())

        # Initialize memory-mapped arrays only once with estimated max shape
        if Z_file is None:
            N = ((Nt - tstep) // tstep) * Z_new.shape[0]
            Z_shape = (N, Z_new.shape[1])
            X_shape = (N, X_new.shape[1]) if X_new.ndim > 1 else (N,)
            B_shape = (N, B_new.shape[1]) if B_new.ndim > 1 else (N,)

            Z_file = np.lib.format.open_memmap(save_dir / 'Z.npy', dtype='float32', mode='w+', shape=Z_shape)
            X_file = np.lib.format.open_memmap(save_dir / 'X.npy', dtype='float32', mode='w+', shape=X_shape)
            B_file = np.lib.format.open_memmap(save_dir / 'B.npy', dtype='float32', mode='w+', shape=B_shape)

        # Append new data
        n_new = Z_new.shape[0]
        Z_file[total_samples:total_samples + n_new] = Z_new
        X_file[total_samples:total_samples + n_new] = X_new
        B_file[total_samples:total_samples + n_new] = B_new
        total_samples += n_new

    # Optionally: truncate if you overestimated the shape
    Z_file.flush(); X_file.flush(); B_file.flush()
    print(f'Done. Total samples saved: {total_samples}')


def create_latent_space_trainig_data_multi_trajectory(m, tau, past_timesteps=1000, latent_dims=5, trajectories=20):
    criterion = nn.MSELoss()
    # Load Autoencoder model
    model_name = f'networks/NN+AE_latent_dims={latent_dims}/input_lagg={past_timesteps}/m={m}_tau={tau}_NN+AE.pkl'
    with open(model_name, 'rb') as file:
        model = torch.load(file, weights_only=False, map_location='cpu')
    
    Nt = 102  # in MTU
    tstep = 50

    # Prepare save directory
    save_dir = Path(f'latent_space_data/multiple_trajectories/dim={latent_dims}_t={Nt}MTU_pt={past_timesteps}_m={m}_tau={tau}')
    save_dir.mkdir(parents=True, exist_ok=True)

    for traj in range(14, trajectories):
        print(f'trajectory: {traj+1}/{trajectories}')
        # Placeholder variables to be initialized later
        Z_file = X_file = B_file = None
        Z_shape, X_shape, B_shape = None, None, None
        total_samples = 0

        # Make simulation data
        L96 = L962LvlMem(m=m, tau=tau)
        L96.iterate(Nt)

        X_train, X_test, B_train, B_test = generate_data(L96, past_timesteps, BATCH_SIZE=3000, train_share=1.0)
        X_present = X_train[:, -1]  # save only the present timestep

        with torch.no_grad():
            xT, bT = transpose_if_1d(X_train), transpose_if_1d(B_train)
            b_pred = model.forward(xT)
            loss = criterion(b_pred, bT)
            R2 = R2Score()(b_pred, bT)

            x_past = xT[:, :-1]
            Z_new = model.encoder(x_past).cpu().numpy()  # (N, latent_dims)
            X_new = X_present.cpu().numpy()
            B_new = B_train.cpu().numpy()

            print('Loss on data:', loss.item())
            print('R2 on data:', R2.item())

        # Initialize memory-mapped arrays only once with estimated max shape
        if Z_file is None:
            N = ((Nt - tstep) // tstep) * Z_new.shape[0]
            Z_shape = (N, Z_new.shape[1])
            X_shape = (N, X_new.shape[1]) if X_new.ndim > 1 else (N,)
            B_shape = (N, B_new.shape[1]) if B_new.ndim > 1 else (N,)

            Z_file = np.lib.format.open_memmap(save_dir / f'Z_traj={str(traj).zfill(2)}.npy', dtype='float32', mode='w+', shape=Z_shape)
            X_file = np.lib.format.open_memmap(save_dir / f'X_traj={str(traj).zfill(2)}.npy', dtype='float32', mode='w+', shape=X_shape)
            B_file = np.lib.format.open_memmap(save_dir / f'B_traj={str(traj).zfill(2)}.npy', dtype='float32', mode='w+', shape=B_shape)

        # Append new data
        n_new = Z_new.shape[0]
        Z_file[total_samples:total_samples + n_new] = Z_new
        X_file[total_samples:total_samples + n_new] = X_new
        B_file[total_samples:total_samples + n_new] = B_new
        total_samples += n_new

        # Optionally: truncate if you overestimated the shape
        Z_file.flush(); X_file.flush(); B_file.flush()
        print(f'Done. Total samples saved: {total_samples}')


    
    
    

import time
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run L96 sensitivity experiment')
    parser.add_argument('--model_type', type=str, default='nn', help='Model type to use (e.g., nn, rf, svm)')
    parser.add_argument('--past_timesteps', type=int, default=1000, help='Number of past timesteps to consider')
    parser.add_argument('--latent_dims', type=int, default=5, help='Number of past timesteps to consider')

    args = parser.parse_args()
    #print(args.model_type, args.past_timesteps, args.latent_dims)
    #sensitivity_experiment(args.past_timesteps, models=['baseline_nn', 'nn', 'NN+AE'])
    #sensitivity_experiment(args.past_timesteps, models=[args.model_type], latent_dims=None)
    #sensitivity_experiment(1000, ['NN+AE'], latent_dims=1)

    m, tau = 1.0, 100.0
    with open(f'online_runs/NO_PARAMETRIZATION/time=10000000_m={m}_tau={tau}.pkl', 'rb') as file:
        L96 = pickle.load(file)

    X = L96.history.X.values
    B = L96.history.B.values
    X = X[:500_000]
    B = B[:500_000]

    Ntrain = int(X.shape[0] * .8)
    def train_test_split(arr, Ntrain):
        return arr[:Ntrain], arr[Ntrain:]

    X_train, X_test = train_test_split(X, Ntrain)
    B_train, B_test = train_test_split(B, Ntrain)

    past_timesteps = 1000

    X_train = torch.tensor(return_lagged_input_vector_opt(X_train.T, past_timesteps), dtype=torch.float32, device=device)
    X_test = torch.tensor(return_lagged_input_vector_opt(X_test.T, past_timesteps), dtype=torch.float32, device=device)
    B_train = torch.tensor(return_lagged_input_vector_opt(B_train.T, past_timesteps), dtype=torch.float32, device=device)[:,-1].reshape(-1,1)
    B_test = torch.tensor(return_lagged_input_vector_opt(B_test.T, past_timesteps), dtype=torch.float32, device=device)[:,-1].reshape(-1,1)

    for w in [1.e-3]:
        print(w)
        model = NNpAEpD(n_neighbours=1, past_timesteps=past_timesteps, latent_dims=8)
        #model = NNpAE(past_timesteps=past_timesteps, latent_dims=8)
        model.memory_cutoff = m
        model.tau = tau
        model = train_model_NNpAEpD(X_train, X_test, B_train, B_test, model, num_epochs=2000, weight_decay=w)
        #model = train_model(X_train, X_test, B_train, B_test, model, num_epochs=2000, weight_decay=w)
        save_model(model, w=w)


    #create_latent_space_trainig_data(m=.001, tau=.001, past_timesteps=1000, latent_dims=5, num_samples=1_000)
    # create_latent_space_trainig_data_multi_trajectory(m=.001, tau=.001, past_timesteps=1000, latent_dims=1)


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

