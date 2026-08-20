import xarray as xr
import torch
import pysindy as ps
from parametrizations import generate_dataloaders, NNpAEpD, process_past_timesteps
import math
import numpy as np
np.math = math
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
from torchmetrics.regression import R2Score
from tqdm import trange
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_preprocessing(path, coord=4):
    ### Load training data XZ
    L96 = xr.open_dataset(path)
    coord = 4
    Z = L96.Z.values[:, coord, :]
    X = L96.X.values[:, coord].reshape(-1, 1)
    Nt = len(X)
    Ntrain = int(0.8 * Nt)
    X_train, X_test = X[:Ntrain], X[Ntrain:]
    Z_train, Z_test = Z[:Ntrain], Z[Ntrain:]

    fd=ps.FiniteDifference()
    Zdot_train = fd._differentiate(Z_train, t=.001)
    Zdot_test = fd._differentiate(Z_test, t=.001)

    return X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test


def move_to_torch(X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test):
    Zdot_train = torch.tensor(Zdot_train, dtype=torch.float32, device=device)
    Zdot_test = torch.tensor(Zdot_test, dtype=torch.float32, device=device)
    Z_train = torch.tensor(Z_train, dtype=torch.float32, device=device)
    Z_test = torch.tensor(Z_test, dtype=torch.float32, device=device)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    return X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test


class NN_predictability(nn.Module):
    def __init__(self, latent_dims, n_forcings, nodes_per_layer=16):
        nn.Module.__init__(self)
        self.latent_dims = latent_dims
        self.n_forcings = n_forcings
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN_predictability'

        self.neural_net = nn.Sequential(
            nn.Linear(self.latent_dims + self.n_forcings, self.nodes_per_layer), 
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, latent_dims)
        )

        self.train_loss = []
        self.test_loss = []
        self.R2 = []
    
    def forward(self, x):
        y_pred = self.neural_net(x)
        return y_pred


def upper_predictaiblity_boundary(model, XZ_train, XZ_test, Zdot_train, Zdot_test):
    model = model.to(device)
    criterion = nn.MSELoss()
    lr = .007
    num_epochs = 3000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Make progress bar
    best_R2 = float('-inf')
    best_R2_epoch = 0
    pbar = trange(num_epochs, desc="Training", ncols=200)

    # Training
    for epoch in pbar:
        model.train()

        # Transpose 1-dimensional data
        dZdt_pred = model.forward(XZ_train)
        loss = criterion(dZdt_pred, Zdot_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item() / len(XZ_train)
        
        model.eval()
        with torch.no_grad():
            # Transpose 1-dimensional data
            dZdt_pred = model.forward(XZ_test)
            loss = criterion(dZdt_pred, Zdot_test)
            R2 = R2Score(multioutput='raw_values').to(device)(dZdt_pred, Zdot_test).detach().cpu().numpy()
            test_loss = loss.item() / len(XZ_test)
        

        model.train_loss.append(train_loss)
        model.test_loss.append(test_loss)
        model.R2.append(R2)

        # Inline update
        pbar.set_postfix({
            "Train Loss": f"{train_loss:.6f}",
            "Test Loss": f"{test_loss:.6f}",
            "R²": f"{np.min(R2):.4f}, {np.mean(R2):.4f}"})


def lower_predictability_boundary(X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test):
    # Learn ODE without other Z_k variables
    opt = ps.STLSQ(threshold=0.0)
    lib = ps.PolynomialLibrary(degree=1)
    model = ps.SINDy(feature_library=lib, 
                    optimizer=opt)

    model.fit(x=Z_train, u=X_train, x_dot=Zdot_train, t=.001)
    model.print()  
    score = model.score(x=Z_test, x_dot=Zdot_test, u=X_test, t=.001, multioutput='raw_values')
    # print('Model score: ', score, flush=True)
    return score



if __name__=='__main__':
    # model_paths = [
    #     'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110511.pkl',
    #     'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.001_20260528103552.pkl',
    #     'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104318.pkl',
    #     'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105035.pkl',
    #     'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105808.pkl'
    # ]
    # paths = [
    #     'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0_20260528212527.nc',
    #     'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.001_20260528191848.nc',
    #     'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0001_20260528202155.nc',
    #     'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-05_20260528222859.nc',
    #     'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-06_20260528181546.nc'
    # ]
    # model_paths = [
    #     'networks/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110523.pkl',
    #     'networks/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.001_20260528103557.pkl',
    #     'networks/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104326.pkl',
    #     'networks/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105032.pkl',
    #     'networks/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl'
    # ]
    # paths = [
    #     'online_runs/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0_20260528205029.nc',
    #     'online_runs/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.001_20260528215242.nc',
    #     'online_runs/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0001_20260528235825.nc',
    #     'online_runs/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-05_20260529010036.nc',
    #     'online_runs/NN+AE+D/latent_dims=4_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-06_20260528225509.nc'
    # ]
    # model_paths = [
    #     'networks/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110528.pkl',
    #     'networks/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.001_20260528103559.pkl',
    #     'networks/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104332.pkl',
    #     'networks/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105041.pkl',
    #     'networks/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105817.pkl'
    # ]
    # paths = [
    #     'online_runs/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0_20260528215558.nc',
    #     'online_runs/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0001_20260528205142.nc',
    #     'online_runs/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.001_20260528225903.nc',
    #     'online_runs/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-05_20260529000244.nc',
    #     'online_runs/NN+AE+D/latent_dims=2_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-06_20260529010650.nc'        
    # ]

    model_paths = [
        'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110521.pkl',
        'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.001_20260528103600.pkl',
        'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104329.pkl',
        'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105037.pkl',
        'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl'
    ]
    paths = [
        'online_runs/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0_20260528205516.nc',
        'online_runs/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.0001_20260528220919.nc',
        'online_runs/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=0.001_20260528232629.nc',
        'online_runs/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-05_20260529131120.nc',
        'online_runs/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_t=10000MTU_hyper_opt_w=1e-06_20260529122754.nc'
    ]

    for path, model_path in zip(paths, model_paths):
        X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test = data_preprocessing(path)
        lower_bound = lower_predictability_boundary(X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test)
        
        X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test = move_to_torch(X_train, X_test, Z_train, Z_test, Zdot_train, Zdot_test)
        XZ_train = torch.cat((X_train, Z_train), dim=-1)
        XZ_test = torch.cat((X_test, Z_test), dim=-1)
        model = NN_predictability(latent_dims=6, n_forcings=1)
        upper_predictaiblity_boundary(model, XZ_train, XZ_test, Zdot_train, Zdot_test)
        R2s = np.array(model.R2)
        best_worst_variable = np.argmax(np.min(R2s, axis=1))
        upper_bound = R2s[best_worst_variable]
        print('upper bound: ', upper_bound)
        print('lower bound:', lower_bound)

        model_param = torch.load(model_path, map_location=device, weights_only=False)
        model_param.linear_predictability = lower_bound
        model_param.nonlinear_predictability = upper_bound
        model_param.save(additional_info=f'hyper_opt_w={model_param.weight_decay}')
