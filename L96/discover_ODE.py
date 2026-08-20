import numpy as np
import pysindy as ps
import pickle
import os
import xarray as xr
from L96 import run_online
from parametrizations import *
import glob
import torch
# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_online_run(path):
    L96 = xr.open_dataset(path)
    X, Z = L96.X.values, L96.Z.values

    Nt = len(X)
    Ntrain = int(0.8 * Nt)
    X_train, X_test = X[:Ntrain], X[Ntrain:]
    Z_train, Z_test = Z[:Ntrain], Z[Ntrain:]

    return X_train, X_test, Z_train, Z_test


def learn_ODE(path_online_run, lmbda, deg, latent_dim, decoder, m=0.001, tau=0.001, diff_method=None, save=True, save_path=None):
    X_train, X_test, Z_train, Z_test = load_online_run(path_online_run)

    z_train = [Z_train[:, k, :] for k in range(8)]
    z_test = [Z_test[:, k, :] for k in range(8)]
    x_train = [X_train[:, k].reshape(-1, 1) for k in range(8)]
    x_test = [X_test[:, k].reshape(-1, 1) for k in range(8)]

    # Set SINDy parameters
    diff_method = ps.FiniteDifference()

    # Learn ODE without other Z_k variables
    opt = ps.STLSQ(threshold=lmbda)
    lib = ps.PolynomialLibrary(degree=deg)
    model = ps.SINDy(feature_library=lib, 
                    differentiation_method=diff_method, 
                    optimizer=opt)

    model.fit(x=z_train, u=x_train, t=.001, multiple_trajectories=True)
    model.print()  
    score = model.score(x=z_test, u=x_test, t=.001, multiple_trajectories=True)
    print('Model score: ', score, flush=True)
    
    if save:
        if not save_path:
            save_path = f'ODEs/m={m}_tau={tau}/dim={latent_dim}_decoder={decoder}_deg={deg}_lambda={lmbda}_finitedifference.npy'

        with open(save_path, 'wb') as file:
            np.save(file, model.coefficients())
        #     pickle.dump(model, file)
        

def run_ODE_allongside_AE(path_AE, path_ODE):
    model_AE = torch.load(path_AE, map_location=device, weights_only=False)  # should only be trained on gpu
    print(model_AE.m, model_AE.past_timesteps)
    model = ODE_Z(model_AE.m, model_AE.tau, model_AE.past_timesteps, model_AE.latent_dims, model_AE.n_neighbours, path_ODE, path_AE, path_AE, model_name='ODE_Z')
    save_path = model.save()
    run_online(save_path, simulation_time=10_000)


def retrain_NN(path_ODE, path_online_run, path_AE):
    L96 = xr.open_dataset(path_online_run)
    model_AE = torch.load(path_AE, map_location=device, weights_only=False)  # should only be trained on gpu
    model_NN = NNpODE(model_AE.m, model_AE.tau, model_AE.past_timesteps, model_AE.latent_dims, model_AE.n_neighbours)

    trainloader, testloader = generate_dataloaders(L96, model_NN, train_share=.5, batch_size=100_000, num_workers=64)
    model_NN = train_model(trainloader, testloader, model_NN, num_epochs=100, weight_decay=.001)
    path_NN = model_NN.save()

    model_ODE = ODE_Z(model_AE.m, model_AE.tau, model_AE.past_timesteps, model_AE.latent_dims, model_AE.n_neighbours, path_ODE, path_AE, path_NN, model_name='ODE_Z_online')
    model_ODE.save()
    



if __name__=='__main__':
    # path_online = 'online_runs/NN+AE_latent_dims=8/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.0.nc'
    # path_online = 'online_runs/NN+AE_latent_dims=8/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.01.nc'
    # path_online = 'online_runs/NN+AE+D/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.001_for_ODE_less_data.nc'
    # path_online = 'online_runs/NN+AE+D/input_lagg=1000/time=10000MTU_m=0.001_tau=0.001_w=0.001_for_ODE_more_data.nc'
    # path_online = 'online_runs/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_t=10000MTU_20250908165421.nc'
    # learn_ODE(path_online, lmbda=0.0, deg=1, latent_dim=8, m=0.001, tau=0.001, decoder=True, save=True)

    # m, tau = 0.001, 0.001
    # for model, dec in zip(['NN+AE+D', 'NN+AE'], [True, False]):
    #     for latent_dim in [3, 5, 8]:
    #         print(f'model: {model}, latent dim: {latent_dim}, m: {m}, tau: {tau}', flush=True)
    #         path_online = f'online_runs/{model}/latent_dims={latent_dim}_past_timesteps=1000_n_neighbours=0/m={m}_tau={tau}_*'
    #         path_online = glob.glob(path_online)[0]
    #         learn_ODE(path_online, lmbda=0.0, deg=1, latent_dim=latent_dim, m=m, tau=tau, decoder=dec, save=True)

    # m, tau = 0.001, 0.001
    m, tau = 1.0, 100.0
    path_ODE = f'ODEs/m={m}_tau={tau}/dim=8_decoder=True_deg=1_lambda=0.0_finitedifference.npy'
    path_online_run = glob.glob(f'online_runs/ODE_Z/latent_dims=8_past_timesteps=1000_n_neighbours=0/m={m}_tau={tau}_t=10000MTU_*')[0]
    path_AE = glob.glob(f'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m={m}_tau={tau}_w=0.001_*')[0]
    retrain_NN(path_ODE, path_online_run, path_AE)
    # run_ODE_allongside_AE(path_AE, path_ODE)
