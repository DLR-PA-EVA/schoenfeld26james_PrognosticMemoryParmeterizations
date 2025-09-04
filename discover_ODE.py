import numpy as np
import pysindy as ps
import pickle
import os
import xarray as xr
from parametrizations import ODE_Z


def load_online_run(path):
    L96 = xr.open_dataset(path)
    X, Z = L96.X.values, L96.Z.values

    Nt = len(X)
    Ntrain = int(0.8 * Nt)
    X_train, X_test = X[:Ntrain], X[Ntrain:]
    Z_train, Z_test = Z[:Ntrain], Z[Ntrain:]

    return X_train, X_test, Z_train, Z_test


def learn_ODE(path_online_run, lmbda, deg, latent_dim, diff_method=None, save=True, save_path=None):
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
    print('Model score: ', score)
    print(model.coefficients())
    print(model.coefficients().shape)

    if save:
        if not save_path:
            save_path = f'ODEs/dim={latent_dim}_deg={deg}_lambda={lmbda}_finitedifference_more_data.npy'

        with open(save_path, 'wb') as file:
            np.save(file, model.coefficients())
        #     pickle.dump(model, file)
        


def retrain_NN(path_ODE, path_online_run):
    with open(path_ODE, 'rb') as file:
        model_ODE = pickle.load(file)
    
    # Run online



if __name__=='__main__':
    # path_online = 'online_runs/NN+AE_latent_dims=8/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.0.nc'
    # path_online = 'online_runs/NN+AE_latent_dims=8/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.01.nc'
    # path_online = 'online_runs/NN+AE+D/input_lagg=1000/time=1000MTU_m=0.001_tau=0.001_w=0.001_for_ODE_less_data.nc'
    path_online = 'online_runs/NN+AE+D/input_lagg=1000/time=10000MTU_m=0.001_tau=0.001_w=0.001_for_ODE_more_data.nc'
    learn_ODE(path_online, lmbda=0.0, deg=1, latent_dim=8, save=True)