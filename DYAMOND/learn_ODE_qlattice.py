import numpy as np
import pandas as pd
import feyn
import pysindy as ps
import argparse
import pickle
from tqdm import trange

'''
Module for training nonlinear ODEs with Qlattice
'''
# Point to training data
XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260507185408_global_ocean_land_rp=False_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy')
XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260507185408_global_ocean_land_rp=False_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy')

def get_derivative(XZ, first_Z_ind):
    dm = ps.FiniteDifference()
    # dm = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
    dZdt = np.array([dm._differentiate(Z[:, first_Z_ind:], t=1) for Z in XZ])
    return dZdt

# PySR data prep
latent_dims = 4
coord_skipper = None
time_skipper = None
tmin = None
tmax = None
input_indices = [1, 2, 3, 4, 5, 6, -4, -3, -2, -1]
n_features = len(input_indices) - latent_dims
n_epochs = 1000

for N_subsample in [500_000]:
    xz_train = XZ_train[::coord_skipper, tmin:tmax, input_indices]
    xz_test = XZ_val[::coord_skipper, :, input_indices]

    dZdt_train = get_derivative(xz_train, first_Z_ind=-latent_dims)
    dZdt_train = dZdt_train[:, ::time_skipper].reshape(-1, latent_dims)
    xz_train = xz_train[:, ::time_skipper].reshape(-1, latent_dims + n_features)

    N_samples = len(xz_train)
    dZdt_test = get_derivative(xz_test, first_Z_ind=-latent_dims)
    dZdt_test = dZdt_test.reshape(-1, latent_dims)
    xz_test = xz_test.reshape(-1, latent_dims + n_features)
    print('subsamples: ', N_subsample)
    subsample = np.random.choice(N_samples, size=N_subsample)
    xz_train = xz_train[subsample]
    dZdt_train = dZdt_train[subsample]
    # 'ts', 'hfss', 'hfls', 'is_land',
    all_inputs = ['coord', 'qv2m', 'Prw', 'T2m', 'Ts', 'Fs', 'Fl', 'l', 'z1', 'z2', 'z3', 'z4']
    inputs = [all_inputs[i] for i in input_indices]
    feature_names = inputs + ['z1_dot', 'z2_dot', 'z3_dot', 'z4_dot']
    data_train = pd.DataFrame(np.concat((xz_train, dZdt_train), axis=1), columns=feature_names, dtype=np.float32)
    data_test = pd.DataFrame(np.concat((xz_test, dZdt_test), axis=1), columns=feature_names, dtype=np.float32)

    parser = argparse.ArgumentParser(description='Run symbolic regression for latent dynamics.')
    parser.add_argument('--target', type=str, default='z1_dot', help='Dimension of the latent space to analyze.')
    args = parser.parse_args()
    target = args.target
    train = data_train[inputs + [target]]
    test = data_test[inputs]
    true = data_test[target]

    # --- Train Qlattice model ---
    print(f'Running QLattice for target: {target}')
    ql = feyn.QLattice()

    # Load priors if previously computed
    priors_precomputed = True
    path_to_priors = f'qlattice_equations/latent_dim=4/priors/priors_target={target}_Nsamples=50000'
    if priors_precomputed:
        with open(path_to_priors, 'rb') as file:
            priors = pickle.load(file)
    else:
        priors = feyn.tools.estimate_priors(train, target)
    ql.update_priors(priors)

    # Training loop
    models = []
    pbar = trange(n_epochs, desc="Training", ncols=100)
    for epoch in range(n_epochs):
        # Sample models from the QLattice, and add them to the list
        models += ql.sample_models(train.columns, target, 'regression', max_complexity=10)

        # Fit the list of models. Returns a list of models sorted by loss or criterion.
        models = feyn.fit_models(
            models, 
            train, 
            loss_function='squared_error', 
            criterion='bic',
            threads=256)

        # Remove redundant and poorly performing models from the list
        models = feyn.prune_models(models)

        # Display the best model in the current epoch
        # feyn.show_model(models[0], label=f"Epoch: {epoch}", update_display=True)

        # Update QLattice with the fitted list of models (sorted by loss)
        ql.update(models)
        pred = models[0].predict(test)
        # print(f'Epoch {epoch + 1}, R^2 = ', feyn.metrics.r2_score(true, pred))
        R2 = feyn.metrics.r2_score(true, pred)
        pbar.set_postfix({
            "R2": f"{R2}",
            "Epoch": f"{epoch}"
            })

    # Find the 10 best and sufficiently diverse models
    best_models = feyn.get_diverse_models(models, n=10)

    # Evaluate and save
    for i in range(len(best_models)):
        model = best_models[i]
        pred = model.predict(test)
        print(f'Model_{i} R^2 = ', feyn.metrics.r2_score(true, pred))
        model.save(f'my_qlattice_model_{target}_{str(i).zfill(2)}.json')


