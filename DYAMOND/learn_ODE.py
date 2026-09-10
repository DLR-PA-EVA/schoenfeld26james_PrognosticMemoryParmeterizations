import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import pickle

'''
Module for training polynomial ODEs with PySINDy
'''
# Point to training data
XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=3_past_timesteps=20/20251215092753_global_ocean_land_rp=False_eps=1.0_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy')
XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=3_past_timesteps=20/20251215092753_global_ocean_land_rp=False_eps=1.0_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy')

# Adjust parameters for ODE inputs 
sep = None  # Reduce the amount of training data for SR 
# last_x_ind = 6  # exclude present precip
x_inds = [1, 2, 3]  # Select certain forcing variables
first_z_ind = -3  # # latent_dims
x_train= list(XZ_train[::sep, :, x_inds])
z_train= list(XZ_train[::sep, :, first_z_ind:])
x_val= list(XZ_val[::sep, :, x_inds])
z_val= list(XZ_val[::sep, :, first_z_ind:])

# Learn ODE without other Z_k variables
degree = 3  # Polynomial degree
#lambdas = np.linspace(1.e-5, 1.e-3, 10)
lambdas = [0.00034]  # Regularization strength
lambdas_scores = np.zeros(shape=(len(lambdas), 2))
for i, lmbda in enumerate(lambdas):
    print("LAMBDA: ", lmbda)
    opt = ps.STLSQ(threshold=lmbda)
    lib = ps.PolynomialLibrary(degree=degree)
    model = ps.SINDy(feature_library=lib, 
                    differentiation_method=ps.FiniteDifference(), #ps.SmoothedFiniteDifference(smoother_kws={'window_length': 100}), 
                    optimizer=opt)

    model.fit(x=z_train, u=x_train, t=1)
    model.print()  
    score = model.score(x=z_val, u=x_val, t=1)
    print("Model score: ", score)
    lambdas_scores[i, 0] = lmbda
    lambdas_scores[i, 1] = score

with open('SINDy_ODEs/equations/model_20251215092753_latent_dims=3_tp=20_degree=3_sep=3_lambda=0.00034.pkl', 'wb') as f:
    pickle.dump(model, f)
#np.save(f'SINDy_ODEs/degree={degree}/20251215092753_global_ocean_land_rp=False_eps=1.0_Ztrain_ConvNN+AE+D_tp=20_wd=0.0_zoom.npy', lambdas_scores)