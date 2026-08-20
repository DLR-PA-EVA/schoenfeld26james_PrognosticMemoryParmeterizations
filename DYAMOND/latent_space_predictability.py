import torch
import numpy as np
import os
from parametrizations import *
from data_utils import get_dataloader, epsilon, load_regional_ds
#from training_gpu import preprocess, preprocess_thresholded
from datetime import datetime
import pysindy as ps

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_call_past(model, x_present, x_past):
    return model.forward(x_past, x_present)[0]

def forward_call_no_past(model, x_present, x_past):
    return model.forward(x_present)


def load_ds_from_model(model):
    ds = load_regional_ds(model.region, model.surface)  # get np array of shape [features, time, coords]
    ds = (ds - model.mean) / model.std

    if model.rescale_precip:
        precip = ds[-1]
        neg_mask_bool = precip < 0.0
        precip[neg_mask_bool] = 0.0
        ds[-1] = np.log(epsilon + precip)
    
    return torch.as_tensor(ds), model.tstart, model.train_ind, model.val_ind, model.mean, model.std
    




if __name__ == '__main__':
    # region, surface = 'tropical', 'ocean'
    # precip_threshold = None
    # model_path = 'networks/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260223172030_tropical_ocean_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'

    # make_Z_vars_for_ODE_learning(*setup_Z_sim(region, surface, model_path))
    XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260129150710_global_ocean_land_rp=False_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy')
    XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260129150710_global_ocean_land_rp=False_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy')
    n_features = 6
    latent_dims = 4
    benchmak_Zdot_predictability(XZ_train, XZ_val, n_features, latent_dims)
