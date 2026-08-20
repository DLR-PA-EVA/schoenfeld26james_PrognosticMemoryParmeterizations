from feyn import Model
import numpy as np
import sympytorch
import torch
import pysindy as ps
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_equations(paths):
    models = []
    for path in paths: 
        models.append(Model.load(path))
    
    return models

def convert_to_torchequations(models, signif=16):
    models_sympy = sympytorch.SymPyModule(expressions=[m.sympify(signif=signif) for m in models])
    models_sympy.to(device)
    used_inputs = []
    used_targets = []
    for m in models:
        used_inputs += m.inputs
        used_targets += [m.output]
    used_inputs = list(set(used_inputs))

    return models_sympy, used_inputs, used_targets


# def get_derivative(XZ, first_Z_ind):
#     dm = ps.FiniteDifference()
#     # dm = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
#     dZdt = np.array([dm._differentiate(Z[:, first_Z_ind:], t=1) for Z in XZ])
#     return dZdt


def load_data(path_train, path_test, coord_skipper_train, coord_skipper_test, all_inputs):
    XZ_train = np.load(path_train)[::coord_skipper_train]
    XZ_test = np.load(path_test)[::coord_skipper_test]

    data_train = pd.DataFrame(X)




if __name__ == "__main__":
    # Load models
    base_dir = 'qlattice_equations/latent_dim=4/5mio_traindata_complexity=10/'
    path1 = base_dir + 'my_qlattice_model_z1_dot_00.json'
    path2 = base_dir + 'my_qlattice_model_z2_dot_00.json'
    path3 = base_dir + 'my_qlattice_model_z3_dot_00.json'
    path4 = base_dir + 'my_qlattice_model_z4_dot_01.json'
    paths = [path1, path2, path3, path4]
    latent_dims = len(paths)
    all_inputs = ['coord', 'qv2m', 'Prw', 'T2m', 'Ts', 'Fs', 'Fl', 'l', 'Pr_true', 'Pr_pred']
    z_inputs = [f'z{i+1}' for i in range(latent_dims)]  
    all_inputs += z_inputs
    models = load_equations(paths)

    # Convert equations to torch modules
    models_torch, used_inputs, used_targets = convert_to_torchequations(models)

    # Load data
    coord_skipper_train = 100
    coord_skipper_test = None
    path_train = 'Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260129150710_global_ocean_land_rp=False_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy'
    path_test = 'Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260129150710_global_ocean_land_rp=False_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy'
    load_data(path_train, path_test, coord_skipper_train, coord_skipper_test)
