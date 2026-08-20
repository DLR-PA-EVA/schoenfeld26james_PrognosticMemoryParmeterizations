from pysr import PySRRegressor
import pysindy as ps
import numpy as np
import argparse


# Compute derivates
def get_derivative(XZ, first_Z_ind):
    dm = ps.FiniteDifference()
    # dm = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
    dZdt = np.array([dm._differentiate(Z[:, first_Z_ind:], t=1) for Z in XZ])
    #dZdt = torch.tensor(dZdt, dtype=torch.float32)
    return dZdt

# def get_C_Pr_XZ_and_dZdt(data, latent_dims, coord_skipper):
#     C = data[::coord_skipper, :, 0]
#     Pr = data[::coord_skipper, :, 1: first_X_index]
#     XZ = data[::coord_skipper, :, first_X_index: first_X_index + n_features + latent_dims]
#     dZdt = data[::coord_skipper, :, first_X_index + n_features + latent_dims:]
#     # Test if last shape of dZdt matches latent_dims
#     assert dZdt.shape[2] == latent_dims, f"dZdt shape {dZdt.shape} does not match latent_dims {latent_dims}"
#     feature_names = ['qv2m', 'Prw', 'T2m', 'Ts', 'Fs', 'Fl', 'l']
#     for i in range(latent_dims):
#         feature_names.append(f'z{i+1}')

#     return C, Pr, XZ, dZdt, feature_names


# --- prepare data ---
XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=3_past_timesteps=20/20251215092753_global_ocean_land_rp=False_eps=1.0_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy')
XZ_test = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=3_past_timesteps=20/20251215092753_global_ocean_land_rp=False_eps=1.0_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy')
latent_dims = 3
coord_skipper = 50
tmin = 1000
tmax = 1200
n_features = 3
input_indices = [1, 2, 3, -3, -2, -1]
# subset_size = 10_000

XZ_train = XZ_train[::coord_skipper, tmin:tmax, input_indices]
# XZ_test = XZ_test[::coord_skipper, tmin:tmax, input_indices]
print(XZ_train.shape)
dZdt_train = get_derivative(XZ_train, first_Z_ind=-latent_dims)
# dZdt_test = get_derivative(XZ_test, first_Z_ind=-latent_dims)
feature_names = ['qv2m', 'Prw', 'T2m', 'z1', 'z2', 'z3']

# Reshape for SR
XZ_train = XZ_train.reshape(-1, n_features + latent_dims)
dZdt_train = dZdt_train.reshape(-1, latent_dims)
# XZ_test = XZ_test.reshape(-1, n_features + latent_dims)
# dZdt_test = dZdt_test.reshape(-1, latent_dims)
print(XZ_train.shape, dZdt_train.shape)
# np.random.seed(1)
# subset = np.random.choice(XZ_train.shape[0], subset_size, replace=False)
# XZ_train = XZ_train[subset]
# dZdt_train = dZdt_train[subset]

# --- SR ---
# Configure Operators
# Removed relu and abs to avoid discontinuities
verylow_ops_complexity = 1
unary_operators = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
                   'exp', 'log', 'inv', 'square', 'cube', 'cbrt', 'sqrt']
binary_operators = ["/", "*", "+", "-", "^"]

# Very low-complexity operators (x)
very_low_complex_ops = ["*", "+", "-", '/'] 

# Low-complexity operators (2x)
low_complex_ops = ["/", "sqrt", "cube", 'square', 'cbrt', 'inv']

# Medium-complexity operators (3x)
medium_complex_ops = ['cos', 'sin', "exp", 'tan', "tanh", "cosh", "sinh", "log"]

# High-complexity operators (9x)
high_complex_ops = ["^"]

# Set up model
parser = argparse.ArgumentParser(description='Run symbolic regression for latent dynamics.')
parser.add_argument('--dim', type=int, default=0, help='Dimension of the latent space to analyze.')
args = parser.parse_args()
fit_dim = args.dim
print('Symbolic regression for dimension: ', fit_dim)
model = PySRRegressor(
    populations= 2 * 30,
    population_size=50,
    niterations=100000000,
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    complexity_of_operators = {**{key: 8*verylow_ops_complexity for key in high_complex_ops},
                               **{key: 3*verylow_ops_complexity for key in medium_complex_ops}, 
                               **{key: 2*verylow_ops_complexity for key in low_complex_ops},
                               **{key: verylow_ops_complexity for key in very_low_complex_ops}},
    complexity_of_variables = 1,
    complexity_of_constants=0,
    model_selection="accuracy",  # or "accuracy"
    progress=False,
    batching=False,
    # batch_size=50_000,
    maxsize=100,
    maxdepth=4
)

model.fit(XZ_train, dZdt_train[:, fit_dim])
