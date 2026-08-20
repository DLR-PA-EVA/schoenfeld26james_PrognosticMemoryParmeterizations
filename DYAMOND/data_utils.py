import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pysindy as ps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = 1.e-0


# Data loading
def load_grid(grid_path='/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/icon_grid_0019_R02B05_G.nc'):
    grid = xr.open_dataset(grid_path)
    grid = grid.assign_coords(
        clon=("cell", np.rad2deg(grid.clon.values)),
        clat=("cell", np.rad2deg(grid.clat.values))
    )
    return grid

def load_ds(ds_path='merged_15min_full_ICON-NWP-2km_DW-ATM_r1i1p1f1_2d_gn_20200120000000-20200301000000_R02B05.nc'):
    ds = xr.open_dataset(ds_path)
    ds = ds.assign_coords(
        clon=("cell", np.rad2deg(ds.clon.values)),
        clat=("cell", np.rad2deg(ds.clat.values))
    )
    return ds

def load_DYAMOND_data_bool_land_mask(grid_path='/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/icon_grid_0019_R02B05_G.nc', 
                      ds_path='merged_15min_ICON-NWP-2km_DW-ATM_r1i1p1f1_2d_gn_20200120000000-20200301000000_R02B05.nc'):
    grid = load_grid(grid_path)
    ds = load_ds(ds_path)

    # Compute land_mask
    mask = np.array([grid.cell_sea_land_mask.values for _ in range(ds.time.shape[0])])
    mask[mask < 0] = 0
    mask[mask > 0] = 1
    ds = ds.assign({'is_land': (('time', 'cell'), mask)})

    # Compute 15-min precipitation increment → convert to mm/h
    precip = ds.pracc.diff('time') * 4
    precip.name = 'precip'
    precip.attrs.update({
        'long_name': 'Precipitation rate',
        'units': 'mm h-1',
        'description': 'Derived from accumulated precipitation (15-min difference)'
    })

    # Drop the first timestep from the dataset to align with `precip` and add new variable
    ds = ds.isel(time=slice(1, None))
    ds['precip'] = precip

    # drop the height dimension for tas and huss
    ds['tas'] = ds.tas.isel(height=0)
    ds['huss'] = ds.huss.isel(height=0)
    ds = ds.drop_dims('height')
    return ds

def load_DYAMOND_data(grid_path='/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/icon_grid_0019_R02B05_G.nc', 
                      ds_path='merged_15min_full_ICON-NWP-2km_DW-ATM_r1i1p1f1_2d_gn_20200120000000-20200301000000_R02B05.nc',
                      ext_par_path = '/work/pd1295/ICON/grids/public/mpim/0019/extpar.2016/r0001/icon_extpar_grid_0019_R02B05_G_20180829.nc'):
    grid = load_grid(grid_path)
    ds = load_ds(ds_path)
    ext_par = xr.open_dataset(ext_par_path)
    ext_par = ext_par.assign_coords(
            clon=("cell", ds.clon.values),
            clat=("cell", ds.clat.values)
        )

    # Compute land_mask
    sea_land_frac = np.array([ext_par.FR_LAND_TOPO.values for _ in range(ds.time.shape[0])])
    ds = ds.assign({'is_land': (('time', 'cell'), sea_land_frac)})

    # Compute 15-min precipitation increment → convert to mm/h
    precip = ds.pracc.diff('time') * 4
    precip.name = 'precip'
    precip.attrs.update({
        'long_name': 'Precipitation rate',
        'units': 'mm h-1',
        'description': 'Derived from accumulated precipitation (15-min difference)'
    })

    # Drop the first timestep from the dataset to align with `precip` and add new variable
    ds = ds.isel(time=slice(1, None))
    ds['precip'] = precip

    # drop the height dimension for tas and huss
    ds['tas'] = ds.tas.isel(height=0)
    ds['huss'] = ds.huss.isel(height=0)
    ds = ds.drop_dims('height')
    return ds

def load_regional_ds(region, surface):
    # Load
    ds = load_DYAMOND_data()
    grid = load_grid()

    # Mask specific region/surface
    region_coords_dict = {'tropical': (-23, 23), 'global': (-180, 180)}
    surface_mask_values_dict = {'ocean': -2, 'ocean_land': None}
    region_mask = (grid.clat <= region_coords_dict[region][1]) & (grid.clat >= region_coords_dict[region][0])
    if surface_mask_values_dict[surface]:
        surface_mask = grid.cell_sea_land_mask == surface_mask_values_dict[surface]
    else: 
        surface_mask = np.ones_like(grid.cell_sea_land_mask, dtype=bool)
    mask = surface_mask & region_mask
    ds = ds[['huss', 'prw', 'tas', 'ts', 'hfss', 'hfls', 'is_land', 'precip']].to_array()
    ds = ds[:,:,mask]
    
    # Standardize except is_land and precip 
    # mean = xr.load_dataarray('merged_DYAMOND_variable_mean.nc')
    # std = xr.load_dataarray('merged_DYAMOND_variable_std.nc')
    # mean = xr.load_dataarray('merged_DYAMOND_variable_mean_unleaked.nc')
    # std = xr.load_dataarray('merged_DYAMOND_variable_std_unleaked.nc')
    # ds = (ds - mean) / std
    ds = ds.to_numpy().astype(np.float32)
    return ds#, mean, std

def get_standardized_ds(region='global', surface='ocean_land', rescale_precip=False, share_train=.8, share_val=.1):
    ds = load_regional_ds(region, surface)

    # Split train/val/test to compute mean and std on train set only
    tstart, train_ind, val_ind = train_val_test_split(ds, share_train, share_val)
    train_data = ds[:, tstart:train_ind, :]
    mean = train_data.mean(axis=(1, 2), keepdims=True)  # [features, 1, 1]
    std = train_data.std(axis=(1, 2), keepdims=True)    # [features, 1, 1]
    mean[-2:] = 0.0  # Don't standardize is_land and precip
    std[-2:] = 1.0
    ds = (ds - mean) / std

    # log scale precip
    if rescale_precip:
        precip = ds[-1]
        neg_mask_bool = precip < 0.0
        precip[neg_mask_bool] = 0.0
        ds[-1] = np.log(epsilon + precip)

    # Convert to tensor and return
    ds = torch.as_tensor(ds)
    return ds, tstart, train_ind, val_ind, mean, std


# Prepare data for ML
def train_val_test_split(ds, share_train, share_val):
    start_ind = 960  # Remove ~10d of simulation spin-up
    train_ind = int(ds.shape[1] * share_train)
    val_ind = int(ds.shape[1] * (share_val + share_train))

    return start_ind, train_ind, val_ind

class CoordDataset(Dataset):
    def __init__(self, X, tstart, tend):
        self.X = X
        self.Nfeatures, self.NT, self.num_coords = X.shape
        self.tstart = tstart
        self.tend = tend

    def __len__(self):
        return self.num_coords

    def __getitem__(self, idx):
        return idx
    
def get_dataloader(ds, tstart, train_ind, val_ind, batch_size=8, num_workers=0, shuffle_train=True):
    # Compute mean and std on train set only **This is what I want to do but it blows up reconstruction loss**
    # ds shape: [features, time, coords] -> compute over time and coords (dims 1 and 2)
    dataset_train, dataset_val, dataset_test = CoordDataset(ds, tstart, train_ind), CoordDataset(ds, train_ind, val_ind), CoordDataset(ds, val_ind, None)
    
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # print(ds.mean(dim=[1, 2]), ds.std(dim=[1, 2]))  # Check that precip is standardized to mean 0 and std 1
    return trainloader, valloader, testloader


# Prepare for Latent Space Predictability
def load_ds_from_model(model):
    ds = load_regional_ds(model.region, model.surface)  # get np array of shape [features, time, coords]
    ds = (ds - model.mean) / model.std
    
    if model.rescale_precip:
        precip = ds[-1]
        neg_mask_bool = precip < 0.0
        precip[neg_mask_bool] = 0.0
        ds[-1] = np.log(epsilon + precip)
    
    # return torch.as_tensor(ds, dtype=torch.float32), 960, 2000, 2500, None, None
    return torch.as_tensor(ds), model.tstart, model.train_ind, model.val_ind, model.mean, model.std
    
class BenchmarkNN(nn.Module):
    def __init__(self, latent_dims, n_features, nodes_per_layer=16):
        nn.Module.__init__(self)
        
        self.latent_dims = latent_dims
        self.n_features = n_features
        self.n_features_in = self.latent_dims + self.n_features
        self.n_features_out = latent_dims
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NNbase'
        self.X, self.Y = None, None
        self.X_test, self.Y_test = None, None

        self.neural_net = nn.Sequential(
            nn.Linear(self.n_features_in, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.n_features_out)
        )

        self.R2 = []
        self.train_loss = []
        self.test_loss = []
    
    def forward(self, x):
        return self.neural_net(x)

class BenchmarkCoordDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.num_coords, self.NT, self.N_features = X.shape

    def __len__(self):
        return self.num_coords

    def __getitem__(self, idx):
        return idx

def get_derivative(XZ, first_Z_ind):
    dm = ps.FiniteDifference()
    # dm = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10})
    dZdt = np.array([dm._differentiate(Z[:, first_Z_ind:], t=1) for Z in XZ])
    dZdt = torch.tensor(dZdt, dtype=torch.float32)
    return dZdt

def get_bench_dataloaders(XZ_train, XZ_test, dZdt_test, n_features, batch_size=32, num_workes=10):
    # inds_except_precip = list(range(XZ_train.shape[-1]))
    # inds_except_precip.pop(n_features)
    # inds_except_precip.pop(n_features)
    inds_except_precip = [1, 2, 3, 4, 5, 6, 7, -4, -3, -2, -1]
    print(inds_except_precip)
    train_dataset = BenchmarkCoordDataset(XZ_train[:, :, inds_except_precip])
    test_dataset = BenchmarkCoordDataset(XZ_test[:, :, inds_except_precip])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workes)
    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workes)

    latent_dims = dZdt_test.shape[-1]
    dZdt_test_flatt = dZdt_test.view(-1, latent_dims)
    dZdt_mean = torch.mean(dZdt_test_flatt, axis=0)
    SStot_test = torch.sum((dZdt_test_flatt - dZdt_mean)**2, dim=0).to(device)
    return trainloader, testloader, SStot_test