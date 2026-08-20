from parametrizations import *
import glob
import pickle
import torch
from pathlib import Path


def get_mask(grid, ds, region, surface, precip_threshold):
    region_coords_dict = {'tropical': (-23, 23), 'global': (-180, 180)}
    surface_mask_values_dict = {'ocean': -2, 'ocean_land': None}
    region_mask = (grid.clat <= region_coords_dict[region][1]) & (grid.clat >= region_coords_dict[region][0])
    if surface_mask_values_dict[surface]:
        surface_mask = grid.cell_sea_land_mask == surface_mask_values_dict[surface]
    else: 
        surface_mask = np.ones_like(grid.cell_sea_land_mask, dtype=bool)
    
    mask = (ds.precip >= precip_threshold) & surface_mask & region_mask
    return mask

def get_X(ds, training_mask, tmax=None):
    tmax = None
    X = torch.tensor(np.array([ds.huss.values[training_mask], ds.prw.values[training_mask], 
                            ds.tas.values[training_mask], ds.ts.values[training_mask], 
                            ds.hfss.values[training_mask], ds.hfls.values[training_mask],
                            ds.precip.values[training_mask]]), dtype=torch.float32)[:, :tmax, :]
    return X


def load_model(model_type=None, region=None, surface=None, threshold=None, path=None):
    if path:
        return torch.load(path, weights_only=False, map_location='cpu')

    base_dirs = {
        'NNbase': Path('networks/NNbase/n_features=6_memory_indices=None_latent_dims=0_past_timesteps=0'),
        'NN+AE+D': Path('networks/NN+AE+D/n_features=6_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=5_past_timesteps=2'),
        'NNpast': Path('networks/NNpast/n_features=6_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=0_past_timesteps=2')
    }

    base_dir = base_dirs.get(model_type)
    if base_dir is None:
        raise ValueError(f"Unknown model_type: {model_type}")

    # search for the file literally; no escaping needed
    matches = list(base_dir.glob(f"*{region}_{surface}_threshold={threshold}.pkl"))

    if len(matches) == 1:
        return torch.load(matches[0], weights_only=False, map_location='cpu')
    else:
        raise ValueError(f"Found {len(matches)} models for {model_type}, {region}, {surface}, threshold={threshold}")


def experiment():
    # Dataset parameters
    region = 'global'  # 'tropical' or 'global'
    surface = 'ocean_land'  # 'ocean' or 'ocean_land'
    precip_threshold = 0.0  # in mm/h

    # ML parameters
    #model = NNpast
    #model = NNbase
    model = ConvNNpAEpD
    past_timesteps = 10
    latent_dims = 5
    n_features = 6
    kernel_size= 3
    memory_indices = [0, 1, 2, 3, 4, 5]

    # Load data
    ds = load_DYAMOND_data()
    grid = load_grid()
    training_mask = get_mask(grid, ds, region, surface, precip_threshold)
    ds = ds[['huss', 'prw', 'tas', 'ts', 'hfss', 'hfls', 'precip']].to_array()
    ds = torch.as_tensor(ds.values, dtype=torch.float32)
    #X = get_X(ds, training_mask)

    # Train model
    model = model(n_features, past_timesteps, latent_dims, memory_indices, kernel_size)
    print(model.model_name, region, surface, precip_threshold)
    trainloader, valloader, testloader = generate_masked_dataloaders(ds, model, training_mask, batch_size=100_000, num_workers=50)
    model = train_model(trainloader, valloader, model, num_epochs=60)
    model.save(f'{region}_{surface}_threshold={precip_threshold}_kernel={kernel_size}')


def compute_R2_on_test(model_type=None, region=None, surface=None, precip_threshold=None, path=None):
    # ---Load data---
    ds = load_DYAMOND_data()
    grid = load_grid()
    training_mask = get_mask(grid, ds, region, surface, precip_threshold)
    ds = ds[['huss', 'prw', 'tas', 'ts', 'hfss', 'hfls', 'precip']].to_array()
    ds = torch.as_tensor(ds.values, dtype=torch.float32)
    model = load_model(model_type, region, surface, precip_threshold, path)
    _, testloader, _ = generate_masked_dataloaders(ds, model, training_mask, batch_size=100_000, num_workers=20)
    if model.model_name == 'NNbase':
        process_func = process_NN
    elif model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
        process_func = process_pt
    elif model.model_name == 'NNpast':
        process_func = process_NN_past

    process_args = ()

    Y_pred, Y = [], []
    for i, (x_batch, y_batch) in enumerate(testloader):
        x_past, x_present, y = process_func(x_batch, y_batch, *process_args)
        Y.append(y)
        model.eval()
        with torch.no_grad():
            if model.model_name in ['NNbase', 'NNpast']:
                y_pred = model(x_present)
            elif model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
                y_pred, _ = model(x_past, x_present)
        Y_pred.append(y_pred)
    
    Y = torch.cat(Y, dim=0)
    Y_pred = torch.cat(Y_pred, dim=0)
    # ---Compute R2---
    r2_metric = R2Score()
    r2 = r2_metric(Y_pred, Y).item()  
    print(r2)
    return r2


def collect_offline_training_results():
    data_dict = {'region': [], 'surface_type': [], 'precip_threshold':[], 
                    'R2_baseline': [], 'R2': [], 'MSE_baseline': [], 'MSE': []}
        
    for region in ['global', 'tropical']:
        for surface in ['ocean', 'ocean_land']:
            for precip_threshold in [0.05, 0.0]:
                print(region, surface, precip_threshold)
                data_dict['region'].append(region)
                data_dict['surface_type'].append(surface)
                data_dict['precip_threshold'].append(precip_threshold)

                model_base = load_model('NNbase', region, surface, precip_threshold)
                data_dict['MSE_baseline'].append(min(model_base.test_loss))
                R2_base = compute_R2_on_test('NNbase', region, surface, precip_threshold)
                data_dict['R2_baseline'].append(R2_base)
                print('finished baseline')

                model_base = load_model('NN+AE+D', region, surface, precip_threshold)
                data_dict['MSE'].append(min(model_base.test_loss))
                R2_base = compute_R2_on_test('NN+AE+D', region, surface, precip_threshold)
                data_dict['R2'].append(R2_base)
    
                with open('experiment_offline_results_different_configs.pkl', 'wb') as file:
                    print('Save notebook')
                    pickle.dump(data_dict, file)  # Save after every new entry


if __name__ == '__main__':
    experiment()
    # collect_offline_training_results()
    # path = 'networks/ConvNN+AE+D/n_features=6_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=5_past_timesteps=2/20251029170726_global_ocean_land_threshold=0.0_kernel=2.pkl'
    # compute_R2_on_test(model_type='ConvNN+AE+D', region='global', surface='ocean_land', precip_threshold=0.0, path=path)
    # path = 'networks/ConvNN+AE+D/n_features=6_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=5_past_timesteps=10/20251029173426_global_ocean_land_threshold=0.0_kernel=3.pkl'
    # compute_R2_on_test(model_type='ConvNN+AE+D', region='global', surface='ocean_land', precip_threshold=0.0, path=path)
    
    



