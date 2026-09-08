from parametrizations import ConvNNpAEpD, NNbase, NNpAEpD, BaseParametrization
import torch.nn as nn
import torch
from torchmetrics.regression import R2Score
from tqdm import trange
import copy
import os
from datetime import datetime
from pathlib import Path
from data_utils import *
# from latent_space_predictability import *
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#--- Train memory-informed ML-parametrizations and local baseline ---
def preprocess(X, coords, past_timesteps, Nf, tstart, tend, memory_indices):
    #  Shape of X: (Nfeatures, NT, Ncoords)
    y_batch = X[-1, past_timesteps + tstart: tend, coords]
    y_batch = y_batch.reshape(-1, 1)
    x_batch = X[:-1, tstart:tend, coords]
    x_batch = x_batch.unfold(1, past_timesteps + 1, 1)
    x_batch = x_batch.permute(0, 3, 1, 2)
    x_batch = x_batch.reshape(Nf, past_timesteps + 1, -1)  # X
    x_batch = x_batch.permute(2, 0, 1)  # Shape: (N_samples, Nfeatures, past_timesteps + 1)
    x_present = x_batch[:, :, -1]  
    x_past = x_batch[:, memory_indices, :-1]
    return x_present, x_past, y_batch


def prerprocess_AE(X, coords, past_timesteps, Nf, tstart, tend, memory_indices):
    x_present, x_past, y_batch = preprocess(X, coords, past_timesteps, Nf, tstart, tend, memory_indices)
    x_past = x_past.reshape(x_past.shape[0], -1)  # (N_samples, Nfeatures * past_timesteps)
    return x_present, x_past, y_batch


def preprocess_thresholded(X, coords, past_timesteps, Nf, tstart, tend, precip_threshold):  # TODO: Compute precip threshold in standardized units
    """
    X: (Nfeatures, NT, Ncoords)
    Returns:
        x_present: (N_samples, Nfeatures-1)
        x_past: (N_samples, Nfeatures-1, past_timesteps)
        y_batch: (N_samples, 1)
    """

    # Target variable
    y = X[-1, tstart + past_timesteps: tend, coords]  # (NT - past_timesteps, Ncoords)
    mask = y > precip_threshold
    y = y[mask].reshape(-1, 1)

    # Input features
    x = X[:-1, tstart: tend, coords]  # (Nfeatures-1, NT, Ncoords)

    # Vectorized unfolding — fully on GPU
    # Same as sliding window without Python loop
    seqs = x.unfold(dimension=1, size=past_timesteps + 1, step=1)
    # shape: (Nfeatures-1, NT - past_timesteps, Ncoords, past_timesteps + 1)

    # Move axes so we can apply the mask
    seqs = seqs.permute(1, 2, 0, 3)  # (NT - past_timesteps, Ncoords, Nfeatures-1, past_timesteps + 1)
    seqs = seqs[mask]  # apply mask on GPU

    # Split
    x_present = seqs[..., -1]          # (N_samples, Nfeatures-1)
    x_past = seqs[..., :-1]            # (N_samples, Nfeatures-1, past_timesteps)

    return x_present, x_past, y


def reconstruction_loss(x_batch, x_present, y_batch, model, criterion, reconstruction_criterion, alpha):
    b_pred, x_recon = model(x_batch, x_present)
    l_precip = alpha * criterion(b_pred, y_batch)
    l_recon = (1 - alpha) * reconstruction_criterion(x_recon, x_batch)
    #loss = criterion(b_pred, y_batch) + alpha * reconstruction_criterion(x_recon, x_batch)
    return l_precip, l_recon, b_pred


def prediction_loss(_, x_batch, y_batch, model, criterion):
    b_pred = model(x_batch)
    loss = criterion(b_pred, y_batch)
    return loss, torch.tensor(0.0), b_pred


def train_model(train_loader, test_loader, val_loader, model, skip, SS_tot, rescale_precip=False, alpha=1.0, num_epochs=5, weight_decay=1.e-9, lr=0.001, precip_threshold=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.learning_rate = lr
    model.weight_decay = weight_decay
    model.alpha = alpha
    best_val_loss = float('inf')
    pbar = trange(num_epochs, desc="Training", ncols=200)
    # Set loss function
    if model.model_name in ['ConvNN+AE+D', 'NN+AE+D']:
        reconstruction_criterion = nn.MSELoss()
        loss_func = reconstruction_loss
        loss_args = (reconstruction_criterion, alpha)
    elif model.model_name in ['NNbase']:
        loss_func = prediction_loss
        loss_args = ()
    else:
        raise ValueError(f'No loss function set for {model.model_name}')
    
    # Set preprocessing function
    if model.model_name in ['ConvNN+AE+D', 'NNbase']:
        if precip_threshold:
            process_func = preprocess_thresholded
            process_args = (precip_threshold,)
        else:
            process_func = preprocess
            process_args = (model.memory_indices,)
    elif model.model_name in ['NN+AE+D']:
        process_func = prerprocess_AE
        process_args = (model.memory_indices,)
    else:
        raise ValueError(f'No pre-processing function set for {model.model_name}')
    
    for epoch in pbar:
        model.train()
        total_train_loss = 0.0

        # Training
        total_precip_loss = 0.0
        total_recon_loss = 0.0
        for i, coords in enumerate(train_loader):
            # if i % skip != 0:
            #     continue
            # Preprocessing
            x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                model.n_features, train_loader.dataset.tstart, 
                                                train_loader.dataset.tend, *process_args)  
            if x_present.shape[0] == 0:  # When no samples meet the precip threshold
                continue
            # Compute loss
            l_precip, l_recon, _ = loss_func(x_past, x_present, y, model, criterion, *loss_args)
            loss = l_precip + l_recon
            total_recon_loss += l_recon.item()
            total_precip_loss += l_precip.item()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss_train = total_recon_loss / len(train_loader)
        avg_precip_loss_train = total_precip_loss / len(train_loader)
        model.train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_precip_loss = 0.0
        total_recon_loss = 0.0
        SS_res = 0
        with torch.no_grad():
            for coords in val_loader:
                # Preprocessing
                x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                    model.n_features, val_loader.dataset.tstart, 
                                                    val_loader.dataset.tend, *process_args)
                if x_present.shape[0] == 0:  # When no samples meet the precip threshold
                    continue

                # Compute loss
                l_precip, l_recon, y_pred = loss_func(x_past, x_present, y, model, criterion, *loss_args)
                loss = l_precip + l_recon
                total_val_loss += loss.item()
                total_precip_loss += l_precip.item()
                total_recon_loss += l_recon.item()
                if rescale_precip:
                    y = torch.exp(y) - epsilon
                    y_pred = torch.exp(y_pred) - epsilon
                SS_res += torch.sum((y - y_pred)**2)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_precip_loss = total_precip_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)

        model.precip_val_loss.append(avg_precip_loss)
        model.recon_val_loss.append(avg_recon_loss)
        model.val_loss.append(avg_val_loss)
        # Compute custom R2 score for monitoring only
        R2 = 1 - SS_res / SS_tot
        model.R2.append(R2)

        pbar.set_postfix({
            "Precip Train L": f"{avg_precip_loss_train:.4f}",
            "Recon Train L": f"{avg_recon_loss_train:.4f}",
            #"Val L": f"{avg_val_loss}",
            "Precip Val l": f"{avg_precip_loss:.4f}",
            "Recon Val l": f"{avg_recon_loss:.4f}",
            "R2": f"{R2:.4f}"
            })
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
    
    # Testing on best model
    model.load_state_dict(best_state_dict)
    model.eval()
    total_test_loss = 0.0
    Y, Y_pred = [], []
    with torch.no_grad():
        for coords in test_loader:
            # Preprocessing
            x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                model.n_features, test_loader.dataset.tstart, 
                                                test_loader.dataset.tend, *process_args)
            if x_present.shape[0] == 0:  # When no samples meet the precip threshold
                continue

            # Compute loss
            if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
                y_pred, _ = model.forward(x_past, x_present)
            elif model.model_name in ['NNbase']:
                y_pred = model.forward(x_present)
            loss = criterion(y_pred, y)
            total_test_loss += loss.item()
            if rescale_precip:
                y = torch.exp(y) - epsilon
                y_pred = torch.exp(y_pred) - epsilon
            Y_pred.append(y_pred)
            Y.append(y)      
    Y = torch.cat(Y, dim=0)
    Y_pred = torch.cat(Y_pred, dim=0)
    R2_Y = R2Score().to(device)(Y_pred, Y).item()
    avg_test_loss = total_test_loss / len(test_loader)
    model.test_loss.append(avg_test_loss)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"R2 Score: {R2_Y:.4f}")
    model.R2.append(R2_Y)

    return model


def get_SStot(ds, tval_start, tval_stop, past_timesteps, rescale_precip):
    precip = ds[-1, tval_start + past_timesteps: tval_stop]
    if rescale_precip:
        precip = torch.exp(precip) - epsilon
    mean_precip = torch.mean(precip)
    SS_tot = torch.sum((precip - mean_precip)**2)
    return SS_tot


# --- Latent space predictability ---
def setup_Z_sim(model_path, model=None):
    if model:
        pass
    else:  # model_path must be passed
        model = torch.load(model_path, weights_only=False, map_location=device)
    print(f'Setup: {model.region}, {model.surface}, precip_threshold={model.precip_threshold}')
    additional_info = f'{model.region}_{model.surface}_rp={model.rescale_precip}'
    # additional_info = None
    
    ds, tstart, train_ind, val_ind, _, _ = load_ds_from_model(model)
    trainloader, valloader, testloader = get_dataloader(ds, tstart, train_ind, val_ind, batch_size=1, shuffle_train=False)
    model = model.to(device)
    model.X = ds 
    model.X = model.X.to(device)
    return model, trainloader, valloader, testloader, additional_info 


def forward_call_past(model, x_present, x_past):
    return model.forward(x_past, x_present)[0]


def forward_call_no_past(model, x_present, x_past):
    return model.forward(x_present)


def make_Z_vars_for_ODE_learning(model, train_loader, val_loader, test_loader, additional_info=None, save=True, test=False):
    print('Simulate Z-data')
    save_dir = Path(f'Z_data/{model.model_name}/n_features={model.n_features}_memory_indices={model.memory_indices}_latent_dims={model.latent_dims}_past_timesteps={model.past_timesteps}')
    if not save_dir.exists(): 
        os.makedirs(save_dir) 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Set preprocessing function
    if model.model_name in ['NN+AE+D', 'ConvNN+AE+D', 'NNbase']:
        if model.precip_threshold:
            process_func = preprocess_thresholded
            process_args = (model.precip_threshold,)
        else:
            process_func = preprocess
            process_args = (model.memory_indices,)
    else:
        raise ValueError(f'No pre-processing function set for {model.model_name}')
    # Set forward call function
    if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
        forward_func = forward_call_past
    elif model.model_name in ['NNbase']:
        forward_func = forward_call_no_past

    model.eval()
    XYZ_train = []
    with torch.no_grad():
        # Iterate over batches
        for coords in train_loader:
            # Preprocessing
            x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                model.n_features, train_loader.dataset.tstart, 
                                                train_loader.dataset.tend, *process_args)  
            y_pred = forward_func(model, x_present, x_past)
            c = coords.to(device)
            c = torch.repeat_interleave(c, x_past.shape[0]).reshape(-1, 1)
            if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
                z = model.encoder(x_past)
                xyz = torch.cat([c, x_present, y, y_pred, z], dim=-1).detach().cpu().numpy()
            elif model.model_name in ['NNbase']:
                xyz = torch.cat([c, x_present, y, y_pred], dim=-1).detach().cpu().numpy()
            XYZ_train.append(xyz)
    XYZ_train = np.array(XYZ_train)
    if additional_info:
        path = f'{save_dir}/{timestamp}_{additional_info}_Ztrain_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
    else:
        path = f'{save_dir}/{timestamp}_Ztrain_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
    if save:
        np.save(path, XYZ_train)

    XYZ_val = []
    with torch.no_grad():
        # Iterate over batches
        for coords in val_loader:
            # Preprocessing
            x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                model.n_features, val_loader.dataset.tstart, 
                                                val_loader.dataset.tend, *process_args)  
            y_pred = forward_func(model, x_present, x_past)

            c = coords.to(device)
            c = torch.repeat_interleave(c, x_past.shape[0]).reshape(-1, 1)
            if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
                z = model.encoder(x_past)
                xyz = torch.cat([c, x_present, y, y_pred, z], dim=-1).detach().cpu().numpy()
            elif model.model_name in ['NNbase']:
                xyz = torch.cat([c, x_present, y, y_pred], dim=-1).detach().cpu().numpy()

            XYZ_val.append(xyz)
    XYZ_val = np.array(XYZ_val)
    if additional_info:
        path = f'{save_dir}/{timestamp}_{additional_info}_Zval_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
    else:
        path = f'{save_dir}/{timestamp}_Zval_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
    if save:
        np.save(path, XYZ_val)
    
    if test:  # Also create test data for quick testing of ODE learning code
        XYZ_test = []
        with torch.no_grad():
            # Iterate over batches
            for coords in test_loader:
                # Preprocessing
                x_present, x_past, y = process_func(model.X, coords, model.past_timesteps, 
                                                    model.n_features, test_loader.dataset.tstart, 
                                                    test_loader.dataset.tend, *process_args)  
                y_pred = forward_func(model, x_present, x_past)

                c = coords.to(device)
                c = torch.repeat_interleave(c, x_past.shape[0]).reshape(-1, 1)
                if model.model_name in ['NN+AE+D', 'ConvNN+AE+D']:
                    z = model.encoder(x_past)
                    xyz = torch.cat([c, x_present, y, y_pred, z], dim=-1).detach().cpu().numpy()
                elif model.model_name in ['NNbase']:
                    xyz = torch.cat([c, x_present, y, y_pred], dim=-1).detach().cpu().numpy()

                XYZ_test.append(xyz)
        XYZ_test = np.array(XYZ_test)
        if additional_info:
            path = f'{save_dir}/{timestamp}_{additional_info}_Ztest_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
        else:
            path = f'{save_dir}/{timestamp}_Ztest_{model.model_name}_tp={model.past_timesteps}_wd={model.weight_decay}.npy'
        if save:
            np.save(path, XYZ_test)
    
    return XYZ_train, XYZ_val


def preprocess_benchmarkNN(ind, x, y, latent_dims, n_features):
    x_batch, y_batch = x[ind], y[ind]
    #x_batch = torch.cat(tuple(x_batch), dim=0)
    x_batch = x_batch.reshape(-1, latent_dims + n_features)
    y_batch = y_batch.reshape(-1, latent_dims)
    #y_batch = torch.cat(tuple(y_batch), dim=0)
    return x_batch, y_batch


def train_predictability_benchmark(model, trainloader, testloader, SS_tot, num_epochs=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    pbar = trange(num_epochs, desc="Training", ncols=150)

    for epoch in pbar:
        model.train()
        train_loss = 0
        for ind in trainloader:
            x_batch, y_batch = preprocess_benchmarkNN(ind, model.X, model.Y, model.latent_dims, model.n_features)
            y_pred = model.forward(x_batch)
            loss = loss_func(y_pred, y_batch)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(trainloader)

        model.eval()
        test_loss = 0
        SS_res = 0
        with torch.no_grad():
            for ind in testloader:
                x_batch, y_batch = preprocess_benchmarkNN(ind, model.X_test, model.Y_test, model.latent_dims, model.n_features)
                y_pred = model.forward(x_batch)
                test_loss += loss_func(y_pred, y_batch).item()
                SS_res += torch.sum((y_batch - y_pred)**2, dim=0)
        test_loss = test_loss / len(testloader)
        # R2 = torch.mean(1 - SS_res / SS_tot).item()
        R2 = 1 - SS_res / SS_tot
        model.R2.append(R2.detach().cpu())
        model.train_loss.append(train_loss)
        model.test_loss.append(test_loss)
        # print(R2)

        pbar.set_postfix({
            "Train Loss": f"{train_loss:.8f}",
            "Test Loss": f"{test_loss:.8f}",
            "R2": f"{torch.mean(R2)}"
            })
    
    mean_R2 = [torch.mean(R2) for R2 in model.R2]
    return model.R2[np.argmax(mean_R2)]


def benchmak_Zdot_predictability(XZ_train, XZ_val, n_features, latent_dims):
    # Targets for ED
    dZdt_train = get_derivative(XZ_train, first_Z_ind=-latent_dims)
    dZdt_test = get_derivative(XZ_val, first_Z_ind=-latent_dims)

    # linear predictability
    sep = 10
    x_inds = list(range(1, n_features + 1))  # skip first x-ind (coordinate)
    first_z_ind = -latent_dims
    x_train= list(XZ_train[::sep, :, x_inds])
    z_train= list(XZ_train[::sep, :, first_z_ind:])
    x_val= list(XZ_val[::sep, :, x_inds])
    z_val= list(XZ_val[::sep, :, first_z_ind:])

    opt = ps.STLSQ(threshold=0.0)
    lib = ps.PolynomialLibrary(degree=1)
    model = ps.SINDy(feature_library=lib, 
                    differentiation_method=ps.FiniteDifference(),
                    #differentiation_method=ps.SmoothedFiniteDifference(smoother_kws={'window_length': 10}), 
                    optimizer=opt)

    model.fit(x=z_train, u=x_train, t=1)
    model.print()  
    R2_linear = model.score(x=z_val, u=x_val, t=1, multioutput='raw_values')
    print('Linear predictability: ', R2_linear)

    # nonlinear predictability
    trainloader, testloader, SStot = get_bench_dataloaders(torch.tensor(XZ_train, dtype=torch.float32), torch.tensor(XZ_val, dtype=torch.float32), 
                                                 dZdt_test, n_features, num_workes=0, batch_size=64)

    model = BenchmarkNN(latent_dims=latent_dims, n_features=n_features)
    inds_except_precip = list(range(1, n_features + 1)) + list(range(-latent_dims, 0))  # get list of indices that map to input features of f
    model.X = torch.tensor(XZ_train[:,:, inds_except_precip], dtype=torch.float32, device=device)
    model.X_test = torch.tensor(XZ_val[:,:, inds_except_precip], dtype=torch.float32, device=device)
    model.Y = dZdt_train.to(device)
    model.Y_test = dZdt_test.to(device)
    best_R2_nonlinear = train_predictability_benchmark(model, trainloader, testloader, SStot, num_epochs=200)
    print('Nonlinear predictability: ', best_R2_nonlinear)

    return R2_linear, best_R2_nonlinear


if __name__=='__main__':
    # Dataset parameters
    region = 'global'  # 'tropical' or 'global'
    surface = 'ocean_land'  # 'ocean' or 'ocean_land'
    precip_threshold = None  # in mm/h
    rescale_precip = False
    train_share, val_share = .8, .1
    print('rescale precip ', rescale_precip)
    print(region, surface)

    # Load data
    ds, tstart, train_ind, val_ind, mean, std = get_standardized_ds(region, surface, rescale_precip)
    # Collect parameters describing the current experiment
    training_params = {'region': region, 'surface': surface, 
                    'precip_threshold': precip_threshold, 'rescale_precip': rescale_precip,
                    'tstart': tstart, 'train_ind': train_ind, 'val_ind': val_ind , 
                    'train_share': train_share, 'val_share': val_share}
    
    parser = argparse.ArgumentParser(description='Train NN+AE and asess predictability')
    parser.add_argument('--latent_dims', type=int, default=4, help='Number of latent space dimensions')
    parser.add_argument('--past_timesteps', type=int, default=20, help='Number of past timesteps to use as input')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Regularization strength for L2 weight decay')
    args = parser.parse_args()
    past_timesteps = args.past_timesteps
    latent_dims = args.latent_dims
    wd = args.weight_decay
    n_features = 7
    kernel_size= 3
    n_epochs = 150
    memory_indices = [0, 1, 2, 3, 4, 5]

    for w in [wd]: 
        lr = .001
        alpha = .5 
        model = ConvNNpAEpD(n_features, past_timesteps, latent_dims, memory_indices, kernel_size, ds)
        # model = NNbase(n_features, past_timesteps=0, latent_dims=0, memory_indices=None)
        # model = NNpAEpD(n_features, past_timesteps, latent_dims, memory_indices, ds)
        # model = torch.load('networks/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=4_past_timesteps=50/20260327172444_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl', weights_only=False, map_location=device)
        # model_path = 'networks/NNbase/n_features=7_memory_indices=None_latent_dims=0_past_timesteps=0/20260506184630_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'
        # model_path = 'networks/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260506213850_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'
        # model = torch.load(model_path, weights_only=False, map_location=device) 
        model.set_metadata(training_params)
        print('t_p:', model.past_timesteps, ', d_z:', latent_dims, ', weigth decay: ', w)
        ds = ds.float()  # Convert to float32 for training
        model.X = ds
        model.X = model.X.to(device)
        model = model.to(device)
        trainloader, valloader, testloader = get_dataloader(ds, tstart, train_ind, val_ind, batch_size=8, num_workers=0)
        model.mean, model.std = mean, std
        tval_start, tval_end = valloader.dataset.tstart, valloader.dataset.tend
        SS_tot = get_SStot(ds, tval_start, tval_end, model.past_timesteps, rescale_precip=rescale_precip)
        model = train_model(trainloader, testloader, valloader, model, skip=1, SS_tot=SS_tot, 
                            rescale_precip=rescale_precip, alpha=alpha, num_epochs=n_epochs, 
                            lr=lr, weight_decay=w, precip_threshold=precip_threshold)
        model_path = model.save(f'{region}_{surface}_rp={rescale_precip}_alpha={alpha}_lr={lr}_weigth_decay={w}_eps={epsilon}')

        # --- Model Predictability ---
        save_Z_data = False
        create_test_data = False
        # model_path = 'networks/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260128182053_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'
        #model_path = 'networks/NNbase/n_features=7_memory_indices=None_latent_dims=0_past_timesteps=0/20251208155535_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'
        #trained_model = torch.load(model_path, weights_only=False, map_location=device)
        # model.load_state_dict(trained_model.state_dict())
        # model.set_metadata(training_params)
        print('Model region: ', model.region)
        XZ_train, XZ_val = make_Z_vars_for_ODE_learning(*setup_Z_sim(model_path=model_path, model=None), save=save_Z_data, test=create_test_data)
        # XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260417112208_global_ocean_land_rp=False_Ztrain_ConvNN+AE+D_tp=20_wd=None.npy')
        # XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260417112208_global_ocean_land_rp=False_Zval_ConvNN+AE+D_tp=20_wd=None.npy')
        n_features_ED = 6  # Reassign as number of features for NN predictability is not necessariliy the same as for the parametrization
        predictabity_linear, predictabity_nonlinear = benchmak_Zdot_predictability(XZ_train, XZ_val, n_features_ED, latent_dims)
        model.linear_predictability = predictabity_linear
        model.nonlinear_predictability = predictabity_nonlinear
        model.save(save_path=model_path)
        # model.save()