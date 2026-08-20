import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from datetime import datetime
import sympytorch
from torch.utils.data import DataLoader
import warnings
import pickle
from parametrizations import BaseParametrization
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN_ODE(nn.Module, BaseParametrization):
    def __init__(self, path_ODE, dt, path_to_AE_model, neural_net=None):
        model_AE = torch.load(path_to_AE_model, map_location=device, weights_only=False)
        past_timesteps, latent_dims =  model_AE.past_timesteps, model_AE.latent_dims
        memory_indices = model_AE.memory_indices
        n_features = model_AE.n_features #len(memory_indices)
        nodes_per_layer = model_AE.nodes_per_layer
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices,)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN_ODE'
        self.path_ODE = path_ODE
        with open(self.path_ODE, 'rb') as file:
            self.ODE = pickle.load(file)
        self.ODE = sympytorch.SymPyModule(expressions=self.ODE)
        self.dt = dt
        self.inputs = [f'z{i+1}' for i in range(latent_dims)] + ['qv2m', 'Prw', 'T2m', 'Ts', 'Fs', 'Fl', 'l', 'Precip'] 
        self.n_forcings = self.n_features + 1  # +1 to keep precip

        self.path_to_AE_model = path_to_AE_model
        self.model_AE = model_AE
        if not neural_net:  # No NN was passed so we use the NN from the NN+AE model
            self.neural_net = self.model_AE.neural_net
        else:
            self.neural_net = neural_net
            warnings.warn('Custom NN was passed as parametrization. If nodes_per_layer was changed you need to set this attribute manauly. Otherwise nodes_per_layer from the AE is used.')

        # Evaluation
        self.loss = []
        self.ODE_gradients = []
        self.NN_gradients = []
        self.train_R2 = []
        self.test_R2 = []
        self.train_diverged_preds = []
    
    def get_metadata_from_AE(self):
        metadata = {}
        AE_attributes = ['region', 'surface', 'mean', 'std', 'alpha', 'rescale_precip', 'precip_threshold', 'linear_predictability', 'nonlinear_predictability']
        for attr in AE_attributes:
            try:
                metadata[attr] = self.model_AE.__getattribute__(attr)
            except AttributeError:
                warnings.warn(f'AE model has no attribute {attr}. Set None instead')
                metadata[attr] = None

        self.set_metadata(metadata)
        return metadata
    
    def get_init_kwargs(self):
        kwargs = {
            'path_ODE': self.path_ODE,
            'dt': self.dt,
            'path_to_AE_model': self.path_to_AE_model
        }
        return kwargs

    def forward(self, coords, XP, Z, t_prediction, t_rollout):
        Z_ED = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.latent_dims), device=device)  # Z_ED saves Z variables at t0 + 1 up to t_prediction
        XP_batch = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.n_forcings), device=device)
        t0 = t_prediction - t_rollout
        z = Z[:, t0, :].reshape(-1, self.latent_dims)  # dims = (coords, prediction_points, laten_dims)
        for delta_t in range(t_rollout): 
            x = XP[:,t0 + delta_t].reshape(-1, self.n_forcings)
            zx = torch.cat((z, x), dim=-1)
            symbols = dict(zip(self.inputs, zx.T))  # Note that we pass precip here, but sympy equations never use the precip symbol
            z = z + self.ODE.forward(**symbols) * self.dt  # Euler Step
            Z_ED[delta_t] = z
            XP_batch[delta_t] = zx[..., self.latent_dims:]  # Can't I just set x instead of zx sliced
            
        Z_ED_batch = Z_ED.reshape(-1, self.latent_dims)
        XP_batch = XP_batch.reshape(-1, self.n_forcings)  # Last forcing dim carries precip
        P_batch = XP_batch[:, -1].reshape(-1, 1)
        P_pred_batch = self.neural_net(torch.cat((Z_ED_batch, XP_batch[:, :-1]), dim=1))  # Predict precip from input except precip
        return P_batch, P_pred_batch  # Return precip and predicted precip to compute parametrization loss

    def safe_forward(self, coords, XP, Z, t_prediction, t_rollout):
        Z_ED = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.latent_dims), device=device)  # Z_ED saves Z variables at t0 + 1 up to t_prediction
        XP_batch = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.n_forcings), device=device)
        t0 = t_prediction - t_rollout
        z = Z[:, t0, :].reshape(-1, self.latent_dims)  # dims = (coords, prediction_points, laten_dims)
        for delta_t in range(t_rollout): 
            x = XP[:,t0 + delta_t].reshape(-1, self.n_forcings)
            zx = torch.cat((z, x), dim=-1)
            symbols = dict(zip(self.inputs, zx.T))  # Note that we pass precip here, but sympy equations never use the precip symbol
            z = z + self.ODE.forward(**symbols) * self.dt  # Euler Step
            Z_ED[delta_t] = z
            XP_batch[delta_t] = zx[..., self.latent_dims:]  # Can't I just set x instead of zx sliced
            
        Z_ED_batch = Z_ED.reshape(-1, self.latent_dims)
        nan_mask = torch.isnan(Z_ED_batch)
        Z_ED_batch = torch.where(
            nan_mask,
            torch.zeros_like(Z_ED_batch),
            Z_ED_batch
        )
        N_diverged_predictions = torch.sum(nan_mask)
        XP_batch = XP_batch.reshape(-1, self.n_forcings)  # Last forcing dim carries precip
        P_batch = XP_batch[:, -1].reshape(-1, 1)
        P_pred_batch = self.neural_net(torch.cat((Z_ED_batch, XP_batch[:, :-1]), dim=1))  # Predict precip from input except precip
        return P_batch, P_pred_batch, N_diverged_predictions  # Return precip and predicted precip to compute parametrization loss


    def forward_Z(self, coords, XP, Z, t_prediction, t_rollout):
        Z_ED = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.latent_dims), device=device)  # Z_ED saves Z variables at t0 + 1 up to t_prediction
        Z_AE = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.latent_dims), device=device)
        XP_batch = torch.zeros(size=(t_rollout, len(coords) * len(t_prediction), self.n_forcings), device=device)
        t0 = t_prediction - t_rollout
        z = Z[:, t0, :].reshape(-1, self.latent_dims)  # dims = (coords, prediction_points, laten_dims)
        for delta_t in range(t_rollout): 
            x = XP[:,t0 + delta_t].reshape(-1, self.n_forcings)
            zx = torch.cat((z, x), dim=-1)
            symbols = dict(zip(self.inputs, zx.T))  # Note that we pass precip here, but sympy equations never use the precip symbol
            z = z + self.ODE.forward(**symbols) * self.dt  # Euler Step
            Z_ED[delta_t] = z
            Z_AE[delta_t] = Z[:, t0 + delta_t, :].reshape(-1, self.latent_dims)
            # XP_batch[delta_t] = zx[..., self.latent_dims:]
            XP_batch[delta_t] = x
            
        Z_ED_batch = Z_ED.reshape(-1, self.latent_dims).reshape(t_rollout, len(coords), len(t_prediction), self.latent_dims)
        Z_AE_batch = Z_AE.reshape(-1, self.latent_dims).reshape(t_rollout, len(coords), len(t_prediction), self.latent_dims)
        XP_batch = XP_batch.reshape(-1, self.n_forcings)  # Last forcing dim carries precip
        P_batch = XP_batch[:, -1].reshape(-1, 1)
        XP_batch = XP_batch.reshape(t_rollout, len(coords), len(t_prediction), self.n_forcings)
        P_batch = P_batch.reshape(t_rollout, len(coords), len(t_prediction), 1)
        return Z_ED_batch, Z_AE_batch, XP_batch, P_batch

    def save_checkpoint(self, additional_info=None):
        metadata = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'ODE_gradients': self.ODE_gradients,
            'NN_gradients': self.NN_gradients
        }
        metadata = self.get_metadata_from_AE() | metadata  # This will combine the two dictionaries, order matters as we want to keep losses from NN_ODE
        
        save_dir = Path(f'networks/{self.model_name}/n_features={self.n_features}_memory_indices={self.memory_indices}_latent_dims={self.latent_dims}_past_timesteps={self.past_timesteps}')
        if not save_dir.exists(): 
            os.makedirs(save_dir) 
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if additional_info:
            save_file = f'{timestamp}_{additional_info}.pkl'
        else:
            save_file = f'{timestamp}.pkl'
        save_path = f'{save_dir}/{save_file}'
    
        torch.save({
            "ODE_state": self.ODE.state_dict(),
            "NN_state": self.neural_net.state_dict(),
            "model_kwargs": self.get_init_kwargs(),
            "metadata": metadata
        }, save_path)


class EDCoordDataset(Dataset):
    def __init__(self, X, tstart, tend):
        self.X = X
        self.num_coords, self.NT, self.Nfeatures = X.shape
        self.tstart = tstart
        self.tend = tend

    def __len__(self):
        return self.num_coords

    def __getitem__(self, idx):
        return idx


def train_ODE_online(model_ODE, XZ_train, XZ_val, t_rollouts, epochs=100, batch_size=1024, lr=1.e-3, wd=0.0):
    '''
    model_ODE_online: A model consisting of a NN parametrization and an ODE that integrates latent space variables in time for parametrization input.
    t_rollouts: List of integers that give rollout times during training. Training will start with the first element.
    epochs: Integer or list of integers that gives the number of epochs that should be trained per rollout time.
    '''

    if isinstance(epochs, int):
        epochs = [epochs] * len(t_rollouts)

    # Set training parameters
    criterion = nn.MSELoss(reduction='sum')  # we need to reduce with sum because train and val have different numbers of prediction points and therefore effective batches have different lengths
    optimizer = torch.optim.Adam(model_ODE.parameters(), lr=lr, weight_decay=wd)
    model_ODE.weight_decay = wd
    model_ODE.learning_rate = lr
    divergence_penalty = 1.e-5

    # Split XZ into model inputs 
    xp_inds = [1, 2, 3, 4, 5, 6, 7, 8]
    XP_train = torch.tensor(XZ_train[:, :, xp_inds], dtype=torch.float32, device=device)
    XP_val = torch.tensor(XZ_val[:, :, xp_inds], dtype=torch.float32, device=device)
    Z_train = torch.tensor(XZ_train[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)
    Z_val = torch.tensor(XZ_val[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)

    # Compute SS_tot to evaluate R2 on validation set during training
    Pr_val = XP_val[:, :, -1]
    mean_precip_val = torch.mean(Pr_val)
    SS_tot = torch.sum((Pr_val - mean_precip_val)**2)

    trainset = EDCoordDataset(XP_train, None, None)  # tstart and tend are not needed here as X is already seperated into train and val set
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    valset = EDCoordDataset(XP_val, None, None)
    valloader = DataLoader(valset, batch_size, shuffle=False)

    tmax_train = XP_train.shape[1]
    tmax_val = XP_val.shape[1]

    for t_rollout, epoch in zip(t_rollouts, epochs):
        print(f'Train with rollout time {t_rollout}')
        t_prediction_train = np.arange(start=t_rollout, stop=tmax_train, step=t_rollout)
        t_prediction_val = np.arange(start=t_rollout, stop=tmax_val, step=t_rollout)
        pbar = trange(epoch, desc="Training", ncols=200)

        for _ in pbar:
            # Train Loop
            train_loss = 0
            n_train = 0
            ODE_grads = 0
            NN_grads = 0
            total_div_preds = 0
            model_ODE.train()
            for coords_train in trainloader:
                XP_batch = XP_train[coords_train] # grab batch
                Z_batch = Z_train[coords_train]
                Pr_batch, Pr_pred_batch = model_ODE.forward(coords_train, XP_batch, Z_batch, t_prediction_train, t_rollout)

                # handle diverged coords
                nan_mask = torch.isnan(Pr_pred_batch)
                N_diverged_preds = nan_mask.sum()
                Pr_pred_batch = torch.where(nan_mask, torch.zeros_like(Pr_pred_batch), Pr_pred_batch)

                loss = criterion(Pr_batch, Pr_pred_batch) #+ divergence_penalty * N_diverged_preds
                total_div_preds += N_diverged_preds
                train_loss += loss.item()
                n_train += Pr_batch.numel()

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                # -- Do gradient clipping to prevent explosion, but only clip ODE gradients, not NN gradients to not interfere with learning of the NN --
                torch.nn.utils.clip_grad_norm_(model_ODE.parameters(), 1.0)
                optimizer.step()
            
            avg_train_loss = train_loss / n_train
            avg_div_preds = total_div_preds / n_train
            model_ODE.train_loss.append(avg_train_loss)
            model_ODE.train_diverged_preds.append(avg_div_preds)

            # Validation Loop
            model_ODE.eval()
            val_loss = 0
            n_val = 0
            SS_res = 0
            with torch.no_grad():
                for coords_val in valloader:
                    XP_batch = XP_val[coords_val] # grab batch
                    Z_batch = Z_val[coords_val]
                    Pr_batch, Pr_pred_batch = model_ODE.forward(coords_val, XP_batch, Z_batch, t_prediction_val, t_rollout)
                    loss = criterion(Pr_batch, Pr_pred_batch)
                    val_loss += loss.item()
                    n_val += Pr_batch.numel()
                    SS_res += torch.sum((Pr_batch - Pr_pred_batch)**2)
                
                avg_val_loss = val_loss / n_val
                model_ODE.val_loss.append(avg_val_loss)
                R2 = 1 -SS_res / SS_tot
                model_ODE.R2.append(R2)
            
            # Update progress bar
            pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.8f}",
            "Val Loss": f"{avg_val_loss:.8f}",
            'diverged pred': f'{N_diverged_preds}',
            "R2": f"{R2:.4f}"
            })
    
    return model_ODE


def train_ODE_online_gradients(
    model_ODE,
    XZ_train,
    XZ_val,
    t_rollouts,
    epochs=100,
    batch_size=1024,
    lr=1.e-3,
    wd=0.0
):

    if isinstance(epochs, int):
        epochs = [epochs] * len(t_rollouts)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model_ODE.parameters(), lr=lr, weight_decay=wd)

    model_ODE.weight_decay = wd
    model_ODE.learning_rate = lr

    # Data
    xp_inds = [1, 2, 3, 4, 5, 6, 7, 8]
    XP_train = torch.tensor(XZ_train[:, :, xp_inds], dtype=torch.float32, device=device)
    XP_val = torch.tensor(XZ_val[:, :, xp_inds], dtype=torch.float32, device=device)

    Z_train = torch.tensor(XZ_train[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)
    Z_val = torch.tensor(XZ_val[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)

    # Validation normalization
    Pr_val = XP_val[:, :, -1]
    mean_precip_val = torch.mean(Pr_val)
    SS_tot = torch.sum((Pr_val - mean_precip_val) ** 2)

    trainset = EDCoordDataset(XP_train, None, None)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = EDCoordDataset(XP_val, None, None)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    tmax_train = XP_train.shape[1]
    tmax_val = XP_val.shape[1]

    def compute_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    for t_rollout, epoch in zip(t_rollouts, epochs):
        print(f'Train with rollout time {t_rollout}')

        t_prediction_train = np.arange(start=t_rollout, stop=tmax_train, step=t_rollout)
        t_prediction_val = np.arange(start=t_rollout, stop=tmax_val, step=t_rollout)

        pbar = trange(epoch, desc="Training", ncols=200)

        for _ in pbar:
            model_ODE.train()

            train_loss = 0.0
            n_train = 0
            total_div_preds = 0

            ODE_batch_grads = []
            NN_batch_grads = []

            for coords_train in trainloader:
                XP_batch = XP_train[coords_train]
                Z_batch = Z_train[coords_train]

                Pr_batch, Pr_pred_batch = model_ODE.forward(
                    coords_train, XP_batch, Z_batch, t_prediction_train, t_rollout
                )

                # Handle NaNs safely
                nan_mask = torch.isnan(Pr_pred_batch)
                N_diverged_preds = nan_mask.sum()

                Pr_pred_batch = torch.where(
                    nan_mask,
                    torch.zeros_like(Pr_pred_batch),
                    Pr_pred_batch
                )

                loss = criterion(Pr_batch, Pr_pred_batch)

                train_loss += loss.item()
                n_train += Pr_batch.numel()
                total_div_preds += N_diverged_preds

                # Backprop
                optimizer.zero_grad()
                loss.backward()

                # Compute gradient norms (before any clipping)
                ODE_grad_norm = compute_grad_norm(model_ODE.ODE)
                NN_grad_norm = compute_grad_norm(model_ODE.neural_net)

                ODE_batch_grads.append(ODE_grad_norm)
                NN_batch_grads.append(NN_grad_norm)

                optimizer.step()

            # ---- Aggregate gradient stats ----
            ODE_batch_grads = np.array(ODE_batch_grads)
            NN_batch_grads = np.array(NN_batch_grads)

            def grad_stats(arr):
                return {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "max": float(arr.max()),
                    "p95": float(np.percentile(arr, 95)),
                }

            model_ODE.ODE_gradients.append(grad_stats(ODE_batch_grads))
            model_ODE.NN_gradients.append(grad_stats(NN_batch_grads))

            avg_train_loss = train_loss / n_train
            avg_div_preds = total_div_preds / n_train

            model_ODE.train_loss.append(avg_train_loss)
            model_ODE.train_diverged_preds.append(avg_div_preds)

            # ---- Validation ----
            model_ODE.eval()
            val_loss = 0.0
            n_val = 0
            SS_res = 0.0

            with torch.no_grad():
                for coords_val in valloader:
                    XP_batch = XP_val[coords_val]
                    Z_batch = Z_val[coords_val]

                    Pr_batch, Pr_pred_batch = model_ODE.forward(
                        coords_val, XP_batch, Z_batch, t_prediction_val, t_rollout
                    )

                    loss = criterion(Pr_batch, Pr_pred_batch)

                    val_loss += loss.item()
                    n_val += Pr_batch.numel()
                    SS_res += torch.sum((Pr_batch - Pr_pred_batch) ** 2)

            avg_val_loss = val_loss / n_val
            R2 = 1 - SS_res / SS_tot

            model_ODE.val_loss.append(avg_val_loss)
            model_ODE.R2.append(R2)

            # ---- Progress bar ----
            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.8f}",
                "Val Loss": f"{avg_val_loss:.8f}",
                "Div": f"{total_div_preds}",
                "R2": f"{R2:.4f}",
            })

    return model_ODE


def train_ODE_online_safe(
    model_ODE,
    XZ_train,
    XZ_val,
    t_rollouts,
    epochs=100,
    batch_size=10_000,
    lr=1.e-3,
    wd=0.0,
    clip_norm=1.0,
    instability_penalty=1e-5,
    value_threshold=1e6   # threshold for explosion detection
):


    if isinstance(epochs, int):
        epochs = [epochs] * len(t_rollouts)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model_ODE.parameters(), lr=lr, weight_decay=wd)

    model_ODE.weight_decay = wd
    model_ODE.learning_rate = lr

    xp_inds = [1, 2, 3, 4, 5, 6, 7, 8]
    XP_train = torch.tensor(XZ_train[:, :, xp_inds], dtype=torch.float32, device=device)
    XP_val = torch.tensor(XZ_val[:, :, xp_inds], dtype=torch.float32, device=device)

    Z_train = torch.tensor(XZ_train[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)
    Z_val = torch.tensor(XZ_val[:, :, -model_ODE.latent_dims:], dtype=torch.float32, device=device)

    Pr_val = XP_val[:, :, -1]
    mean_precip_val = torch.mean(Pr_val)
    SS_tot = torch.sum((Pr_val - mean_precip_val) ** 2)

    trainset = EDCoordDataset(XP_train, None, None)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = EDCoordDataset(XP_val, None, None)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    tmax_train = XP_train.shape[1]
    tmax_val = XP_val.shape[1]

    def is_invalid(x):
        return (
            torch.isnan(x).any() or
            torch.isinf(x).any() or
            (x.abs() > value_threshold).any()
        )

    def has_invalid_params(model):
        return any(is_invalid(p) for p in model.parameters())

    def compute_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    for t_rollout, epoch in zip(t_rollouts, epochs):
        print(f'Train with rollout time {t_rollout}')

        t_prediction_train = np.arange(start=t_rollout, stop=tmax_train, step=t_rollout)
        t_prediction_val = np.arange(start=t_rollout, stop=tmax_val, step=t_rollout)

        pbar = trange(epoch, desc="Training", ncols=200)

        for _ in pbar:
            model_ODE.train()

            train_loss = 0.0
            n_train = 0
            total_div_preds = 0

            ODE_batch_grads = []
            NN_batch_grads = []

            for coords_train in trainloader:
                XP_batch = XP_train[coords_train]
                Z_batch = Z_train[coords_train]

                # ---- SAVE STATE FOR ROLLBACK ----
                old_state = copy.deepcopy(model_ODE.state_dict())

                Pr_batch, Pr_pred_batch, N_diverged_preds = model_ODE.safe_forward(
                    coords_train, XP_batch, Z_batch, t_prediction_train, t_rollout
                )

                # ---- CHECK FOR FORWARD INSTABILITY ----
                # if is_invalid(Pr_pred_batch) or is_invalid(Z_batch):
                #     model_ODE.load_state_dict(old_state)
                #     optimizer.zero_grad()
                #     continue

                # ---- HANDLE NAN VALUES (SAFE MASKING) ----
                # nan_mask = torch.isnan(Pr_pred_batch)
                # N_diverged_preds = nan_mask.sum()

                # Pr_pred_batch = torch.where(
                #     nan_mask,
                #     torch.zeros_like(Pr_pred_batch),
                #     Pr_pred_batch
                # )

                loss = criterion(Pr_batch, Pr_pred_batch)

                # ---- LOSS CHECK ----
                # if not torch.isfinite(loss):
                #     model_ODE.load_state_dict(old_state)
                #     optimizer.zero_grad()
                #     continue

                # ---- INSTABILITY PENALTY ----
                instability_penalty_term = N_diverged_preds * instability_penalty
                loss = loss + instability_penalty_term

                train_loss += loss.item()
                n_train += Pr_batch.numel()
                total_div_preds += N_diverged_preds

                # ---- BACKWARD ----
                optimizer.zero_grad()
                loss.backward()

                # ---- GRADIENT CLIPPING ----
                # torch.nn.utils.clip_grad_norm_(model_ODE.parameters(), clip_norm)

                # ---- GRADIENT STATS ----
                ODE_batch_grads.append(compute_grad_norm(model_ODE.ODE))
                NN_batch_grads.append(compute_grad_norm(model_ODE.neural_net))

                optimizer.step()

                # ---- POST-STEP PARAMETER CHECK ----
                if has_invalid_params(model_ODE):
                    for p in model_ODE.ODE.parameters():
                        print(p)
                    model_ODE.load_state_dict(old_state)
                    optimizer.zero_grad()
                    continue

            # ---- SAFE AGGREGATION ----
            ODE_batch_grads = np.array(ODE_batch_grads) if len(ODE_batch_grads) > 0 else np.array([0.0])
            NN_batch_grads = np.array(NN_batch_grads) if len(NN_batch_grads) > 0 else np.array([0.0])

            def grad_stats(arr):
                return {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "max": float(arr.max()),
                    "p95": float(np.percentile(arr, 95)),
                }

            model_ODE.ODE_gradients.append(grad_stats(ODE_batch_grads))
            model_ODE.NN_gradients.append(grad_stats(NN_batch_grads))

            avg_train_loss = train_loss / max(n_train, 1)
            avg_div_preds = total_div_preds / (max(n_train, 1) * Pr_batch.shape[0])

            model_ODE.train_loss.append(avg_train_loss)
            model_ODE.train_diverged_preds.append(avg_div_preds)

            # ---- VALIDATION ----
            model_ODE.eval()
            val_loss = 0.0
            n_val = 0
            SS_res = 0.0

            with torch.no_grad():
                for coords_val in valloader:
                    XP_batch = XP_val[coords_val]
                    Z_batch = Z_val[coords_val]

                    Pr_batch, Pr_pred_batch, N_diverged_preds_val = model_ODE.safe_forward(
                        coords_val, XP_batch, Z_batch, t_prediction_val, t_rollout
                    )

                    nan_mask = torch.isnan(Pr_pred_batch)
                    Pr_pred_batch = torch.where(
                        nan_mask,
                        torch.zeros_like(Pr_pred_batch),
                        Pr_pred_batch
                    )

                    # if is_invalid(Pr_pred_batch):
                    #     continue

                    loss = criterion(Pr_batch, Pr_pred_batch)

                    # if not torch.isfinite(loss):
                    #     continue

                    val_loss += loss.item()
                    n_val += Pr_batch.numel()
                    SS_res += torch.sum((Pr_batch - Pr_pred_batch) ** 2)

            avg_val_loss = val_loss / max(n_val, 1)
            R2 = 1 - SS_res / SS_tot if SS_tot > 0 else float("nan")

            model_ODE.val_loss.append(avg_val_loss)
            model_ODE.R2.append(R2)

            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.6f}",
                "Val Loss": f"{avg_val_loss:.6f}",
                "Div": f"{total_div_preds}",
                "R2": f"{R2:.4f}",
            })

    return model_ODE


def load_NNODE_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model = NN_ODE(**checkpoint['model_kwargs'])
    model.ODE.load_state_dict(checkpoint['ODE_state'], )
    model.neural_net.load_state_dict(checkpoint['NN_state'])
    model.set_metadata(checkpoint['metadata'])

    return model


if __name__=='__main__':
    # --- Training block ---
    XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260507185408_global_ocean_land_rp=False_Ztrain_ConvNN+AE+D_tp=20_wd=0.0.npy')
    XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260507185408_global_ocean_land_rp=False_Zval_ConvNN+AE+D_tp=20_wd=0.0.npy')
    path_NNpAE = 'networks/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260507185301_global_ocean_land_rp=False_alpha=0.5_lr=0.001_weigth_decay=0.0_eps=1.0.pkl'
    # path_ODE = 'SINDy_ODEs/equations/model_20260507185408_linear_ODE_4d_past_timesteps=20.pkl'
    #path_ODE = 'nonlinear_stable_ODE_4d_sympy_equations.pkl'
    #path_ODE = 'qlattice_equations/laten_dims=4_complexity=10_n_epochs=1000/nonlinear_models_selection_sympy.pkl'
    #model_online = NN_ODE(path_ODE, dt=1, path_to_AE_model=path_NNpAE)
    # model_online = load_NNODE_checkpoint('networks/NN_ODE/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260508161522_linear_4d_corrected_00_timesteps.pkl')
    model_online = load_NNODE_checkpoint('networks/NN_ODE/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5]_latent_dims=4_past_timesteps=20/20260511173500_nonlinear_4d_corrected_00_timesteps_0.pkl')
    model_online.to(device)
    #model_online = train_ODE_online(model_online, XZ_train, XZ_val, t_rollouts=[5, 10, 20, 50, 100, 200, 300, 360], epochs=[10, 10, 10, 5, 5, 5, 5, 10], lr=1.e-3)
    for i in range(1):
        print(i)
        model_online = train_ODE_online(model_online, XZ_train, XZ_val, t_rollouts=[360], epochs=[10], lr=1.e-3)
        model_online.save_checkpoint(f'nonlinear_4d_corrected_00_timesteps_{i}')
     



