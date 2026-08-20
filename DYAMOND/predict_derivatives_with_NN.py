import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange
from torchmetrics.regression import R2Score
from torch.utils.data import Dataset, DataLoader
from parametrizations import load_DYAMOND_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Compute derivates
def get_derivative(XZ, first_Z_ind):
    dm = ps.FiniteDifference()
    dZdt = np.array([dm._differentiate(Z[:, first_Z_ind:], t=1) for Z in XZ])
    dZdt = torch.tensor(dZdt, dtype=torch.float32)
    return dZdt


class CoordDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.num_coords, self.NT, self.N_features = X.shape

    def __len__(self):
        return self.num_coords

    def __getitem__(self, idx):
        return idx


def get_dataloaders(XZ_train, XZ_test, dZdt_test, n_features, batch_size=32, num_workes=10):
    inds_except_precip = list(range(XZ_train.shape[-1]))
    inds_except_precip.pop(n_features)
    print(inds_except_precip)
    train_dataset = CoordDataset(XZ_train[:, :, inds_except_precip])
    test_dataset = CoordDataset(XZ_test[:, :, inds_except_precip])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workes)
    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workes)

    latent_dims = dZdt_test.shape[-1]
    dZdt_test_flatt = dZdt_test.view(-1, latent_dims)
    dZdt_mean = torch.mean(dZdt_test_flatt, axis=0)
    SStot_test = torch.sum((dZdt_test_flatt - dZdt_mean)**2, dim=0).to(device)
    return trainloader, testloader, SStot_test


class NN(nn.Module):
    def __init__(self, latent_dims, n_features, nodes_per_layer=16):
        nn.Module.__init__(self)
        
        self.latent_dims = latent_dims
        self.n_features = n_features
        self.n_features_in = latent_dims + self.n_features
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
    
    def forward(self, x):
        return self.neural_net(x)
    

def preprocess(ind, x, y, latent_dims, n_features):
    x_batch, y_batch = x[ind], y[ind]
    #x_batch = torch.cat(tuple(x_batch), dim=0)
    x_batch = x_batch.reshape(-1, latent_dims + n_features)
    y_batch = y_batch.reshape(-1, latent_dims)
    #y_batch = torch.cat(tuple(y_batch), dim=0)
    return x_batch, y_batch


def simple_train(model, trainloader, testloader, SS_tot, num_epochs=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.MSELoss()
    pbar = trange(num_epochs, desc="Training", ncols=150)

    for epoch in pbar:
        model.train()
        train_loss = 0
        for ind in trainloader:
            x_batch, y_batch = preprocess(ind, model.X, model.Y, model.latent_dims)
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
                x_batch, y_batch = preprocess(ind, model.X_test, model.Y_test, model.latent_dims)
                y_pred = model.forward(x_batch)
                test_loss += loss_func(y_pred, y_batch).item()
                SS_res += torch.sum((y_batch - y_pred)**2, dim=0)
        test_loss = test_loss / len(testloader)
        R2 = torch.mean(1 - SS_res / SS_tot).item()


        pbar.set_postfix({
            "Train Loss": f"{train_loss:.8f}",
            "Test Loss": f"{test_loss:.8f}",
            "R2": f"{R2:.4f}"
            })


XZ_train = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=9_past_timesteps=50/20251120131619_Ztrain_ConvNN+AE+D_tp=50_wd=0.0001.npy')[:,:,1:]  # Remove the coordinate variable
XZ_val = np.load('Z_data/ConvNN+AE+D/n_features=7_memory_indices=[0, 1, 2, 3, 4, 5, 6]_latent_dims=9_past_timesteps=50/20251120131619_Zval_ConvNN+AE+D_tp=50_wd=0.0001.npy')[:,:,1:]  # Remove the coordinate variable
n_features = 7
latent_dims = 9

dZdt_train = get_derivative(XZ_train, first_Z_ind=n_features + 1)
dZdt_test = get_derivative(XZ_val)
trainloader, testloader, SStot = get_dataloaders(torch.tensor(XZ_train, dtype=torch.float32), torch.tensor(XZ_val, dtype=torch.float32), 
                                                 dZdt_test, num_workes=0, batch_size=64)

model = NN(latent_dims=latent_dims, n_features=n_features)
inds_except_precip = list(range(XZ_train.shape[-1]))
inds_except_precip.pop(n_features)
model.X = torch.tensor(XZ_train[:,:, inds_except_precip], dtype=torch.float32, device=device)
model.X_test = torch.tensor(XZ_val[:,:, inds_except_precip], dtype=torch.float32, device=device)
model.Y = dZdt_train.to(device)
model.Y_test = dZdt_test.to(device)
print(model.X.device)
simple_train(model, trainloader, testloader, SStot, num_epochs=200)
