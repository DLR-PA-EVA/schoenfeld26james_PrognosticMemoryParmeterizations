from datetime import datetime
import os 
from pathlib import Path
import torch
import torch.nn as nn

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model classes
class BaseParametrization:
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices):
        super().__init__()

        # Parametrization params
        self.latent_dims = latent_dims
        self.past_timesteps = past_timesteps
        self.memory_indices = memory_indices
        if self.memory_indices:
            self.n_memory_features = len(self.memory_indices)
        else:
            self.n_memory_features = 0
        self.n_features = n_features
        self.model_name = None
        
        # Training data params
        self.mean = None
        self.std = None
        self.region = None
        self.surface = None
        self.precip_threshold = None
        self.rescale_precip = None
        self.tstart = None
        self.train_ind = None
        self.val_ind = None
        self.train_share = None
        self.val_share = None

        # training params
        self.weight_decay = None
        self.learning_rate = None
        self.alpha = None
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.R2 = []
        self.precip_val_loss = []
        self.recon_val_loss = []

        # interpretability metrics
        self.linear_predictability = None
        self.nonlinear_predictability = None

        # other
        self.save_path = None

    def set_metadata(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f'Warning: BaseParametrization or Subclass has no attribute {key}, but got value {value} in kwargs. This key-value pair will be ignored.')

    def save(self, additional_info=None, save_path=None):
        # Save the model
        if hasattr(self, 'X'):
            self.X = None  # remove data before saving
        
        if save_path:
            torch.save(self, save_path)
        else:
            save_dir = Path(f'networks/{self.model_name}/n_features={self.n_features}_memory_indices={self.memory_indices}_latent_dims={self.latent_dims}_past_timesteps={self.past_timesteps}')
            if not save_dir.exists(): 
                os.makedirs(save_dir) 
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            if additional_info:
                save_file = f'{timestamp}_{additional_info}.pkl'
            else:
                save_file = f'{timestamp}.pkl'
            save_path = f'{save_dir}/{save_file}'
            torch.save(self, save_path)
            
        return save_path

    
class NNpAEpD(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices, X, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices,)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NN+AE+D'
        self.X = X
        self.X = self.X.to(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.past_timesteps * self.n_memory_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.latent_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.past_timesteps * self.n_memory_features)
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + self.n_features, self.nodes_per_layer), 
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )

    def forward(self, x_memory, x_present):
        latent_space = self.encoder(x_memory)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


class ConvNNpAEpD(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps, latent_dims, memory_indices, kernel_size, X, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices)
        
        self.kernel_size = kernel_size
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'ConvNN+AE+D'
        self.X = X
        self.X = self.X.to(device)

        # Length after convolution
        self.conv_out_len = self.past_timesteps - self.kernel_size + 1
        self.conv_out_features = self.past_timesteps * self.n_memory_features

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_memory_features, self.conv_out_features, self.kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_out_features * self.conv_out_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, self.latent_dims)
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, 12),
            nn.ReLU(),
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.conv_out_features * self.conv_out_len),
            nn.ReLU(),
            nn.Unflatten(1, (self.conv_out_features, self.conv_out_len)),
            nn.ConvTranspose1d(
                in_channels=self.conv_out_features,
                out_channels=self.n_memory_features,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0
            ),
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + self.n_features, self.nodes_per_layer), 
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )

    def forward(self, x_memory, x_present):
        latent_space = self.encoder(x_memory)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


class NNbase(nn.Module, BaseParametrization):
    def __init__(self, n_features, past_timesteps=0, latent_dims=0, memory_indices=None, nodes_per_layer=16):
        nn.Module.__init__(self)
        BaseParametrization.__init__(self, n_features, past_timesteps, latent_dims, memory_indices)
        
        self.nodes_per_layer = nodes_per_layer
        self.model_name = 'NNbase'

        self.neural_net = nn.Sequential(
            nn.Linear(self.n_features, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer),
            nn.ReLU(),
            nn.Linear(self.nodes_per_layer, 1)
        )
    
    def forward(self, x_present):
        return self.neural_net(x_present)


if __name__=='__main__':
    pass
    
