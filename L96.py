import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
from pathlib import Path
import os
import pickle
import pysr
import torch
import re
import torch.nn as nn
import argparse

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class L96TwoLevelMemory(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001, parametrization=None, save_derivative=False,
                 X_init=None, Y_init=None, memory_tau=1.0, memory_cutoff=1.0, save_dt=0.1, memory_activation_func=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt

        # Parametrization
        '''
        parametrization is expected to be a class with: 
        - a forward method such that param.forward(X) -> B
        - a variable past_timesteps that informs L96 how many past X-steps need to be saved for the parametrization
        '''
        self.parametrization = parametrization
        
        # Initialize state variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()

        # Memory parameters
        self.memory_tau = memory_tau  # Decay rate of the kernel
        self.memory_cutoff = memory_cutoff  # Maximum memory time
        self.memory_steps = int(memory_cutoff / dt)  # Number of stored steps
        print('Memory steps:', self.memory_steps)
        self.memory_X = []  # Store past X values
        self.param_memory_X = [self.X]  # Store past X values for the parametrization
        self.times = np.array(range(self.memory_steps - 1, -1, -1)) * self.dt # Delta ts for the memory kernel, from memory_cutoff to zero
        #self.time_weights = np.array([np.exp(-self.memory_tau * t / self.dt) for t in self.times])
        self.time_weights = self.memory_tau / (1 - np.exp(-self.memory_tau * self.memory_cutoff)) * np.array([np.exp(-self.memory_tau * t) for t in self.times])
        self.memory_activation_func = memory_activation_func
        
        # Initialize history storage
        self._history_X = []
        self._history_dXdt = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_Y = []
        
        # Save interval
        #self.save_derivative = save_derivative
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.step_count = 0

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(np.array(self._history_X), dims=['time', 'x'], name='X')  # 2D array for X
        dic['B'] = xr.DataArray(np.array(self._history_B), dims=['time', 'x'], name='B')
        #dic['Y_mean'] = xr.DataArray(np.array(self._history_Y_mean), dims=['time', 'x'], name='Y_mean')
        #dic['Y2_mean'] = xr.DataArray(np.array(self._history_Y2_mean), dims=['time', 'x'], name='Y2_mean')
        #dic['Y'] = xr.DataArray(np.array(self._history_Y), dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
            np.atleast_1d(fn(h.Y2_mean, ax))
        ])

    def _rhs_X_dt(self, X, B):
        """Compute the right-hand side of the X equation."""
        dXdt = (-np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + self.F + B)
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y, memory_term):
        """Compute the right-hand side of the Y equation with memory effect."""
        dYdt = (-self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) - Y +
                self.h / self.J * np.repeat(memory_term, self.J)) * self.c
        return self.dt * dYdt

    def compute_memory_term(self):
        """Compute the memory integral using the trapezoidal rule."""
        if len(self.memory_X) < 2:
            return self.X  # No history yet, use current X

        memory_term = self.dt * np.sum([w * X_past for w, X_past in zip(self.time_weights[-len(self.memory_X):], self.memory_X)], axis=0)
        
        if self.memory_activation_func:
            memory_term = self.memory_activation_func(memory_term)
        
        return memory_term

    def step(self):
        """Integrate one time step with memory effects."""
        # Compute memory integral
        memory_term = self.compute_memory_term()
        
        # Compute backreaction term for X
        B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)
        
        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        k1_Y = self._rhs_Y_dt(self.X, self.Y, memory_term)
        k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2, memory_term)
        k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2, memory_term)
        k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y, memory_term)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += (1 / 6) * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        
        # Store X for memory effects
        if len(self.memory_X) >= self.memory_steps:
            self.memory_X.pop(0)  # Remove oldest entry
        self.memory_X.append(self.X.copy())

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.parametrization.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        
        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            #Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            #Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            #self._history_Y_mean.append(Y_mean.copy())
            #self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            #self._history_Y.append(self.Y.copy())
    

    def step_parametrized(self):
        # Check if enough data is available to run the parametrization
        if self.step_count <= self.parametrization.past_timesteps:
            # Run step with fast variables
            self.step()
            return

        """Integrate one time step with parametrized fast varibales."""
        # Compute backreaction term for X
        self.parametrization.eval()
        with torch.no_grad():
            x = np.array(self.param_memory_X).astype(np.float32).T
            x = torch.from_numpy(x)
            B = self.parametrization.forward(x)
            B = B.numpy().flatten()

        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.parametrization.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self._history_X.append(self.X.copy())
            self._history_B.append(B.copy())
        

    def iterate(self, time):
        if self.parametrization is not None:
            step_func = self.step_parametrized
        else:
            step_func = self.step

        steps = int(time / self.dt)
        for _ in tqdm(range(steps)):
            step_func()


class L962LvlMem(object):
    def __init__(self, K=8, J=32, h=1, F=20, c=10, b=10, dt=0.001, parametrization=None,
                 X_init=None, Y_init=None, tau=.001, m=.001, save_dt=0.001, memory_activation_func=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt

        # Parametrization
        '''
        parametrization is expected to be a class with: 
        - a forward method such that param.forward(X) -> B
        - a variable past_timesteps that informs L96 how many past X-steps need to be saved for the parametrization
        '''
        self.parametrization = parametrization
        
        # Initialize state variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()

        # Memory parameters
        self.tau = tau  # Decay rate of the kernel
        self.m = m  # Maximum memory time
        self.memory_steps = int(self.m / dt)  # Number of stored steps
        print('Memory steps:', self.memory_steps)
        self.memory_X = [self.X]  # Store past X values
        self.param_memory_X = [self.X]  # Store past X values for the parametrization
        self.times = np.array(range(self.memory_steps - 1, -1, -1)) * self.dt # Delta ts for the memory kernel, from memory_cutoff to zero
        # self.time_weights = self.tau / (1 - np.exp(-self.tau * self.m)) * np.array([np.exp(-self.tau * t) for t in self.times])  # old version
        self.time_weights = 1 /(self.tau * (1 - np.exp(- self.m/self.tau))) * np.array([np.exp(-t/self.tau) for t in self.times])  # tau -> 1/tau
        self.memory_activation_func = memory_activation_func
        
        # Initialize history storage
        self._history_X = []
        self._history_dXdt = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_Y = []
        self._history_Z = []
        
        # Save interval
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.step_count = 0

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_Y = []

    @property
    def history(self):
        coords={'time': np.arange(len(self._history_X)) * self.dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        
        dic = {}
        dic['X'] = xr.DataArray(np.array(self._history_X), dims=['time', 'x'], name='X')  # 2D array for X
        dic['B'] = xr.DataArray(np.array(self._history_B), dims=['time', 'x'], name='B')
        if self.parametrization:
            if 'NN+AE' in self.parametrization.model_name:
                dic['Z'] = xr.DataArray(np.array(self._history_Z), dims=['time', 'x', 'latent_dims'])
                coords['latent_dims'] = np.arange(self.parametrization.latent_dims)
            

        return xr.Dataset(
            dic,
            coords=coords
            
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
            np.atleast_1d(fn(h.Y2_mean, ax))
        ])

    def _rhs_X_dt(self, X, B):
        """Compute the right-hand side of the X equation."""
        dXdt = (-np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + self.F + B)

        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y, memory_term):
        """Compute the right-hand side of the Y equation with memory effect."""
        dYdt = (-self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) - Y +
                self.h / self.b * np.repeat(memory_term, self.J)) * self.c
        return self.dt * dYdt

    def compute_memory_term(self):
        """Compute the memory integral using the trapezoidal rule."""
        if len(self.memory_X) < 2:
            return self.X  # No history yet, use current X

        memory_term = self.dt * np.sum([w * X_past for w, X_past in zip(self.time_weights[-len(self.memory_X):], self.memory_X)], axis=0)
        
        if self.memory_activation_func:
            memory_term = self.memory_activation_func(memory_term)
        
        return memory_term

    def step(self):
        """Integrate one time step with memory effects."""
        # Compute memory integral
        memory_term = self.compute_memory_term()
        
        # Compute backreaction term for X
        # B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)  # coupling term in Rasp et al. configuration
        B = - (self.h /self.b) * self.c * self.Y.reshape(self.K, self.J).sum(1)  # coupling term in Christensen et al. configuration
        
        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        k1_Y = self._rhs_Y_dt(self.X, self.Y, memory_term)
        k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2, memory_term)
        k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2, memory_term)
        k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y, memory_term)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += (1 / 6) * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        
        # Store X for memory effects
        if len(self.memory_X) >= self.memory_steps:
            self.memory_X.pop(0)  # Remove oldest entry
        self.memory_X.append(self.X.copy())

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.parametrization.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        
        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            #Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            #Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            #self._history_Y_mean.append(Y_mean.copy())
            #self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            #self._history_Y.append(self.Y.copy())
            if 'NN+AE' in self.parametrization.model_name:
                self._history_Z.append(np.zeros((self.K, self.parametrization.latent_dims)))
           
    def step_parametrized(self):
        # Check if enough data is available to run the parametrization
        if self.step_count <= 1000:  # Do spinup with regular L96 for 1 MTU
            # Run step with fast variables
            self.step()
            return

        """Integrate one time step with parametrized fast varibales."""
        # Compute backreaction term for X
        self.parametrization.eval()

        if self.parametrization.model_name == 'NN+AE':
            with torch.no_grad():
                x_past = torch.tensor(np.array(self.param_memory_X).astype(np.float32).T[:,:-1])
                x_present_shifted = torch.tensor(np.array([np.roll(self.X, -i) for i in range(self.K)]), dtype=torch.float32)
                Z = self.parametrization.encoder(x_past)
                x = torch.cat((Z, x_present_shifted), dim=1)
                B = self.parametrization.neural_net(x).numpy().flatten()
        elif self.parametrization.model_name == 'NN':
            with torch.no_grad():
                x_present_shifted = torch.tensor(np.array([np.roll(self.X, -i) for i in range(self.K)]), dtype=torch.float32)
                B = self.parametrization.forward(x_present_shifted).numpy().flatten()

        else:
            with torch.no_grad():
                x = np.array(self.param_memory_X).astype(np.float32).T
                x = torch.from_numpy(x)
                if hasattr(self.parametrization, 'encoder'):
                    x_past = x[:, :-1]
                    x_present = torch.unsqueeze(x[:, -1], 1)
                    Z = self.parametrization.encoder(x_past)
                    x_nn = torch.cat((Z, x_present), dim=1)
                    B = self.parametrization.neural_net(x_nn).numpy().flatten()
                else:
                    B = self.parametrization.forward(x)
                    B = B.numpy().flatten()

        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.parametrization.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self._history_X.append(self.X.copy())
            self._history_B.append(B.copy())
            if 'NN+AE' in self.parametrization.model_name:
                self._history_Z.append(Z.numpy().copy())

    def iterate(self, time, save_interval=500000000, model='baseline_nn'):
        if self.parametrization is not None:
            step_func = self.step_parametrized
        else:
            step_func = self.step

        steps = int(time / self.dt)
        for _ in tqdm(range(steps)):
            step_func()

            if self.step_count % save_interval == 0:
                if model == 'baseline_nn':
                    pt = 0
                else:
                    pt = self.parametrization.past_timesteps
                
                if 'NN+AE' in model:
                    mn = 'NN+AE'
                else:
                    mn = model
                h = self.history
                h.attrs['m'] = self.m
                h.attrs['tau'] = self.tau
                h.attrs['model'] = self.parametrization.model_name if self.parametrization is not None else 'NO_PARAMETRIZATION'
                h.attrs['past_timesteps'] = self.parametrization.past_timesteps if self.parametrization is not None else 0
                path = f'networks/{model}/input_lagg={pt}/m={m}_tau={tau}_{mn}.pkl'
                h.attrs['model_path'] = path
                if hasattr(self.parametrization, 'latent_dims') and self.parametrization.latent_dims is None:
                    self.parametrization.latent_dims = 0
                h.attrs['latent_dims'] = self.parametrization.latent_dims if hasattr(self.parametrization, 'latent_dims') else 0
                h.to_netcdf(f'./online_runs/{model}/input_lagg={pt}/time={int(self.step_count * self.dt - save_interval * self.dt)}_{int(self.step_count * self.dt)}MTU_m={self.m}_tau={self.tau}.nc', mode='w')


class L962LvlMemZ(object):
    def __init__(self, K=8, J=32, h=1, F=20, c=10, b=10, dt=0.001, encoder=None, parametrization=None, 
                 latent_dims=3, past_timesteps=None, equations=None,
                 X_init=None, Y_init=None, tau=1.0, m=1.0, save_dt=0.001, memory_activation_func=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt

        # Parametrization
        '''
        parametrization is expected to be a class with: 
        - a forward method such that param.forward(X) -> B
        - a variable past_timesteps that informs L96 how many past X-steps need to be saved for the parametrization
        '''
        self.encoder = encoder
        self.parametrization = parametrization
        self.past_timesteps = past_timesteps
        self.equations = [load_equation_models(dim) for dim in range(latent_dims)]
        self.latent_dims = latent_dims
        self.xm = np.array([3.7499406, 3.8337057, 3.737856,  3.7358584, 2.6165977, 3.2850852, 3.7044418, 2.6879525, 3.238627,  3.679761,  2.5996115, 3.2479622]) 
        self.xs = np.array([5.1449647, 5.134747,  5.092099,  4.217163,  3.6469073, 3.5644076, 4.212367, 3.6456118, 3.5587962, 4.163716,  3.5897295, 3.5596335]) 
        self.dzm, self.dzs = np.array([0.01590547, -0.00163114, 0.00207441]),  np.array([66.69067,  43.466976, 48.401077])
        
        # Initialize state variables
        self.X = np.random.rand(self.K, dtype=np.float32) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self.Z = np.random.uniform(-20, 20, size=(self.K, latent_dims))

        # Memory parameters
        self.tau = tau  # Decay rate of the kernel
        self.m = m  # Maximum memory time
        self.memory_steps = int(self.m / dt)  # Number of stored steps
        print('Memory steps:', self.memory_steps)
        self.memory_X = [self.X]  # Store past X values
        self.param_memory_X = [self.X]  # Store past X values for the parametrization
        self.times = np.array(range(self.memory_steps - 1, -1, -1)) * self.dt # Delta ts for the memory kernel, from memory_cutoff to zero
        # self.time_weights = self.tau / (1 - np.exp(-self.tau * self.m)) * np.array([np.exp(-self.tau * t) for t in self.times])  # old version
        self.time_weights = 1 /(self.tau * (1 - np.exp(- self.m/self.tau))) * np.array([np.exp(-t/self.tau) for t in self.times])  # tau -> 1/tau
        self.memory_activation_func = memory_activation_func
        
        # Initialize history storage
        self._history_X = []
        self._history_dXdt = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_B_param = []
        self._history_Y = []
        self._history_Z = []
        self._history_Z_encoder = []
        self._history_dZdt = []
        
        # Save interval
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.step_count = 0

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        self._history_Y = []

    @property
    def history(self):
        dic = {}
        print(np.array(self._history_Z).shape, np.array(self._history_Z_encoder).shape, np.array(self._history_B).shape)
        dic['X'] = xr.DataArray(np.array(self._history_X), dims=['time', 'x'], name='X')  # 2D array for X
        dic['B'] = xr.DataArray(np.array(self._history_B), dims=['time', 'x'], name='B')
        dic['Z'] = xr.DataArray(np.array(self._history_Z), dims=['time', 'x', 'd'], name='Z')
        dic['Z_enc'] = xr.DataArray(np.array(self._history_Z_encoder), dims=['time', 'x', 'd'], name='Z_enc')
        dic['dZdt'] = xr.DataArray(np.array(self._history_dZdt), dims=['time', 'x', 'd'], name='dZdt')
        
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J), 'd': np.arange(self.latent_dims)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
            np.atleast_1d(fn(h.Y2_mean, ax))
        ])

    def _rhs_X_dt(self, X, B):
        """Compute the right-hand side of the X equation."""
        dXdt = (-np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + self.F + B)

        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y, memory_term):
        """Compute the right-hand side of the Y equation with memory effect."""
        dYdt = (-self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) - Y +
                self.h / self.b * np.repeat(memory_term, self.J)) * self.c
        return self.dt * dYdt
    
    def _rhs_Z_dt(self, X, Z):
        """Compute right-hand side of the Z equation"""
        input = [X, np.roll(X, -1), np.roll(X, 1), 
                Z[:,0], Z[:,1], Z[:,2],
                np.roll(Z[:,0], -1), np.roll(Z[:,1], -1), np.roll(Z[:,2], -1),
                np.roll(Z[:,0], 1), np.roll(Z[:,1], 1), np.roll(Z[:,2], 1)]
        input = (np.array(input).T - self.xm) / self.xs
        dzdts = []
        for i, eq in enumerate(self.equations):
            dzdt = eq.predict(input, -1)
            dzdt = dzdt * self.dzs[i] + self.dzm[i]
            dzdts.append(dzdt)
        
        dZdt = np.stack(dzdts, axis=0).T
        return self.dt * dZdt

    def compute_memory_term(self):
        """Compute the memory integral using the trapezoidal rule."""
        if len(self.memory_X) < 2:
            return self.X  # No history yet, use current X

        memory_term = self.dt * np.sum([w * X_past for w, X_past in zip(self.time_weights[-len(self.memory_X):], self.memory_X)], axis=0)
        
        if self.memory_activation_func:
            memory_term = self.memory_activation_func(memory_term)
        
        return memory_term

    def step(self):
        """Integrate one time step with memory effects."""
        # Compute memory integral
        memory_term = self.compute_memory_term()
        
        # Compute backreaction term for X
        # B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)  # coupling term in Rasp et al. configuration
        B = - (self.h /self.b) * self.c * self.Y.reshape(self.K, self.J).sum(1)  # coupling term in Christensen et al. configuration
        
        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        k1_Y = self._rhs_Y_dt(self.X, self.Y, memory_term)
        k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2, memory_term)
        k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2, memory_term)
        k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y, memory_term)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += (1 / 6) * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        
        # Store X for memory effects
        if len(self.memory_X) >= self.memory_steps:
            self.memory_X.pop(0)  # Remove oldest entry
        self.memory_X.append(self.X.copy())

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        
        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self._history_X.append(self.X.copy())
            self._history_B.append(B.copy())
            #self._history_B_param.append(B_param.copy())
            self._history_Z.append(np.zeros((self.K, self.latent_dims)))  # Placeholder for Z
            self._history_Z_encoder.append(np.zeros((self.K, self.latent_dims)))
            self._history_dZdt.append(np.zeros((self.K, self.latent_dims)))
    
    def step_parametrized(self):
        # Check if enough data is available to run the parametrization
        if self.encoder:
            if self.step_count <= self.past_timesteps:
                # Run step with fast variables
                Z_encoder = np.random.uniform(-20, 20, size=(self.K, self.latent_dims))
                self.step()
                return
            if self.step_count == self.past_timesteps + 1:
                with torch.no_grad():
                    x = np.array(self.param_memory_X).astype(np.float32).T
                    x = torch.from_numpy(x[:, :-1])  # Exclude present time step for memory encoder
                    self.Z = self.encoder(x).numpy()  # Init Z with encoder
            if self.step_count >= self.past_timesteps + 1:
                with torch.no_grad():
                    x = np.array(self.param_memory_X).astype(np.float32).T
                    x = torch.from_numpy(x[:, :-1])
                    Z_encoder = self.encoder(x).numpy()
                    #print(np.max(np.abs(Z_encoder)))

        """Integrate one time step with parametrized fast varibales."""
        # Compute backreaction term for X
        with torch.no_grad():
            x = torch.from_numpy(self.X.T)
            z = torch.from_numpy(self.Z)
            zx = torch.stack([z[:,0], z[:,1], z[:,2], x], dim=1)  
            B_param = self.parametrization(zx)
            B_param = B_param.numpy().flatten()

        memory_term = self.compute_memory_term()
        B = - (self.h /self.b) * self.c * self.Y.reshape(self.K, self.J).sum(1)

        # X
        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        # Y
        k1_Y = self._rhs_Y_dt(self.X, self.Y, memory_term)
        k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2, memory_term)
        k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2, memory_term)
        k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y, memory_term)
        # Z
        k1_Z = self._rhs_Z_dt(self.X, Z_encoder)
        k2_Z = self._rhs_Z_dt(self.X, Z_encoder + k1_Z / 2)
        k3_Z = self._rhs_Z_dt(self.X, Z_encoder + k2_Z / 2)
        k4_Z = self._rhs_Z_dt(self.X, Z_encoder + k3_Z)

        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += (1 / 6) * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        self.Z += (1 / 6) * (k1_Z + 2 * k2_Z + 2 * k3_Z + k4_Z)

        # Store X for parametrization
        if self.parametrization is not None: 
            if len(self.param_memory_X) >= self.past_timesteps + 1:
                self.param_memory_X.pop(0)  # Remove oldest entry
            self.param_memory_X.append(self.X.copy())

        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self._history_X.append(self.X.copy())
            self._history_B.append(B.copy())
            #self._history_B_param.append(B_param.copy())
            self._history_Z.append(self.Z.copy())
            self._history_Z_encoder.append(Z_encoder.copy())
            self._history_dZdt.append(k1_Z / self.dt)
        

    def iterate(self, time, save_interval=5_000_000, model='baseline_nn'):
        if self.parametrization is not None:
            step_func = self.step_parametrized
        else:
            step_func = self.step

        steps = int(time / self.dt)
        for _ in tqdm(range(steps)):
            step_func()


class L96TwoLevel(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.001,
                 integration_type='uncoupled', parameterization=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.noprog, self.noYhist, self.integration_type = noprog, noYhist, integration_type
        self.step_count = 0
        self.save_dt = save_dt
        self.parameterization = parameterization
        if self.parameterization is not None: self.integration_type = 'parameterization'
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()]
        self._history_Y2_mean = [(self.Y.reshape(self.K, self.J)**2).mean(1).copy()]
        self._history_B = [-self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)]
        if not self.noYhist:
            self._history_Y = [self.Y.copy()]

    def _rhs_X_dt(self, X, Y=None, B=None):
        """Compute the right hand side of the X-ODE."""
        if Y is None:
            dXdt = (
                    -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                    X + self.F + B
            )
        else:
            dXdt = (
                    -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                    X + self.F - self.h * self.c * Y.reshape(self.K, self.J).mean(1)
            )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, 1) * (np.roll(Y, 2) - np.roll(Y, -1)) -
                       Y + self.h / self.J * np.repeat(X, self.J)
               ) * self.c
        return self.dt * dYdt

    def _rhs_dt(self, X, Y):
        return self._rhs_X_dt(X, Y=Y), self._rhs_Y_dt(X, Y)

    def step(self, add_B=True, B=None):
        """Integrate one time step"""
        if self.parameterization is None:
            B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1) if B is None else B
            if self.integration_type == 'coupled':
                k1_X, k1_Y = self._rhs_dt(self.X, self.Y)
                k2_X, k2_Y = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2)
                k3_X, k3_Y = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2)
                k4_X, k4_Y = self._rhs_dt(self.X + k3_X, self.Y + k3_Y)
            elif self.integration_type == 'uncoupled':
                k1_X = self._rhs_X_dt(self.X, B=B)
                k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
                k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
                k4_X = self._rhs_X_dt(self.X + k3_X, B=B)
                # Then update Y with unupdated X
                k1_Y = self._rhs_Y_dt(self.X, self.Y)
                k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2)
                k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2)
                k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y)

            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        else:  # Parameterization case
            k1_X = self._rhs_X_dt(self.X, B=0)
            k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=0)
            k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=0)
            k4_X = self._rhs_X_dt(self.X + k3_X, B=0)

            B = self.parameterization(self.X) if B is None else B
            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            if add_B: self.X += B * self.dt

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

    def iterate(self, time, save_interval=500_000):
        steps = int(time / self.dt)
        for _ in tqdm(range(steps), disable=self.noprog):
            self.step()
            if self.step_count % save_interval == 0:
                print(f'Saving history at step {self.step_count}')
                np.save(f'/work/bd1179/b309297/L96/X_{self.step_count-save_interval}_{save_interval}.npy', self.history.X.values)
                self.erase_history()


    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        if not self.noYhist:
            self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['B'] = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        dic['Y2_mean'] = xr.DataArray(self._history_Y2_mean, dims=['time', 'x'], name='Y2_mean')
        if not self.noYhist:
            dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
            dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
            np.atleast_1d(fn(h.Y2_mean, ax))
        ])


def save_L96(L96):
    run_time = int(L96.step_count * L96.dt)
    try:
        m, tau = L96.memory_cutoff, L96.memory_tau
    except AttributeError:
        m, tau = L96.m, L96.tau

    if L96.parametrization is not None:
        # This is an online run
        param = L96.parametrization.model_name
        input_lagg = L96.parametrization.past_timesteps
        save_dir = Path(f'./online_runs/{param}/input_lagg={input_lagg}')
        save_path = f'{save_dir}/time={run_time}MTU_m={m}_tau={tau}.pkl'
        if not save_dir.exists(): 
            os.makedirs(save_dir) 
    else:
        save_path = f'./online_runs/NO_PARAMETRIZATION/time={run_time}MTU_m={m}_tau={tau}.pkl'
    
    with open(save_path, 'wb') as file:
        pickle.dump(L96, file)


class L96TwoLevel_updated(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.1,
                 integration_type='uncoupled', parameterization=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.noprog, self.noYhist, self.integration_type = noprog, noYhist, integration_type
        self.step_count = 0
        self.save_dt = save_dt
        self.parameterization = parameterization
        if self.parameterization is not None: self.integration_type = 'parameterization'
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()]
        self._history_Y2_mean = [(self.Y.reshape(self.K, self.J)**2).mean(1).copy()]
        self._history_B = [-self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)]
        if not self.noYhist:
            self._history_Y = [self.Y.copy()]

    def _rhs_X_dt(self, X, Y=None, B=None):
        """Compute the right hand side of the X-ODE."""
        if Y is None:
            dXdt = (
                    -np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) -
                    X + self.F + B
            )
        else:
            dXdt = (
                    -np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) -
                    X + self.F - (self.h /self.b) * self.c * Y.reshape(self.K, self.J).sum(1)
            )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       Y + self.h / self.b * np.repeat(X, self.J)
               ) * self.c
        return self.dt * dYdt

    def _rhs_dt(self, X, Y):
        return self._rhs_X_dt(X, Y=Y), self._rhs_Y_dt(X, Y)

    def step(self, add_B=True, B=None):
        """Integrate one time step"""
        if self.parameterization is None:
            B = -(self.h / self.b) * self.c * self.Y.reshape(self.K, self.J).sum(1) if B is None else B
            if self.integration_type == 'coupled':
                k1_X, k1_Y = self._rhs_dt(self.X, self.Y)
                k2_X, k2_Y = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2)
                k3_X, k3_Y = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2)
                k4_X, k4_Y = self._rhs_dt(self.X + k3_X, self.Y + k3_Y)
            elif self.integration_type == 'uncoupled':
                k1_X = self._rhs_X_dt(self.X, B=B)
                k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
                k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
                k4_X = self._rhs_X_dt(self.X + k3_X, B=B)
                # Then update Y with unupdated X
                k1_Y = self._rhs_Y_dt(self.X, self.Y)
                k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2)
                k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2)
                k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y)

            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        else:  # Parameterization case
            k1_X = self._rhs_X_dt(self.X, B=0)
            k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=0)
            k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=0)
            k4_X = self._rhs_X_dt(self.X + k3_X, B=0)

            B = self.parameterization(self.X) if B is None else B
            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            if add_B: self.X += B * self.dt

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

    def iterate(self, time, save_interval=500_000):
        steps = int(time / self.dt)
        for _ in tqdm(range(steps), disable=self.noprog):
            self.step()
            if self.step_count % save_interval == 0:
                print(f'Saving history at step {self.step_count}')
                np.save(f'/work/bd1179/b309297/L96/X_{self.step_count-save_interval}_{self.step_count}.npy', self.history.X.values)
                self.erase_history()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        if not self.noYhist:
            self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['B'] = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        dic['Y2_mean'] = xr.DataArray(self._history_Y2_mean, dims=['time', 'x'], name='Y2_mean')
        if not self.noYhist:
            dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
            dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
            np.atleast_1d(fn(h.Y2_mean, ax))
        ])


class NNpAEpD(nn.Module):
    def __init__(self, past_timesteps, n_neighbours, latent_dims, nodes_per_layer=16):
        super().__init__()

        self.past_timesteps = past_timesteps
        self.n_neighbours = n_neighbours
        self.latent_dims = latent_dims
        self.nodes_per_layer = nodes_per_layer

        self.encoder = nn.Sequential(
            nn.Linear(self.past_timesteps, 32),
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
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, past_timesteps)
        )

        self.neural_net = nn.Sequential(
            nn.Linear(latent_dims + n_neighbours, self.nodes_per_layer),
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

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'NN+AE+D'

    def forward(self, x):
        x_past = x[:, :self.past_timesteps]
        x_present = x[:, self.past_timesteps:]
        latent_space = self.encoder(x_past)
        x_reconstructed = self.decoder(latent_space)
        x_nn = torch.cat((latent_space, x_present), dim=1)
        y_pred = self.neural_net(x_nn)
        
        return y_pred, x_reconstructed


class NN(nn.Module):
    def __init__(self,past_timesteps, n_neighbours, nodes_per_layer=16):
        super().__init__()
        self.past_timesteps = past_timesteps
        self.n_neighbours = n_neighbours
        self.nodes_per_layer = nodes_per_layer

        self.neural_net = nn.Sequential(
            nn.Linear(self.past_timesteps + self.n_neighbours, self.nodes_per_layer),
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

        # Loss
        self.train_loss = []
        self.test_loss = []

        # R^2
        self.R2 = []

        # Metadata
        self.memory_cutoff = None
        self.tau = None
        self.model_name = 'NN'
        self.latent_dims = 0

    def forward(self, x):
        y_pred = self.neural_net(x)
        return y_pred
    
      
def run_online(ms, taus, models, past_timesteps):
    initX = np.load('initX.npy')[:8]
    initY = np.load('initY.npy')[:8*32]
    np.random.seed(123)
    print(models)
    simulation_time = 1000

    if not isinstance(models, list):
        models = [models]

    for m, tau in zip(ms, taus):
        for model in models:
            if model == 'baseline_nn':
                pt = 0
            else:
                pt = past_timesteps
            
            if 'NN+AE_latent_dims' in model:
                mn = 'NN+AE'
            else:
                mn = model
            
            print(m, tau)
            path = f'networks/{model}/input_lagg={pt}/m={m}_tau={tau}_{mn}.pkl'
            parametrization = torch.load(path, weights_only=False, map_location='cpu')
            parametrization.model_name = model
            L96 = L962LvlMem(X_init=initX, Y_init=initY, save_dt=.001, m=m, tau=tau, memory_activation_func=None, parametrization=parametrization)
            L96.iterate(simulation_time)
            h = L96.history
            h.attrs['m'] = L96.m
            h.attrs['tau'] = L96.tau
            h.attrs['model'] = model
            h.attrs['past_timesteps'] = L96.parametrization.past_timesteps if L96.parametrization is not None else 0
            h.attrs['model_path'] = path
            if hasattr(L96.parametrization, 'latent_dims') and L96.parametrization.latent_dims is None:
                L96.parametrization.latent_dims = 0
            h.attrs['latent_dims'] = L96.parametrization.latent_dims if hasattr(L96.parametrization, 'latent_dims') else 0

            # Save run
            save_dir = Path(f'./online_runs/{model}/input_lagg={pt}')
            save_path = f'{save_dir}/time={simulation_time}MTU_m={m}_tau={tau}.nc'
            if not save_dir.exists(): 
                os.makedirs(save_dir) 
            #h.to_netcdf(f'./online_runs/{model}/input_lagg={pt}/time={simulation_time}MTU_m={m}_tau={tau}.nc', mode='w')
            h.to_netcdf(save_path, mode='w')
            print(h)



def run_NNpAE_online(model_path, sim_time=10_000):
    model = torch.load(model_path, weights_only=False, map_location='cpu')
    initX = np.load('initX.npy')[:8]
    initY = np.load('initY.npy')[:8*32]
    np.random.seed(123)

    # Init L96
    L96 = L962LvlMem(m=.001, tau=.001, parametrization=model, X_init=initX, Y_init=initY)
    L96.iterate(sim_time)
    print(L96.history)
    save_L96(L96)



def load_equation_models(dim):
    path_dict = {0:'outputs/20250704_120045_4bP2CT/checkpoint.pkl', 1: 'outputs/20250704_120106_netRvS/checkpoint.pkl', 2: 'outputs/20250704_120117_92TOuO/checkpoint.pkl'}
    #path = f'equations/latent_variabel={dim}_3dim_3d_training.csv'
    #path = 'outputs/20250703_144254_GtkNcN/checkpoint.pkl'
    path = path_dict[dim]
    with open(path, 'rb') as file:
        model = pickle.load(file)
    model.equations_ = model.get_hof()  # I lost the equations_ df at some point, luckily I can reconstruct them

    return model


if __name__=='__main__':
    #L96 = L962LvlMem(X_init=None, Y_init=None, save_dt=.005, dt=.001, K=8, J=32, h=1, F=20, b=10, c=10)
    #MTUs = 100_000
    #L96.iterate(MTUs)
    #print(L96.time_weights)
    #np.save(f'/work/bd1179/b309297/X_{MTUs}.npy', L96.history.X.values)
    #np.save(f'/work/bb1153/b309297/B_{MTUs}.npy', L96.history.B.values)

    # parser = argparse.ArgumentParser(description='Run L96 sensitivity experiment')
    # parser.add_argument('--model_type', type=str, default='nn', help='Model type to use (e.g., nn, rf, svm)')
    # args = parser.parse_args()
    # if args.model_type == 'nn':
    #     path = 'networks/nn/input_lagg=1000/m=0.001_tau=0.001_nn.pkl'
    # elif args.model_type == 'baseline_nn':
    #     path = 'networks/baseline_nn/input_lagg=0/m=0.001_tau=0.001_baseline_nn.pkl'

    # # path = 'networks/nn/input_lagg=1000/m=0.001_tau=0.001_nn.pkl'
    # # path = 'networks/NN+AE_latent_dims=5/input_lagg=1000/m=0.001_tau=0.001_NN+AE.pkl'
    # path = 'networks/baseline_nn/input_lagg=0/m=0.001_tau=0.001_baseline_nn.pkl'

    # print(path)
    # path = 'networks/NN+AE_latent_dims=3/input_lagg=1000/m=0.001_tau=0.001_NN+AE.pkl'
    # initX = np.load('initX.npy')[:8].astype(np.float32)
    # initY = np.load('initY.npy')[:8*32].astype(np.float32)
    # np.random.seed(123)

    # past_timesteps = int(re.search(r'input_lagg=(\d+)', path).group(1))
    # m = float(re.search(r'm=([\d.]+)', path).group(1))
    # tau = float(re.search(r'tau=([\d.]+)', path).group(1))
    # model = torch.load(path, weights_only=False, map_location='cpu')

    # L96 = L962LvlMemZ(X_init=initX, Y_init=initY, save_dt=.001, m=0.001, tau=0.001, 
    #                   parametrization=model.neural_net, encoder=model.encoder, 
    #                   latent_dims=model.latent_dims, past_timesteps=past_timesteps)
    # L96.iterate(5)
    # print(L96.history)
    # L96.parametrization = None
    # save_L96(L96)


    # Run NN+AE with full information online while saving Z for ODE learning
    #path = 'networks/phi_1_1.0_8_m=0.001_tau=0.001_20250717104927.pkl' # Phi_1,1,8
    #path = 'networks/phi_0_0.0_8_m=None_tau=None_20250717112413.pkl'  # Phi_0,0,8
    #print(path)
    #run_NNpAE_online(path)
    run_online([1.0], [100.0], ['NN'], past_timesteps=0)


    # initX = np.load('initX.npy')[:8]
    # initY = np.load('initY.npy')[:8*32]
    # np.random.seed(123)
    ms = [np.logspace(-3, 0, 10)[0]]
    taus = [np.logspace(-3, 2, 10)[0]]
    # print(ms, taus)
    # for m, tau in zip(ms, taus):
    #     L96 = L962LvlMem(X_init=initX, Y_init=initY, save_dt=.001, m=m, tau=tau, memory_activation_func=None)
    #     L96.iterate(10_000)
    #     save_L96(L96)

    #run_online(ms, taus, models=args.model_type, past_timesteps=1000)
