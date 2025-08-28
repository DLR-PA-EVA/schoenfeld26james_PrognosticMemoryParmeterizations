import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
from pathlib import Path
import os
import pickle
import torch
import torch.nn as nn
from parametrizations import NN, NNpAE, NNpAEpD, NNpODE, FCNN, ODE_Z

# Set device to gpu if avaible, else to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if self.parametrization:
            try:
                self.parametrization.eval()
            except AttributeError:
                print(f'{self.parametrization.model_name} is not directly a pytorch class \nAssume that you passed on ODE parametrization')
                self.parametrization.AE.eval()
                self.parametrization.NN.eval()
            
        # Initialize state variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)
        if hasattr(self.parametrization, 'latent_dims'):
            self.Z = np.zeros((self.K, self.parametrization.latent_dims))  # Z0 = 0
            self.Z_ODE = np.zeros((self.K, self.parametrization.latent_dims))  # Z_ODE0 = 0

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
        self._history_B = []
        self._history_Y = []
        self._history_Z = []
        self._history_Z_ODE = []
        
        # Save interval
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.step_count = 0

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

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
            if self.parametrization.model_name == 'ODE_Z':
                dic['Z_ODE'] = xr.DataArray(np.array(self._history_Z_ODE), dims=['time', 'x', 'latent_dims'])
                dic['Z'] = xr.DataArray(np.array(self._history_Z), dims=['time', 'x', 'latent_dims'])
                coords['latent_dims'] = np.arange(self.parametrization.latent_dims)
            if self.parametrization.model_name == 'ODE_Z_online':
                dic['Z'] = xr.DataArray(np.array(self._history_Z), dims=['time', 'x', 'latent_dims'])
                coords['latent_dims'] = np.arange(self.parametrization.latent_dims)

        return xr.Dataset(
            dic,
            coords=coords
        )

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
            self._history_X.append(self.X.copy())
            self._history_B.append(B.copy())
            if 'NN+AE' in self.parametrization.model_name:
                self._history_Z.append(self.Z)
            if 'ODE_Z' in self.parametrization.model_name:
                self._history_Z_ODE.append(self.Z_ODE)
                self._history_Z.append(self.Z)
        
        return B
           
    def step_parametrized(self):
        # Check if enough data is available to run the parametrization
        if self.step_count <= 1000:  # Do spinup with regular L96 for 1 MTU
            # Run step with fast variables
            self.step()
            return

        """Integrate one time step with parametrized fast varibales."""
        # Compute backreaction term for X
        with torch.no_grad():
            if self.parametrization.model_name == 'NN+AE':
                # Parametrization for NN+AE
                x_past = torch.tensor(np.array(self.param_memory_X).astype(np.float32).T[:,:-1])
                x_present_shifted = torch.tensor(np.array([np.roll(self.X, -i) for i in range(self.K)]), dtype=torch.float32)
                self.Z = self.parametrization.encoder(x_past)
                x = torch.cat((self.Z, x_present_shifted), dim=1)
                B = self.parametrization.neural_net(x).numpy().flatten()
            
            elif self.parametrization.model_name == 'NN':
                x_present_shifted = torch.tensor(np.array([np.roll(self.X, -i) for i in range(self.K)]), dtype=torch.float32)
                B = self.parametrization.forward(x_present_shifted).numpy().flatten()

            elif 'ODE_Z' in self.parametrization.model_name:
                if self.step_count <= 2000:  # Run another MTU to spinup AE
                    x = np.array(self.param_memory_X).astype(np.float32).T
                    x = torch.from_numpy(x)
                    x_past = x[:, :-1]
                    x_present = torch.unsqueeze(x[:, -1], 1)
                    self.Z = self.parametrization.AE.encoder(x_past)
                    self.Z_ODE = self.Z#.numpy().copy()
                    self.step()  # do unparametrized step
                    return  # return, otherwise two RK steps will be made for X
                else:
                    if 'online' in self.parametrization.model_name:
                        x_present = torch.from_numpy(self.X)
                        self.Z = self.parametrization.forward(self.Z, x_present)  # Predict Z derivatives and integrate using RK4
                        # self.Z = torch.from_numpy(self.Z.astype(np.float32))
                        x_present = x_present.reshape(-1, 1)
                        x_nn = torch.cat((self.Z, x_present), dim=1)
                        # with torch.no_grad():
                        B = self.parametrization.NN.forward(x_nn).numpy().flatten()
                    else:
                        # Compute Z_ODE allongside Z from the AE, but dont use it for the parametrization
                        self.Z_ODE = self.parametrization.forward(self.Z_ODE, torch.from_numpy(self.X))  # Predict Z derivatives and integrate using RK4
                        x = np.array(self.param_memory_X).astype(np.float32).T
                        x = torch.from_numpy(x)
                        x_past = x[:, :-1]
                        x_present = torch.unsqueeze(x[:, -1], 1)
                        self.Z = self.parametrization.AE.encoder(x_past)
                        self.step()  # do unparametrized step and return
                        return  # return, otherwise two RK steps will be made for X

            else:
                x = np.array(self.param_memory_X).astype(np.float32).T
                x = torch.from_numpy(x)
                if hasattr(self.parametrization, 'encoder'):
                    x_past = x[:, :-1]
                    x_present = torch.unsqueeze(x[:, -1], 1)
                    self.Z = self.parametrization.encoder(x_past)
                    x_nn = torch.cat((self.Z, x_present), dim=1)
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
            if ('NN+AE' in self.parametrization.model_name) or ('ODE_Z' in self.parametrization.model_name):
                self._history_Z.append(self.Z.numpy().copy())
            if (self.parametrization.model_name == 'ODE_Z'):
                self._history_Z_ODE.append(self.Z_ODE.copy())
                
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


def run_online(ms, taus, models, past_timesteps, simulation_time=1000):
    initX = np.load('initX.npy')[:8]
    initY = np.load('initY.npy')[:8*32]
    np.random.seed(123)
    # initX = None
    # initY = None
    print(models)

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
            path = f'networks/{model}/input_lagg={pt}/m={m}_tau={tau}_{mn}_faster.pkl'
            parametrization = torch.load(path, weights_only=False, map_location='cpu')
            print(path)
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
            print(save_path)
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


def L96_pickle_to_nc(path):
    with open(path, 'rb') as file:
        L96 = pickle.load(file)
    
    h = L96.history
    m, tau = L96.m, L96.tau
    h.attrs['m'] = L96.m
    h.attrs['tau'] = L96.tau
    if L96.parametrization:
        model = L96.parametrization.model_name
        h.attrs['model'] = model
    else:
        model = 'NO_PARAMETRIZATION'
        h.attrs['model'] = model

    pt = L96.parametrization.past_timesteps if L96.parametrization is not None else 0
    h.attrs['past_timesteps'] = pt
    h.attrs['model_path'] = path
    if hasattr(L96.parametrization, 'latent_dims') and L96.parametrization.latent_dims is None:
        L96.parametrization.latent_dims = 0
    h.attrs['latent_dims'] = L96.parametrization.latent_dims if hasattr(L96.parametrization, 'latent_dims') else 0
    simulation_time = int(h.X.values.shape[0] * .001)

    # Save run
    save_dir = Path(f'./online_runs/{model}/input_lagg={pt}')
    save_path = f'{save_dir}/time={simulation_time}MTU_m={m}_tau={tau}.nc'
    if not save_dir.exists(): 
        os.makedirs(save_dir) 
    #h.to_netcdf(f'./online_runs/{model}/input_lagg={pt}/time={simulation_time}MTU_m={m}_tau={tau}.nc', mode='w')
    h.to_netcdf(save_path, mode='w')


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
    # run_online([0.001], [0.001], ['NN+AE+D'], past_timesteps=1000, simulation_time=10_000)
    run_online([0.001], [0.001], ['ODE_Z_online'], past_timesteps=1000, simulation_time=10_000)


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
