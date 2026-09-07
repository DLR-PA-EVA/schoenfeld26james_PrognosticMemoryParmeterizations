import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
from pathlib import Path
import os
import pickle
import torch
import argparse
torch.set_num_threads(1)   # limit intra-op threads
torch.set_num_interop_threads(1)  # limit inter-op parallelism
import torch.nn as nn
from parametrizations import NN, NNpAE, NNpAEpD, ODE_Z, get_circular_neighbours
from datetime import datetime

# L96 implementation adapted from Stephan Rasp
# https://github.com/raspstephan/Lorenz-Online
# with permission from the author.
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
        # Initialize state variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)
        self.t0 = 0
        
        # Inint parametrization
        self.parametrization = parametrization
        if self.parametrization:
            self.param_memory_X = torch.tensor(np.zeros(shape=(self.K, self.parametrization.past_timesteps + 1)), dtype=torch.float32)  # pre
            self.param_memory_X[:, 0] = torch.as_tensor(self.X)
            try:
                self.parametrization.eval()
            except AttributeError:
                print(f'{self.parametrization.model_name} is not directly a pytorch class \nAssume that you passed on ODE parametrization')
                self.parametrization.AE.eval()
                self.parametrization.NN.eval()
            if hasattr(self.parametrization, 'latent_dims'):
                self.Z = np.zeros((self.K, self.parametrization.latent_dims))  # Z0 = 0
                self.Z_ODE = np.zeros((self.K, self.parametrization.latent_dims))  # Z_ODE0 = 0
            if self.parametrization.n_neighbours > 0:
                self.circular_neighbours = get_circular_neighbours(self.parametrization.n_neighbours, self.K)

        # Memory parameters
        self.tau = tau  # Decay rate of the kernel
        self.m = m  # Maximum memory time
        self.memory_steps = int(self.m / dt)  # Number of stored steps
        self.memory_X = [self.X]  # Store past X values
        # self.param_memory_X = [self.X]  # Store past X values for the parametrization
        self.times = np.array(range(self.memory_steps - 1, -1, -1)) * self.dt # Delta ts for the memory kernel, from memory_cutoff to zero
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
        coords={'time': np.arange(self.t0, self.t0 + len(self._history_X)) * self.dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        
        dic = {}
        dic['X'] = xr.DataArray(np.array(self._history_X), dims=['time', 'x'], name='X')  # 2D array for X
        dic['B'] = xr.DataArray(np.array(self._history_B), dims=['time', 'x'], name='B')
        if self._history_Y:
            dic['Y'] = xr.DataArray(np.array(self._history_Y), dims=['time', 'y'], name='Y')
        if self.parametrization:
            if self._history_Z:
                coords['latent_dims'] = np.arange(self.parametrization.latent_dims)
                dic['Z'] = xr.DataArray(np.array(self._history_Z), dims=['time', 'x', 'latent_dims'])
            if self._history_Z_ODE:
                # if not coords['latent_dims']:  #check if latent_dim coords where already initilized and init if not
                #     coords['latent_dims'] = np.arange(self.parametrization.latent_dims)
                dic['Z_ODE'] = xr.DataArray(np.array(self._history_Z_ODE), dims=['time', 'x', 'latent_dims'])

        return xr.Dataset(
            dic,
            coords=coords
        )
    
    def clear_history(self):
        self._history_X = []
        self._history_B = []
        self._history_Y = []
        self._history_Z = []
        self._history_Z_ODE = []

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

        # Save X and Y to history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self.save_step(B)

        # Store X for parametrization
        if (self.parametrization is not None) and (self.parametrization.past_timesteps > 0): 
            # if len(self.param_memory_X) >= self.parametrization.past_timesteps + 1:
            #     self.param_memory_X.pop(0)  # Remove oldest entry
            # self.param_memory_X.append(self.X.copy())
            self.param_memory_X[:, self.step_count % (self.parametrization.past_timesteps + 1)]  = torch.as_tensor(self.X)  # % is necessary if ODE_step calls normal step allongside parametrization
        
        return B

    def NNpAE_step(self):
        if self.step_count <= 2000:
            self.step()
            return
        x_past = self.param_memory_X[:, :-1]
        x_present = torch.unsqueeze(self.param_memory_X[:, -1], 1)
        self.Z = self.parametrization.encoder(x_past)
        x_nn = torch.cat((self.Z, x_present), dim=1)
        B = self.parametrization.neural_net(x_nn).numpy().flatten()
        return B

    def NN_step(self):
        if self.step_count <= 2000:
            self.step()
            return
        x_present_shifted = torch.from_numpy(self.X[self.circular_neighbours])
        B = self.parametrization.forward(x_present_shifted).numpy().flatten()
        return B
    
    def NNpast_step(self):
        if self.step_count <= 2000:
            self.step()
            return
        B = self.parametrization.forward(self.param_memory_X).numpy().flatten()
        return B

    def baseline_step(self):
        if self.step_count <= 2000:
            self.step()
            return
        B = self.parametrization.forward(self.param_memory_X)
        B = B.numpy().flatten()
        return B

    def ODE_Z_step(self):
        x_past = self.param_memory_X[:, :-1]
        self.Z = self.parametrization.AE.encoder(x_past)

        if self.step_count <= 2000:  # Run another MTU to spinup AE
            self.Z_ODE = self.Z
        else:
            # Compute Z_ODE allongside Z from the AE, but dont use it for the parametrization
            self.Z_ODE = self.parametrization.forward(self.Z_ODE, torch.from_numpy(self.X))  # Predict Z derivatives and integrate using RK4
        
        self.step()  # do unparametrized step
        return  # return, otherwise two RK steps will be made for X

    def ODE_Z_online_step(self):
        if self.step_count <= 2000:
            self.ODE_Z_step()  # wait for Z spin up
        else:
            x_present = torch.from_numpy(self.X)
            self.Z = self.parametrization.forward(self.Z, x_present)  # Predict Z derivatives and integrate using RK4
            x_present = x_present.reshape(-1, 1)
            x_nn = torch.cat((self.Z, x_present), dim=1)
            B = self.parametrization.NN.forward(x_nn).numpy().flatten()
            return B

    def save_XB(self, B):
        self._history_X.append(self.X.copy())
        self._history_B.append(B.copy())
    
    def save_XBY(self, B):
        self._history_X.append(self.X.copy())
        self._history_B.append(B.copy())
        self._history_Y.append(self.Y.copy())

    def save_XBZ(self, B):
        self._history_X.append(self.X.copy())
        self._history_B.append(B.copy())
        self._history_Z.append(self.Z)

    def save_XBZODEZ(self, B):
        self._history_X.append(self.X.copy())
        self._history_B.append(B.copy())
        self._history_Z.append(self.Z)
        self._history_Z_ODE.append(self.Z_ODE)

    def step_parametrized(self):
        # Check if enough data is available to run the parametrization
        if self.step_count <= 999:  # Do spinup with regular L96 for 1 MTU
            # Run step with fast variables
            self.step()
            return

        """Integrate one time step with parametrized fast varibales."""
        # Compute backreaction term for X
        with torch.no_grad():
            B = self.forward_step()

        if B is None:  # forward_step made unparametrized step
            return

        # RK4 with parametrization
        k1_X = self._rhs_X_dt(self.X, B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B)
        
        self.X += (1 / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        # Save history after a set number of steps
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            self.save_step(B)

        # Store X for parametrization
        self.param_memory_X = torch.cat([self.param_memory_X[:, 1:], torch.as_tensor(self.X).unsqueeze(1)], dim=1)
                
    def iterate(self, time, save_interval=10_000_100):
        self.save_interval = save_interval
        if self.parametrization is not None:
            step_func = self.step_parametrized
            if self.parametrization.model_name == 'ODE_Z_online':
                self.forward_step = self.ODE_Z_online_step
                # self.save_step = self.save_XBZ
                self.save_step = self.save_XB
            elif self.parametrization.model_name == 'ODE_Z':
                self.forward_step = self.ODE_Z_step
                self.save_step = self.save_XBZODEZ
                # self.save_step = self.save_XB
            elif (self.parametrization.model_name == 'NN') and (self.parametrization.n_neighbours > 0):
                self.forward_step = self.NN_step
                self.save_step = self.save_XB
            elif (self.parametrization.model_name == 'NN') and (self.parametrization.n_neighbours == 0):
                self.forward_step = self.baseline_step
                self.save_step = self.save_XB
            elif self.parametrization.model_name == 'NNpast':
                self.forward_step = self.NNpast_step
                self.save_step = self.save_XB
            elif self.parametrization.model_name == 'NN+AE+D':
                self.forward_step = self.NNpAE_step
                self.save_step = self.save_XBZ
            elif self.parametrization.model_name == 'NN+AE':
                self.forward_step = self.NNpAE_step
                self.save_step = self.save_XBZ
        else:
            step_func = self.step
            self.save_step = self.save_XB

        steps = int(time / self.dt)
        for _ in tqdm(range(steps)):
            step_func()

            if self.step_count % save_interval == 0:
                if self.parametrization:
                    ld, pt, Nk = self.parametrization.latent_dims, self.parametrization.past_timesteps, self.parametrization.n_neighbours
                    model_name = f'{self.parametrization.model_name}/latent_dims={ld}_past_timesteps={pt}_n_neighbours={Nk}'
                else:
                    model_name = 'NO_PARAMETRIZATION'

                self.t0 = self.step_count - self.save_interval
                h = self.history
                save_dir = Path(f'./temp/{model_name}')
                if not save_dir.exists(): 
                    os.makedirs(save_dir) 
                h.to_netcdf(save_dir / f'time={int(self.step_count * self.dt - save_interval * self.dt)}_{int(self.step_count * self.dt)}MTU_m={self.m}_tau={self.tau}.nc', mode='w')
                self.clear_history()


def run_online(model_path, simulation_time, m=None, tau=None, additional_info=None, 
               weather=False, seed=123, initX=None, initY=None):
    '''
    Function for running parametrizations online with L96
    model_path: path to trained pytorch model
    simulation_time: duration, in MTU, of the simulation
    m, tau: memory kernel parameters. If not passed directly they are inferred from the model 
    '''
    if simulation_time > 10_000:
        save_interval = 1000_000
    else:
        save_interval = 1.e9  # Set to some large value as we wont save in between anyway
    # Initial conditions for online run
    if seed == 123:
        initX = np.load('initX.npy')[:8]
        initY = np.load('initY.npy')[:8*32]

    np.random.seed(seed)

    # Load parametrization from model_path
    if weather:
        x_runs = 'weather_runs'
    else:
        x_runs = 'online_runs'

    if not model_path:  # Run with no parametrization, m and tau must be provided
        parametrization  = None
        save_dir = Path(f'{x_runs}/NO_PARAMETRIZATION/')
    else:
        print(model_path)
        parametrization = torch.load(model_path, map_location='cpu', weights_only=False)

        m, tau = parametrization.m, parametrization.tau
        save_dir = Path(f'{x_runs}/{parametrization.model_name}/latent_dims={parametrization.latent_dims}_past_timesteps={parametrization.past_timesteps}_n_neighbours={parametrization.n_neighbours}/')
    if not save_dir.exists(): 
            os.makedirs(save_dir) 
    
    # Perform simulation
    L96 = L962LvlMem(X_init=initX, Y_init=initY, save_dt=0.001, m=m, tau=tau, parametrization=parametrization)
    L96.iterate(simulation_time, save_interval)

    # Save simulation
    if save_interval >= 30_000_000:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if additional_info:
            save_file = f'm={m}_tau={tau}_t={simulation_time}MTU_{additional_info}_{timestamp}.nc'
        else:
            save_file = f'm={m}_tau={tau}_t={simulation_time}MTU_{timestamp}.nc'
        h = L96.history
        h.to_netcdf(save_dir / save_file, mode='w')
        return str(save_dir / save_file)


def make_weather_runs(nruns, model_path, m, tau):
    #path = glob.glob(f'weather_runs/m={m}_tau={tau}_t=50000MTU_savedt=10MTU_*')[0]
    #path = 'online_runs/NO_PARAMETRIZATION/m=0.001_tau=0.001_t=50000MTU_20250918101458.nc'
    path = 'weather_runs/m=0.001_tau=0.001_t=50000MTU_savedt=10MTU_20250918020858.nc'
    L96 = xr.open_dataset(path)
    X, Y = L96.X.values, L96.Y.values
    tmax = X.shape[0]
    # seeds = np.random.sample(np.arange(tmax), size=nruns, replace=False)  # Sample nruns random time points from the weather run to use as initial conditions for the online runs
    seeds =np.arange(nruns)
    # seeds = np.arange(nruns * time_between_init_conds, step=time_between_init_conds)
    print('tmax: ', tmax)
    for seed in seeds:
        print(seed)
        initX = X[seed]
        initY = Y[seed]
        run_online(model_path, simulation_time=7, m=m, tau=tau, 
                   additional_info=f'seed={seed}', seed=seed, weather=True,
                   initX=initX, initY=initY)


if __name__=='__main__':
    #L96 = L962LvlMem(X_init=None, Y_init=None, save_dt=.005, dt=.001, K=8, J=32, h=1, F=20, b=10, c=10)
    #MTUs = 100_000
    #L96.iterate(MTUs)
    #print(L96.time_weights)
    #np.save(f'/work/bd1179/b309297/X_{MTUs}.npy', L96.history.X.values)
    #np.save(f'/work/bb1153/b309297/B_{MTUs}.npy', L96.history.B.values)

    parser = argparse.ArgumentParser(description='Run L96 sensitivity experiment')
    parser.add_argument('--latent_dims', type=int, default='8')
    args = parser.parse_args()

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
    #run_online([0.001], [0.001], ['NN+AE+D'], past_timesteps=1000, simulation_time=10_000)
    # run_online([0.001], [0.001], ['ODE_Z'], past_timesteps=1000, simulation_time=10_000)


    # initX = np.load('initX.npy')[:8]
    # initY = np.load('initY.npy')[:8*32]
    # np.random.seed(123)
    # ms = [np.logspace(-3, 0, 10)[-1]]
    # taus = [np.logspace(-3, 2, 10)[-1]]
    # for m, tau in zip(ms, taus):
    #     print(m, tau)
    #     run_online(None, 10_000, m, tau)
    # print(ms, taus)
    # for m, tau in zip(ms, taus):
    #     L96 = L962LvlMem(X_init=initX, Y_init=initY, save_dt=.001, m=m, tau=tau, memory_activation_func=None)
    #     L96.iterate(10_000)
    #     save_L96(L96)

    #run_online(ms, taus, models=args.model_type, past_timesteps=1000)

    # paths_weak_kernel = ['networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905173346.pkl', 
    #                      'networks/NN+AE+D/latent_dims=5_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905163313.pkl',
    #                      'networks/NN+AE+D/latent_dims=3_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905153331.pkl',
    #                      'networks/NN+AE/latent_dims=3_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905150348.pkl',
    #                      'networks/NN+AE/latent_dims=5_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905152851.pkl',
    #                      'networks/NN+AE/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905155405.pkl',
    #                      'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905153939.pkl',
    #                      'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=7/m=0.001_tau=0.001_w=0.001_20250905161835.pkl',
    #                      'networks/NNpast/latent_dims=0_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_w=0.001_20250905152318.pkl'
    #                      ]
    # paths_strong_kernel = ['networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905173228.pkl',
    #                        'networks/NN+AE+D/latent_dims=5_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905163102.pkl',
    #                        'networks/NN+AE+D/latent_dims=3_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905153037.pkl',
    #                        'networks/NN+AE/latent_dims=3_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905150650.pkl',
    #                        'networks/NN+AE/latent_dims=5_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905153144.pkl',
    #                        'networks/NN+AE/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905155709.pkl',
    #                        'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905170515.pkl',
    #                        'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=7/m=1.0_tau=100.0_w=0.001_20250905174304.pkl',
    #                        'networks/NNpast/latent_dims=0_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905164945.pkl'
    #                        ]

    # for path in paths_weak_kernel[1:]:
    #     run_online(path, 10_000)
    
    m, tau = 0.001, 0.001
    paths8 = ['m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105808.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.001_20260528103552.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104318.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.0_20260528110511.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105035.pkl']
    
    paths6 = ['m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110521.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104329.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.001_20260528103600.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105037.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl']
    
    paths4 = ['m=0.001_tau=0.001_hyper_opt_w=0.0_20260528110523.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.001_20260528103557.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104326.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105032.pkl']

    paths2 = ['m=0.001_tau=0.001_hyper_opt_w=0.0001_20260528104332.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.0_20260528110528.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=0.001_20260528103559.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105041.pkl',
                'm=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105817.pkl']
    
    paths = {2: paths2, 4: paths4, 6: paths6, 8: paths8}

    # m, tau = 1.0, 100.0
    # path1 = 'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905170515.pkl'
    # path2 = 'networks/ODE_Z_online/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_20250912143641.pkl'
    # path3 = 'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=7/m=1.0_tau=100.0_w=0.001_20250905174304.pkl'
    # path4 = 'networks/NN+AE+D/latent_dims=8_past_timesteps=1000_n_neighbours=0/m=1.0_tau=100.0_w=0.001_20250905173228.pkl'
    # run_online(path1, 50_000,m, tau, seed=123)

    # for path in [path1, path2]:
    #     make_weather_runs(500, path, m, tau)
    print('latent dims: ', args.latent_dims)
    #for model_path in paths[args.latent_dims]:
    #model_path = 'm=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl'
    #path = 'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-06_20260528105809.pkl'
    #path = 'networks/NN+AE+D/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_hyper_opt_w=1e-05_20260528105037.pkl'
    #path = 'networks/ODE_Z_online/latent_dims=6_past_timesteps=1000_n_neighbours=0/m=0.001_tau=0.001_20260601162804.pkl'
    #path = f'networks/NN+AE+D/latent_dims={args.latent_dims}_past_timesteps=1000_n_neighbours=0/' + model_path
    path = 'networks/NN/latent_dims=0_past_timesteps=0_n_neighbours=7/m=0.001_tau=0.001_w=0.0_20250905155154.pkl'
    model = torch.load(path, map_location='cpu', weights_only=False)  
    run_online(path, simulation_time=50_000, m=.001, tau=.001, additional_info=f'hyper_opt_w={model.weight_decay}')
    # make_weather_runs(500, path, m, tau)
