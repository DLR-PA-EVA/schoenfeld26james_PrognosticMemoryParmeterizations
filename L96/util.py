import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import xarray as xr


def merge_time(model, timestamp, m, tau):
    files = os.listdir(f'temp/{model}')
    dss = []
    for file in files:
        if (f'm={m}' in file) and (f'tau={tau}' in file): 
            ds = xr.open_dataset(f'temp/{model}/' + file)
            if hasattr(ds, 'Z'):
                ds = ds.drop_vars('Z')
            dss.append(ds)

    # Concatenate along time lazily
    merged = xr.concat(dss, dim="time")

    # Write to disk (actual computation happens here)
    merged.to_netcdf(f"online_runs/{model}/m={m}_tau={tau}_t=50000MTU_{timestamp}.nc")


def return_lagged_input_vector(X, m):
    """
    Optimized version of the function to compute lagged input vectors.
    Expects X to be in shape (K, NT).
    """
    K, NT = X.shape
    if NT <= m:
        raise ValueError("Number of time steps (NT) must be greater than the lag (m).")

    # Preallocate the output array with the correct shape
    X_past_reordered = np.zeros((K * (NT - m), m + 1), dtype=np.float32)

    # Fill the output array directly using slicing
    for lag in range(m + 1):
        X_past_reordered[:, lag] = X[:, m - lag:NT - lag].T.ravel()

    return X_past_reordered[:,::-1]  # return with native time direction


def return_lagged_input_vector_opt(X, m):
    """
    Constructs lagged input vectors with row order grouped by feature.
    
    Parameters:
    - X: np.ndarray of shape (K, NT)
    - m: int, number of lags (excluding current time step)
    
    Returns:
    - A 2D array of shape (K * (NT - m), m + K)
      Each row: [X_k[t-m], ..., X_k[t-1], X_1[t], ..., X_K[t]]
      Row order: All time steps for k=0, then for k=1, ..., up to k=K-1
    """
    K, NT = X.shape
    if NT <= m:
        raise ValueError("Number of time steps (NT) must be greater than lag m.")

    num_time_steps = NT - m
    out = np.zeros((K * num_time_steps, m + 1), dtype=np.float32)

    #for k in range(K):
    for i, t in enumerate(range(m, NT)):
        #row_idx = k * num_time_steps + i
        row_idx = i * K
        out[row_idx: row_idx + K, :m + 1] = X[:, t - m:t + 1]           # past m values of variable k

    return out


def return_lagged_input_vector_and_present_k(X, m):
    """
    Constructs lagged input vectors with row order grouped by feature.
    
    Parameters:
    - X: np.ndarray of shape (K, NT)
    - m: int, number of lags (excluding current time step)
    
    Returns:
    - A 2D array of shape (K * (NT - m), m + K)
      Each row: [X_k[t-m], ..., X_k[t-1], X_{k+1}%K[t], ..., X_{k+K}%K[t]]
      Row order: chronological such that the first 80% of rows correspond to the first 80% of time points
    """
    K, NT = X.shape
    if NT <= m:
        raise ValueError("Number of time steps (NT) must be greater than lag m.")

    num_time_steps = NT - m
    out = np.zeros((K * num_time_steps, m + K), dtype=np.float32)
    ks = np.array([list(range(i, i + K)) for i in range(K)]) % K
    ks = ks[:, 1:]
    #ks = ks.flatten()
    #for k in range(K):
    #ks = np.arange(k+1, k + K) % K # for present values select k + 1, k + 2, ... Divide by K to start from the beginning of k-valriables
    
    for i, t in enumerate(range(m, NT)):
        row_idx = K * i
        out[row_idx: row_idx + K, :m + 1] = X[:, t - m:t + 1]           # past m values of variable k
        out[row_idx: row_idx + K, m + 1:] = X[ks, t]                 # current values of all K variables at time t

    return out


def transpose_if_1d(x):
    # Transpose 1-dimensional data
    if len(x.shape) == 1:
        xT = torch.unsqueeze(x, 1)
    else:
        xT = x
    
    return xT


def plot_predictions_vs_data(X_test, B_test, model, L96):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test, B_test = X_test[::L96.K], B_test[::L96.K]
    Npoints = len(X_test)
    Nstart = 0
    X_plot, B_plot = X_test[Nstart:Nstart+Npoints], B_test[Nstart:Nstart+Npoints]
    X_plot = transpose_if_1d(X_plot).to(device)
    model.eval()
    with torch.no_grad():
        B_plot_pred = model.forward(X_plot).flatten().cpu()


    plt.plot(np.arange(Npoints), B_plot.cpu(), label='data')
    plt.plot(np.arange(Npoints), B_plot_pred.cpu(), label='model')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$B$')
    plt.title(f'm={L96.memory_cutoff}, tau={L96.memory_tau}')
    plt.legend()


def linear_regression_fit(L96, fig=None, ax=None):
    h = L96.history
    slope, intercept = np.polyfit(np.ravel(h.X), np.ravel(h.B), 1)
    # sns.set_context('talk')
    
    if fig:
        pass
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    try:
        ax.set_title(f'memory cutoff={L96.memory_cutoff}, tau={L96.memory_tau}')
    except AttributeError:
        ax.set_title(f'memory cutoff={L96.m}, tau={L96.tau}')
    ax.scatter(np.ravel(h.X)[::100], np.ravel(h.B)[::100], alpha=0.1, label='True data points')
    plt.xlabel('X (resolved variable)'); plt.ylabel('B (sub-grid term)')
    a = [-6, 12]
    ax.plot(a, np.polyval([slope, intercept], a), c='orange', lw=4, label=f'Linear regression, {slope:.2f}X + {intercept:.2f}')
    ax.legend()

    return fig, ax


def collect_model_scores(netw_dir, inds=None):
        models, taus, ms, R2s = [], [], [], []
        files = os.listdir(netw_dir)
        files.sort()
        if inds:
            files = [files[i] for i in inds]

        for file in files:
            if '.pkl' in file:
                model = torch.load(netw_dir + file, weights_only=False, map_location=device)
                tau, m, R2 = model.tau, model.memory_cutoff, float(max(model.R2))
                if not tau:
                    tau = float(file.split('tau=')[1].split('.pkl')[0])
                if not m:
                    m = float(file.split('m=')[1].split('_tau=')[0])
                taus.append(tau)
                ms.append(m)
                R2s.append(R2)
                models.append(model)
        
        return models, np.array(taus), np.array(ms), np.array(R2s)


def memory_sensitivity(model_type, past_timesteps, minus=None, vmin=0, vmax=.9, cmap='viridis', fig=None, ax=None):
    netw_dir = f'./networks/{model_type}/input_lagg={past_timesteps}/'
    _, taus, ms, R2s = collect_model_scores(netw_dir)
    print(R2s)
    title = f'{model_type} network performance, past timesteps={past_timesteps}'


    if minus:
        netw_dir_baseline = f'./networks/{minus[0]}/input_lagg={minus[1]}/'
        _, _, _, R2s_minus = collect_model_scores(netw_dir_baseline)

        R2s = R2s - R2s_minus
        title = f'{model_type}  - {minus[0]}'

    if not fig:
        fig, ax = plt.subplots()

    scatter = ax.scatter(taus, ms, c=R2s, cmap=cmap, vmin=vmin, vmax=vmax, s=600, marker='s', edgecolor='k')
    cbar = fig.colorbar(scatter)
    cbar.set_label(r'$R^2$ Value')
    ax.set(xscale='log', yscale='log', xlabel=r'$\tau$', ylabel=r'$m$')
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()


def training_history_overview(past_timesteps, n_samples=1):
    model_inds = np.random.randint(0, 100, size=n_samples)
    fig, axs = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for model_type, c in zip(['nn', 'nn+autoencoder'], colors):
        netw_dir = f'./networks/{model_type}/input_lagg={past_timesteps}/'
        models, taus, ms, R2s = collect_model_scores(netw_dir, model_inds)
    
        for model in models:
            epochs = np.arange(1, len(model.train_loss) + 1)
            axs.plot(epochs, model.train_loss, c=c, linestyle='--', alpha=.5, label=model_type)
            axs.plot(epochs, model.test_loss, c=c, linestyle=':', alpha=.5)
    
    axs.set(xlabel='epoch', ylabel='loss')
    axs.set_title(f'{model_type}, tau={taus}, m={ms}')
    axs.grid()
    axs.legend()


if __name__=='__main__':
    model_name = 'NO_PARAMETRIZATION'  # Set model_name according to what shows up in your temp folder
    timestamp = '20260911190000'  # Set a timestamp or something to uniquely identify your model
    merge_time(model_name, timestamp, m=0.001, tau=0.001)
