# Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation
### Summary
Earth system models evolve the Earth's climate through a set of governing equations on a coarse resolution grid to maintain computational feasibility. Physical processes below the computed resolution need to be approximated with parameterizations. Atmospheric patterns, like organized moisture patches, leave a fingerprint on the time series of grid-scale state variables inducing a memory of the past. We improve parameterizations by leveraging this information with a Machine Learning (ML) model creating a new set of variables that expand the governing equations of the dynamical core. The ML model can be replaced by approximating its behavior with a set of evolution equations that can be interpreted and simulated efficiently unlike the original ML model. We test our approach with an atmospheric toy model and a realistic parameterization of precipitation. We find that the new equations recover most of the predictive skill added by ML unlike parameterizations without memory. 

This repository accompanies the paper *[unpublished]*, containing the two experiments:

- [`L96/`](L96) — Lorenz-96 (L96) experiments.
- [`DYAMOND/`](DYAMOND) — DYAMOND experiments.

Trained networks and scripts to reproduce all findings are shared in the respective folders. 

Contact: Jurij Schönfeld, jurij.schoenfeld@dlr.de

## Lorenz-96
### online runs:
Simulation data from different parameterizations covers several GB of data and is not shared in this repository. Instead data can be reproduced by calling the `run_online` function in `L96/L96.py`. Specify the parameterization that should be used to recreate the simulations. If model_path is not given as a variable, the full set of L96 equations is simulated creating the reference run. The default behavior automaticall chooses seed and initial conditions used to create the simulation data. <br>
To recreate the weather statistics use `make_weather_runs`. The function automatically points to the published initial conditions and sets the correct seeds. 

### training:
All parameterizations trained for this paper are available from the [`networks_paper/`](L96/networks_paper) folder. You can train your own parameterizations in the `parameterizations.py` module, for that you need to recreate the reference simulation data as explained in the previous section.

### evaluation:
Once simulation data was recreated, make sure to adjust the paths in the evaluation scripts. The notebook `plots.ipynb` recreates Figure 3 and `regime_analysis.ipynb` recreates Figure 4. 

## DYAMOND precipitation
### preprocessing:
The DYAMOND data is available from the DKRZ tape archieve. Please refere to the DYAMOND Data Library documentation to get access https://easy.gems.dkrz.de/DYAMOND/dyamond-library/index.html#dyamond-library. Once the high-resolution data is accessible you can use the scripts [`preprocessing_scripts`](DYAMOND/preprocessing_scripts) to coarse grain the data. 

### training:
After obtaining the training data you can use `training_gpu.py` to train your own parameterization and to recreate the predicted latent space variables. To generate parameterization predictions of precipitaion use the `predict_precip.ipynb` notebook. For joint training of ODE-coefficients and NN-parameterization you can refer to `train_ODE_and_parameterization.py`. All trained parameterizations for the paper are given in [`hyper_opt_precip`](DYAMOND/hyper_opt_precip) and [`initial_pt_hyper_opt`](DYAMOND/initial_pt_hyper_opt).

### evaluation:
Figures 2, 6, 7 can be reproduced with the notebook `plots_paper.ipynb` and Figure 5 with `evaluate_Xi_coefficients.ipynb`. For Figures 6, 7 you first need to recreate the precipitation predictions of the parameterizations. 

## Environment
The respective experiment folders contain `requirements.txt` files to generate exact copies of the used environments. 

For the L96 environment do
```
conda create -n L96_env python=3.11.12
conda activate L96_env
pip install -r L96/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```
and for the DYAMOND environment do
```
conda create -n DYAMOND_env python=3.13.7
conda activate DYAMOND_env
pip install -r DYAMOND/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Both `requirements.txt` files were generated from the exact environments used to produce the paper's results and are therefore more extensive than what the code actually needs. If you run into installation issues or want to run a minimal setup, installing just the core libraries by hand will likely give you a working environment: `numpy`, `scipy`, `pandas`, `xarray`, `torch`, `torchmetrics`, `matplotlib`, `pysindy` and `feyn`.

