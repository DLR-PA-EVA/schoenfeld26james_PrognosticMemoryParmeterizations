# Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation

This repository accompanies the paper *[...]*, containing the two experiments:

- [`L96/`](L96) — Lorenz-96 (L96) experiments.
- [`DYAMOND/`](DYAMOND) — DYAMOND experiments.

Trained networks and scripts to reproduce all findings are shared in the respective folders. 

## Lorenz-96
### online runs:
Simulation data from different parameterizations covers several GB of data and is not shared in this repository. Instead data can be reproduced by calling the `run_online` function in `L96/L96.py`. Specify the parameterization that should be used e.g. `L96/networks_paper/M=1000_dz=6_w=1e-06_20260529142124.pkl` to recreate the simulation used for analysis of the best parameterization found in the paper. If model_path is not given as a variable, the full set of L96 equations is simulated creating the reference run. The default behavior automaticall chooses seed and initial conditions used to create the simulation data. <br>
To recreate the weather statistics point to your local copy of the reference simulation to draw initial conditions, all possible initial conditions are expected to be seperated by a time interval of 10 MTU (10.000 time steps). <br>

### training:
All parameterizations trained for this paper are available from the [`networks_paper/`](networks_paper) folder. You can train your own parameterizations in the `parameterizations.py` module, for that you need to recreate the reference simulation data as explained in the previous section.

### evaluation:
Once simulation data was recreated, make sure to adjust the paths in the evaluation scripts. The notebook `plots.ipynb` recreates Figure 3 and `regime_analysis.ipynb` recreates Figure 4. 

## DYAMOND precipitation
### preprocessing:
The DYAMOND data is available from the DKRZ tape archieve. Please refere to the DYAMOND Data Library documentation to get access. Once the high-resolution data is accessible you can use the scripts [`preprocessing_scripts`](preprocessing_scripts) to coarse grain the data. 

### training:
After obtaining the training data you can use `training_gpu.py` to train your own parameterization and to recreate the predicted latent space variables and precipitation data. For joint training of ODE-coefficients and NN-parameterization you can reffer to `train_ODE_and_parameterization.py`. All trained parameterizations for the paper are given in [`hyper_opt_precip`](hyper_opt_precip) and [`initial_pt_hyper_opt`](initial_pt_hyper_opt).

### evaluation:
Figures 2, 6, 7 can be reproduced with the notebook `plots_paper.ipynb` and Figure 5 with `evaluate_Xi_coefficients.ipynb`. For Figures 6, 7 you first need to recreate the precipitation predictions of the parameterizations. 

## Environment
The respective experiment folders contain `requirements.txt` files to generate exact copies of the used environments. 

