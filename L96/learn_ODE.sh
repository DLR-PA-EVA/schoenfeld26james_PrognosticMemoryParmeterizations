#!/bin/bash

#SBATCH --output=logs/slurm_%j.out      # Stdout and stderr go to this file

# Load necessary modules 
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate ODE_discovery_env

# Run
python discover_ODE.py