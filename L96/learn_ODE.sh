#!/bin/bash
#SBATCH --output=logs/slurm_%j.out      # Stdout and stderr go to this file

# This bash script is used to schedule equation discovery runs, to learn an ODE from the autoencoder

# Load necessary modules 
echo "Loading modules"
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate ODE_discovery_env

# Run
echo "Running discover_ODE.py"
python discover_ODE.py