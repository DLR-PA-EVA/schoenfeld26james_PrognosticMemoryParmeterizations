#!/bin/bash

# Construct job name
job_name="L96_online_run"

#SBATCH --job-name="L96_ML"                # Job name
#SBATCH --output=logs/L96_online_run_%j.out      # Stdout and stderr go to this file


# Load necessary modules 
# module load pytorch
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate L96_env_sklearn

# Run
python L96.py --latent_dims=6
