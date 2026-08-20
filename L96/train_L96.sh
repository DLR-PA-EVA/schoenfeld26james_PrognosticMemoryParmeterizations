#!/bin/bash

#SBATCH --job-name="L96_ML"                # Job name
#SBATCH --output=logs/${job_name}_%j.out      # Stdout and stderr go to this file

# Load necessary modules 
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate L96_env_upgradet

# Run
python parametrizations.py --past_timesteps 1000 --latent_dims=4
