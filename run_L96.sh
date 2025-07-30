#!/bin/bash

# Construct job name
job_name="L96_ML"

#SBATCH --job-name="L96_ML"                # Job name
#SBATCH --output=logs/${job_name}_%j.out      # Stdout and stderr go to this file
# #SBATCH --gpus=1
# #SBATCH --mem=128G
# #SBATCH --time=16:00:00

# Load necessary modules 
# module load pytorch
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate L96_env_sklearn

# Run
python L96.py # --model_type=NN+AE_latent_dims=5
