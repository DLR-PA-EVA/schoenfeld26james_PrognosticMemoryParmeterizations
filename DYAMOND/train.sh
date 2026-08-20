#!/bin/bash
#SBATCH --job-name=precip_hyper_opt      # Specify job name
#SBATCH --partition=gpu            # Specify partition name
#SBATCH --nodes=1                  # Specify number of nodes
#SBATCH --ntasks-per-node=1        # Specify number of (MPI) tasks on each node
#SBATCH --gpus-per-task=1          # Specify number of GPUs per task
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --account=bd1179          # Charge resources on this project account

#SBATCH --output=logs/slurm_%j.out      # Stdout and stderr go to this file

# Load necessary modules 
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate DYAMOND_env

# Run
python training_gpu.py  --past_timesteps 20 --latent_dims 8 --weight_decay 0.001


