#!/bin/bash
#SBATCH --job-name=precip_hyper_opt
#SBATCH --partition=gpu

# One independent job per hyperparameter configuration
#SBATCH --array=0-4

# Resources per job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G

#SBATCH --time=08:00:00
#SBATCH --account=bd1179
#SBATCH --mail-type=FAIL

#SBATCH --output=logs/slurm_%A_%a.out

# Activate environment
source /sw/spack-levante/mambaforge-22.9.0-2-Linux-x86_64-kptncg/etc/profile.d/conda.sh
conda activate DYAMOND_env

# Hyperparameter configurations
LATENT_DIMS=(2 2 2 4 4)
WEIGHT_DECAYS=(0.0 1e-6 1e-4 1e-6 1e-4)

# Select configuration based on array index
LD=${LATENT_DIMS[$SLURM_ARRAY_TASK_ID]}
WD=${WEIGHT_DECAYS[$SLURM_ARRAY_TASK_ID]}

echo "Running array task $SLURM_ARRAY_TASK_ID"
echo "latent_dims=$LD"
echo "weight_decay=$WD"

# Launch training
python training_gpu.py \
    --past_timesteps 20 \
    --latent_dims $LD \
    --weight_decay $WD