#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=950
#SBATCH --output=/home/b/b309297/slurm_scripts/%j.out
#SBATCH --error=/home/b/b309297/slurm_scripts/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=temporal_interp
#=============================================================================

# Load CDO module
module load cdo

#=============================================================================
# Paths
variable="hus"  # Change to your variable (e.g., to, hus, prw)
inpath="DYAMOND/hcg_data/$variable"
outpath="/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/temporal_interp/$variable"
mkdir -p $outpath

# Reference file with correct 15min time steps
# reference_file="/fastdata/ka1081/DYAMOND/data/winter_data/MPIM-DWD-DKRZ/ICON-SAP-5km/DW-CPL/atmos/15min/tas/dpp0029/2d/gn/tas_15min_ICON-SAP-5km_DW-CPL_dpp0029_2d_gn_20200120000000-20200120234500.nc"

#=============================================================================
# Parallelization settings
N=1   # Max concurrent jobs

# Semaphore functions
open_sem(){
    mkfifo /home/b/b309297/scratch/pipe-$$
    exec 3<>/home/b/b309297/scratch/pipe-$$
    rm /home/b/b309297/scratch/pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
        "$@"
        printf '%.3d' $? >&3
    )&
}

#=============================================================================
# Function to interpolate one file
interp_time () {
    file=$1
    echo "Processing $file"
    file_name=$(basename "$file" .nc)

    if ! [[ -f ${outpath}/${file_name}_15min.nc ]]; then
        touch ${outpath}/${file_name}_15min.nc

        # --- Get first/last timestamp and number of steps from reference file ---
        # start_time="$(cdo showtimestamp $reference_file | head -1)"
        # end_time="$(cdo showtimestamp $reference_file | tail -1)"
        # nsteps=$(cdo ntime $reference_file)

        # --- Interpolate in time to reference 15min time axis ---
        cdo intntime,12 $file ${outpath}/${file_name}_15min.nc
    fi
}

#=============================================================================
# Main loop
files=$(ls $inpath/*.nc)
open_sem $N
for file in $files; do
    run_with_lock interp_time $file
done

wait
