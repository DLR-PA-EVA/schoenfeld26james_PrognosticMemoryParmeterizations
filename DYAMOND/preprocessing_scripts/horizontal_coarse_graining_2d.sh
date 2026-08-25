#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=compute     # Specify partition name
#SBATCH --time=08:00:00         # Set a limit on the total run time
#SBATCH --nodes=1               # I think one could try more than one node
#SBATCH --mem=950G              # Limit N to 11 (need 80GB per process)          
#SBATCH --output=/home/b/b309297/slurm_scripts/%j.out
#SBATCH --error=/home/b/b309297/slurm_scripts/%j.out
#SBATCH --mail-type=END,FAIL

#SBATCH --job-name=hcg.sh

#=============================================================================

## Coarse-grains the data (specified variables) from $inpath and stores the output in $outpath.

# DO NOT RUN MULTIPLE INSTANCES OF THIS JOB IN PARALLEL; THEY WILL INTERFERE

# 90 files in 8 hours (on Mistral)
# We process (no_nodes)*N files concurrently! So 100 files with the current setting.

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo /home/b/b309297/scratch/pipe-$$
    exec 3<>/home/b/b309297/scratch/pipe-$$
    rm /home/b/b309297/scratch/pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

foo () {
    echo $file
    # Read filename
    file_name=`echo $file | cut -d "." -f1`
    
    
    if ! [[ -f ${outpath}/${file_name}_R02B05.nc ]] # Do not overwrite
        then
            # Add resolution to the end of the filename
            # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
            touch ${outpath}/${file_name}_R02B05.nc
            echo ${file_name}_R02B05.nc
            
            # I think we can run 36 processes per compute node
            # cdo -f nc -P 36 remapcon,${target_grid} -setgrid,${source_grid} ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
            
            # We use CDO version 2.0.6 (https://code.mpimet.mpg.de/projects/cdo)
            cdo -f nc remapcon,${target_grid} -setgrid,${source_grid} ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
            # cdo -f nc remapcon,${target_grid} ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
    fi
    
}

## Main
module load cdo
# Set variable that you want to regrid
variable=hfls
# Adjust this path to point to the DYAMOND data
inpath=/fastdata/ka1081/DYAMOND/data/winter_data/MPIM-DWD-DKRZ/ICON-NWP-2km/DW-ATM/atmos/15min/$variable/r1i1p1f1/2d/gn/

source_grid=/fastdata/ka1081/DYAMOND/data/winter_data/MPIM-DWD-DKRZ/ICON-NWP-2km/DW-ATM/atmos/fx/gn/icon_grid_0017_R02B10_G.nc
target_grid="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"
outpath=/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/hcg_data/$variable

files="`ls $inpath`"
# No more than N processes run at the same time
N=11
open_sem $N
for file in $files; do
    run_with_lock foo $file
done 

# Wait until all threads are finished
wait 
