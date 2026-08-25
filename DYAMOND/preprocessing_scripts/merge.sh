#!/bin/bash
module load cdo

# variables=("hfss" "hfls")

# for var in "${variables[@]}"; do
#     echo "$var"
#     cd "/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/hcg_data/$var"
#     cdo mergetime *.nc "/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/hcg_data/$var.nc"
# done

cd "/work/bd1179/b309297/DYAMOND_experiment/DYAMOND/hcg_data"
cdo merge *.nc /work/bd1179/b309297/DYAMOND_experiment/DYAMOND/merged_15min_ICON-NWP-2km_DW-ATM_r1i1p1f1_2d_gn_20200120000000-20200301000000_R02B05.nc
echo Finished. You may start cleaning up now
