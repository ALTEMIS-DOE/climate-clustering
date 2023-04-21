#!/bin/bash

### IMPORTANT PARAM
cmip_meta='GFDL_ESM3G CONUS downscaled 1/8_degree' # format=(modelname, area, downscaled or not, reso)
###



data_basedir="/home/jupyter/climate-data/downscaled/GFDL-ESM2G/CONUS/hydro5"
precip_filename="${data_basedir}/Extraction_pr.nc"
et_filename="${data_basedir}/Extraction_et.nc"

#output_datadir="/home/jupyter/climate-data/tfrecords"
output_datadir="/home/jupyter/climate-data/tfrecords-hist-future"

stride=8
patch_size=16
timewindow=3
mhist=852

#patches_per_record=20000
patches_per_record=50000
rcps=( 'rcp26' 'rcp45' 'rcp60' 'rcp85')  # array to list

cwd="/home/jupyter/digitaltwin-climate-explorer/climate_explorer/pipelines"
python ${cwd}/pipeline.py \
    --precip_filename ${precip_filename} \
    --et_filename ${et_filename} \
    --output_datadir ${output_datadir} \
    --stride $stride \
    --patch_size ${patch_size} \
    --timewindow $timewindow \
    --patches_per_record ${patches_per_record} \
    --rcps "${rcps[@]}" \
    --cmip_meta "${cmip_meta}" \
    --mhist ${mhist}
    &