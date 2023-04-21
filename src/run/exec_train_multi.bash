#!/bin/bash

export PYTHONPATH="/home/jupyter/digitaltwin-climate-explorer"
 
expname=`echo $EPOCHSECONDS`
echo "JOB START",  $expname $1 $2 $3

input_datadir="/home/jupyter/climate-data/tfrecords"
logdir="./logs"
output_modeldir="./outputs/${expname}"

learning_rate=0.01 # Adam:0.001 SGD:0.01
num_epoch=100   # 96min for 25 epochs on one-V100
batch_size=1024
#npatches=3520000
npatches=368998

height=16
width=16
channel=6 # 3month x 2 variables

## if you want to specify designated gpu
assigned_gpu=$1

## For model arch.
kernel_size=$3 #5
nblocks=4  # 32x32 for 4x4 x256
base_dim=3  # 32x32-->2,2,512 for nblocks 5 1x1x1024 for nblock 5 #### <--- This is default
nstack_layer=$2 #1 
save_every=20


cwd="/home/jupyter/digitaltwin-climate-explorer/climate_explorer/train"

#export CUDA_VISIBLE_DEVICES=${assigned_gpu}
#mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 3 \
mpirun -np 2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
python ${cwd}/train_autoencoder.py \
    --input_datadir ${input_datadir} \
    --logdir ${logdir} \
    --output_modeldir $output_modeldir \
    --num_epoch $num_epoch \
    --lr ${learning_rate} \
    --batch_size ${batch_size} \
    --conv_kernel_size ${kernel_size} \
    --height ${height} \
    --width $width \
    --channel $channel \
    --expname $expname \
    --save_every $save_every \
    --nblocks ${nblocks} \
    --base_dim ${base_dim} \
    --nstack_layer ${nstack_layer} \
     --npatches ${npatches} \
     &


