#!/bin/bash
declare -a txt=("0001" "00025" "0005"  "001"  "005" "01" )
declare -a lmd=("0.001" "0.0025" "0.005"  "0.01"  "0.05" "0.1" )
# get length of an array
arraylength=${#txt[@]}
#Hyperparameters:
steps=20000
lpips_w=0
total_w=10
lr_decay=1

#Export CUDA Libraries
export LD_LIBRARY_PATH=/scratch_net/unclemax/styannic/apps/cuda_10.0/lib64:$LD_LIBRARY_PATH
mkdir -p $1
cd $1
# Copy source files
cp /scratch_net/unclemax/styannic/JPEGLerarning/batch_job_attention.sh .
cp /scratch_net/unclemax/styannic/JPEGLerarning/ljpeg_attention.py .
for (( i=1; i<${arraylength}+1; i++ ));
do
  mkdir lambda_${txt[$i-1]}
  cd lambda_${txt[$i-1]}
  venv/bin/python3 models/ljpeg_attention.py train --train_glob="/scratch_net/unclemax/styannic/JPEGLerarning/gallery_20171023/*" --lr 0.000001 --lr_decay $lr_decay --lambda ${lmd[$i-1]} --batchsize 8 --last_step $steps --lpips_weight $lpips_w --total_weight_regularizer $total_w
  cd ..
done
