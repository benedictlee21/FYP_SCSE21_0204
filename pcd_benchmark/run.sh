#!/bin/bash

mode=0 # 0 for train, 1 for test
model_dir="" # for test only
batch_size=32
num_workers=12
nepoch=100
model_name=$1
load_model=""
resume_epoch=0
num_points=2048
loss=$2
log_env="${model_name}_${num_points}_${loss}"
manual_seed=""
lr=0.0001
lr_clip=1e-6
n_primitives=16
step_ratio=2
use_mean_feature=0 # 0 if don't use, 1 if use

partition=vi_irdc_v100_32g
job_name=$log_env
gpus=1
g=$((${gpus}<8?${gpus}:8))

srun -u --partition=${partition} --job-name=${job_name} -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
     python main.py --mode $mode --batch_size $batch_size --workers $num_workers \
            --nepoch $nepoch --model_name $model_name --num_points $num_points \
            --log_env $log_env --loss $loss --lr $lr --lr_clip $lr_clip \
            --n_primitives $n_primitives --step_ratio $step_ratio --use_mean_feature $use_mean_feature \
            #--load_model $load_model --resume_epoch $resume_epoch # --model_dir $model_dir # --manual_seed $manual_seed





