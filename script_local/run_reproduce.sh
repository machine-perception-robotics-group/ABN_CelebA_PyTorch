#!/bin/bash


### run_reproduce.sh
# This script describes training and evaluation commands for reproducing experiments of Fukui and Mitsuhara.

### NOTE:
#   Original MT-ABN uses ResNet-101 architecture.


#################################################
# Fukui experiment
# (ABN V1, no residual attention, fine-tuning)
#################################################

# training
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet101 --pretrained \
    --logdir ./runs_reproduce/rep_fukui \
    --gpu_id 0

# evaluation
python3 eval_celeba.py --data_root /raid/hirakawa/dataset \
    --logdir ./runs_reproduce/rep_fukui \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 0


#################################################
# Mitsuhara experiment
# (ABN V2, residual attention, fine-tuning)
#################################################

# training
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet101 --residual_attention --pretrained \
    --logdir ./runs_reproduce/rep_mitsuhara \
    --gpu_id 0

# evaluation
python3 eval_celeba.py --data_root /raid/hirakawa/dataset \
    --logdir ./runs_reproduce/rep_mitsuhara \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 0
