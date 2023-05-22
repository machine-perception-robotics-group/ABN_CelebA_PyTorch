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
python3 train_celeba.py --data_root ./data \
    --model mtabn_v1_resnet101 --pretrained \
    --logdir ./runs_reproduce/rep_fukui \
    --gpu_id 0

# evaluation
python3 eval_celeba.py --data_root ./data \
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
python3 train_celeba.py --data_root ./data \
    --model mtabn_v2_resnet101 --residual_attention --pretrained \
    --logdir ./runs_reproduce/rep_mitsuhara \
    --gpu_id 0

# evaluation
python3 eval_celeba.py --data_root ./data \
    --logdir ./runs_reproduce/rep_mitsuhara \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 0


#################################################
# Hirakawa original experiment
# (ABN V3, no residual attention, fine-tuning)
#################################################

# training
python3 train_celeba_mtabn_v3.py --data_root ./data \
    --model mtabn_v3_resnet101 --pretrained \
    --logdir ./runs_reproduce/orig_hirakawa \
    --gpu_id 0

# evaluation
python3 eval_celeba_mtabn_v3.py --data_root ./data \
    --logdir ./runs_reproduce/orig_hirakawa \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 0


#################################################
# Hirakawa original experiment
# (ABN V3, no residual attention, fine-tuning, weighted focal loss)
#################################################

# training
python3 train_celeba_wfl.py --data_root ./data \
    --model mtabn_v3_resnet101 --pretrained \
    --logdir ./runs_reproduce/wfl \
    --gpu_id 1

# evaluation
python3 eval_celeba_mtabn_v3.py --data_root ./data \
    --logdir ./runs_reproduce/wfl \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 1



#################################################
# Hirakawa original experiment
# (ABN V3, no residual attention, fine-tuning, weighted focal loss)
#################################################

# training
python3 train_celeba_wfl_norm.py --data_root ./data \
    --model mtabn_v3_resnet101 --pretrained \
    --logdir ./runs_reproduce/wfl_norm \
    --gpu_id 0

# evaluation
python3 eval_celeba_mtabn_v3.py --data_root ./data \
    --logdir ./runs_reproduce/wfl_norm \
    --no_eval_train --no_eval_val \
    --save_attention --attention_type pos \
    --resume checkpoint-final.pt \
    --gpu_id 1

