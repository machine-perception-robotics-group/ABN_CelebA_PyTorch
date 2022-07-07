#!/bin/bash


#################################################
# without residual attention
# train from scratch
#################################################
### Multitask ABN V2 with ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet18 --logdir ./runs/abn_v2_no_res_att/v2_r18 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet34 --logdir ./runs/abn_v2_no_res_att/v2_r34 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet50 --logdir ./runs/abn_v2_no_res_att/v2_r50 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet101 --logdir ./runs/abn_v2_no_res_att/v2_r101 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet152 --logdir ./runs/abn_v2_no_res_att/v2_r152 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# without residual attention
# pre-trained
#################################################
### Multitask ABN V2 with ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet18 --pretrained --logdir ./runs/abn_v2_no_res_att/v2_r18_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet34 --pretrained --logdir ./runs/abn_v2_no_res_att/v2_r34_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet50 --pretrained --logdir ./runs/abn_v2_no_res_att/v2_r50_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet101 --pretrained --logdir ./runs/abn_v2_no_res_att/v2_r101_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet152 --pretrained --logdir ./runs/abn_v2_no_res_att/v2_r152_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# with residual attention
# train from scratch
#################################################
### Multitask ABN V2 with ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet18 --residual_attention --logdir ./runs/abn_v2_res_att/v2_r18 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet34 --residual_attention --logdir ./runs/abn_v2_res_att/v2_r34 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet50 --residual_attention --logdir ./runs/abn_v2_res_att/v2_r50 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet101 --residual_attention --logdir ./runs/abn_v2_res_att/v2_r101 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet152 --residual_attention --logdir ./runs/abn_v2_res_att/v2_r152 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# with residual attention
# pre-trained
#################################################
### Multitask ABN V2 with ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet18 --pretrained --residual_attention --logdir ./runs/abn_v2_res_att/v2_r18_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet34 --pretrained --residual_attention --logdir ./runs/abn_v2_res_att/v2_r34_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet50 --pretrained --residual_attention --logdir ./runs/abn_v2_res_att/v2_r50_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet101 --pretrained --residual_attention --logdir ./runs/abn_v2_res_att/v2_r101_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V2 with ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v2_resnet152 --pretrained --residual_attention --logdir ./runs/abn_v2_res_att/v2_r152_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0
