#!/bin/bash


#################################################
# without residual attention
# train from scratch
#################################################
### Multitask ABN V3 with ResNet 18
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet18 --logdir ./runs/abn_v1_no_res_att/v1_r18 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 34
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet34 --logdir ./runs/abn_v1_no_res_att/v1_r34 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 50
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet50 --logdir ./runs/abn_v1_no_res_att/v1_r50 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 101
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet101 --logdir ./runs/abn_v1_no_res_att/v1_r101 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 152
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet152 --logdir ./runs/abn_v1_no_res_att/v1_r152 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# without residual attention
# pre-trained
#################################################
### Multitask ABN V3 with ResNet 18
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet18 --pretrained --logdir ./runs/abn_v1_no_res_att/v1_r18_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 34
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet34 --pretrained --logdir ./runs/abn_v1_no_res_att/v1_r34_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 50
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet50 --pretrained --logdir ./runs/abn_v1_no_res_att/v1_r50_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 101
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet101 --pretrained --logdir ./runs/abn_v1_no_res_att/v1_r101_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 152
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet152 --pretrained --logdir ./runs/abn_v1_no_res_att/v1_r152_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# with residual attention
# train from scratch
#################################################
### Multitask ABN V3 with ResNet 18
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet18 --residual_attention --logdir ./runs/abn_v1_res_att/v1_r18 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 34
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet34 --residual_attention --logdir ./runs/abn_v1_res_att/v1_r34 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 50
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet50 --residual_attention --logdir ./runs/abn_v1_res_att/v1_r50 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 101
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet101 --residual_attention --logdir ./runs/abn_v1_res_att/v1_r101 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 152
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet152 --residual_attention --logdir ./runs/abn_v1_res_att/v1_r152 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# with residual attention
# pre-trained
#################################################
### Multitask ABN V3 with ResNet 18
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet18 --pretrained --residual_attention --logdir ./runs/abn_v1_res_att/v1_r18_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 34
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet34 --pretrained --residual_attention --logdir ./runs/abn_v1_res_att/v1_r34_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 50
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet50 --pretrained --residual_attention --logdir ./runs/abn_v1_res_att/v1_r50_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 101
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet101 --pretrained --residual_attention --logdir ./runs/abn_v1_res_att/v1_r101_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### Multitask ABN V3 with ResNet 152
python3 train_celeba_mtabn_v3.py --data_root /raid/hirakawa/dataset \
    --model mtabn_v1_resnet152 --pretrained --residual_attention --logdir ./runs/abn_v1_res_att/v1_r152_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0
