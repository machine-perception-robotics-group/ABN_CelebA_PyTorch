#!/bin/bash


#################################################
# ResNet (train from scratch)
#################################################
### ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet18 --logdir ./runs/resnet/res18 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet34 --logdir ./runs/resnet/res34 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet50 --logdir ./runs/resnet/res50 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet101 --logdir ./runs/resnet/res101 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet152 --logdir ./runs/resnet/res152 \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0


#################################################
# ResNet (pre-trained)
#################################################
### ResNet 18
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet18 --pretrained --logdir ./runs/resnet/res18_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 34
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet34 --pretrained --logdir ./runs/resnet/res34_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 50
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet50 --pretrained --logdir ./runs/resnet/res50_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 101
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet101 --pretrained --logdir ./runs/resnet/res101_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0

### ResNet 152
python3 train_celeba.py --data_root /raid/hirakawa/dataset \
    --model resnet152 --pretrained --logdir ./runs/resnet/res152_pre \
    --epochs 100 --lr_steps 50 75 \
    --gpu_id 0
