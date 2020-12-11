# -*- coding: utf-8 -*-
'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.celeba as customized_models

from torch.utils.data.sampler import Sampler
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from celeba import CelebA_Dataset
from os import path
import cv2


# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-s', '--save_attention_map', dest='save_attention_map', action='store_true',
                    help='save Attention map')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def min_max(x, axis=None):
    min = 0#x.min(axis=axis, keepdims=True)
    max = 1#x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def flip_img(sample):
    """
    画像だけを左右反転
    """
    image, landmarks = sample['image'], sample['landmarks']
    image = image[:, ::-1]  # 画像データを左右反転
    return {'image': image, 'landmarks': landmarks}


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[1., 1., 1.],
                                      std=[1., 1., 1.])

    # trans_train_dataset = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.train_batch, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # trans_test_dataset = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.test_batch, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    train_dataset = CelebA_Dataset(
        root_dir=args.data,
        train=True,
        transform=transforms.Compose([  # transform引数にcomposeを与える
           transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           normalize,
        ]))
    trans_train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch,
                        shuffle=True, num_workers=args.workers)

    test_dataset = CelebA_Dataset(
        root_dir=args.data,
        train=False,
        transform=transforms.Compose([  # transform引数にcomposeを与える
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize,
        ]))
    trans_test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch,
                        shuffle=False, num_workers=args.workers)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss(size_average=True).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'CelebA-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(trans_test_dataset, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trans_train_dataset, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(trans_test_dataset, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trans_train_dataset, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    alosses = AverageMeter()
    plosses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trans_train_dataset))
    for batch_idx, (sample_batch) in enumerate(trans_train_dataset):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = sample_batch['image']
        targets = sample_batch['landmarks']

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        att_outputs, outputs, _ = model(inputs)
        att_loss = criterion(att_outputs, targets)
        per_loss = 0
        for i in range(40):
            item_target = torch.reshape(targets[:,i], (len(targets[:,i]), 1))
            per_loss += criterion(outputs[i], item_target)
        per_loss /= 40.
        loss = att_loss + per_loss

        # measure accuracy and record loss
        alosses.update(att_loss.item(), inputs.size(0))
        plosses.update(per_loss.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Att.Loss: {aloss:.4f} | Per.Loss: {ploss:.4f} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trans_train_dataset),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    aloss=alosses.avg,
                    ploss=plosses.avg,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(trans_test_dataset, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc_tp = np.zeros(40, dtype=np.float32)
    acc_tn = np.zeros(40, dtype=np.float32)
    acc_np = np.zeros(40, dtype=np.float32)
    acc_nn = np.zeros(40, dtype=np.float32)

    with open('attribute_name.pkl', mode='rb') as f:
        names = pickle.load(f)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(trans_test_dataset))
    for batch_idx, (sample_batch) in enumerate(trans_test_dataset):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = sample_batch['image']
        targets = sample_batch['landmarks']

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        att_outputs, outputs, attention = model(inputs)

        out_dir = path.join('output', '{0:06d}'.format(batch_idx))
        if not path.exists(out_dir):
            os.mkdir(out_dir)

        d_inputs = inputs.data.cpu()
        d_inputs = d_inputs.numpy()

        for idx_att in range(40):
            batch_output = outputs[idx_att]
            item_att = attention[idx_att].data.cpu()
            item_att = item_att.numpy()

            for idx_batch in range(len(outputs[idx_att])):
                if args.save_attention_map:
                    c_att = item_att[idx_batch] 
                    resize_att = cv2.resize(c_att[0], (224, 224))
                    resize_att = min_max(resize_att)
                    resize_att *= 255.

                    attention_name = path.join(out_dir, '{0:06d}_'.format(batch_idx) + '{}.png'.format(names[idx_att]))
                    input_name = path.join(out_dir, '{0:06d}_input.png'.format(batch_idx))

                    v_img = ((d_inputs[idx_batch].transpose((1,2,0)) * [1., 1., 1.]) + [1., 1., 1.])* 255
                    v_img = v_img[:, :, ::-1]
                    v_img = np.uint8(v_img)
                    vis_map = np.uint8(resize_att)
                    jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                    jet_map = cv2.add(v_img, jet_map)

                    cv2.imwrite(attention_name, jet_map)
                    cv2.imwrite(input_name, v_img)


                item_output = batch_output[idx_batch].cpu()
                item_output = item_output.detach().numpy()
                if item_output >= 0.5:
                    fin_output = 1.0
                else:
                    fin_output = 0.0

                item_target = targets[idx_batch, idx_att].cpu()
                item_target = item_target.detach().numpy()

                if item_target == 1.0:
                    acc_np[idx_att] += 1.0
                    if fin_output == item_target:
                        acc_tp[idx_att] += 1.0
                else:
                    acc_nn[idx_att] += 1.0
                    if fin_output == item_target:
                        acc_tn[idx_att] += 1.0

        acc = (acc_tp + acc_tn) / (acc_np + acc_nn)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | acc.: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trans_test_dataset),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=np.average(acc),
                    )
        bar.next()
    bar.finish()
    return (np.average(acc), np.average(acc))

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
