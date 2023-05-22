#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train_celeba.py
"""


import os
from time import time
from argparse import ArgumentParser

import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.datasets import CelebA

from mtabn.datasets.celeba import CELEBA_TRANS_TRAIN, CELEBA_TRANS_EVAL, CELEBA_NUM_CLASSES, CELEBA_ATTRIBUTE_NAMES, get_celeba_frequency_histogram
from mtabn.models import load_model, MODEL_NAMES
from mtabn.nn import WeightedBFLossWithLogits
from mtabn.metrics import MultitaskConfusionMatrix
from mtabn.utils import load_checkpoint, save_checkpoint, save_args, load_args


LOG_STEP = 500
CHECKPOINT_STEP = 1


def parser():
    arg_parser = ArgumentParser(add_help=True)

    ### network settings
    arg_parser.add_argument('--model', type=str, default='mtabn_v1_resnet101', choices=MODEL_NAMES, help='network model')
    arg_parser.add_argument('--residual_attention', action='store_true', help='use residual attention mechanism')
    arg_parser.add_argument('--pretrained', action='store_true', help='use pretrained network model as initial parameter')

    ### dataset path
    arg_parser.add_argument('--data_root', type=str, required=True, help='path to CelebA dataset directory')

    ### traininig settings
    arg_parser.add_argument('--logdir', type=str, required=True, help='directory for storing train log and checkpoints')
    arg_parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    arg_parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
    arg_parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    arg_parser.add_argument('--lr_steps', type=int, nargs='+', default=[4, 8], help='epochs to decrease learning rate')
    arg_parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD optimizer')
    arg_parser.add_argument('--wd', type=float, default=1e-4, help='weight decay of SGD optimizer')
    arg_parser.add_argument('--use_nesterov', action='store_true', help='use nesterov accelerated SGD')
    arg_parser.add_argument('--num_workers', type=int, default=8, help='the number of multiprocess workers for data loader')

    ### weighted focal loss settings
    arg_parser.add_argument('--use_wfl', action='store_true', help='use weighted focal loss')
    arg_parser.add_argument('--alpha', type=float, default=-1, help='alpha parameter of weighted focal loss')
    arg_parser.add_argument('--gamma', type=float, default=2, help='gamma parameter of weighted focal loss')

    ### resume settings
    arg_parser.add_argument('--resume', type=str, default=None, help='filename of checkpoint for resuming the training')

    ### random seed settings
    arg_parser.add_argument('--seed', type=int, default=1701, help='random seed')

    ### GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    args = arg_parser.parse_args()

    ### resume train args
    if args.resume is not None:
        _resume_path = os.path.join(args.logdir, args.resume)
        assert os.path.exists(_resume_path), "ERROR: resume file does not exist: %s" % _resume_path

        print("resume: duplicate arguments ...")
        _resume_args = load_args(os.path.join(args.logdir, "args.json"))
        args.model        = _resume_args['model']
        args.residual_attention = _resume_args['residual_attention']
        args.pretrained   = _resume_args['pretrained']
        args.batch_size   = _resume_args['batch_size']
        args.epochs       = _resume_args['epochs']
        args.lr           = _resume_args['lr']
        args.lr_steps     = _resume_args['lr_steps']
        args.momentum     = _resume_args['momentum']
        args.wd           = _resume_args['wd']
        args.use_nesterov = _resume_args['use_nesterov']
        args.use_wfl      = _resume_args['use_wfl']
        args.alpha        = _resume_args['alpha']
        args.gamma        = _resume_args['gamma']
        args.seed         = _resume_args['seed']

    return args


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    args = parser()

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    assert use_cuda, "This training script needs CUDA (GPU)."

    # set random seed (optional) ##########################
    if args.seed is not None:
        fix_random_seed(args.seed)

    # dataset #############################################
    print("load dataset")

    train_dataset = CelebA(root=args.data_root, split='train', target_type='attr', transform=CELEBA_TRANS_TRAIN, download=False)
    val_dataset = CelebA(root=args.data_root, split='valid', target_type='attr', transform=CELEBA_TRANS_EVAL, download=False)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, **kwargs)

    # network model & loss ################################
    _is_abn = True if "mtabn_" in args.model else False
    model = load_model(
        model_name=args.model, num_classes=CELEBA_NUM_CLASSES,
        residual_attention=args.residual_attention, pretrained=args.pretrained
    )
    if args.use_wfl:
        freq_hist = get_celeba_frequency_histogram(train_dataset)
        criterion_bce = WeightedBFLossWithLogits(freq_hist=freq_hist, alpha=args.alpha, gamma=args.gamma)
    else:
        criterion_bce = nn.BCEWithLogitsLoss()

    # optimizer ###########################################
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.use_nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

    # CPU or GPU
    model = nn.DataParallel(model).cuda()
    criterion_bce = criterion_bce.cuda()
    cudnn.benchmark = True

    initial_epoch = 1
    iteration = 0
    best_score = 0.0
    loss_sum, loss_sum_att, loss_sum_per = 0.0, 0.0, 0.0

    # resume ##############################################
    if args.resume is not None:
        print("Load checkpoint for resuming a training ...")
        print("    checkpoint:", os.path.join(args.logdir, args.resume))
        model, optimizer, scheduler, initial_epoch, iteration, best_score = load_checkpoint(os.path.join(args.logdir, args.resume), model, optimizer, scheduler)
        initial_epoch += 1

    # tensorboardX ########################################
    writer = SummaryWriter(log_dir=args.logdir)
    log_dir = writer.file_writer.get_logdir()
    save_args(os.path.join(log_dir, 'args.json'), args)

    #######################################################
    # the beginning of train loop
    #######################################################
    _start = time()
    for epoch in range(initial_epoch, args.epochs + 1):
        print("epoch:", epoch)

        # train #######################
        model.train()
        for image, label in train_loader:
            iteration += 1

            image, label = image.cuda(), label.to(torch.float32).cuda()
            output = model(image)

            if _is_abn:
                loss_per = criterion_bce(output[0], label)
                loss_att = criterion_bce(output[1], label)
                loss = loss_per + loss_att
            else:
                loss = criterion_bce(output, label)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if _is_abn:
                loss_sum_per += loss_per.item()
                loss_sum_att += loss_att.item()

            if iteration % LOG_STEP == 0:
                print("iteration: %06d loss: %0.8f (per: %0.8f, att: %0.8f) elapsed time: %0.1f" % (
                    iteration, loss_sum / LOG_STEP, loss_sum_per / LOG_STEP, loss_sum_att / LOG_STEP, time() - _start
                ))
                writer.add_scalar("loss/all", loss_sum / LOG_STEP, iteration)
                if _is_abn:
                    writer.add_scalar("loss/per", loss_sum_per / LOG_STEP, iteration)
                    writer.add_scalar("loss/att", loss_sum_att / LOG_STEP, iteration)
                loss_sum, loss_sum_att, loss_sum_per = 0.0, 0.0, 0.0

        # validation #######################
        print("evaluation ...")
        mt_conf_mat_per = MultitaskConfusionMatrix(num_attributes=CELEBA_NUM_CLASSES, attr_name_list=CELEBA_ATTRIBUTE_NAMES)
        mt_conf_mat_att = MultitaskConfusionMatrix(num_attributes=CELEBA_NUM_CLASSES, attr_name_list=CELEBA_ATTRIBUTE_NAMES)
        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image = image.cuda()
                output = model(image)

                # binarize output
                label = label.data
                if _is_abn:
                    mt_conf_mat_per.update(label_trues=label, label_preds=torch.sigmoid(output[0]), use_cuda=True)
                    mt_conf_mat_att.update(label_trues=label, label_preds=torch.sigmoid(output[1]), use_cuda=True)
                else:
                    mt_conf_mat_per.update(label_trues=label, label_preds=torch.sigmoid(output), use_cuda=True)

        # compute accuracy
        mean_acc_per = mt_conf_mat_per.get_average_accuracy()
        score_per = mt_conf_mat_per.get_attr_score()
        if _is_abn:
            mean_acc_att = mt_conf_mat_att.get_average_accuracy()
            score_att = mt_conf_mat_att.get_attr_score()

        # print accuracy
        print("    accuracy (per):", mean_acc_per)
        if _is_abn:
            print("    accuracy (att):", mean_acc_att)

        # write log to tensorboard
        writer.add_scalar("accuracy/per", mean_acc_per, epoch)
        for k in score_per.keys():
            writer.add_scalar("attr_acc_per/%s" % str(k), score_per[k]['accuracy'], epoch)
        if _is_abn:
            writer.add_scalar("accuracy/att", mean_acc_att, epoch)
            for k in score_att.keys():
                writer.add_scalar("attr_acc_att/%s" % str(k), score_att[k]['accuracy'], epoch)

         # change learning rate ########
        scheduler.step()

        # save model ##################
        print("save model ...")
        ### 1. best validation accuracy
        if best_score < mean_acc_per:
            print("    best score")
            best_score = mean_acc_per
            save_checkpoint(os.path.join(log_dir, "checkpoint-best.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 2. at regular intervals
        if epoch % CHECKPOINT_STEP == 0:
            print("    regular intervals")
            save_checkpoint(os.path.join(log_dir, "checkpoint-%04d.pt" % epoch), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 3. for resume (latest checkpoint)
        print("    latest")
        save_checkpoint(os.path.join(log_dir, "checkpoint-latest.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        print("epoch:", epoch, "; done.\n")
    #######################################################
    # the end of train loop
    #######################################################

    # save final model & close tensorboard writer
    print("save final model")
    save_checkpoint(os.path.join(log_dir, "checkpoint-final.pt"), model, optimizer, scheduler, best_score, epoch, iteration)
    writer.close()
    print("training; done.")


if __name__ == '__main__':
    main()
