#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
eval_celeba.py
"""


import os
import json
from time import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.datasets import CelebA
from torchvision import transforms

from mtabn.datasets.celeba import CELEBA_NUM_CLASSES, CELEBA_ATTRIBUTE_NAMES
from mtabn.models import load_model, MODEL_NAMES
from mtabn.metrics import MultitaskConfusionMatrix
from mtabn.attention import save_multi_task_attention_map
from mtabn.utils import load_checkpoint, load_args


RESULT_DIR_TRAIN = "result_train"
RESULT_DIR_VAL = "result_val"
RESULT_DIR_TEST = "result_test"


def parser():
    arg_parser = ArgumentParser(add_help=True)

    ### basic settings
    arg_parser.add_argument('--logdir', type=str, required=True, help='directory stored trained model and settings')
    arg_parser.add_argument('--arg_file', type=str, default='args.json', help='json file name saved training settings')
    arg_parser.add_argument('--resume', type=str, default='checkpoint-best.pt', help='trained model file')

    ### dataset path
    arg_parser.add_argument('--data_root', type=str, required=True, help='path to CelebA dataset directory')

    ### evaluation settings
    arg_parser.add_argument('--no_eval_train', action='store_false', help='do not evaluate training data')
    arg_parser.add_argument('--no_eval_val', action='store_false', help='do not evaluate validation data')
    arg_parser.add_argument('--no_eval_test', action='store_false', help='do not evaluate test data')
    arg_parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size for evaluation')
    arg_parser.add_argument('--num_workers', type=int, default=32, help='the number of multiprocess workers for data loader')
    arg_parser.add_argument('--save_attention', action='store_true', help='save attention maps')

    ### GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    return arg_parser.parse_args()


def main():
    args = parser()

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    assert use_cuda, "Please use GPUs for evaluation."
    cudnn.deterministic = True
    cudnn.benchmark = False

    # load train args #####################################
    train_args = load_args(os.path.join(args.logdir, args.arg_file))

    # dataset #############################################
    print("load dataset")
    trans = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean=[1., 1., 1.], std=[1., 1., 1.])
    ])
    train_dataset = CelebA(root=args.data_root, split='train', target_type='attr', transform=trans, download=False)
    val_dataset   = CelebA(root=args.data_root, split='valid', target_type='attr', transform=trans, download=False)
    test_dataset  = CelebA(root=args.data_root, split='test', target_type='attr', transform=trans, download=False)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # network model #######################################
    _is_abn = True if "mtabn_" in train_args['model'] else False
    model = load_model(
        model_name=train_args['model'], num_classes=CELEBA_NUM_CLASSES,
        residual_attention=train_args['residual_attention'], pretrained=train_args['pretrained']
    )
    model = nn.DataParallel(model).cuda()
    ### load checkpoint
    print("    load checkpont:", os.path.join(args.logdir, args.resume))
    model, _, _, _, _, _ = load_checkpoint(os.path.join(args.logdir, args.resume), model, None, None)

    #######################################################
    # the beginning of evaluation
    #######################################################
    ### train data
    if args.no_eval_train:
        print("evaluate train data ...")
        evaluation(model, train_loader, RESULT_DIR_TRAIN, args, _is_abn)

    ### validation data
    if args.no_eval_val:
        print("evaluate validation data ...")
        evaluation(model, val_loader, RESULT_DIR_VAL, args, _is_abn)

    ### test data
    if args.no_eval_test:
        print("evaluate test data ...")
        evaluation(model, test_loader, RESULT_DIR_TRAIN, args, _is_abn)
    #######################################################
    # the end of evaluation
    #######################################################

    print("evaluation; done.")


def evaluation(model, data_loader, result_dir_name, args, is_abn):
    ### make result directory in logdir
    result_dir = os.path.join(args.logdir, result_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    mt_conf_mat_per = MultitaskConfusionMatrix(num_attributes=CELEBA_NUM_CLASSES, attr_name_list=CELEBA_ATTRIBUTE_NAMES)
    mt_conf_mat_att = MultitaskConfusionMatrix(num_attributes=CELEBA_NUM_CLASSES, attr_name_list=CELEBA_ATTRIBUTE_NAMES)

    image_counter = 0

    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.cuda()
            output = model(image)

            # binarize output
            label = label.data
            if is_abn:
                mt_conf_mat_per.update(label_trues=label, label_preds=output[0], use_cuda=True)
                mt_conf_mat_att.update(label_trues=label, label_preds=output[1], use_cuda=True)
            else:
                mt_conf_mat_per.update(label_trues=label, label_preds=torch.sigmoid(output), use_cuda=True)

            if is_abn and args.save_attention:
                image = image.cpu().data.numpy()              # [batch, channel, height, width]
                attention_map = output[2].cpu().data.numpy()  # [batch, attribute, height, width]
                for img_index in range(image.shape[0]):
                    save_multi_task_attention_map(
                        image[img_index], attention_map[img_index],
                        os.path.join(result_dir, "%06d.jpg" % image_counter), attr_name_list=CELEBA_ATTRIBUTE_NAMES
                    )
                    image_counter += 1

    # compute accuracy
    mean_acc_dict = {}
    mean_acc_dict['mean accuracy (per)'] = mt_conf_mat_per.get_average_accuracy()
    score_per = mt_conf_mat_per.get_attr_score()
    if is_abn:
        mean_acc_dict['mean accuracy (att)'] = mt_conf_mat_att.get_average_accuracy()
        score_att = mt_conf_mat_att.get_attr_score()

    # print
    print("mean accuracy:", mean_acc_dict)
    # save as json
    with open(os.path.join(result_dir, "mean_accuracy.json"), 'w') as f:
        json.dump(mean_acc_dict, f, indent=4)
    with open(os.path.join(result_dir, "class-wise_accuracy_per.json"), 'w') as f:
        json.dump(score_per, f, indent=4)
    if is_abn:
        with open(os.path.join(result_dir, "class-wise_accuracy_att.json"), 'w') as f:
            json.dump(score_att, f, indent=4)


if __name__ == '__main__':
    main()
