#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from .datasets.celeba import CELEBA_ATTRIBUTE_NAMES


def destandardize(image, mean, std):
    image[0, :, :] = (image[0, :, :] * std[0]) + mean[0]
    image[1, :, :] = (image[1, :, :] * std[1]) + mean[1]
    image[2, :, :] = (image[2, :, :] * std[2]) + mean[2]
    return image


def normalize(input_data):
    return (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))


def normalize_pos_neg(input_data):
    return (input_data / (np.max(np.abs(input_data)) * 2)) + 0.5


def save_celeba_attention_map(image, att_map, label, out_per, out_att, save_name, att_type, rgb_mean=None, rgb_std=None):
    ### expected inputs, image = np.array, att_map = np.array

    ### fixed params. for CelebA dataset
    _N_COL, _N_ROW = 10, 4

    ### binarize label_preds
    out_per[out_per >= 0.5] = 1.0
    out_per[out_per < 0.5] = 0.0
    out_att[out_att >= 0.5] = 1.0
    out_att[out_att < 0.5] = 0.0
    out_per = out_per.to(torch.int32)
    out_att = out_att.to(torch.int32)

    ### scale RGB values of an image from [0.0, 1.0] to [0, 255]
    if rgb_mean is None and rgb_std is None:
        image = image * 255.0
    else:
        image = destandardize(image, rgb_mean, rgb_std) * 255.0
    image = image.transpose(1, 2, 0).astype(np.uint8)
    H, W, C = image.shape

    ### scale values of attention maps to [0.0, 1.0]
    if att_type == 'both':   # type 1: scale (-inf, inf) to [0, 1]
        att_map = normalize_pos_neg(att_map)
    elif att_type == 'pos':  # type 2: clip [0, inf), then scale to [0, 1]
        att_map[att_map <= 0] = 0
        if np.max(att_map) > 0:
            att_map = normalize(att_map)
    elif att_type == 'neg':  # type 3: clip (-inf, 0], then scale to [0, 1]
        att_map[att_map >= 0] = 0
        if np.min(att_map) < 0:
            att_map = normalize(att_map)
    att_map = att_map * 255.0

    ### make overlaid attention maps
    result_map = []
    for i in range(att_map.shape[0]):
        a_map = cv2.resize(att_map[i], (W, H))
        a_map = cv2.applyColorMap(a_map.astype(np.uint8), cv2.COLORMAP_JET)
        a_map = cv2.cvtColor(a_map, cv2.COLOR_BGR2RGB)
        dst_map = cv2.addWeighted(image, 0.55, a_map, 0.45, 0.0)
        result_map.append(dst_map)

    ### plot attention maps
    fig, ax = plt.subplots(_N_ROW, _N_COL, figsize=(15, 1.6 * _N_ROW))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(_N_ROW):
        for j in range(_N_COL):
            _a_idx = _N_COL * i + j
            if _a_idx == len(result_map):
                break
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(result_map[_a_idx])
            _title = "%s\nTrue: %s, Per: %s, Att: %s" % (CELEBA_ATTRIBUTE_NAMES[_a_idx], str(label[_a_idx].item()), str(out_per[_a_idx].item()), str(out_att[_a_idx].item()))
            ax[i, j].set_title(_title, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
