#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def denormalize(image, mean, std):
    image[0, :, :] = (image[0, :, :] * std[0]) + mean[0]
    image[1, :, :] = (image[1, :, :] * std[1]) + mean[1]
    image[2, :, :] = (image[2, :, :] * std[2]) + mean[2]
    return image


def save_multi_task_attention_map(image, att_map, save_name, attr_name_list, rgb_mean=[1., 1., 1.], rgb_std=[1., 1., 1.]):
    ### expected inputs, image = np.array, att_map = np.array

    # convert image tensor to numpy array (uint8)
    image = denormalize(image, rgb_mean, rgb_std)
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    H, W, C = image.shape

    # make heat maps of attention maps
    att_map = att_map * 255.0
    att_map = att_map.clip(0, 255).astype(np.uint8)

    # plot map for each attribute
    result_map = []
    for i in range(att_map.shape[0]):
        a_map = cv2.applyColorMap(att_map[i], cv2.COLORMAP_JET)
        a_map = cv2.cvtColor(a_map, cv2.COLOR_BGR2RGB)
        a_map = cv2.resize(a_map, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        dst_map = cv2.addWeighted(image, 0.5, a_map, 0.5, 0.0)
        result_map.append(dst_map)

    # plot attribute
    _n_col = 10
    _n_row = math.ceil(len(attr_name_list) / _n_col)

    fig, ax = plt.subplots(_n_row, _n_col, figsize=(15, 1.6 * _n_row))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(_n_row):
        for j in range(_n_col):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(result_map[_n_col * i + j])
            ax[i, j].set_title(attr_name_list[_n_col * i + j], fontsize=8)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
