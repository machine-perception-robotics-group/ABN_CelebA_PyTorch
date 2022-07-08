#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
check_rgb_mean_std_celeba.py

Compute mean and std of RGB values of CelebA dataset.
"""


from email.mime import image
import os
from glob import glob
import numpy as np
import cv2


IMAGE_DIR_PATH = "./celeba/img_align_celeba"
EVAL_PARTITIOIN_FILE_PATH = "./celeba/list_eval_partition.txt"


def get_image_names(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    image_names = []
    for l in lines:
        fname, label = l.split(" ")
        if int(label.strip()) == 0:
            image_names.append(fname)

    return image_names


def main():
    image_names = get_image_names(EVAL_PARTITIOIN_FILE_PATH)

    num_images = len(image_names)

    B = np.zeros(num_images, dtype=np.float32)
    G = np.zeros(num_images, dtype=np.float32)
    R = np.zeros(num_images, dtype=np.float32)

    for i, i_name in enumerate(image_names):
        if i % 10000 == 0:
            print("num:", i)
        src = cv2.imread(os.path.join(IMAGE_DIR_PATH, i_name), 1)
        src = src.astype(np.float32) / 255.0
        B[i] += np.mean(src[:, :, 0])
        G[i] += np.mean(src[:, :, 1])
        R[i] += np.mean(src[:, :, 2])
    
    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)

    B_std = np.std(B)
    G_std = np.std(G)
    R_std = np.std(R)

    print("mean (R, G, B):", R_mean, G_mean, B_mean)
    print("std (R, G, B) :", R_std, G_std, B_std)


if __name__ == '__main__':
    main()
