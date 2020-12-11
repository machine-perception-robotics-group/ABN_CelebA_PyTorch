#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import PIL
import torch
import random
import pickle

from glob import glob
from os import path
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


class CelebA_Dataset(Dataset):

    def __init__(self, root_dir, train, transform):
        self.train = train
        self.transform = transform
        self.dataset = []
        max_v = 5000

        print('Converting CelebA Dataset...')
        if self.train:
            label = [0, 1]
        else:
            label = [2]

        data_ann = path.join(root_dir, 'Anno', 'list_attr_celeba.txt')
        att_data = self.convert_attribute(data_ann)

        data_list = path.join(root_dir, 'list_eval_partition.txt')
        f_list = open(data_list, 'r')
        for idx, line in enumerate(f_list):
            ipath, raw_no = line[:-1].split(' ')
            sp_no = int(raw_no)
            for item_label in label:
                if item_label == sp_no:
                    img_full_path = path.join(root_dir, 'Img/img_celeba', ipath)

                    # if (idx < max_v) and (self.train):
                    #     self.dataset.append([img_full_path, att_data[idx + 1]])
                    # elif not self.train:
                    self.dataset.append([img_full_path, att_data[idx + 1]])

        print('  - # of samples :', len(self.dataset))
        print('\n')


    def convert_attribute(self, ann_path):
        f_ann = open(ann_path, 'r')
        data = []
        for idx, line in enumerate(f_ann):
            itemList = line[:-1].split(' ')
            if idx == 1:
                with open('attribute_name.pkl', mode='wb') as f:
                    pickle.dump(itemList, f)

            else:
                filter_item = [x for x in itemList if not x == '']
                item_data = []
                for item_filter in filter_item[1:]:
                    raw_label = int(item_filter)
                    if raw_label == -1:
                        raw_label = 0
                    item_data.append(raw_label)

                data.append(item_data)
        return data


    def get_path(self, i):
        data_path, att = self.dataset[i]
        return data_path


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, i):
        data_path, att = self.dataset[i]
        image = PIL.Image.open(data_path)

        # r_img = cv2.resize(image, (225, 225))
        # trans_img = np.asarray(image, dtype=np.float32)
        # r_img -= [103.939, 116.779, 123.68]
        # trans_img = r_img.transpose((2, 0, 1))#.reshape((1, 3, 225, 225))

        # trans_img = np.asarray(image, dtype=np.float32)
        att = np.asarray(att, dtype=np.float32).reshape((40))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'landmarks': att}

        return sample


