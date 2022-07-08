#!/usr/bin/env python3
# -*- coding: utf-8 -*-


CELEBA_NUM_CLASSES = 40


# NOTE:
#   this attribute order is based on list_attr_celeba.txt.
#   Please do NOT change.
CELEBA_ATTRIBUTE_NAMES = (
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
)


### mean and std of RGB values on train dataset (order: R, G, B)
## In case of RGB value range: [0, 255]
# CELEBA_TRAIN_RGB_MEAN = (129.11807, 108.58144, 97.71261)
# CELEBA_TRAIN_RGB_STD = (38.48049, 37.01807, 37.42858)
## In case of RGB value range: [0, 1]
CELEBA_TRAIN_RGB_MEAN = (0.50634545, 0.42580956, 0.38318673)
CELEBA_TRAIN_RGB_STD = (0.1509039, 0.1451689, 0.14677875)


# NOTE:
# we use torchvision.datasets.CelebA and do not implement Dataset class by ourselves.
# Therefore, please see official document and source code.
