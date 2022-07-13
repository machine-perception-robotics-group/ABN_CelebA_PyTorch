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
### NOTE: We assume that RGB value range is [0, 1]
CELEBA_TRAIN_RGB_MEAN = (0.5063454195744012, 0.42580961977997583, 0.38318672615935173)
CELEBA_TRAIN_RGB_STD  = (0.310506447934692, 0.2903443482746604, 0.2896806573348839)


# NOTE:
# we use torchvision.datasets.CelebA and do not implement Dataset class by ourselves.
# Therefore, please see official document and source code.
