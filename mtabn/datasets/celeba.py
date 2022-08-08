#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torchvision import transforms


CELEBA_NUM_CLASSES = 40


# NOTE: tuple of 40 attribute names of CelebA
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


### NOTE: MEAN and STD of CelebA
#   mean and std of RGB values on train dataset (order: R, G, B)
#   We assume that RGB value range is [0, 1]
CELEBA_TRAIN_RGB_MEAN = (0.5063454195744012, 0.42580961977997583, 0.38318672615935173)
CELEBA_TRAIN_RGB_STD  = (0.310506447934692, 0.2903443482746604, 0.2896806573348839)


### NOTE: transforms for image data
#   We do not know optimal transforms for obtaining clear attention maps...
CELEBA_TRANS_TRAIN = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
])
CELEBA_TRANS_EVAL = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
])


### NOTE: CelebA Dataset Class
#   We use torchvision.datasets.CelebA and do not implement Dataset class by ourselves.
#   Therefore, please see official document and source code for more details.


if __name__ == '__main__':

    #######################################################
    ### NOTE:
    # Here is debug code to check data augmentation.

    # ./figure/celeba_dataloader_check_001: default settings of Mitsuhara (no chages for scale and ratio)
    # ./figure/celeba_dataloader_check_002: only fix ratio with (1.0, 1.0)
    # ./figure/celeba_dataloader_check_003: only fix scale with (0.8, 1.0)
    # ./figure/celeba_dataloader_check_004: fix scale with (0.8, 1.0) and ratio with (1.0, 1.0)
    # ./figure/celeba_dataloader_check_005: add ColorJitter to 004
    # ./figure/celeba_dataloader_check_006: adjust params. from 005 settings
    #######################################################

    from torchvision.datasets import CelebA
    from torchvision import transforms

    DATA_ROOT = "/raid/hirakawa/dataset"

    # remove Normalize for visualizing augmentation result
    trans1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans2 = transforms.Compose([
        transforms.RandomResizedCrop(224, ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans3 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans4 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans5 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor()
    ])

    trans6 = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
        transforms.ToTensor()
    ])

    def _debug_plot_augmented_image(tranform, save_image_name):
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs("./figure", exist_ok=True)

        dataset = CelebA(root=DATA_ROOT, split='valid', target_type='attr', transform=tranform, download=False)

        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        fig.subplots_adjust(hspace=0, wspace=0)

        for ind in range(100):
            i = ind % 10
            j = ind // 10
            image, label = dataset[i]
            image = image.numpy()
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(image.transpose(1, 2, 0))

        plt.tight_layout()
        plt.savefig(save_image_name)
        plt.close()

    _debug_plot_augmented_image(trans1, "figure/celeba_dataloader_check_001.png")
    _debug_plot_augmented_image(trans2, "figure/celeba_dataloader_check_002.png")
    _debug_plot_augmented_image(trans3, "figure/celeba_dataloader_check_003.png")
    _debug_plot_augmented_image(trans4, "figure/celeba_dataloader_check_004.png")
    _debug_plot_augmented_image(trans5, "figure/celeba_dataloader_check_005.png")
    _debug_plot_augmented_image(trans6, "figure/celeba_dataloader_check_006.png")
