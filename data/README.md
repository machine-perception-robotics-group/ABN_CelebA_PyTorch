# Data Preparation


Here, we describe how to prepare dataset for Multitask ABN scripts.


## CelebA Dataset

### Download dataset

First, you need to download [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

From Google Drive or Baido Drive provided by [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), please download the following files:

* `Anno/identity_CelebA.txt`
* `Anno/list_attr_celeba.txt`
* `Anno/list_bbox_celeba.txt`
* `Anno/list_landmarks_align_celeba.txt`
* `Anno/list_landmarks_celeba.txt`
* `Eval/list_eval_partition.txt`
* `Img/img_align_celeba.zip`

After finishing download, please unzip `img_align_celeba.zip`.


### Dataset organization

In our program, we use `torchvision.datasets.CelebA()` as a dataset class for CelebA Dataset.

`torchvision.datasets.CelebA()` needs to prepare CelebA dataset as follows:

```
foo/
  |- bar/
      |- celeba/
          |- identity_CelebA.txt
          |- list_attr_celeba.txt
          |- list_bbox_celeba.txt
          |- list_eval_partition.txt
          |- list_landmarks_align_celeba.txt
          |- list_landmarks_celeba.txt
          |- img_align_celeba/
              |- 000000.jpg
              |- 000001.jpg
              |- 000002.jpg
              |- ...
```


### MEAN and STD values on train dataset

For calculating mean and std values on train dataset.
Please use `check_rgb_mean_std_celeba.py`.

As a reference, we describe those values:

* pixel-wise mean and std (recommended)
    * mean (R, G, B): `(0.5063454195744012 0.42580961977997583 0.38318672615935173)`
    * std (R, G, B): `(0.310506447934692 0.2903443482746604 0.2896806573348839)`
* image-wise mean and std
    * mean (R, G, B): `(0.50634545, 0.42580956, 0.38318673)`
    * std (R, G, B): `(0.1509039, 0.1451689, 0.14677875)`
