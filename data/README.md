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
[your directory]/
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
