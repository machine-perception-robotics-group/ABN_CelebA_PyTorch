# Docker for Multitask ABN


## Change Log

* xx Jul 2022: Add a Docker image with PyTorch 1.11.0 (nvcr.io/nvidia/pytorch:22.02-py3).
    * NGC container: `nvcr.io/nvidia/pytorch:22.02-py3`
    * Our built image: `cumprg/mtlabn:1.11.0`
* xx Apr 2023: Add a Docker image with PyTorch 2.1.0 (nvcr.io/nvidia/pytorch:23.04-py3).
    * NGC container: `nvcr.io/nvidia/pytorch:23.04-py3`
    * Our built image: `cumprg/mtlabn:2.1.0`

## Pull Docker Images

First, you need to pull docker images.
You can use the following two images:

* NGC container: `nvcr.io/nvidia/pytorch:23.04-py3`
* Our built image: `cumprg/mtlabn:2.1.0`

To pull docker images, please run the following commands.
```bash
# 1. NGC container
docker pull nvcr.io/nvidia/pytorch:23:04-py3

# 2. Our built image
docker pull cumprg/mtlabn:2.1.0
```

`cumprg/mtlabn:2.1.0` runs docker daemon as a general user with the same user ID in host OS.
The detailed usage is described below (see Run).

## Build image

If you want to build docker image `cumprg/mtlabn:2.1.0` on your environment by yourself, please run following command.

```bash
bash build.sh
```

## Run docker

### 1. NGC Container

Please execute `docker run` command. For example:

```bash
docker run --gpus all -ti --rm --ipc=host \
    --name=[container name] -v [volume mount] \
    nvcr.io/nvidia/pytorch:23:04-py3
```

### 2. Our built image

By using `cumprg/mtlabn:2.1.0`, you can run docker with a general user with the same user ID in host OS.

You can run a docker daemon by following command. (User ID is automatically set.)

```bash
./run.sh [container name] [volume mount 1] [volume mount 2] ...
```

If you want run manually, please add user ID option. For example,

```bash
docker run --gpus all -ti --rm --ipc=host \
    --name=[container name] -v [volume mount] -u [uid]:[gid] \
    cumprg/mtlabn:2.1.0
```
