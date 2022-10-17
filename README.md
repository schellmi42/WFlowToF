# Weakly-Supervised Optical Flow Estimation for Time of Flight

### [Arxiv](https://arxiv.org/abs/2210.05298) | [Project Page](https://viscom.uni-ulm.de/publications/weakly-supervised-optical-flow-estimation-for-time-of-flight/)
> Weakly-Supervised Optical Flow Estimation for Time of Flight <br />
> [Michael Schelling](https://viscom.uni-ulm.de/members/michael-schelling/), [Pedro Hermosilla](https://viscom.uni-ulm.de/members/pedro-hermosilla/), [Timo Ropinski](https://viscom.uni-ulm.de/members/timo-ropinski/) <br />
> Winter Conference on Applications of Computer Vision - 2023 (Accepted)



This repository contains the PyTorch code the for the WACV paper 'Weakly-Supervised Optical Flow Estimation for Time of Flight'.

The code was tested using PyTorch 1.10.1+cu102 and Python 3.6.9 on Ubuntu 18.

## Dockerfile

To setup the environment it is advised to use the following dockerfile
```
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
	
RUN apt-get update
RUN apt-get -y install python3-opencv

RUN git clone https://github.com/schellmi42/Weakly_Supervised_ToF_Motion /tof_motion
RUN pip install -r /tof_motion/requirements.txt
RUN python -c 'import imageio; imageio.plugins.freeimage.download()'

WORKDIR /tof_motion
```

Installation of [NVIDIA-Docker-Support](https://github.com/NVIDIA/nvidia-docker) is necessary.

To create the docker image run the following (sudo) in the location you pasted the `Dockerfile`
```
nvidia-docker build -t tof_motion .
```
Start the docker container using the  `nvidia-container-toolkit` and `--gpus all` flags.


## Dataset

### Cornell-Box Dataset

The original Cornell-Box Dataset is avaiable at this GIT repository

>https://github.com/schellmi42/RADU/tree/main/data/data_CB

### Dataset Extension

The additional scenes containing object movements cam be downloaded from this url

>https://viscom.datasets.uni-ulm.de/weakly_sup_tof/CB_motion_extension.zip

### Loading of the Dataset

Unpack both ZIP-files into the `data` directory, such that the folders named `o_*, ca_*, cp_*` are at the same level.

To load the dataset in a docker container it is advised to mount the data folders into the container at `/tof_motion/data/` using the `docker --volume` flag.

The paths to the datasets may also be specified indiviually in the `DATA_PATH` variable inside the [data_loader.py](code/data_ops/data_loader.py) file.

## Pretrained model weights

Pretrained model weights of the FFN, MOM and CFN Network for Experiments 5.1, 5.2 and 5.3 are available at this URL:

> https://viscom.datasets.uni-ulm.de/weakly_sup_tof/trained_weights.zip

To evaluate the network using the pretrained weights use for example:


```
python code/train_FFN.py --log trained_weights/FFN_SF_1Tap/ --epochs 0 --eval_test --taps 1
```

Pre-trained weights for the RGB-networks are linked in the `ptlflow` documentation.

## Citing this work

If you use this code in your work, please kindly cite the following paper:

```
@inproceedings{schelling2020weakly-supervised,
	title={Weakly-Supervised Optical Flow Estimation for Time-of-Flight},
	author={Schelling, Michael and Hermosilla, Pedro and Ropinski, Timo},
	bookTitle={Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision}
	year={2023}
}
```