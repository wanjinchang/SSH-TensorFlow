# SSH-TensorFlow

### Introduction

This is a implementation of [SSH: Single Stage Headless Face Detector](https://arxiv.org/pdf/1708.03979.pdf) reproduced using TensorFlow. 

This code is modified from [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn).	

### Prerequisites

1. You need a CUDA-compatible GPU to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) for face detection.

### Dependencies

* TensorFlow 1.4.1
* TF-Slim
* Python3.6
* Ubuntu 16.04
* Cuda 8.0

### WIDER Face Eval result

SSH eval result without using an image pyramid between different backbones:

| backbone |  easy | medium | hard | inference time(GTX 1060) | training method |
|:-------:|:-----:|:-----:|:-----:|:-------:|:-------:|
|  VGG16 | 0.908 | 0.885 | 0.746 | 56ms(400x600) | random training |
|  VGG16 | 0.917 | 0.903 | 0.799 | 56ms(400x600) | group training |
| ResNet50 | 0.902 | 0.880 | 0.689 | 68ms(400x600) | random training |
| ResNet50 | 0.918 | 0.906 | 0.791 | 68ms(400x600) | group training |
| MobileNetV1 | 0.897 | 0.874 | 0.720 | 23ms(400x600) | random training |
| MobileNetV1 | 0.910 | 0.886 | 0.754 | 23ms(400x600) | group training |
| MobileNetV2 | 0.865 | 0.820 | 0.576 | 21ms(400x600) | random training |

**Note**: About the difference between random training and group training, please see details in the script ``lib/model/train_val.py``.

### Result

demo_result(MobileNetV1_SSH result):

**Result on FDDB**


### Contents

1. [Installation](#installation)
2. [Setup_data](#setup_data)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Demo](#demo)
6. [Models](#models)

## Installation

-  Clone the repository
  ```Shell
  git clone https://github.com/wanjinchang/SSH-TensorFlow.git
  ```

-  Update your -arch in setup script to match your GPU
  ```Shell
  cd SSH-TensorFlow/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | :-------------: | :-------------: |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.

-  Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

## Setup_data

Generate your own annotation file from WIDER FACE dataset(eliminate the invalid data that x <=0 or y <=0 or w <=0 or h <= 0).
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min y_min x_max y_max`.  
    Here is an example:
```
0--Parade/0_Parade_marchingband_1_849 449.0 330.0 570.0 478.0
0--Parade/0_Parade_Parade_0_904 361.0 98.0 623.0 436.0
...
```

Or you can use my annotation files `wider_face_train.txt` and `wider_face_val.txt` under the folder ``data/`` directly.
And you should have a directory structure as follows:  
```
data
   |--WIDER
         |--WIDER_train
             |--Annotations/
             |--images/ 
         |--WIDER_val
             |--Annotations/
             |--images/ 
```

Or you can follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup WIDER Face datasets. The steps involve downloading data and optionally creating soft links in the ``data/`` folder. 
If you find it useful, the ``data/cache`` folder created on my side is also shared [here](https://drive.google.com/open?id=1L7QpZm5qVgGO8HtDvQbrFcfTIoGY4Jzh).

## Training

-  Download pre-trained models and weights of backbones.The current code supports VGG16/ResNet_V1/MobileNet_Series models. 
-  Pre-trained models are provided by slim, you can get the pre-trained models from [Google Driver](https://drive.google.com/open?id=1iqOZNA9nwvITvwTDvK2gZUHAI1fo_XHI) or [BaiduYun Driver](https://pan.baidu.com/s/1m7uv9Sqs6hEb3VcMy3gFzg). Uzip and place them in the folder ``data/imagenet_weights``. For example, for VGG16 model, you also can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For ResNet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

-  Train
  ```Shell
  ./experiments/scripts/train_ssh.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU id you want to train on
  # NET in {vgg16, res50, res101, res152, mobile, mobile_v2} is the backbone network arch to use
  # DATASET {wider_face} is defined in train_ssh.sh
  # Examples:
  ./experiments/scripts/train_ssh.sh 0 wider_face vgg16
  ./experiments/scripts/train_ssh.sh 1 wider_face res101
  ```
  **Note**: Only support IMS_PER_BATCH=1 for training now, see details in the cfg files under the foder ``experiments/cfgs/``.
 
By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

## Evaluation

Evaluation the trained models on wider face val dataset
  ```Shell
  ./experiments/scripts/test_ssh.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the backbone network arch to use
  # DATASET {wider_face} is defined in test_ssh.sh
  # Examples: 
  ./experiments/scripts/test_ssh.sh 0 wider_face vgg16
  ./experiments/scripts/test_ssh.sh 0 wider_face mobile
  ```

## Demo

-  For ckpt demo
Download trained models from [Models](#models), then uzip to the folder ``output/``, modify your path of trained model
  ```Shell
  # at repository root
  GPU_ID=0
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py.
  ```
or run ``python tools/demo.py`` directly.

-  For frozen graph inference
Download the pb models(contained in [Models](#models)) or frozen your model by yourself using script ``tools/convert_ckpt_to_pb.py``, modify your path of trained model, then run ``python tools/inference.py``.


## Models

* vgg16_ssh(contained random_training and group training models) [Google Driver](https://drive.google.com/open?id=1D2B_PueKvYpY-oMX4p2HImJaZAz6cM5f), [BaiduYun Driver](https://pan.baidu.com/s/1AD61gsBR8QP-Zo_Ej2-d_w)
* res50_ssh(contained random_training and group training models) [Google Driver](https://drive.google.com/open?id=1zCzzxi3N1dWZG-i9XdsL0sSynzZ0MAC_), [BaiduYun Driver](https://pan.baidu.com/s/1797jTFrLgGfCVRmC4XK8Bg)
* mobile_ssh(contained random training and group training models/mobilenetv1_ssh) [Google Driver](https://drive.google.com/open?id=1U63x2sYS8pt8NCrKunRsdxGRB9vwMzOp), [BaiduYun Driver](https://pan.baidu.com/s/1_iZnkPMy3Xsxzp7R2BUNkw)
* mobilev2_ssh((random training/mobilenetv2_ssh) [Google Driver](https://drive.google.com/open?id=1nhW-a6xEuvTPteVFztWXXm2Coozj-s4X), [BaiduYun Driver](https://pan.baidu.com/s/1mZVksGh4KlCmJ-pP-R6dcg)

### License
MIT LICENSE

### TODOs
- [ ] Fix some bugs in training MobileNetV2_SSH
- [ ] Support multi-batch images training
- [ ] Multi-GPUs training
- [ ] Add facial landmarks detection to SSH

### References
1. SSH: Single Stage Headless Face Detector(https://arxiv.org/pdf/1708.03979.pdf). Mahyar Najibi, Pouya Samangouei, Rama Chellappa, Larry S. Davis.ICCV 2017.
2. [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
3. [SSH(offical)](https://github.com/mahyarnajibi/SSH)
