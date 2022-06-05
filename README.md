# FNS

This repository contains the PyTorch code and a fast implementation for our proposed defending method based on 
**Feature Norm Suppressing (FNS)** against the universal adversarial patch attack.

## Installation

This repository requires, among others, the following packages:

* Python >= 3.5
* PyTorch and torchvision
* numpy
* matplotlib
* wget

## Implementation on CIFAR10

Please change directory to `CIFAR10`. The patch size is 10% of the whole image.

To attack the model without **FNS** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1000 --dr 1.0 --advp 0(LaVAN) or 1(AdvP) (--nonrob (without DOA) or None (with DOA)) --save YOUR_SAVE_PATH

To attack the model with **FNS-C** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1.2 --dr 1.0 --advp 0(LaVAN) or 1(AdvP) (--nonrob (without DOA) or None (with DOA)) --save YOUR_SAVE_PATH

To attack the model with **FNS-E** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1.25 --dr 1.5 --advp 0(LaVAN) or 1(AdvP) (--nonrob (without DOA) or None (with DOA)) --save YOUR_SAVE_PATH

To attack the model with **FNS-G** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_gaussian.py --clp 1.25 --dr 2.0 --advp 0(LaVAN) or 1(AdvP) (--nonrob (without DOA) or None (with DOA)) --save YOUR_SAVE_PATH

## Implementation on ImageNet

Please change directory to `CIFAR10`, and put the ImageNet validation set (separated by class labels) in `ImageNet`. The patch size is 5% of the whole image.

First, to modify the BatchNorm statistics for model with **FNS-C** on `GPU 0,1,2,3`, please run

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_exp.py --dist-url \'tcp://127.0.0.1:6008\' --dist-backend \'nccl\' --multiprocessing-distributed --world-size 1 --rank 0 /home/yuc/patchattack/imagenet --start-epoch 60 --model_type 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --clp 1.1 --dr 1.0

For model with **FNS-E** on `GPU 0,1,2,3`, please run

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_exp.py --dist-url \'tcp://127.0.0.1:6008\' --dist-backend \'nccl\' --multiprocessing-distributed --world-size 1 --rank 0 /home/yuc/patchattack/imagenet --start-epoch 60 --model_type 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --clp 1.15 --dr 1.5

For model with **FNS-G** on `GPU 0,1,2,3`, please run

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gaussian.py --dist-url \'tcp://127.0.0.1:6008\' --dist-backend \'nccl\' --multiprocessing-distributed --world-size 1 --rank 0 /home/yuc/patchattack/imagenet --start-epoch 60 --model_type 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --clp 1.13 --dr 2.0

Then, to attack the model without **FNS** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1000 --dr 1.0 --advp 0(LaVAN) or 1(AdvP) --model 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --nonrob --save YOUR_SAVE_PATH

To attack the model with **FNS-C** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1.1 --dr 1.0 --advp 0(LaVAN) or 1(AdvP) --model 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --save YOUR_SAVE_PATH

To attack the model with **FNS-E** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_exp.py --clp 1.15 --dr 1.5 --advp 0(LaVAN) or 1(AdvP) --model 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --save YOUR_SAVE_PATH

To attack the model with **FNS-G** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack_gaussian.py --clp 1.13 --dr 2.0 --advp 0(LaVAN) or 1(AdvP) --model 0(ResNet50) or 1(InceptionV3) or 2(MobileNetV2) --save YOUR_SAVE_PATH

## Implementation on VOC07

Please change directory to `VOC07`.
Put the test images (padded and resized to 416) and the annotations in `clean/voc_test` and `RawImage/voc_test/Annotations` respectively.
Then download the pretrained weights by running `weights/download_weights.sh`.
The patch size is 8% of the whole image.

To attack the model without **FNS** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack.py --clp 1000 --dr 1.0 --save YOUR_SAVE_PATH

To attack the model with **FNS-C** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack.py --clp 1.35 --dr 1.0 --save YOUR_SAVE_PATH

To attack the model with **FNS-E** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack.py --clp 1.4 --dr 1.5 --save YOUR_SAVE_PATH

To attack the model with **FNS-G** on `GPU 0` , please run

    CUDA_VISIBLE_DEVICES=0 python patchattack.py --clp 1.4 --dr 2.0 --gaussian --save YOUR_SAVE_PATH