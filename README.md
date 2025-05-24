# Releasing Inequality Phenomena In $L_{\infty}$-Adversarial Training Via Input Gradient Distillation

This repository is the official implementation of [Releasing Inequality Phenomena In $L_{\infty}$-Adversarial Training Via Input Gradient Distillation].

## Requirements

To install requirements:

```setup
conda env create -f CondaEnv.yml
pip install -r pipList.txt
```

## How to prepare datasets

For CIFAR, just type a folder path into arg's 'imageFolder' or 'trainDatasetFolder' and 'testDatasetFolder'. Pytorch will do the rest for you.

For Imagenet and TinyImagenet, type the path of val and train folder of the dataset into arg's 'trainDatasetFolder' and 'testDatasetFolder' respectively in train.py. Type the path of val folder of the dataset into arg's 'imageFolder' in test.py.

For Imagenet-100, type the root path of Imagenet into arg's 'trainDatasetFolder' and 'testDatasetFolder' and 'imageFolder'.

For Imagenet-C(https://zenodo.org/record/2235448#.ZGI0TnZBygs), arrange it like:
```
--Imagenet-C
    |
    |----gaussian_noise
    |        |
    |        |----1
    |        |     |-train
    |        |     |   |-n01440764
    |        |     |   |...
    |        |     |   |-n15075141
    |        |     |-val
    |        |        |-n01440764
    |        |        |...
    |        |        |-n15075141
    |        |----2
    |        |    |-...
    |        |----3
    |        |    |-...
    |        |----4
    |        |    |-...
    |        |----5
    |        |    |-...
    |----shot_noise
    |        |...
    |----impulse_noise
             |...
```
Imagenet-C only contains images for evaluation. Thus, each class folder in 'train' only contains one useless image. Type the root of Imagenet-C into arg's 'imageFolder' in test.py and choose the attack by editing 'type' and 'strength'. You can use buildImagenetC.py to build it.


## Training

To train the model(s) in the paper, edit arguments in train.py and run this command:

```train
python train.py
```

## Evaluation

To evaluate, edit arguments in test.py and run this command:

```eval
python test.py
```

## Pre-trained weight

Please download weight in https://drive.google.com/file/d/1JV6QVcpRSm6qpAFFfBj12jMe3pCECkpl/view?usp=sharing

Specifically, when evaluating models on Imagenet-C, use weights in Imagenet-100.
## Results

Please refer to our paper.


## Contributing

> We propose a method, called "Input Gradient Distillation"(IGD for short), to release the inequality phenomena in $l_{\infty}$-adversarial training. Experiments show that the model trained with IGD tends to have a more equal attribution map and better robustness to i.i.d. noise and occlusion compared to the PGDAT-trained model while preserving adversarial robustness.

> We formally analyze the relationship between the Gini value(a metric to evaluate inequality) of saliency maps and models' robustness to noise and occlusion and claim that the equal decision pattern promotes the model's robustness by suppressing the deviation of the class score. We also explain why such robustness improvement is not notable on low-resolution datasets like CIFAR100.