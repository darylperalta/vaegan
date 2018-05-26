# VAEGAN
Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300v2) in Keras.

## Prerequisites

- tensorflow >=1.4
- keras >= 2.1.4
- OpenCV >= 3.4.0
- numpy

## Dataset

[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- 202599 celebrity images

## Image Generation from Noise Results


VAEGAN generation from noise:
![](img/generator_out.jpg)

VAEGAN autoencoder:
![](img/autoencoder_input.jpg)



![](img/autoencoder_output.jpg)


## Problems Encountered

- Hard to train.
- Numerical instability in loss.
