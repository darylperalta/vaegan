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

## VAE model

## VAE results

VAE generation from noise:

![](imgs/vae_generator_out.jpg)

VAE autoencoder input:

![](imgs/vae_autoencoder_input.jpg)

VAE autoencoder reconstruction:

![](imgs/vae_autoencoder_output.jpg)

## GAN model

## GAN results

GAN generation from noise:

![](imgs/GAN_51273.jpg)

## VAEGAN model

<img src="imgs/VAEGANpaperfigure.jpg" width="500">

VAEGAN encoder model:

<img src="imgs/vaegan_encoder_complete.png" width="500">

VAEGAN decoder model:

<img src="imgs/vaegan_decoder_complete.png" width="500">

VAEGAN discriminator model:

<img src="imgs/vaegan_discriminator.png" width="300">

VAEGAN encoder model for training:

<img src="imgs/model1_enc.png" width="400">

VAEGAN decoder model for training:

<img src="imgs/model2_dec.png" width="400">

## VAEGAN results

VAEGAN generation from noise:

![](imgs/generator_out.jpg)

VAEGAN autoencoder input:

![](imgs/autoencoder_input.jpg)

VAEGAN autoencoder reconstruction:

![](imgs/autoencoder_output.jpg)


## VAEGAN results with Checkerboard Artifacts

VAEGAN generation from noise:

![](imgs/generator_out_grid.jpg)

VAEGAN autoencoder input:

![](imgs/autoencoder_input_grid.jpg)

VAEGAN autoencoder reconstruction:

![](imgs/autoencoder_output_grid.jpg)

## Problems Encountered

- Hard to train.
- Numerical instability in loss.
