# VAEGAN
Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300v2) in Keras.

EE298 Group 4;

Peralta, Daryl
Mendaros, Jonathan
Aslan, Cha-dash


## Prerequisites

- tensorflow >=1.4
- keras >= 2.1.4
- OpenCV >= 3.4.0
- numpy

Usage

For training:

python train.py --dataset [path to dataset]

Ex.

python train.py --dataset /home/daryl/datasets/img_align_celeba

For testing:

python test.py --dataset_path [path dataset] --encoder_path [path to encoder] --decoder_path [path to decoder]

Ex.

python test.py --dataset_path '/home/daryl/datasets/img_align_celeba' --encoder_path checkpoints/encoder_chk-vaegan_complete_demo.hdf5 --decoder_path checkpoints/decoder_chk-vaegan_complete_demo.hdf5 --dataset_path /home/daryl/datasets/img_align_celeba


Checkpoints can be found [here](https://drive.google.com/drive/folders/1hoU9QXccq6M1OkmtJbetyi5UoC9apgkU?usp=sharing).

## Dataset

[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- 202599 celebrity images


## VAEGAN model

<img src="imgs/VAEGANpaperfigure.jpg" width="500">

VAEGAN encoder model:

<img src="imgs/vaegan_encoder_complete.png" width="600">

VAEGAN decoder model:

<img src="imgs/vaegan_decoder_complete.png" width="400">

VAEGAN discriminator model:

<img src="imgs/vaegan_discriminator.png" width="300">

VAEGAN encoder model for training (only the encoder is trainable):

<img src="imgs/model1_enc.png" width="400">

VAEGAN decoder model for training (only the decoder is trainable):

<img src="imgs/model2_dec.png" width="400">


## VAE results

VAE generation from noise:

![](imgs/vae_generator_out.jpg)

VAE autoencoder input:

![](imgs/vae_autoencoder_input.jpg)

VAE autoencoder reconstruction:

![](imgs/vae_autoencoder_output.jpg)

## GAN results

GAN generation from noise:

![](imgs/GAN_51273.jpg)

## VAEGAN results

VAEGAN generation from noise:

![](imgs/generator_out.jpg)

VAEGAN autoencoder input:

![](imgs/autoencoder_input.jpg)

VAEGAN autoencoder reconstruction:

![](imgs/autoencoder_output.jpg)

VAEGAN generation from noise:

![](imgs/generator_out_633.jpg)

VAEGAN autoencoder input:

![](imgs/autoencoder_input_633.jpg)

VAEGAN autoencoder reconstruction:

![](imgs/autoencoder_output_633.jpg)



## VAEGAN results with Checkerboard Artifacts

VAEGAN generation from noise:

![](imgs/generator_out_grid.jpg)

VAEGAN autoencoder input:

![](imgs/autoencoder_input_grid.jpg)

VAEGAN autoencoder reconstruction:

![](imgs/autoencoder_output_grid.jpg)
