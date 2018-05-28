from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Lambda, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers import LeakyReLU

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import math

#from dataloader import dataloader
from models import vaegan_complete_predict
import argparse


def main():

    parser = argparse.ArgumentParser(description="Test code")
    parser.add_argument("--dataset_path", type = str, required=True, help="path to celebA dataset")
    parser.add_argument("--encoder_path", type = str, required=True, help="path to encoder")
    parser.add_argument("--decoder_path", type = str, required=True, help="path to decoder")

    args = parser.parse_args()

    #path_encoder = 'checkpoints/encoder_chk-vaegan_complete_lessdense_meannll_plusganloss_reducedlrencoder0_44642.hdf5'
    #path_decoder = 'checkpoints/decoder_chk-vaegan_complete_lessdense_meannll_plusganloss_reducedlrencoder0_44642.hdf5'

    print(args.encoder_path)
    print(args.decoder_path)

    vaegan_complete_predict(path_encoder = args.encoder_path, path_decoder=args.decoder_path, datapath = args.dataset_path, latent_dim = 128, save_out=False)


if __name__ == '__main__':
    main()
