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
from models import vaegan_complete_train
import argparse

def main():
    parser = argparse.ArgumentParser(description="Training code")
    parser.add_argument("--dataset_path", type = str, required=True, help="path to celebA dataset")
    args = parser.parse_args()

    vaegan_complete_train(latent_size=128, epochs=6, retrain = True, dataset_path=args.dataset_path)


if __name__ == '__main__':
    main()
