'''Trains DCGAN on MNIST using Keras
DCGAN is a Generative Adversarial Network (GAN) using CNN.
The generator tries to fool the discriminator by generating fake images.
The discriminator learns to discriminate real from fake images.
The generator + discriminator form an adversarial network.
DCGAN trains the discriminator and adversarial networks alternately.
During training, not only the discriminator learns to distinguish real from
fake images, it also coaches the generator part of the adversarial on how
to improve its ability to generate fake images.
[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import plot_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

from models import vaegan_model, vae_discriminator_model
from dataloader import dataloader
import cv2


def build_generator(latent_dim = 2048):
    """Build a Generator Model
    Stacks of BN-ReLU-Conv2DTranpose to generate fake images
    Output activation is sigmoid instead of tanh in [1].
    Sigmoid converges easily.
    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        image_size: Target size of one side (assuming square image)
    # Returns
        Model: Generator Model
    """
    '''
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    '''
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(8*8*256)(latent_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(256, (5,5), strides=(2,2), padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128,(5,5),strides=(2,2),padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32,(5,5),strides=(2,2),padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3,(5,5),strides=(1,1),padding ='same')(x)
    #outputs = Activation('tanh')(x)
    outputs = Activation('sigmoid')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    #generator = Model(inputs, x, name='generator')
    #return generator
    return decoder


def build_discriminator(inputs):
    """Build a Discriminator Model
    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in [1]
    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator


def train(models, params):
    """Train the Discriminator and Adversarial Networks
    Alternately train Discriminator and Adversarial networks by batch
    Discriminator is trained first with properly real and fake images
    Adversarial is trained next with fake images pretending to be real
    Generate sample images per save_interval
    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train images
        params (list) : Networks parameters
    """
    print('Training started.')
    generate_batch = dataloader(batch_size =64, normalized = True)

    generator, discriminator, adversarial = models
    batch_size, latent_size, train_steps, model_name = params
    num_images = 202599
    num_batches = num_images//batch_size
    save_interval = 633
    #save_interval = num_batches
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    for i in range(train_steps):
        # Random real images
        #rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        #real_images = x_train[rand_indexes]
        real_images, _ = next(generate_batch)
        # Generate fake images
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_images = generator.predict(noise)
        x = np.concatenate((real_images, fake_images))
        # Label real and fake images
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        # Train the Discriminator network
        metrics = discriminator.train_on_batch(x, y)
        loss = metrics[0]
        acc = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # Generate random noise
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # Label fake images as real
        y = np.ones([batch_size, 1])
        # Train the Adversarial network
        metrics = adversarial.train_on_batch(noise, y)
        loss = metrics[0]
        acc = metrics[1]
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            print('epoch: '+str((i+1)//save_interval))
            print(log)
            #filename = os.path.join(model_name, "check%05d.png" % step)
            filename = 'checkpoints/'+'chk-'+model_name+str((i+1))+'.hdf5'
            '''
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator,
                        noise_input=noise_input,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)
            '''
            generator.save_weights(filename)
    generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them
    For visualization purposes, generate fake images
    then plot them in a square grid
    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name
    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    print('max', np.max(images))
    print('min', np.min(images))
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        print(i)
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size, 3])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        #print(images)
        cv2.imshow('out', image)
        cv2.waitKey(0)
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def build_and_train_models():
    # MNIST dataset
    #(x_train, _), (_, _) = mnist.load_data()

    #image_size = x_train.shape[1]
    image_size = 64
    #x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    #x_train = x_train.astype('float32') / 255

    model_name = "dcgan_celeb_sigmoid"
    # Network parameters
    # The latent or z vector is 100-dim
    latent_size = 2048
    batch_size = 64
    #train_steps = 40000
    num_images = 202599
    epochs = 10
    train_steps = num_images*10
    lr = 0.0003*0.5
    decay = 6e-8*0.5
    input_shape = (image_size, image_size, 3)

    # Build discriminator model
    #inputs = Input(shape=input_shape, name='discriminator_input')
    #discriminator = build_discriminator(inputs)
    discriminator = vae_discriminator_model()
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    print('discriminator')
    discriminator.summary()

    # Build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator()
    #encoder, decoder, vae = vaegan_model(mse_flag=False)
    #generator = decoder
    print('generator')

    generator.summary()

    # Build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.05, decay=decay*0.5)
    discriminator.trainable = False
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()
    plot_model(adversarial, to_file='adversarial.png', show_shapes=True)
    # Train Discriminator and Adversarial Networks
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, params)


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 2048])
    images = generator.predict(noise_input)

    num_images = images.shape[0]
    image_size = images.shape[1]
    for i in range(num_images):
        image = np.reshape(images[i], [image_size, image_size, 3])
        cv2.imshow('out', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator= build_generator()
        #generator = load_model(args.generator)
        generator.load_weights(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()
