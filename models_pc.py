'''Models for VAE/GAN'''


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

from dataloader import dataloader
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def vae_model(original_dim=32*32,input_shape = (32*32,), intermediate_dim = 512, batch_size =128, latent_dim = 2, epochs=50):
    '''VAE model.'''
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = binary_crossentropy(inputs,
                                              outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae, inputs, outputs

def vaegan_model(original_dim=(64,64,3), batch_size =64, latent_dim = 2048, epochs=50, mse_flag=True):
    '''VAE model.'''
    # VAE model = encoder + decoder
    # build encoder model
    input_shape = original_dim
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(64,(5,5), strides =(2,2),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x_mean = Dense(latent_dim, name='x_mean')(x)
    x_mean = BatchNormalization()(x_mean)
    z_mean = Activation('relu', name='z_mean')(x_mean)

    x_log_var = Dense(latent_dim, name='x_log_var')(x)
    x_log_var = BatchNormalization()(x_log_var)
    z_log_var = Activation('relu', name='z_log_var')(x_log_var)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    encoder.summary()
    plot_model(encoder, to_file='vaegan_encoder.png', show_shapes=True)

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
    outputs = Activation('tanh')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vaegan_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    #outputs = Dense(original_dim, activation='sigmoid')(x)
    if mse_flag:
        reconstruction_loss = mse(inputs,
                              outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
    reconstruction_loss *= original_dim[0]*original_dim[1]*original_dim[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=RMSprop(lr=0.0003))
    vae.summary()
    plot_model(vae,
               to_file='vae.png',
               show_shapes=True)

    return encoder, decoder, vae

def vaegan_actual_model(original_dim=(64,64,3), batch_size =64, latent_dim = 128, epochs=50, mse_flag=True):
    '''VAE model.'''
    # VAE model = encoder + decoder
    # build encoder model
    input_shape = original_dim
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(64,(5,5), strides =(2,2),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='z_mean')(x)

    z_mean = Dense(latent_dim, name='x_mean')(x)

    z_log_var = Dense(latent_dim, name='x_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    encoder.summary()
    plot_model(encoder, to_file='vaegan_encoder.png', show_shapes=True)

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
    outputs = Activation('tanh')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vaegan_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    #outputs = Dense(original_dim, activation='sigmoid')(x)
    if mse_flag:
        reconstruction_loss = mse(inputs,
                              outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
    reconstruction_loss *= original_dim[0]*original_dim[1]*original_dim[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=RMSprop(lr=0.0003))
    vae.summary()
    plot_model(vae,
               to_file='vae.png',
               show_shapes=True)

    return encoder, decoder, vae


def vaegan_actual_train(batch_size = 64, epochs=10, final_chk = 'vae.h5',mse_flag=True):
    ''' TRAIN VAEGAN model on CELEBA'''
    num_images = 202599
    num_batches = num_images//batch_size
    print('num_batches', num_batches)

    #print('mse: ', mse)
    '''
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('original_dim: ', original_dim)
    input_shape = (original_dim, )
    '''
    encoder, decoder, vae = vaegan_actual_model(mse_flag=mse_flag)

    models = (encoder, decoder)
    #data = (x_test, y_test)

    chkpath="/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-negative-May24-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(chkpath, verbose=1)

    vae.fit_generator(dataloader(negative=True),
                    epochs=epochs, steps_per_epoch=num_batches,
                    verbose=1, callbacks =[checkpoint]
                    )

    vae.save_weights(final_chk)

    plot_results(models,
                     data,
                     batch_size=batch_size,
                     model_name="vae_mlp")

def vae_discriminator_model(original_dim=(64,64,3)):

    input_shape = original_dim
    input = Input(shape=input_shape)
    x = Conv2D(32,(5,5), strides =(2,2),padding='same')(input)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(128,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Flatten()(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    discriminator = Model(input, output, name='discriminator')
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator.png', show_shapes=True)


    return discriminator

def vae_train(batch_size = 128,epochs=50):
    ''' Test VAE model on mnist'''
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('original_dim: ', original_dim)
    input_shape = (original_dim, )

    encoder, decoder, vae, inputs, outputs = vae_model(original_dim,input_shape = (original_dim,))

    models = (encoder, decoder)
    data = (x_test, y_test)


    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    vae.fit(x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                     data,
                     batch_size=batch_size,
                     model_name="vae_mlp")

def vaegan_train(batch_size = 64, epochs=10, final_chk = 'vae.h5',mse_flag=True):
    ''' TRAIN VAEGAN model on CELEBA'''
    num_images = 202599
    num_batches = num_images//batch_size
    print('num_batches', num_batches)

    #print('mse: ', mse)
    '''
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('original_dim: ', original_dim)
    input_shape = (original_dim, )
    '''
    encoder, decoder, vae = vaegan_model(mse_flag=mse_flag)

    models = (encoder, decoder)
    #data = (x_test, y_test)

    chkpath="/home/daryl/EE298Z/vaegan/checkpoints/chkpt-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(chkpath, verbose=1)

    vae.fit_generator(dataloader(),
                    epochs=epochs, steps_per_epoch=num_batches,
                    verbose=1, callbacks =[checkpoint]
                    )

    vae.save_weights(final_chk)

    plot_results(models,
                     data,
                     batch_size=batch_size,
                     model_name="vae_mlp")

def vaegan_predict(weights_path = 'vae_mlp_mnist.h5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 2048, save_out=True):
    encoder, decoder, vae = vaegan_model()
    batch = 10
    out_dir = 'vaegan_vae_out'

    vae.load_weights(weights_path)

    '''Generator prediction.'''

    #z = np.random.normal(size=(batch,latent_dim))
    z = np.random.uniform(-1.0, 1.0, size=[batch, latent_dim])
    print('z shape', z.shape)
    out = decoder.predict(z)
    print('min', np.min(out))
    os.makedirs(out_dir, exist_ok = True)
    for i in range(batch):

        print('predict', out.shape)
        cv2.imshow('asdfa', out[i])
        cv2.waitKey(0)
        if save_out == True:
            cv2.imwrite(out_dir+'/'+'out'+str(i)+'.jpg', (out[i]*255).astype(np.uint8))

    '''Autoencoder prediction.'''
    image_size =64
    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:batch]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        #cv2.imshow('temp',img_temp)
        #cv2.waitKey(0)
        batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))

    batch_images = batch_images/255.0
    out_vae = vae.predict(batch_images)

    for i in range(batch):
        cv2.imshow('Input', batch_images[i,:,:,:])
        cv2.waitKey(0)
        cv2.imshow('Output', out_vae[i,:,:,:])
        cv2.waitKey(0)

def vaegan_actual_predict(weights_path = 'vae_mlp_mnist.h5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 2048, save_out=True):
    encoder, decoder, vae = vaegan_actual_model()
    batch = 10
    out_dir = 'vaegan_vae_out_actual_128'

    vae.load_weights(weights_path)

    '''Generator prediction.'''

    #z = np.random.normal(size=(batch,latent_dim))
    z = np.random.uniform(-1.0, 1.0, size=[batch, latent_dim])
    print('z shape', z.shape)
    out = decoder.predict(z)
    print('min', np.min(out))
    os.makedirs(out_dir, exist_ok = True)
    for i in range(batch):

        print('predict', out.shape)
        cv2.imshow('asdfa', (out[i]*127.5+127.5).astype(np.uint8) )
        cv2.waitKey(0)
        if save_out == True:
            cv2.imwrite(out_dir+'/'+'out'+str(i)+'.jpg', (out[i]*127.5+127.5).astype(np.uint8))

    '''Autoencoder prediction.'''
    image_size =64
    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:batch]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        #cv2.imshow('temp',img_temp)
        #cv2.waitKey(0)
        batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))

    batch_images = batch_images/255.0
    out_vae = vae.predict(batch_images)

    for i in range(batch):
        cv2.imshow('Input', batch_images[i,:,:,:])
        cv2.waitKey(0)
        cv2.imshow('Output', out_vae[i,:,:,:])
        cv2.waitKey(0)


def nll_loss(mean, x):
    '''
    sigma = 1.0
    multiplier = 1.0/(2.0*sigma**2)
    c = -0.5*np.log(2*np.pi)
    tmp = y_pred - y_true
    tmp **= 2.0
    tmp *= -multiplier
    tmp += c
    #return K.sum(tmp)
    return K.mean(tmp)
    '''
    ln_var =0
    x_prec = math.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    return K.sum(loss)
    #return K.mean(loss)

def vaegan_complete_model(original_dim=(64,64,3), batch_size =64, latent_dim = 128, epochs=50, mse_flag=True, lr = 0.0003):
        '''VAEGAN complete model.'''
        # VAE model = encoder + decoder
        # build encoder model
        input_shape = original_dim
        inputs = Input(shape=input_shape, name='encoder_input')

        x = Conv2D(64,(5,5), strides =(2,2),padding='same', name= 'enc_conv1')(inputs)
        x = BatchNormalization(name= 'enc_bn1')(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU1')(x)


        x = Conv2D(128,(5,5), strides =(2,2),padding='same', name= 'enc_conv2')(x)
        x = BatchNormalization(name= 'enc_bn2')(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU2')(x)


        x = Conv2D(256,(5,5), strides =(2,2),padding='same', name= 'enc_conv3')(x)
        x = BatchNormalization(name= 'enc_bn3')(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU3')(x)

        x = Flatten()(x)
        #x = Dense(2048, name = 'enc_dense1')(x)
        #x = BatchNormalization(name = 'enc_bn4')(x)
        #x = Activation('relu', name='z_mean')(x)
        #x = LeakyReLU(alpha = 0.2, name = 'enc_dense2')(x)



        x_mean = Dense(latent_dim, name='x_mean')(x)
        x_mean = BatchNormalization()(x_mean)
        z_mean = LeakyReLU(alpha = 0.2, name = 'z_mean')(x_mean)


        x_log_var = Dense(latent_dim, name='x_log_var')(x)
        x_log_var = BatchNormalization()(x_log_var)
        z_log_var = LeakyReLU(alpha = 0.2, name='z_log_var')(x_log_var)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        print('encoder')
        encoder.summary()
        plot_model(encoder, to_file='vaegan_encoder_complete.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(8*8*256)(latent_inputs)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Reshape((8, 8, 256))(x)

        x = Conv2DTranspose(256, (5,5), strides=(2,2), padding ='same')(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Conv2DTranspose(128,(5,5),strides=(2,2),padding ='same')(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Conv2DTranspose(32,(5,5),strides=(2,2),padding ='same')(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Conv2DTranspose(3,(5,5),strides=(1,1),padding ='same')(x)
        outputs = Activation('tanh')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        print('decoder')
        decoder.summary()
        plot_model(decoder, to_file='vaegan_decoder_complete.png', show_shapes=True)

        #instantiate discriminator
        x_recon = Input(shape=input_shape)
        #x = Conv2D(32,(5,5), strides =(2,2),padding='same')(x_recon)
        x = Conv2D(32,(5,5), strides =(1,1),padding='same')(x_recon)
        #x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Conv2D(128,(5,5), strides =(2,2),padding='same')(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        l_layer = Conv2D(256,(5,5), strides =(2,2),padding='same')(x)


        l_layer_shape = (8,8,256)

        input_disc2 = Input(shape=l_layer_shape)

        x = BatchNormalization()(input_disc2)
        #x = BatchNormalization()(l_layer)

        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Flatten()(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Dense(1)(x)
        output_dis = Activation('sigmoid')(x)
        #discriminator_2 = Model(input_disc2, output_dis, name='discriminator_1')

        '''construct discriminator with l_layer output'''
        discriminator_l = Model(x_recon, l_layer, name='discriminator_l')
        print('discriminator_l')
        discriminator_l.summary()

        ''' construct discriminator second part'''
        discriminator_2 = Model(input_disc2, output_dis, name='discriminator_2')
        print('discriminator_2')
        discriminator_2.summary()





        ''' construct discriminator (discriminator trainable) '''

        discriminator = Model(x_recon, discriminator_2(discriminator_l(x_recon)), name='discriminator')
        print('discriminator')
        #optimizer = RMSprop(lr=lr)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=RMSprop(lr=lr*0.8),
                              metrics=['accuracy'])
        print('discriminator')
        discriminator.summary()
        plot_model(discriminator, to_file='vaegan_discriminator.png', show_shapes=True)





        '''construct model 1 (encoder trainable) '''

        encoder.trainable =True
        decoder.trainable = False
        discriminator_l.trainable = False
        discriminator_2.trainable = False
        print('model1_enc')

        disc_xtilde = discriminator_l(decoder(encoder(inputs)[2]))
        disc_x = discriminator_l(inputs)
        out_recon = decoder(encoder(inputs)[2])
        model1_enc = Model(inputs, [discriminator_2(disc_x),discriminator_2(disc_xtilde)],name = 'model_encoder1')
        model1_enc.summary()
        plot_model(model1_enc, to_file='model1_enc.png', show_shapes=True)

        '''
        model1_enc = Model(inputs, discriminator_l(decoder(encoder(inputs)[2])), name='model1_encoder')
        print('model1 encoder trainable')
        plot_model(model1_enc, to_file='model1_enc.png', show_shapes=True)
        '''
        '''Define losses for encoder parameter update'''



        reconstruction_loss = nll_loss(disc_x,disc_xtilde)
        #reconstruction_loss *= original_dim[0]*original_dim[1]*original_dim[2]
        #recon_mse = mse(inputs,out_recon)
        #recon_mse *= original_dim[0]*original_dim[1]*original_dim[2]

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        #vae_loss = K.mean(reconstruction_loss + kl_loss+recon_mse)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        model1_enc.add_loss(vae_loss)
        model1_enc.compile(optimizer=RMSprop(lr=lr*0.1))
        #model1_enc.compile(optimizer=RMSprop(lr=0.003*0.001))

        #model1_enc.summary()

        ''' construct model 2 (decoder trainable) '''

        encoder.trainable =False
        decoder.trainable = True
        discriminator_l.trainable = False
        discriminator_2.trainable = False

        zp = Input(shape=(latent_dim,), name='zp')
        out_zp = discriminator_2(discriminator_l(decoder(zp)))

        model2_dec = Model([inputs,zp], [discriminator_2(disc_x),discriminator_2(disc_xtilde), out_zp], name='model2_encoder')
        print('model2 decoder trainable')
        model2_dec.summary()
        plot_model(model2_dec, to_file='model2_dec.png', show_shapes=True)

        #reconstruction_loss = nll_loss(disc_x,disc_xtilde)
        #reconstruction_loss *= original_dim[0]*original_dim[1]*original_dim[2]
        gamma = 1e-6

        #vae_loss = K.mean(reconstruction_loss + kl_loss)

        #gan_real_loss = binary_crossentropy(K.ones_like(discriminator_2(disc_x)),discriminator_2(disc_x))
        gan_fake_loss1 = binary_crossentropy(K.ones_like(discriminator_2(disc_xtilde)),discriminator_2(disc_xtilde))
        gan_fake_loss2 = binary_crossentropy(K.ones_like(out_zp),out_zp)
        #gan_fake_loss1 = binary_crossentropy(K.zeros_like(discriminator_2(disc_xtilde)),discriminator_2(disc_xtilde))
        #gan_fake_loss2 = binary_crossentropy(K.zeros_like(out_zp),out_zp)
        gan_fake_loss=K.mean(gan_fake_loss1+gan_fake_loss2)
        #dec_loss = K.mean(gamma*reconstruction_loss - gan_fake_loss)
        #dec_loss = gamma*reconstruction_loss - gan_fake_loss
        dec_loss = gamma*reconstruction_loss + gan_fake_loss

        model2_dec.add_loss(dec_loss)
        model2_dec.compile(optimizer=RMSprop(lr=lr*0.8))

        #optimizer = RMSprop(lr=lr)
        #discriminator.compile(loss='binary_crossentropy',
        #                      optimizer=optimizer,
        #                      metrics=['accuracy'])
        #print('discriminator')


        return encoder, decoder, discriminator, model1_enc, model2_dec

def vaegan_complete_train(batch_size = 64, final_chk = 'vae_complete.h5',mse_flag=True,latent_size = 128, epochs=11, retrain = False):
    image_size = 64
    #x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    #x_train = x_train.astype('float32') / 255

    model_name = "vaegan_complete_sumnll_plus_ganloss_1benc_lr_01_retrain4_enctrain2x_"
    # Network parameters
    # The latent or z vector is 100-dim
    #latent_size = 2048
    batch_size = 64
    #train_steps = 40000
    num_images = 202599
    #epochs = 11
    train_steps = int((num_images//64)*epochs)
    #lr = 0.0003*0.5
    #decay = 6e-8*0.5
    lr = 0.0003
    input_shape = (image_size, image_size, 3)


    encoder, decoder, discriminator, model1_enc, model2_dec = vaegan_complete_model( latent_dim = latent_size)

    if retrain == True:
        encoder.load_weights('checkpoints/encoder_chk-vaegan_complete_sumnll_plus_ganloss_1benc_lr_05_retrain3_may272954.hdf5')
        decoder.load_weights('checkpoints/decoder_chk-vaegan_complete_sumnll_plus_ganloss_1benc_lr_05_retrain3_may272954.hdf5')
        discriminator.load_weights('checkpoints/model2_dec_chk-vaegan_complete_sumnll_plus_ganloss_1benc_lr_05_retrain3_may272954.hdf5')


    print('Training started.')
    generate_batch = dataloader(batch_size =64, normalized = True, negative=True)

    #generator, discriminator, adversarial = models
    #batch_size, latent_size, train_steps, model_name = params
    num_images = 202599
    num_batches = num_images//batch_size
    save_interval = 211
    #save_interval = 2
    #noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    for i in range(train_steps):
        # Random real images
        #rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        #real_images = x_train[rand_indexes]
        real_images, _ = next(generate_batch)

        metrics = model1_enc.train_on_batch(real_images, None)
        log = "%d [encoder loss:%f]" % (i, metrics)

        real_images, _ = next(generate_batch)

        metrics = model1_enc.train_on_batch(real_images, None)
        log = "%d [encoder loss:%f]" % (i, metrics)
        '''
        real_images, _ = next(generate_batch)

        metrics = model1_enc.train_on_batch(real_images, None)
        log = "%d [encoder loss:%f]" % (i, metrics)

        real_images, _ = next(generate_batch)

        metrics = model1_enc.train_on_batch(real_images, None)
        log = "%d [encoder loss:%f]" % (i, metrics)
        '''
        real_images, _ = next(generate_batch)


        y_real = np.ones([batch_size, 1])
        metrics = discriminator.train_on_batch(real_images,y_real)
        log = "%s: [discriminator (real) loss:%f acc:%f]" % (log, metrics[0],metrics[1])

        y_fake = np.zeros([batch_size, 1])
        x_tilde = decoder.predict(encoder.predict(real_images)[2])

        metrics =discriminator.train_on_batch(x_tilde,y_fake)
        log = "%s [discriminator (z) loss:%f acc:%f]" % (log, metrics[0],metrics[1])
        #y = np.zeros([batch_size, 1])
        zp = np.random.normal(0,1,size=(batch_size, latent_size))
        metrics =discriminator.train_on_batch(decoder.predict(zp),y_fake)
        log = "%s [discriminator (zp) loss:%f acc:%f]" % (log, metrics[0], metrics[1])

        real_images, _ = next(generate_batch)
        zp = np.random.normal(0,1,size=(batch_size, latent_size))
        #metrics = model2_dec.train_on_batch([real_images,zp],[y_real,y_fake,y_fake])
        #metrics = model2_dec.train_on_batch([real_images,zp],[None,None,None])
        metrics = model2_dec.train_on_batch([real_images,zp],None)
        #print(metrics)
        log = "%s [decoder loss:%f]" % (log, metrics)

        print(log)

        if (i + 1) % save_interval == 0:
            print('step: '+str((i+1)))
            #print(log)
            #filename = os.path.join(model_name, "check%05d.png" % step)
            filename = 'chk-'+model_name+str((i+1))+'.hdf5'
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
            encoder.save_weights('checkpoints/encoder_'+filename)
            decoder.save_weights('checkpoints/decoder_'+filename)
            #model1_enc.save_weights('checkpoints/model1_enc_'+filename)
            #model2_dec.save_weights('checkpoints/model2_dec_'+filename)
            discriminator.save_weights('checkpoints/model2_dec_'+filename)

            plot_images(decoder,encoder, i+1, model_name,latent_size,'/home/daryl/datasets/img_align_celeba', 25, save_out =True)
            # (generator, encoder, steps, model_name, latent_size, datapath = '/home/daryl/datasets/img_align_celeba', num_images=25,save_out =False)
    encoder.save(model_name +'_encoder'+ ".h5")
    decoder.save(model_name +'_decoder'+ ".h5")
    discriminator.save(model_name +'_discriminator'+ ".h5")

def vaegan_complete_predict(path_encoder = 'checkpoints/encoder_chk-vaegan_complete_lessdense_meannll_minusganloss9073.hdf5', path_decoder='checkpoints/decoder_chk-vaegan_complete_lessdense_meannll_minusganloss9073.hdf5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 128, save_out=False):
    #encoder, decoder, vae = vaegan_actual_model()
    encoder, decoder, discriminator, model1_enc, model2_dec = vaegan_complete_model( latent_dim = latent_dim)
    image_size =64
    batch = 25
    rows = 5
    columns = 5

    out_dir = 'imgs'
    generator = decoder
    generator.load_weights(path_decoder)
    encoder.load_weights(path_encoder)

    '''Generator prediction.'''

    #z = np.random.normal(size=(batch,latent_dim))
    z = np.random.uniform(-1.0, 1.0, size=[batch, latent_dim])
    print('z shape', z.shape)
    out = decoder.predict(z)
    print('min', np.min(out))

    image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)

    os.makedirs(out_dir, exist_ok = True)

    for r in range(rows):
        for c in range(columns):
            print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

        #print('predict', out.shape)
        #cv2.imshow('asdfa', (out[i]*127.5+127.5).astype(np.uint8))
    cv2.imshow('out',image_holder)
    cv2.waitKey(0)
    if save_out == True:
        cv2.imwrite(out_dir+'/'+'generator_out_grid.jpg', (image_holder).astype(np.uint8))

    '''Autoencoder prediction.'''

    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:batch]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        #cv2.imshow('temp',img_temp)
        #cv2.waitKey(0)
        batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))

    batch_images = (batch_images-127.5)/127.5
    z = encoder.predict(batch_images)[2]
    print('z: ', z.shape, np.max(z), np.min(z))
    out_vae = decoder.predict(encoder.predict(batch_images)[2])
    input_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    recon_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    #print('max', np.max(out_vae))
    #print('min', np.min (out_vae))
    for r in range(rows):
        for c in range(columns):
            #print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            input_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (batch_images[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)
            recon_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out_vae[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

    cv2.imshow('Input', (input_image_holder).astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow('Output', (recon_image_holder).astype(np.uint8))
    cv2.waitKey(0)

    if save_out == True:
        cv2.imwrite(out_dir+'/'+'autoencoder_input_grid.jpg', (input_image_holder).astype(np.uint8))
        cv2.imwrite(out_dir+'/'+'autoencoder_output_grid.jpg', (recon_image_holder).astype(np.uint8))


def vaegan_actual_predict_docu(weights_path = 'vae_mlp_mnist.h5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 2048, save_out=True):
    encoder, decoder, vae = vaegan_actual_model()
    batch = 25
    out_dir = 'imgs'
    image_size =64
    rows = 5
    columns = 5

    vae.load_weights(weights_path)

    '''Generator prediction.'''

    #z = np.random.normal(size=(batch,latent_dim))
    z = np.random.uniform(-1.0, 1.0, size=[batch, latent_dim])
    #print('z shape', z.shape)
    out = decoder.predict(z)
    #print('min', np.min(out))
    os.makedirs(out_dir, exist_ok = True)
    image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)

    for r in range(rows):
        for c in range(columns):
            #print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

    cv2.imshow('vae_generator_out',image_holder)
    cv2.waitKey(0)
    if save_out == True:
        cv2.imwrite(out_dir+'/'+'vae_generator_out.jpg', (image_holder).astype(np.uint8))


    '''Autoencoder prediction.'''
    '''
    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:batch]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        #cv2.imshow('temp',img_temp)
        #cv2.waitKey(0)
        batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))

    batch_images = batch_images/255.0
    out_vae = vae.predict(batch_images)

    for i in range(batch):
        cv2.imshow('Input', batch_images[i,:,:,:])
        cv2.waitKey(0)
        cv2.imshow('Output', out_vae[i,:,:,:])
        cv2.waitKey(0)
    '''
    '''Autoencoder prediction.'''

    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:batch]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        #cv2.imshow('temp',img_temp)
        #cv2.waitKey(0)
        batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))

    batch_images = (batch_images-127.5)/127.5
    out_vae = vae.predict(batch_images)
    input_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    recon_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    #print('max', np.max(out_vae))
    #print('min', np.min (out_vae))
    for r in range(rows):
        for c in range(columns):
            #print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            input_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (batch_images[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)
            recon_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out_vae[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

    cv2.imshow('Input', (input_image_holder).astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow('Output', (recon_image_holder).astype(np.uint8))
    cv2.waitKey(0)


    if save_out == True:
        cv2.imwrite(out_dir+'/'+'vae_autoencoder_input.jpg', (input_image_holder).astype(np.uint8))
        cv2.imwrite(out_dir+'/'+'vae_autoencoder_output.jpg', (recon_image_holder).astype(np.uint8))

def plot_images(generator, encoder, steps, model_name, latent_size, datapath = '/home/daryl/datasets/img_align_celeba', num_images=25,save_out =False):
    decoder = generator
    out_dir = model_name+'_output_img'
    image_size = 64
    os.makedirs(out_dir, exist_ok=True)
    rows = 5
    columns = 5

    '''Generator prediction.'''

    #z = np.random.normal(size=(batch,latent_dim))
    z = np.random.uniform(-1.0, 1.0, size=[num_images, latent_size])
    print('z shape', z.shape)
    out = decoder.predict(z)
    print('min', np.min(out))

    image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)

    os.makedirs(out_dir, exist_ok = True)

    for r in range(rows):
        for c in range(columns):
            print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

        #print('predict', out.shape)
        #cv2.imshow('asdfa', (out[i]*127.5+127.5).astype(np.uint8))
    #cv2.imshow('out',image_holder)
    #cv2.waitKey(0)
    if save_out == True:
        cv2.imwrite(out_dir+'/'+'generator_out_'+str(steps)+'.jpg', (image_holder).astype(np.uint8))

    '''Autoencoder prediction.'''

    image_list = glob.glob(os.path.join(datapath,'*.jpg'))

    np.random.shuffle(image_list)
    batch_image_list = image_list[:num_images]
    batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
    for i in range(len(batch_image_list)):
        img_temp = cv2.imread(batch_image_list[i])
        # cv2.imshow('temp',img_temp)
        # cv2.waitKey(0)
        batch_images[i, :, :, :] = cv2.resize(img_temp, (image_size, image_size))

    batch_images = (batch_images-127.5)/127.5
    z = encoder.predict(batch_images)[2]
    print('z: ', z.shape, np.max(z), np.min(z))
    out_vae = decoder.predict(encoder.predict(batch_images)[2])
    input_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    recon_image_holder = np.zeros((image_size*5,image_size*5,3), dtype = np.uint8)
    #print('max', np.max(out_vae))
    #print('min', np.min (out_vae))
    for r in range(rows):
        for c in range(columns):
            #print('shape ', image_holder[r:((r+1)*image_size),c:((c+1)*image_size),:].shape)
            input_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (batch_images[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)
            recon_image_holder[r*image_size:((r+1)*image_size),c*image_size:((c+1)*image_size),:] = (out_vae[(r*5) + (c+1) - 1]*127.5+127.5).astype(np.uint8)

    #cv2.imshow('Input', (input_image_holder).astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.imshow('Output', (recon_image_holder).astype(np.uint8))
    #cv2.waitKey(0)

    if save_out == True:
        cv2.imwrite(out_dir+'/'+'autoencoder_input_'+str(steps)+'.jpg', (input_image_holder).astype(np.uint8))
        cv2.imwrite(out_dir+'/'+'autoencoder_output_'+str(steps)+'.jpg', (recon_image_holder).astype(np.uint8))


def main():
    #some_gen = dataloader()
    #a,b = next(some_gen)
    #print('a', type(a))
    #vaegan_actual_train(epochs=20,final_chk='vae_actual_negative_May24.h5', mse_flag=True)
    #'/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-03.hdf5'
    #vaegan_train(epochs=10,final_chk='vae.h5', mse_flag=True)
    #vaegan_actual_predict(weights_path = '/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-negative-10.hdf5',latent_dim= 128,save_out=True)
    #vaegan_actual_predict_docu(weights_path = '/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-negative-10.hdf5',latent_dim= 128,save_out=False)
    #vaegan_actual_predict_docu(weights_path = '/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-negative-May24-20.hdf5',latent_dim= 128,save_out=True)

    #vaegan_predict(weights_path = 'checkpoints/chkpt-01.hdf 5',save_out=False)

    #encoder, decoder, vae = vaegan_model()
    #vae_discriminator_model()
    #vaegan_complete_model()
    vaegan_complete_train(latent_size=128,epochs=6,retrain = True)
    #vaegan_complete_predict()
    #vaegan_complete_predict(path_encoder = 'checkpoints/encoder_chk-vaegan_complete_lessdense_meannll_minusganloss9073.hdf5', path_decoder='checkpoints/decoder_chk-vaegan_complete_lessdense_meannll_minusganloss9073.hdf5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 128, save_out=True)
    #vaegan_complete_predict(path_encoder = 'checkpoints/encoder_chk-vaegan_complete_lessdense_meannll_plusganloss_reducedlrencoder0_44642.hdf5', path_decoder='checkpoints/decoder_chk-vaegan_complete_lessdense_meannll_plusganloss_reducedlrencoder0_44642.hdf5', datapath = '/home/daryl/datasets/img_align_celeba',latent_dim = 128, save_out=True)

if __name__ == '__main__':
    main()
