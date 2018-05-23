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

    chkpath="/home/daryl/EE298Z/vaegan/checkpoints/chkpt-actual-{epoch:02d}.hdf5"
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




def main():
    #some_gen = dataloader()
    #a,b = next(some_gen)
    #print('a', type(a))
    vaegan_actual_train(epochs=10,final_chk='vae_actual.h5', mse_flag=True)
    #vaegan_train(epochs=10,final_chk='vae.h5', mse_flag=True)
    #vaegan_predict(weights_path = 'checkpoints/chkpt-01.hdf 5',save_out=False)
    #vaegan_predict(weights_path = 'checkpoints/chkpt-01.hdf 5',save_out=False)

    #encoder, decoder, vae = vaegan_model()
    #vae_discriminator_model()

if __name__ == '__main__':
    main()
