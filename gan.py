import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import keras.backend as K
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import mnist

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
# from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils
from make_tensorboard import make_tensorboard
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(100, ), activation="tanh"))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((7, 7, 128), input_shape=(7 * 7 * 128, )))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     input_shape=(28, 28, 1),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    return model

adversal_model = AdversarialModel(base_model=M,
                                  player_params=[generator.trainable_weights,
                                                 discriminator.trainable_weights],
                                  player_names=["generator", "discriminator"])
adversal_model = AdversarialModel(player_models=[gan_g, gan_d],
                                  player_params=[generator.trainable_weights,
                                                 discriminator.trainable_weights],
                                  player_names=["generator", "discriminator"])
mpl.use("Agg")

def gan_targets(n):
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_fake = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]

def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14)(g_input)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(int(nch / 4), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(1, (1, 1), padding="same")(H)
    g_V = Activation("sigmoid")(H)
    return Model(g_input, g_V)

def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    H = Conv2D(int(nch / 2), (5, 5),
               strides=(2, 2),
               padding="same",
               activation="relu",
    )(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(nch, (5, 5),
               strides=(2, 2),
               padding="same",
               activation="relu",
    )(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation="sigmoid")(H)
    return Model(d_input, d_V)

def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def mnist_data():
    (x_train, y_train), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

latent_dim = 100
input_shape = (1, 28, 28)
generator = model_generator()
discriminator = model_discriminator(input_shape=input_shape)
gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim, )))
generator.summary()
discriminator.summary()
gan.summary()

model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights,
                                        discriminator.trainable_weights],
                         player_names=["generator", "discriminator"])

model.adversarial_compile(
    adversarial_optimizer=AdversarialOptimizerSimultaneous(),
    player_optimizers=[Adam(1e-4, decay=1e-4),
                       Adam(1e-3, decay=1e-4)]
)

generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",
                                 generator_sampler(latent_dim,
                                                   generator))
callbacks = [generator_cb]
if K.backend() == "tensorflow":
    callbacks.append(
        TensorBoard(log_dir=os.path.join("output/gan_convolutional/", "logs/"),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True))
xtrain, xtest = mnist_data()
xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))
y = gan_targets(xtrain.shape[0])
history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest),
                    callbacks=[generator_cb], epochs=100,
                    batch_size=32)
df = pd.DataFrame(history.history)
df.to_csv("output/gan_convolutional/history.csv")
generator.save("output/gan_convolutional/generator.h5")
discriminator.save("output/gan_convolutional/discriminator.h5")
models = {}
models["generator"] = generator
models["discriminator"] = discriminator
