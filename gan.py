import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import keras.backend as K
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation, Dense, BatchNormalization, Reshape, UpSampling2D
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

def cifar10_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)

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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return mnist_process(x_train), mnist_process(x_test)

latent_dim = 100
zsamples = np.random.normal(size=(10 * 10, latent_dim))
def generator_sampler():
    xpred = generator.predict(zsamples)
    xpred = dim_ordering_unfix(xpred.transpose((0, 2, 3, 1)))
    return xpred.reshape((10, 10) + xpred.shape[1:])


input_shape = (1, 28, 28)
generator = model_generator()
discriminator = model_discriminator(input_shape=input_shape)
gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim, )))
generator.summary()
discriminator.summary()
gan.summary()

path="output/gan_convolutional/"
generator_cb = ImageGridCallback(
    os.path.join(path, "epoch-{:03d}.png"),
    generator_sampler, cmap=None
)

model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights,
                                        discriminator.trainable_weights],
                         player_names=["generator", "discriminator"])

model.adversarial_compile(
    adversarial_optimizer=AdversarialOptimizerSimultaneous(),
    player_optimizers=[Adam(1e-4, decay=1e-4),
                       Adam(1e-3, decay=1e-4)],
    loss="binary_crossentropy")

callbacks = []
callbacks.append(generator_cb)
if K.backend() == "tensorflow":
    callbacks.append(
        TensorBoard(log_dir=os.path.join("output/gan_convolutional/", "logs/"),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True))


xtrain, xtest = cifar10_data()
ytrain = gan_targets(xtrain.shape[0])
ytest = gan_targets(xtest.shape[0])
history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest),
                    callbacks=callbacks, epochs=100,
                    batch_size=64)
df = pd.DataFrame(history.history)
df.to_csv("output/gan_convolutional/history.csv")

models = {}
models["generator"] = generator
models["discriminator"] = discriminator

score=(None, None)
