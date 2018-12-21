import os
import numpy as np
import keras
import boto3
from pathlib import Path
from datetime import datetime
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
# from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils
from make_tensorboard import make_tensorboard
from keras.callbacks import ModelCheckpoint, TensorBoard

image_width = 32
start = datetime.now()

def network(input_shape, num_classes):
    model = Sequential()
    l = Conv2D(32,
               kernel_size=3,
               padding="same",
               input_shape=input_shape,
               activation="relu")
    model.add(l)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model

class CIFAR10Dataset():
    def __init__(self):
        self.image_shape = (32,32,3)
        self.num_classes = 10

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True)
                           for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0],) + self.image_shape
            data - data.reshape(shape)
        return data

        
class Trainer():
    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss,
                             optimizer=optimizer,
                             metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logdir")
        self.model_file_name = "model_file.hdf5"
        
    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        self._target.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_split=validation_split,
                         callbacks=[
                             TensorBoard(log_dir=self.log_dir),
                             ModelCheckpoint(os.path.join(self.log_dir,
                                                          self.model_file_name),
                                             save_best_only=True)
                         ],
                         verbose=self.verbose)

dataset = CIFAR10Dataset()

model = network(dataset.image_shape, dataset.num_classes)
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss="categorical_crossentropy",
                  optimizer=RMSprop())
trainer.train(x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)

end = datetime.now()

model_path = "model.h5"
model.save(model_path)
result_path = "result.txt".format(datetime.now())
with open(result_path, "w") as f:
    f.write("Test loss:{}".format(score[0]))
    f.write("Test accuracy:{}".format(score[1]))
    f.write("Run time: {}".format(end - start).seconds)
s3 = boto3.resource("s3")
bucket_name = "machine-learning"
bucket = s3.Bucket(bucket_name)
date = datetime.now()
upload_dir = date.strftime("%y-%m-%d-%H-%M-%S"
bucket.upload(model_path,"{}/{}".format(upload_dir, model_path)))
bucket.upload(result_path, "{}/{}".format(upload_dir, result_path))
Path(model_path).unlink()
Path(result_path).unlink()
