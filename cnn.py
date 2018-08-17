from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np


class BaseCNN:
    def __init__(self):
        self.model = None

    def process_tile(self, tile):
        tiles = np.zeros((1,) + tile.shape)
        tiles[0] = tile
        return self.model.predict(tiles)

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)


class SimpleCNN:
    def __init__(self):
        model = Sequential()

        model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
        model.add(Flatten())
        model.add(Dense(1024))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.model = model


class UnetCNN(BaseCNN):
    pass


def get_cnn(name):
    if name == 'simple':
        return SimpleCNN()
    elif name == 'unite':
        return UnetCNN()
    else:
        raise Exception('unknown name {}'.format(name))