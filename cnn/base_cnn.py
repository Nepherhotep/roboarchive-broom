from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D

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

    def input_img_to_cnn(self, tile, tile_size, padding):
        tile = tile.astype('float32')
        tile = tile.reshape((tile_size + 2 * padding, tile_size + 2 * padding, 1))
        tile /= 255
        return tile

    def cnn_output_to_img(self, arr, tile_size):
        tile = arr.reshape((tile_size, tile_size))
        tile *= 255
        tile = tile.clip(0, 255)
        tile = tile.astype(np.uint8)
        return tile
