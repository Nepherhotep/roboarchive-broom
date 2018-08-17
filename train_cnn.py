import argparse
import matplotlib.pyplot as plt
import functools

import cv2
import os

import h5py
import numpy as np

from cnn import get_cnn


class XTileLoader:
    """
    Load square tiles with channel
    """
    def __init__(self, tiles_dir, tile_size):
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size

    def get_shape(self):
        return (self.tile_size, self.tile_size, 1)

    def load(self):
        tiles_list = sorted(os.listdir(self.tiles_dir))
        shape = (len(tiles_list),) + self.get_shape()
        train_data = np.zeros(shape)
        for i, fname in enumerate(tiles_list):
            path = os.path.join(self.tiles_dir, fname)
            tile = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            tile = tile.reshape(self.get_shape())
            train_data[i] = tile

        train_data = train_data.astype('float32')
        train_data /= 255

        return train_data


class YTileLoader(XTileLoader):
    """
    Load tile as flat array without channel
    """
    def get_shape(self):
        return (self.tile_size * self.tile_size,)


def load_data(x_path, y_path):
    """
    Check raw/clean (X/Y) data consistency and load data array
    """
    raw_files = sorted(os.listdir(x_path))
    clean_files = sorted(os.listdir(y_path))

    assert raw_files == clean_files, 'X/Y files are not the same'

    print('Loading x train data')
    x_train = XTileLoader(x_path, 64).load()

    print('Loading y train data')
    y_train = YTileLoader(y_path, 32).load()

    return x_train, y_train


def train(cnn_name, weights_file):
    np.random.seed(123)  # for reproducibility

    x_train, y_train = load_data('samples-raw', 'samples-clean')

    print('Creating CNN')

    cnn = get_cnn(cnn_name)
    cnn.model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)

    if weights_file:
        print('Saving weights')
        cnn.save(weights_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5')
    parser.add_argument('-c', '--cnn', dest='cnn_name', choices=['simple', 'unet'], help='CNN', required=True)

    args = parser.parse_args()

    train(args.cnn_name, weights_file=args.weights_file)


