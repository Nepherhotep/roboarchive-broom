import functools

import cv2
import os

import h5py
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


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


def cache_train_data(filename):
    """
    Cache loading images into hd5f file to speedup loading
    """
    def deco(fun):
        def new_func(*args, **kwargs):
            if os.path.exists(filename):
                h5f = h5py.File(filename, 'r')
                x_train = h5f['x_train'][:]
                y_train = h5f['y_train'][:]
                h5f.close()
                return x_train, y_train
            else:
                x_train, y_train = fun(*args, **kwargs)

                h5f = h5py.File(filename, 'w')
                h5f.create_dataset('x_train', data=x_train)
                h5f.create_dataset('y_train', data=y_train)
                h5f.close()
                return x_train, y_train
        return new_func
    return deco


@cache_train_data('train_data.hdf5')
def load_data(x_path, y_path):
    """
    Check raw/clean (X/Y) data consistency and load data array
    """
    raw_files = sorted(os.listdir(x_path))
    clean_files = sorted(os.listdir(y_path))

    assert raw_files == clean_files, 'X/Y files are not the same'

    print('Loading y train data')
    y_train = YTileLoader(y_path, 32).load()

    print('Loading x train data')
    x_train = XTileLoader(x_path, 64).load()

    print(y_train)
    # print(x_train)

    return x_train, y_train


def main():
    np.random.seed(123)  # for reproducibility

    x_train, y_train = load_data('samples-raw', 'samples-clean')

    print('Creating CNN')
    model = Sequential()

    model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    # 8. Compile model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])

    # 9. Fit model on training data
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)


if __name__ == '__main__':
    main()
