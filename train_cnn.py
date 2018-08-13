import cv2
import os

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


class TileLoader:
    def __init__(self, tiles_dir, tile_size):
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size

    def load(self):
        files = sorted(os.listdir(self.tiles_dir))

        train_data = np.zeros((len(files), self.tile_size, self.tile_size, 1))
        for i, fname in enumerate(files):
            train_data[i] = cv2.imread(os.path.join(self.tiles_dir, fname), cv2.IMREAD_GRAYSCALE)\
                .reshape((self.tile_size, self.tile_size, 1))
            train_data = train_data.astype('float32')
            train_data /= 255

        return train_data


def load_data(x_path, y_path):
    """
    Check raw/clean (X/Y) data consistency and load data array
    """
    raw_files = sorted(os.listdir(x_path))
    clean_files = sorted(os.listdir(y_path))

    assert raw_files == clean_files, 'X/Y files are not the same'

    x_loader = TileLoader(x_path, 64)
    y_loader = TileLoader(y_path, 32)
    return x_loader.load(), y_loader.load()


def main():
    np.random.seed(123)  # for reproducibility

    x_train, y_train = load_data('sample1-raw', 'sample1-clean')

    # 7. Define model architecture
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(1, 64, 64)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data
    model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1)


if __name__ == '__main__':
    main()