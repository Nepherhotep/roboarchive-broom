#!/usr/bin/env python3
import argparse
import functools
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import cv2
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


def configure_backend(args):
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    if args.cpu:
        config.device_count['GPU'] = 0
    sess = tf.Session(config=config)
    K.set_session(sess)


def train(args):
    configure_backend(args)

    np.random.seed(123)  # for reproducibility

    x_train, y_train = load_data('samples-raw', 'samples-clean')

    print('Creating CNN')

    cnn = get_cnn(args)
    model_checkpoint = ModelCheckpoint(
        args.weights_file, monitor='acc', verbose=1, save_best_only=args.best, period=args.period
    )
    cnn.model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[model_checkpoint],
    )


def add_common_arguments(parser):
    parser.add_argument(
        '-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5'
    )
    parser.add_argument(
        '-c', '--cnn', dest='cnn_name', choices=['simple', 'unet'], help='CNN', required=True
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-b', '--batch-size', default=4, type=int)


def display(*images):
    for image in images:
        if len(image.shape) == 4:
            image = image[0, :, :, 0]
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--period', default=2, type=int)
    parser.add_argument('-e', '--epochs', default=5000, type=int)

    args = parser.parse_args()

    train(args)
