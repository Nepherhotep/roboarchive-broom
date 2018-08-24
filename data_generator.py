import os
import random
from glob import glob
from itertools import chain, cycle

import numpy as np

import cv2
from split_image import slice_tile
from utils import display

VALIDATION_RATE = 8
SAME_RATE = 7


class XTileLoader:
    """
    Load square tiles with channel
    """

    def __init__(self, args, cnn, tiles_dir):
        self.args = args
        self.cnn = cnn
        self.tiles_dir = tiles_dir
        self.tile_size = args.tile_size

    def get_shape(self):
        return (self.tile_size, self.tile_size, 1)

    def load(self):
        tiles_list = self.file_name()
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

    def file_names(self):
        if not self.args.filter:
            src = os.listdir(self.tiles_dir)
        else:
            src = glob(os.path.join(self.tiles_dir, self.args.filter))
        for fname in src:
            yield os.path.basename(fname)


class SplitTileLoader(XTileLoader):
    def split_image(self, path):
        tile_size = self.tile_size
        print(f'Load: {path}')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        i = 0
        j = 0

        while tile_size * (i * 1) < (width + tile_size):
            while tile_size * (j + 1) < (height + tile_size):
                tile, orig_size = slice_tile(img, i, j, tile_size, 0, bg_color=255)
                if not orig_size[0] or not orig_size[1]:
                    j += 1
                    continue
                # convert to CNN format
                cnn_tile = self.cnn.input_img_to_cnn(tile, tile_size)
                yield cnn_tile
                j += 1
            i += 1
            j = 0

    def load(self):
        for fname in self.file_names():
            yield from self.split_image(os.path.join(self.tiles_dir, fname))


class Batch:
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.imgs = []

    def append(self, img):
        self.imgs.append(img)

    @property
    def is_ready(self):
        return len(self.imgs) >= self.size

    def get_data(self, reset=True):
        assert self.is_ready
        b = np.zeros((self.size, *self.input_size, 1), dtype='float32')
        assert len(b.shape) == 4
        for idx, img in enumerate(self.imgs):
            b[idx] = img
        if reset:
            self.reset()
        return b

    def reset(self):
        self.imgs = []


class ImageWrapper:
    def __init__(self, path, args, model):
        self.path = path
        self.args = args
        self.model = model
        self.ready = False
        self.all_tiles = list(SplitTileLoader(self.args, self.model, None).split_image(self.path))
        self.len_tiles = len(self.all_tiles)

    def shuffle(self, order):
        self.all_tiles = [y for x, y in sorted(zip(order, self.all_tiles))]
        self.validation_tiles = []
        assert len(self.all_tiles) > 1
        num_validation = int(len(self.all_tiles) / VALIDATION_RATE + 1)
        print(f'Add {num_validation} samples as validation samples')
        for i in range(num_validation):
            self.validation_tiles.append(self.all_tiles.pop())

        self.data_generator = cycle(self.all_tiles)
        self.validation_data_generator = cycle(self.all_tiles)
        self.ready = True

    def next_data(self):
        return next(self.data_generator)

    def next_validation_data(self):
        return next(self.validation_data_generator)

    def debug(self, tile, count):
        if self.args.display:
            print(f'Show: {self.path} {count}')
            # display(tile)

    def get_generation(self):
        return SplitTileLoader(self.args, self.model, None).split_image(self.path)


class ImagePair:
    def __init__(self, args, model, src, dst):
        assert os.path.basename(src) == os.path.basename(dst)
        self.src = ImageWrapper(src, args, model)
        self.dst = ImageWrapper(dst, args, model)

        order = list(range(self.src.len_tiles))
        random.shuffle(order)
        self.src.shuffle(order), self.dst.shuffle(order)


    def next_data(self):
        return self.src.next_data(), self.dst.next_data()

    def next_validation_data(self):
        return self.src.next_validation_data(), self.dst.next_validation_data()


class DataSource:
    """
    load file names: (x, y)
    load tiles from file
    optional - apply transformation and return tiles
    get_data_generator - shouldn't return validation tiles
    get_validation_generator - should return same tiles

    [default_images] + [same_images] + [generated_images]
    + each should have validation_data (tiles that aren't used for data generator)
    """

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.src_dir = 'train/raw/samples'
        self.dst_dir = 'train/clean/samples'
        self.pure_images = []
        self.generated_images = []
        self.fill_pure_images()

    def get_dir_file_names(self, directory):
        if not self.args.filter:
            src = sorted(os.listdir(directory))
        else:
            src = [os.path.basename(x) for x in glob(os.path.join(directory, self.args.filter))]
            src = sorted(src)
        return src

    def full_path(self, directory, src):
        return [os.path.join(directory, x) for x in src]

    def file_names(self):
        src = self.get_dir_file_names(self.src_dir)
        dst = self.get_dir_file_names(self.dst_dir)
        assert src == dst
        return zip(self.full_path(self.src_dir, src), self.full_path(self.dst_dir, dst))

    def fill_pure_images(self):
        for x, y in self.file_names():
            self.pure_images.append(ImagePair(self.args, self.model, x, y))

    def generate_images(self):
        pass

    def data_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.pure_images, self.generated_images):
                x_tile, y_tile = img.next_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)
            self.generate_images()

    def validation_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.pure_images, self.generated_images):
                x_tile, y_tile = img.next_validation_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)
