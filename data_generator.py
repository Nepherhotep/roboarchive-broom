import os
import random
from glob import glob
from itertools import chain, cycle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import cv2
from split_image import slice_tile

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


def load_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def split_image(img, model, tile_size):
    height, width = img.shape
    i, j = 0, 0
    while tile_size * (i * 1) < (width + tile_size):
        while tile_size * (j + 1) < (height + tile_size):
            tile, orig_size = slice_tile(img, i, j, tile_size, 0, bg_color=255)
            if not orig_size[0] or not orig_size[1]:
                j += 1
                continue
            # convert to CNN format
            cnn_tile = model.input_img_to_cnn(tile, tile_size)
            yield cnn_tile
            j += 1
        i += 1
        j = 0


class SplitTileLoader(XTileLoader):
    def split_image(self, path):
        tile_size = self.tile_size
        print(f'Load: {path}')
        img = load_img(path)
        yield from split_image(img, self.cnn, tile_size)

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


dg = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # brightness_range=(-0.3, 0.3),
    horizontal_flip=False,
    vertical_flip=False,
)


class ImageWrapper:
    def __init__(self, path, args, model, trans=None, validation=True):
        self.path = path
        self.args = args
        self.model = model
        self.ready = False
        self.trans = trans
        self.validation = validation
        image = load_img(path)
        self.original_shape = image.shape
        image = self.post_load(image)
        self.all_tiles = list(split_image(image, model, args.tile_size))
        self.len_tiles = len(self.all_tiles)

    def get_transformated(self, trans):
        return ImageWrapper(self.path, self.args, self.model, trans, validation=False)

    def conv_to(self, img):
        img = img.astype('float32')
        img = img.reshape((*self.original_shape, 1))
        img /= 255
        return img

    def conv_from(self, img):
        img = img.reshape(self.original_shape)
        img *= 255
        img = img.clip(0, 255)
        img = img.astype(np.uint8)
        return img

    def post_load(self, image):
        if self.trans:
            return self.conv_from(dg.apply_transform(self.conv_to(image), self.trans))
        return image

    def shuffle(self, order):
        self.all_tiles = [y for x, y in sorted(zip(order, self.all_tiles))]
        self.validation_tiles = []
        assert len(self.all_tiles) > 1
        if self.validation:
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

    def custom_data_generator(self):
        return cycle(self.all_tiles)

    def debug(self, tile, count):
        if self.args.display:
            print(f'Show: {self.path} {count}')
            # display(tile)

    def get_generation(self):
        return SplitTileLoader(self.args, self.model, None).split_image(self.path)


class ImagePair:
    def __init__(self, args, model, src, dst, should_trans=False):
        assert os.path.basename(src) == os.path.basename(dst)
        self.src_path = src
        self.dst_path = dst
        self.args = args
        self.model = model
        self.src = ImageWrapper(src, args, model)
        self.dst = ImageWrapper(dst, args, model)
        if should_trans:
            td = dg.get_random_transform(self.src.original_shape)
            self.src = self.src.get_transformated(td)
            self.dst = self.dst.get_transformated(td)
        self.shuffle(self.src, self.dst)

    def shuffle(self, x, y):
        order = list(range(x.len_tiles))
        random.shuffle(order)
        x.shuffle(order), y.shuffle(order)

    def next_data(self):
        return self.src.next_data(), self.dst.next_data()

    def next_validation_data(self):
        return self.src.next_validation_data(), self.dst.next_validation_data()

    def transformated(self):
        return ImagePair(self.args, self.model, self.src_path, self.dst_path, should_trans=True)


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

    @property
    def ideal_steps(self):
        tiles = 0
        for img in self.pure_images:
            tiles += len(img.src.all_tiles)
        return int(tiles / self.args.batch_size / self.args.transformated) + 1

    def fill_pure_images(self):
        for x, y in self.file_names():
            self.pure_images.append(ImagePair(self.args, self.model, x, y))

    def generate_images(self):
        if self.args.no_generated:
            return
        print('Generate new images')
        self.generated_images = [x.transformated() for x in self.pure_images]

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

    def trans_data_generator(self):
        x, y = (
            Batch(self.args.batch_size, self.model.input_size),
            Batch(self.args.batch_size, self.model.input_size),
        )
        while True:
            for img in chain(self.generated_images):
                x_tile, y_tile = img.next_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)

                x_tile, y_tile = img.next_data()
                x.append(x_tile), y.append(y_tile)
                if x.is_ready:
                    yield x.get_data(reset=True), y.get_data(reset=True)
            self.generate_images()
