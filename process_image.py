import argparse
import numpy as np
from matplotlib import pyplot as plt

import cv2

from cnn import get_model
from split_image import slice_tile


def input_img_to_cnn(tile, tile_size, padding):
    tile = tile.astype('float32')
    tile = tile.reshape((tile_size + 2 * padding, tile_size + 2 * padding, 1))
    tile /= 255
    return tile


def cnn_output_to_img(arr, tile_size):
    tile = arr.reshape((tile_size, tile_size))
    tile *= 255
    tile = tile.clip(0, 255)
    tile = tile.astype(np.uint8)
    return tile


def process_file(weights_file, input_file, output_file, scale_to_width=1024, tile_size=32, padding=16, bg_color=0):
    """
    Scale image to width 1024, convert to grayscale and than slice by tiles.
    It's possible to slice image with padding and each tile will contain pixels from surrounding tiles
    """
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    width = scale_to_width
    height = int(width * h / w)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

    output_img = np.zeros(img.shape)
    cnn = CNN(weights_file)

    i = 0
    j = 0

    while tile_size * (i * 1) < width:
        while tile_size * (j + 1) < height:
            tile = slice_tile(img, i, j, tile_size, padding, bg_color=bg_color)

            # convert to CNN format
            cnn_tile = input_img_to_cnn(tile, tile_size, padding)

            # process output
            print('processing tile {}, {}'.format(i, j))
            out_arr = cnn.process_tile(cnn_tile)

            # convert to img format
            out_tile = cnn_output_to_img(out_arr, tile_size)

            output_img[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size] = out_tile

            j += 1
        i += 1
        j = 0

    cv2.imwrite(output_file, output_img)


def fake_processing(tile):
    return tile[16:48, 16:48].flatten()


class CNN:
    def __init__(self, weight_file):
        self.model = get_model()
        self.model.load_weights(weight_file)

    def process_tile(self, tile):
        tiles = np.zeros((1,) + tile.shape)
        tiles[0] = tile
        return self.model.predict(tiles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input file to process', required=True)
    parser.add_argument('-o', '--output-file', dest='output_file', help='Processed output file', default='output.png')
    parser.add_argument('-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5')

    args = parser.parse_args()

    process_file(args.weights_file, args.input_file, args.output_file)
