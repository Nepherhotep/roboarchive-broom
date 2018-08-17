import argparse
import numpy as np

import cv2

from cnn import get_model
from split_image import slice_tile



def process_file(weights_file, input_file, output_file, scale_to_width=1024, tile_size=32, padding=16, bg_color=0):
    """
    Scale image to width 1024, convert to grayscale and than slice by tiles.
    It's possible to slice image with padding and each tile will contain pixels from surrounding tiles
    """
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    width = scale_to_width
    height = int(width * h / w)
    resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    resized = resized

    output_file = np.zeros(resized.shape)


    i = 0
    j = 0

    while tile_size * i < width:
        while tile_size * j < height:
            tile_img = slice_tile(resized, i, j, tile_size, padding, bg_color=bg_color)


            j += 1
        i += 1
        j = 0




def fake_processing(tile):
    return tile[16:48, 16:48]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input file to process', required=True)
    parser.add_argument('-o', '--output-file', dest='output_file', help='Processed output file', default='output.png')
    parser.add_argument('-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5')

    args = parser.parse_args()

    process_file(args.weights_file, args.input_file, args.output_file)
