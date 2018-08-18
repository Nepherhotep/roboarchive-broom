#!/usr/bin/env python3
import argparse

import numpy as np

import cv2
from cnn import get_cnn
from split_image import slice_tile
from train_cnn import add_common_arguments, configure_backend, display


class FileProcessor:
    def process(
        self,
        args,
        cnn_name,
        weights_file,
        input_file,
        output_file,
        scale_to_width=1024,
        tile_size=32,
        padding=16,
        bg_color=0,
    ):
        """
        Scale image to width 1024, convert to grayscale and than slice by tiles.
        It's possible to slice image with padding and each tile will contain pixels from surrounding tiles
        """
        configure_backend(args)
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

        h, w = img.shape

        width = scale_to_width
        height = int(width * h / w)
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        output_img = np.zeros(img.shape)

        cnn = get_cnn(args)

        i = 0
        j = 0

        while tile_size * (i * 1) < width:
            while tile_size * (j + 1) < height:
                tile = slice_tile(img, i, j, tile_size, padding, bg_color=bg_color)

                # convert to CNN format
                cnn_tile = cnn.input_img_to_cnn(tile, tile_size, padding)

                # process output
                print('processing tile {}, {}'.format(i, j))
                # TODO: fix this, we should be able to batch processing
                out_arr = cnn.process_tile(cnn_tile)

                # convert to img format
                out_tile = cnn.cnn_output_to_img(out_arr, tile_size)

                output_img[
                    j * tile_size : (j + 1) * tile_size, i * tile_size : (i + 1) * tile_size
                ] = out_tile

                j += 1
            i += 1
            j = 0

        cv2.imwrite(output_file, output_img)
        display(output_img)


def fake_processing(tile):
    return tile[16:48, 16:48].flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        '-i', '--input-file', dest='input_file', help='Input file to process', required=True
    )

    parser.add_argument(
        '-o',
        '--output-file',
        dest='output_file',
        help='Processed output file',
        default='output.png',
    )
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-b', '--batch-size', default=4, type=int)

    args = parser.parse_args()

    p = FileProcessor()
    p.process(args, args.cnn_name, args.weights_file, args.input_file, args.output_file)
