#!/usr/bin/env python3
from train_cnn import parse_args, DataSource
from utils import display


class FakeModel:
    def __init__(self, args):
        self.input_size = (args.tile_size, args.tile_size)

    def input_img_to_cnn(self, tile, tile_size):
        tile = tile.astype('float32')
        tile = tile.reshape((tile_size, tile_size, 1))
        tile /= 255
        return tile


def main():
    args = parse_args()
    data = DataSource(args, FakeModel(args))
    dg = data.data_generator()
    vdg = data.validation_generator()

    print(data.ideal_steps)

    while True:
        display(*next(dg))
        # display(*next(vdg))

if __name__ == '__main__':
    main()
