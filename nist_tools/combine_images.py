import os
import random

import nist_tools
from nist_tools.extract_nist_text import BaseMain, parse_args, display

import cv2


class CombineMain(BaseMain):
    SRC_DIR = 'blurred'
    DST_DIR = 'combined'
    BG_DIR = 'backgrounds'

    def __init__(self):
        self.backgrounds = os.listdir(os.path.join(args.data_dir, self.BG_DIR))
        self.backgrounds.sort()

    def get_random_bg(self):
        filename = random.choice(self.backgrounds)
        return os.path.join(args.data_dir, self.BG_DIR, filename)

    def main(self, args):
        lst = self.get_sorted_files(args)

        a = lst[::2]
        b = lst[1::2]

        pairs = list(zip(a, b))

        skipped = 0
        for i, pair in enumerate(pairs):
            a_path = os.path.join(args.data_dir, self.SRC_DIR, pair[0])
            b_path = os.path.join(args.data_dir, self.SRC_DIR, pair[1])
            bg_path = self.get_random_bg()
            output_path = os.path.join(args.data_dir, self.DST_DIR, 'combined-{}.png'.format(i))

            print('Processing {}/{}, omitted {}'.format(i, len(pairs), skipped))

            result = self.process_file(args, a_path, b_path, bg_path, output_path)

            if not result:
                skipped += 1

    def random_bool(self):
        return random.choice([True, False])

    def process_file(self, args, a_path, b_path, bg_path, output_path):
        a_img = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        b_img = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)

        bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)

        # random horizontal flip
        if self.random_bool():
            bg_img = cv2.flip(bg_img, 0)

        # random vertical flip
        if self.random_bool():
            bg_img = cv2.flip(bg_img, 1)

        cv2.imwrite(output_path, bg_img)


if __name__ == '__main__':
    random.seed(123)
    args = parse_args()
    CombineMain().main(args)
    print('done')
