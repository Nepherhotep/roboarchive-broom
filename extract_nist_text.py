import argparse
import cv2
import os
import math

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='train/gen')
    return parser.parse_args()


def display(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def extract_text(args, input_path, output_path):
    img = cv2.imread(input_path)

    assert img is not None, 'Image {} could not be read'.format(input_path)

    img = img[2100:, :]

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = (255 - grayscale)

    im2, contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    print(cnt)
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    print(approx)
    cv2.drawContours(img, [cnt], 0, (255, 0, 0), 3)

    display(img)
    # cv2.imwrite(output_path, img)


def main(args):
    d = os.path.join(args.data_dir, 'nist_orig')

    lst = [f for f in os.listdir(d) if (f.endswith('.png') or f.endswith('.jpg'))]
    lst.sort()

    for i, f in enumerate(lst[6:7]):
        input_path = os.path.join(args.data_dir, 'nist_orig', f)
        output_path = os.path.join(args.data_dir, 'text_extracted', f)

        print('Processing {}/{}'.format(i, len(lst)))
        extract_text(args, input_path, output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done')
