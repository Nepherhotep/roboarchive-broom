import argparse
import cv2
import os

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
    img = img[2100:, :]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_w = grayscale.sum(axis=0)
    hist_h = grayscale.sum(axis=1)

    quarter_width = int(len(hist_w) / 4)
    quarter_height = int(len(hist_h) / 4)

    x1 = np.argmin(hist_w[:quarter_width])
    x2 = np.argmin(hist_w[len(hist_w) - quarter_width:]) + 3 * quarter_width

    y1 = np.argmin(hist_h[:quarter_height])
    y2 = np.argmin(hist_h[len(hist_h) - quarter_height:]) + 3 * quarter_height

    print(x1, y1, x2, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    display(img)


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
