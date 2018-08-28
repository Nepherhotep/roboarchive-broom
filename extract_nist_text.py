import argparse
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='train/gen')
    return parser.parse_args()


def extract_text(args, input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    


def main(args):
    d = os.path.join(args.data_dir, 'nist_orig')

    for f in os.listdir(d)[:1]:
        input_path = os.path.join(args.data_dir, 'nist_orig', f)
        output_path = os.path.join(args.data_dir, 'text_extracted', f)

        extract_text(args, input_path, output_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)
