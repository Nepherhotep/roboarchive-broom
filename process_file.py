import argparse
import cv2

from cnn import get_model


def process_file(weights_file, input_file, output_file):
    model = get_model()
    model.load_weights(weights_file)

    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input file to process', required=True)
    parser.add_argument('-o', '--output-file', dest='output_file', help='Processed output file', default='output.png')
    parser.add_argument('-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5')

    args = parser.parse_args()

    process_file(args.weights_file, args.input_file, args.output_file)
