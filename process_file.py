import argparse


def process_file(wieghts_file, input_file, output_file):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5')
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input file to process', required=True)
    parser.add_argument('-o', '--output-file', dest='output_file', help='Processed output file', required=True)

    args = parser.parse_args()

    process_file()