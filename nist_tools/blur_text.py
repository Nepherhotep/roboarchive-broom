from .extract_nist_text import BaseMain, parse_args


class BlurMain(BaseMain):
    SRC_DIR = 'text_extracted'
    DST_DIR = ''

    def process_file(self, args, input_path, output_path):
        pass


if __name__ == '__main__':
    args = parse_args()
    BlurMain().main(args)
    print('done')
