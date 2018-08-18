from .simple_cnn import SimpleCNN
from .unet import UnetCNN


def get_cnn(args):
    if args.cnn_name == 'simple':
        return SimpleCNN(args)
    elif args.cnn_name == 'unet':
        return UnetCNN(args)
    else:
        raise Exception('unknown name {}'.format(args.cnn_name))
