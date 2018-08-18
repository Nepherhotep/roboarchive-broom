from .simple_cnn import SimpleCNN
from .unet import UnetCNN


def get_cnn(name):
    if name == 'simple':
        return SimpleCNN()
    elif name == 'unet':
        return UnetCNN()
    else:
        raise Exception('unknown name {}'.format(name))
