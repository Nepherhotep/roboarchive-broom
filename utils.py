import matplotlib.cm as cm
import matplotlib.pyplot as plt


def display(*images):
    for image in images:
        if len(image.shape) == 4:
            image = image[0, :, :, 0]
        if len(image.shape) == 3:
            image = image[:, :, 0]
        plt.imshow(image, cmap=cm.gray)
    plt.show()
