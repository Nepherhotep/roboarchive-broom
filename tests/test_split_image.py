import pytest
import numpy as np

from split_image import slice_tile


@pytest.fixture
def img():
    a = np.zeros((10, 10), np.uint8)
    for i in range(10):
        for j in range(10):
            a[i, j] = i * 10 + j
    return a


def test_slice_tile(img):
    tile, (h, w) = slice_tile(img, 0, 0, 3, 0)

    expected = np.array([[0,  1,  2],
                         [10, 11, 12],
                         [20, 21, 22]], np.uint8)

    assert np.array_equal(tile, expected), tile


def test_slice_tile2(img):
    tile, (h, w) = slice_tile(img, 0, 0, 3, 1)
    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 1, 2, 3],
                         [0, 10, 11, 12, 13],
                         [0, 20, 21, 22, 23],
                         [0, 30, 31, 32, 33]], np.uint8)

    assert np.array_equal(tile, expected), tile
