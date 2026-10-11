import numpy as np

from tras.utils import image as image_module

from .util import IMAGE_SHAPE
from .util import get_img_and_data
from .util import make_image
from .util import make_jpeg_bytes


def test_img_b64_to_arr():
    img, _ = get_img_and_data()
    assert img.dtype == np.uint8
    assert img.shape == IMAGE_SHAPE


def test_img_arr_to_b64():
    img_arr = make_image()
    img_b64 = image_module.img_arr_to_b64(img_arr)
    img_arr2 = image_module.img_b64_to_arr(img_b64)
    np.testing.assert_allclose(img_arr, img_arr2)


def test_img_data_to_png_data():
    img_data = make_jpeg_bytes()
    png_data = image_module.img_data_to_png_data(img_data)
    assert isinstance(png_data, bytes)
