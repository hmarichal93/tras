"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import cv2
from PIL import Image
import numpy as np


WHITE=255
NONE=0
def get_image_shape(im_in: np.array):
    """
    Get image shape
    @param im_in: input image
    @return: heigh and width of the image
    """
    if im_in.ndim > 2:
        height, width, _ = im_in.shape
    else:
        height, width = im_in.shape
    return height, width


def resize(im_in: np.array, height_output, width_output, cy=1, cx=1):
    """
    Resize image and keep the center of the image in the same position.
    @param im_in: Gray image to resize.
    @param height_output: output image height_output. If None, the image is not resized
    @param width_output: output image width_output. If None, the image is not resized.
    @param cy: y's center coordinate in pixel.
    @param cx: x's center coordinate in pixel.
    @return: 
    """
    img_r = resize_image_using_pil_lib(im_in, height_output, width_output)
    height, width = get_image_shape(im_in)
    cy_output, cx_output = convert_center_coordinate_to_output_coordinate(cy, cx, height, width, height_output,
                                                                          width_output)
    return img_r, cy_output, cx_output


def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    #flag = Image.ANTIALIAS
    flag = Image.Resampling.LANCZOS
    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r


def convert_center_coordinate_to_output_coordinate(cy, cx, height, width, height_output, width_output):
    """
    Convert center coordinate from input image to output image
    @param cy: y's center coordinate in pixel.
    @param cx: x's center coordinate in pixel
    @param height: input image height_output
    @param width: input image width_output
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: resized pith coordinates
    """
    hscale = height_output / height
    wscale = width_output / width

    cy_output = cy * hscale
    cx_output = cx * wscale

    return cy_output, cx_output


def change_background_intensity_to_mean(im_in):
    """
    Change background intensity to mean intensity
    @param im_in: input gray scale image. Background is white (255).
    @param mask: background mask
    @return:
    """
    im_eq = im_in.copy()
    mask = np.where(im_in == 255, 1, 0)
    im_eq = change_background_to_value(im_eq, mask, np.mean(im_in[mask == 0]))
    return im_eq, mask

def equalize_image_using_clahe(img_eq):
    """
    Equalize image using CLAHE algorithm
    :param img_eq: image
    :return: image equalized
    """
    clahe = cv2.createCLAHE(clipLimit=10)
    img_eq = clahe.apply(img_eq)
    return img_eq

def equalize(im_g):
    """
    Equalize image using CLAHE algorithm.
    @param im_g: gray scale image
    @return: equalized image
    """
    # Line 1
    im_pre, mask = change_background_intensity_to_mean(im_g)
    # Line 2
    im_pre = equalize_image_using_clahe(im_pre)
    # Line 3
    im_pre = change_background_to_value(im_pre, mask, WHITE)
    # Line 4
    return im_pre

def change_background_to_value(im_in, mask, value=255):
    """
    Change background intensity to white.
    @param im_in: input image
    @param mask: background mask
    @return:
    """
    im_in[mask > 0] = value

    return im_in


def rgb2gray(img_r):
    return cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)


def preprocessing(im_in, height_output=None, width_output=None, cy=None, cx=None):
    """
    Image preprocessing steps. Following actions are made
    - image resize
    - image is converted to gray scale
    - gray scale image is equalized
    @param im_in: segmented image
    @param height_output: new image img_height
    @param width_output: new image img_width
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - im_pre: equalized image
    - cy: pith y's coordinate after resize
    - cx: pith x's coordinate after resize
    """
    if NONE in [height_output, width_output]:
        im_r, cy_output, cx_output = ( im_in, cy, cx)
    else:
        im_r, cy_output, cx_output = resize( im_in, height_output, width_output, cy, cx)

    im_g = rgb2gray(im_r)

    im_pre = equalize(im_g)
    return im_pre, cy_output, cx_output
