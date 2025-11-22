import numpy as np
import cv2

from pathlib import Path
from PIL import Image

def load_image(image_path):
    """Load image from path"""
    return cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)

def write_image(image_path, image):
    """Write image to path"""
    cv2.imwrite(str(image_path), image)
    return

def resize_image(image_path : Path, resize_factor : float, output_path : str = None):
    image = load_image(image_path)
    H, W = image.shape[:2]
    H_new = int(H  / resize_factor)
    W_new = int(W  / resize_factor)
    image = resize_image_using_pil_lib(image,  H_new, W_new)
    image_path = image_path if output_path is None else Path(output_path)
    write_image(str(image_path), image)
    height, width = image.shape[:2]
    return str(image_path), height, width

def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object, keep_ratio= True) -> np.ndarray:
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
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r
