

import numpy as np
#import cupy as np
from PIL import Image
import cv2

from lib.geometry import Line


def compute_intersection_with_block_boundaries( p1, p2, img):
    # Get the image dimensions
    if img.ndim == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    a, b, c = Line.compute_line_coefficients(p1, p2)
    if b == 0 and np.abs(a) > 0:
        # Vertical line
        x = int(p2[0])
        x1, y1 = x, 0
        x2, y2 = x, height - 1
        return x1, y1, x2, y2, None, None, None, None
    if a == 0 and np.abs(b) > 0:
        # Horizontal line
        y = int(p2[1])
        x1, y1 = 0, y
        x2, y2 = width - 1, y
        return x1, y1, x2, y2, None, None, None, None
    x1, y1 = 0, None
    x2, y2 = width - 1, None
    x3, y3 = None, 0
    x4, y4 = None, height - 1
    x1, y1 = Line.get_line_coordinates(a, b, c, x1, y1)
    x2, y2 = Line.get_line_coordinates(a, b, c, x2, y2)
    x3, y3 = Line.get_line_coordinates(a, b, c, x3, y3)
    x4, y4 = Line.get_line_coordinates(a, b, c, x4, y4)

    return x1, y1, x2, y2, x3, y3, x4, y4

class Color:
    """BGR"""
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    dark_yellow = (0, 204, 204)
    cyan = (255, 255, 0)
    orange = (0, 165, 255)
    purple = (255, 0, 255)
    maroon = (34, 34, 178)
    green = (0, 255, 0)
    white = (255,255,255)
    black = (0,0,0)
    gray_white = 255
    gray_black = 0

    def __init__(self):
        self.list = [Color.yellow, Color.red,Color.blue, Color.dark_yellow, Color.cyan,Color.orange,Color.purple,Color.maroon]
        self.idx = 0

    def get_next_color(self):
        self.idx = (self.idx + 1 ) % len(self.list)
        return self.list[self.idx]


class Drawing:

    @staticmethod
    def rectangle(image, top_left_point, bottom_right_point, color=Color.black, thickness=2):
        # Define the rectangle coordinates
        x1, y1 = top_left_point
        x2, y2 = bottom_right_point

        # Draw the rectangle on the image
        return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color,
                      thickness)  # (0, 255, 0) is the color in BGR format, and 2 is the line thickness

    @staticmethod
    def circle(image, center_coordinates,thickness=-1, color=Color.black, radius=3):
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, tuple(center_coordinates), radius, color, thickness)
        return image

    @staticmethod
    def put_text(text, image, org, color = (0, 0, 0), fontScale = 1 / 4):
        # font
        font = cv2.FONT_HERSHEY_DUPLEX
        # fontScale

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        return image

    @staticmethod
    def intersection(dot, img, color=Color.red):
        img[int(dot.y),int(dot.x),:] = color

        return img

    @staticmethod
    def curve(curva, img, color=(0, 255, 0), thickness = 1):
        y, x = curva.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x,y)).T
        isClosed=False
        img = cv2.polylines(img, [pts],
                              isClosed, color, thickness)

        return img

    @staticmethod
    def chain(chain, img, color=(0, 255, 0), thickness=5):
        y, x = chain.get_nodes_coordinates()
        pts = np.vstack((y, x)).T.astype(int)
        isClosed = False
        img = cv2.polylines(img, [pts],
                            isClosed, color, thickness)

        return img

    @staticmethod
    def radii(rayo, img, color=(255, 0, 0), debug=False, thickness=1):
        y, x = rayo.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        start_point = (x[0], y[0])
        end_point = (x[1], y[1])
        image = cv2.line(img, start_point, end_point, color, thickness)

        return image

    @staticmethod
    def draw_lsd_lines(lines, img, output_path, lines_all=None, thickness=3):
        """Draw m_lsd from matrix m_lsd in image img using opencv"""
        # lsd = cv2.createLineSegmentDetector(0)
        drawn_img = img.copy()
        if lines_all is not None:
            for line in lines_all:
                x1, y1, x2, y2 = line.ravel()
                cv2.line(drawn_img, (int(x1), int(y1)), (int(x2), int(y2)), Color.red, thickness)
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            cv2.line(drawn_img, (int(x1), int(y1)), (int(x2), int(y2)), Color.blue, thickness)

        # drawn_img = lsd.drawSegments(img.copy(), m_lsd)
        cv2.imwrite(output_path, resize_image_using_pil_lib(drawn_img, 640, 640))
        return

    @staticmethod
    def draw_lines(lines, img, output_path, lines_all=None, thickness=1):
        """Draw m_lsd from matrix m_lsd in image img using opencv"""
        # lsd = cv2.createLineSegmentDetector(0)
        thickness = 1
        drawn_img = img.copy()
        h,w,_ = drawn_img.shape
        drawn_img = np.zeros_like(drawn_img)
        if lines_all is not None:
            for line in lines_all:
                x1, y1, x2, y2 = line.ravel()
                x1, y1, x2, y2, x3, y3, x4, y4 = compute_intersection_with_block_boundaries(np.array((x1, y1)),np.array((x2, y2)), img)
                # Draw the line on the image
                if x1 is None or x2 is None or x3 is None or x4 is None or y1 is None or y2 is None or y3 is None or y4 is None:
                    continue
                if (x1<0 or x2<0 or x3<0 or x4<0 or y1<0 or y2<0 or y3<0 or y4<0 or
                    x1>=w or x2 >=w or x3>=w or x4>=w or y1>=h or y2>=h or y3>=h or y4>=h):
                    continue
                drawn_img = Line.draw_line(x1, y1, x2, y2, drawn_img, thickness, Color.white)
                #cv2.line(drawn_img, (int(x1), int(y1)), (int(x2), int(y2)), Color.red, thickness)
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            x1, y1, x2, y2, x3, y3, x4, y4 = compute_intersection_with_block_boundaries(np.array((x1, y1)),np.array((x2, y2)), img)
            if x1 is None or x2 is None or x3 is None or x4 is None or y1 is None or y2 is None or y3 is None or y4 is None:
                continue
            if (x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0 or y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0 or
                    x1 >= w or x2 >= w or x3 >= w or x4 >= w or y1 >= h or y2 >= h or y3 >= h or y4 >= h):
                continue
            # Draw the line on the image
            drawn_img = Line.draw_line(x1, y1, x2, y2, drawn_img, thickness, Color.white)
            #cv2.line(drawn_img, (int(x1), int(y1)), (int(x2), int(y2)), Color.blue, thickness)

        drawn_img = ((drawn_img / drawn_img.max())*255)
        drawn_img = np.where(drawn_img > 255, 255,
                                                 drawn_img).astype(np.uint8)


        # drawn_img = lsd.drawSegments(img.copy(), m_lsd)
        cv2.imwrite(output_path, resize_image_using_pil_lib(drawn_img, 640, 640))
        return

    @staticmethod
    def draw_cross(img, center, color=Color.red, size=10, thickness=1):
        x, y = center
        img = cv2.line(img, (x - size, y), (x + size, y), color, thickness)
        img = cv2.line(img, (x, y - size), (x, y + size), color, thickness)
        return img

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
    flag = Image.ANTIALIAS
    # flag = Image.Resampling.LANCZOS
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r

def change_background_to_value(im_in, mask, value=255):
    """
    Change background intensity to white.
    @param im_in:
    @param mask:
    @return:
    """
    im_in[mask > 0] = value

    return im_in


def rgb2gray(img_r):
    return cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

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
    clahe = cv2.createCLAHE(clipLimit=10)
    img_eq = clahe.apply(img_eq)
    return img_eq

def equalize(im_g):
    """
    Equalize image using CLAHE algorithm. Implements Algorithm 4 in the paper
    @param im_g: gray scale image
    @return: equalized image
    """
    # equalize image
    im_pre, mask = change_background_intensity_to_mean(im_g)
    im_pre = equalize_image_using_clahe(im_pre)
    im_pre = change_background_to_value(im_pre, mask, Color.gray_white)
    return im_pre