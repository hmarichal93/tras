import os
import numpy as np
import cv2

from abc import ABC, abstractmethod
from PIL import Image
from shapely.geometry import Polygon

from lib.io import load_json

class UserInterface(ABC):
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

    @abstractmethod
    def interface(self):
        pass


class LabelMeInterface(UserInterface):
    def __init__(self, image_path, output_path, edit=False):
        super().__init__(image_path, output_path)
        self.edit = edit

    def interface(self):
        if self.edit:
            command = f"labelme {self.output_path}"

        else:
            command = f"labelme {self.image_path} -O {self.output_path}  --nodata "

        print(command)
        os.system(command)

    @abstractmethod
    def parse_output(self):
        pass

    @staticmethod
    def load_shapes(output_path):
        try:
            json_content = load_json(output_path)
            l_rings = []
            for ring in json_content['shapes']:
                l_rings.append(Polygon(np.array(ring['points'])[:, [1, 0]].tolist()))

        except FileNotFoundError:
            l_rings = []

        return l_rings



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
    def arrow(image, start_point, end_point, color, thickness=2):
        start_point = start_point.coords.xy
        end_point = end_point.coords.xy
        start = (int(start_point[0][0]), int(start_point[1][0]))[::-1]
        end = (int(end_point[0][0]), int(end_point[1][0]))[::-1]
        # Draw a line with arrow
        image = cv2.arrowedLine(image, start, end, color, thickness)
        return image
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
        y,x = curva.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x,y)).T
        isClosed=False
        img = cv2.polylines(img, [pts],
                              isClosed, color, thickness)

        return img

    @staticmethod
    def fill(poly, image, color, opacity=0):
        y, x = poly.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x, y)).T
        # Create a copy of the original image
        overlay = image.copy()

        # Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        return image

    @staticmethod
    def chain(chain, img, color=(0, 255, 0), thickness=5):
        y, x = chain.get_nodes_coordinates()
        pts = np.vstack((y, x)).T.astype(int)
        isClosed = False
        img = cv2.polylines(img, [pts],
                            isClosed, color, thickness)

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


def load_image(image_path):
    """Load image from path"""
    return cv2.imread(image_path)