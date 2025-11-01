"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2



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
    def circle(image, center_coordinates,thickness=-1, color=Color.black, radius=3):
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
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
    def fill_region(exterior, interior, image, color):
        mask_exterior = np.zeros(image.shape[:2], dtype=np.uint8)
        exterior = np.array(exterior.get_nodes_coordinates()).astype(int)
        cv2.fillPoly(mask_exterior, [exterior.T], 255)
        mask_interior = np.zeros(image.shape[:2], dtype=np.uint8)
        interior = np.array(interior.get_nodes_coordinates()).astype(int)
        cv2.fillPoly(mask_interior, [interior.T], 255)
        mask = cv2.bitwise_xor(mask_exterior, mask_interior)
        image[mask == 255] = color
        return image

