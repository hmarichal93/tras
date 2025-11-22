"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2

from shapely.geometry import Polygon


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
        isClosed=True
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
    def fill_growth_ring(inner_ring: Polygon, outer_ring: Polygon, image: np.ndarray, color=(0, 255, 0),
                         background_color=(0, 0, 0)):
        """
        Fill the area between two polygons representing growth rings.
        ---usage
        from urudendro.labelme import AL_LateWood_EarlyWood

        al = AL_LateWood_EarlyWood(ann_path, None)
        shapes = al.read()
        #draw area
        inner_ring = shapes[3].points[:,[1, 0]]  # swap x and y coordinates
        poly_inner = Polygon(inner_ring)
        outer_ring = shapes[4].points[:,[1, 0]]  # swap x and y coordinates
        poly_outer = Polygon(outer_ring)
        #draw the area between the inner and outer polygons
        image_draw = Drawing.fill_growth_ring(poly_inner, poly_outer, image_draw, color=Color.red,
                                              background_color=Color.white)


        :param inner_ring: Polygon representing the inner ring.
        :param outer_ring: Polygon representing the outer ring.
        :param image: Image on which to draw the filled area.
        :param color: Color to fill the area with.
        :param thickness: Thickness of the fill. -1 means fill the polygon.
        :return: Image with filled area.
        """
        cv2.fillPoly(image, [np.array(outer_ring.exterior.coords).astype(np.int32)], color)
        cv2.fillPoly(image, [np.array(inner_ring.exterior.coords).astype(np.int32)], background_color)
        return image
